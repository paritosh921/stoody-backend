"""Auditable cohort migrations for frozen PCR grading contracts.

The grading contract is immutable during ordinary processing.  A version
change therefore happens only through this module, for the complete exam
cohort, with optimistic concurrency and a durable audit record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


SOURCE_PROMPT_VERSION = "pcr-full-document-visual-v5"
TARGET_PROMPT_VERSION = "pcr-full-document-visual-v6"
EVIDENCE_GRAPH_PAPER_VERSION = "canonical-full-document-visual-v2"

EXAMS_COLLECTION = "exampen_exams"
SUBMISSIONS_COLLECTION = "evalpen_submissions"
PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
GRADING_RUNS_COLLECTION = "evalpen_document_grading_runs"
MIGRATIONS_COLLECTION = "exampen_grading_contract_migrations"
PAPER_VERSIONS_COLLECTION = "exampen_paper_versions"

ACTIVE_JOB_STATUSES = ("queued", "processing")


class GradingContractMigrationError(RuntimeError):
    """Raised when a cohort cannot be migrated without violating invariants."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _published_filter(exam_id: str) -> dict[str, Any]:
    return {
        "exam_id": exam_id,
        "$or": [
            {"publication_status": "published"},
            {"published_at": {"$exists": True, "$ne": None}},
        ],
    }


async def _validate_frozen_paper_version(
    tenant_db: Any,
    exam: dict[str, Any],
) -> list[str]:
    """Validate the immutable paper snapshot used by the grading worker."""

    paper_version_id = str(exam.get("paper_version_id") or "").strip()
    if not paper_version_id:
        return ["exam has no frozen paper version"]

    paper_version = await tenant_db[PAPER_VERSIONS_COLLECTION].find_one(
        {"paper_version_id": paper_version_id},
        {
            "_id": 0,
            "document_id": 1,
            "paper_context": 1,
            "paper_assets": 1,
        },
    )
    if not paper_version:
        return [f"frozen paper version {paper_version_id} was not found"]

    blockers: list[str] = []
    prepared_document_id = str(exam.get("prepared_document_id") or "").strip()
    version_document_id = str(paper_version.get("document_id") or "").strip()
    if prepared_document_id and version_document_id != prepared_document_id:
        blockers.append("frozen paper version does not belong to the prepared document")

    paper_context = dict(paper_version.get("paper_context") or {})
    if not paper_context.get("ready"):
        blockers.append("frozen paper context is not ready")
    if str(paper_context.get("version") or "") != EVIDENCE_GRAPH_PAPER_VERSION:
        blockers.append(
            "frozen paper context is not the canonical full-document evidence graph"
        )

    paper_assets = dict(paper_version.get("paper_assets") or {})
    question_asset = dict(paper_assets.get("question_paper") or {})
    if not question_asset.get("asset_id") or not question_asset.get("storage_uri"):
        blockers.append("frozen question-paper asset is unavailable")
    elif paper_context.get("question_paper_asset_id") != question_asset.get("asset_id"):
        blockers.append("frozen question-paper asset does not match its context")

    if paper_context.get("has_teacher_solution_asset"):
        solution_asset = dict(paper_assets.get("teacher_solution") or {})
        if not solution_asset.get("asset_id") or not solution_asset.get("storage_uri"):
            blockers.append("frozen teacher-solution asset is unavailable")
        elif paper_context.get("teacher_solution_asset_id") != solution_asset.get(
            "asset_id"
        ):
            blockers.append("frozen teacher-solution asset does not match its context")
    return blockers


async def inspect_v5_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {
        "pcr_grading_contract.prompt_version": SOURCE_PROMPT_VERSION,
    }
    if exam_id:
        query["exam_id"] = exam_id

    exams = await tenant_db[EXAMS_COLLECTION].find(
        query,
        {
            "_id": 0,
            "exam_id": 1,
            "exam_name": 1,
            "title": 1,
            "prepared_document_id": 1,
            "paper_version_id": 1,
            "pcr_grading_contract": 1,
        },
    ).to_list(length=None)

    plans: list[dict[str, Any]] = []
    for exam in exams:
        current_exam_id = str(exam.get("exam_id") or "")
        submission_count = await tenant_db[SUBMISSIONS_COLLECTION].count_documents(
            {"exam_id": current_exam_id}
        )
        published_count = await tenant_db[SUBMISSIONS_COLLECTION].count_documents(
            _published_filter(current_exam_id)
        )
        job_submission_ids = await tenant_db[PROCESSING_JOBS_COLLECTION].distinct(
            "submission_id", {"exam_id": current_exam_id}
        )
        active_job_count = await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
            {"exam_id": current_exam_id, "status": {"$in": list(ACTIVE_JOB_STATUSES)}}
        )
        missing_job_count = max(submission_count - len(job_submission_ids), 0)

        blockers = await _validate_frozen_paper_version(tenant_db, exam)
        if published_count:
            blockers.append(
                f"{published_count} published submission(s) require an explicit "
                "unpublish/regrade decision"
            )
        if active_job_count:
            blockers.append(
                f"{active_job_count} processing job(s) still own an active lease or queue slot"
            )
        if missing_job_count:
            blockers.append(
                f"{missing_job_count} submitted copy/copies have no durable processing job"
            )

        plans.append(
            {
                "db_name": db_name,
                "exam_id": current_exam_id,
                "exam_name": exam.get("exam_name") or exam.get("title") or current_exam_id,
                "source_prompt_version": SOURCE_PROMPT_VERSION,
                "target_prompt_version": TARGET_PROMPT_VERSION,
                "submission_count": submission_count,
                "published_count": published_count,
                "active_job_count": active_job_count,
                "missing_job_count": missing_job_count,
                "eligible": not blockers,
                "blockers": blockers,
            }
        )
    return plans


async def migrate_v5_exam_to_v6(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
) -> dict[str, Any]:
    """Migrate one complete unpublished cohort and queue exactly one new run per job."""

    exam = await tenant_db[EXAMS_COLLECTION].find_one({"exam_id": exam_id})
    if not exam:
        raise GradingContractMigrationError(f"Exam {exam_id} was not found in {db_name}")

    contract = dict(exam.get("pcr_grading_contract") or {})
    current_version = str(contract.get("prompt_version") or "")
    migration_state = dict(exam.get("pcr_grading_contract_migration") or {})
    if current_version == TARGET_PROMPT_VERSION and migration_state.get("status") == "complete":
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "already_migrated",
            "migration_id": migration_state.get("migration_id"),
        }
    if current_version not in {SOURCE_PROMPT_VERSION, TARGET_PROMPT_VERSION}:
        raise GradingContractMigrationError(
            f"Exam {exam_id} is locked to {current_version or 'no contract'}, not "
            f"{SOURCE_PROMPT_VERSION}"
        )

    plans = await inspect_v5_contracts(tenant_db, db_name=db_name, exam_id=exam_id)
    if current_version == SOURCE_PROMPT_VERSION:
        if not plans:
            raise GradingContractMigrationError(
                f"Exam {exam_id} no longer matches the expected source contract"
            )
        if not plans[0]["eligible"]:
            raise GradingContractMigrationError(
                f"Exam {exam_id} cannot be migrated: {'; '.join(plans[0]['blockers'])}"
            )
    else:
        active_jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
            {"exam_id": exam_id, "status": {"$in": list(ACTIVE_JOB_STATUSES)}}
        )
        if active_jobs:
            raise GradingContractMigrationError(
                f"Exam {exam_id} has {active_jobs} active processing job(s)"
            )

    now = _utcnow()
    migration_id = str(migration_state.get("migration_id") or f"PCR-MIG-{uuid4().hex}")
    audit = {
        "migration_id": migration_id,
        "db_name": db_name,
        "exam_id": exam_id,
        "source_prompt_version": SOURCE_PROMPT_VERSION,
        "target_prompt_version": TARGET_PROMPT_VERSION,
        "requested_by": requested_by,
        "started_at": migration_state.get("started_at") or now,
        "updated_at": now,
        "status": "applying",
    }
    await tenant_db[MIGRATIONS_COLLECTION].update_one(
        {"migration_id": migration_id},
        {"$set": audit, "$setOnInsert": {"contract_before": contract}},
        upsert=True,
    )

    try:
        if current_version == SOURCE_PROMPT_VERSION:
            migrated_contract = {
                **contract,
                "prompt_version": TARGET_PROMPT_VERSION,
                "migrated_from": SOURCE_PROMPT_VERSION,
                "migrated_at": now,
                "migration_id": migration_id,
            }
            updated = await tenant_db[EXAMS_COLLECTION].update_one(
                {
                    "exam_id": exam_id,
                    "paper_version_id": exam.get("paper_version_id"),
                    "prepared_document_id": exam.get("prepared_document_id"),
                    "pcr_grading_contract.prompt_version": SOURCE_PROMPT_VERSION,
                },
                {
                    "$set": {
                        "pcr_grading_contract": migrated_contract,
                        "pcr_grading_contract_migration": {
                            "migration_id": migration_id,
                            "source_prompt_version": SOURCE_PROMPT_VERSION,
                            "target_prompt_version": TARGET_PROMPT_VERSION,
                            "requested_by": requested_by,
                            "started_at": now,
                            "status": "applying",
                        },
                        "updated_at": now,
                    }
                },
            )
            if updated.modified_count != 1:
                raise GradingContractMigrationError(
                    f"Exam {exam_id} changed concurrently; no migration was applied"
                )

        superseded_runs = await tenant_db[GRADING_RUNS_COLLECTION].update_many(
            {
                "exam_id": exam_id,
                "prompt_version": SOURCE_PROMPT_VERSION,
                "status": {"$ne": "superseded"},
            },
            {
                "$set": {
                    "status": "superseded",
                    "superseded_at": now,
                    "superseded_reason": "grading_contract_migration",
                    "superseded_by_migration_id": migration_id,
                    "updated_at": now,
                }
            },
        )

        history_entry = {
            "requested_at": now,
            "requested_by": requested_by,
            "reason": "grading_contract_migration",
            "migration_id": migration_id,
        }
        queued_jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].update_many(
            {"exam_id": exam_id, "status": {"$nin": list(ACTIVE_JOB_STATUSES)}},
            {
                "$set": {
                    "status": "queued",
                    "attempts": 0,
                    "queued_at": now,
                    "updated_at": now,
                    "reprocess_requested_at": now,
                    "reprocess_requested_by": requested_by,
                    "reprocess_reason": "grading_contract_migration",
                    "required_processing_path": "full_document_visual",
                },
                "$unset": {
                    "last_error": "",
                    "failure_code": "",
                    "retry_at": "",
                    "next_retry_at": "",
                    "started_at": "",
                    "finished_at": "",
                    "lease_token": "",
                    "lease_owner": "",
                    "lease_expires_at": "",
                    "heartbeat_at": "",
                },
                "$inc": {"reprocess_count": 1},
                "$push": {
                    "reprocess_history": {"$each": [history_entry], "$slice": -20}
                },
            },
        )
        await tenant_db[SUBMISSIONS_COLLECTION].update_many(
            {"exam_id": exam_id},
            {
                "$set": {
                    "review_state": "processing",
                    "updated_at": now,
                    "grading_contract_migration_id": migration_id,
                }
            },
        )
        completed_at = _utcnow()
        completion = {
            "status": "complete",
            "completed_at": completed_at,
            "updated_at": completed_at,
            "queued_job_count": queued_jobs.modified_count,
            "superseded_run_count": superseded_runs.modified_count,
        }
        await tenant_db[EXAMS_COLLECTION].update_one(
            {"exam_id": exam_id, "pcr_grading_contract.migration_id": migration_id},
            {
                "$set": {
                    "pcr_grading_contract_migration.status": "complete",
                    "pcr_grading_contract_migration.completed_at": completed_at,
                    "updated_at": completed_at,
                }
            },
        )
        await tenant_db[MIGRATIONS_COLLECTION].update_one(
            {"migration_id": migration_id}, {"$set": completion}
        )
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "migrated",
            "migration_id": migration_id,
            "queued_job_count": queued_jobs.modified_count,
            "superseded_run_count": superseded_runs.modified_count,
        }
    except Exception as exc:
        failed_at = _utcnow()
        await tenant_db[MIGRATIONS_COLLECTION].update_one(
            {"migration_id": migration_id},
            {
                "$set": {
                    "status": "failed",
                    "failure_code": exc.__class__.__name__,
                    "failure_detail": str(exc),
                    "failed_at": failed_at,
                    "updated_at": failed_at,
                }
            },
        )
        await tenant_db[EXAMS_COLLECTION].update_one(
            {"exam_id": exam_id, "pcr_grading_contract.migration_id": migration_id},
            {
                "$set": {
                    "pcr_grading_contract_migration.status": "failed",
                    "pcr_grading_contract_migration.failure_code": exc.__class__.__name__,
                    "pcr_grading_contract_migration.failed_at": failed_at,
                    "updated_at": failed_at,
                }
            },
        )
        raise
