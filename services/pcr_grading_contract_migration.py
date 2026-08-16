"""Auditable cohort migrations for frozen PCR grading contracts.

The grading contract is immutable during ordinary processing.  A version
change therefore happens only through this module, for the complete exam
cohort, with optimistic concurrency and a durable audit record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from services.exampen_workflow import (
    CAPABILITY_QUEUED_JOB_STATUS,
    V14_CAPABILITY_QUEUED_JOB_STATUS,
    V15_CAPABILITY_QUEUED_JOB_STATUS,
    V16_CAPABILITY_QUEUED_JOB_STATUS,
)


V5_PROMPT_VERSION = "pcr-full-document-visual-v5"
V6_PROMPT_VERSION = "pcr-full-document-visual-v6"
V11_PROMPT_VERSION = "pcr-full-document-visual-v11"
V12_PROMPT_VERSION = "pcr-full-document-visual-v12"
V13_PROMPT_VERSION = "pcr-full-document-visual-v13"
V14_PROMPT_VERSION = "pcr-full-document-visual-v14"
V14_PIPELINE_VERSION = 5
V14_MAPPING_PIPELINE_VERSION = "bounded-evidence-visual-v5"
V14_REQUIRED_PROCESSING_PATH = "full_document_visual"
V15_PROMPT_VERSION = "pcr-full-document-visual-v15"
V15_PIPELINE_VERSION = 6
V15_MAPPING_PIPELINE_VERSION = "bounded-evidence-visual-v6"
V15_REQUIRED_PROCESSING_PATH = "full_document_visual"
V16_PROMPT_VERSION = "pcr-full-document-visual-v16"
V16_PIPELINE_VERSION = 7
V16_MAPPING_PIPELINE_VERSION = "whole-copy-rubric-v7"
V16_REQUIRED_PROCESSING_PATH = "full_document_visual"
# The apply path is intentionally impossible to invoke accidentally.  The
# operator-facing migration script uses the same value and requires it in
# addition to ``--apply``.
V13_TO_V14_CONFIRMATION_TOKEN = "MIGRATE_PCR_V13_TO_V14"
V14_TO_V15_CONFIRMATION_TOKEN = "MIGRATE_PCR_V14_TO_V15"
V15_TO_V16_CONFIRMATION_TOKEN = "MIGRATE_PCR_V15_TO_V16"
# Backward-compatible exports used by the existing v5 migration CLI.
SOURCE_PROMPT_VERSION = V5_PROMPT_VERSION
TARGET_PROMPT_VERSION = V6_PROMPT_VERSION
EVIDENCE_GRAPH_PAPER_VERSION = "canonical-full-document-visual-v2"
PIPELINE_VERSION_BY_PROMPT = {
    V5_PROMPT_VERSION: 2,
    V6_PROMPT_VERSION: 2,
    V11_PROMPT_VERSION: 3,
    V12_PROMPT_VERSION: 3,
    V13_PROMPT_VERSION: 4,
    V14_PROMPT_VERSION: V14_PIPELINE_VERSION,
    V15_PROMPT_VERSION: V15_PIPELINE_VERSION,
    V16_PROMPT_VERSION: V16_PIPELINE_VERSION,
}
MAPPING_PIPELINE_BY_PROMPT = {
    V11_PROMPT_VERSION: "whole-copy-rubric-v3",
    V12_PROMPT_VERSION: "whole-copy-rubric-v3",
    V13_PROMPT_VERSION: "evidence-first-visual-v4",
    V14_PROMPT_VERSION: V14_MAPPING_PIPELINE_VERSION,
    V15_PROMPT_VERSION: V15_MAPPING_PIPELINE_VERSION,
    V16_PROMPT_VERSION: V16_MAPPING_PIPELINE_VERSION,
}

EXAMS_COLLECTION = "exampen_exams"
SUBMISSIONS_COLLECTION = "evalpen_submissions"
PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
GRADING_RUNS_COLLECTION = "evalpen_document_grading_runs"
MIGRATIONS_COLLECTION = "exampen_grading_contract_migrations"
PAPER_VERSIONS_COLLECTION = "exampen_paper_versions"

ACTIVE_JOB_STATUSES = (
    "queued",
    "processing",
    "retryable_error",
    "enqueue_failed",
    CAPABILITY_QUEUED_JOB_STATUS,
    V14_CAPABILITY_QUEUED_JOB_STATUS,
    V15_CAPABILITY_QUEUED_JOB_STATUS,
    V16_CAPABILITY_QUEUED_JOB_STATUS,
)


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


async def _inspect_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    source_prompt_version: str,
    target_prompt_version: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {
        "pcr_grading_contract.prompt_version": source_prompt_version,
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
                "source_prompt_version": source_prompt_version,
                "target_prompt_version": target_prompt_version,
                "submission_count": submission_count,
                "published_count": published_count,
                "active_job_count": active_job_count,
                "missing_job_count": missing_job_count,
                "eligible": not blockers,
                "blockers": blockers,
            }
        )
    return plans


async def _inspect_v13_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect v13 cohorts with the stricter v14 publication/job gates.

    The legacy inspection functions intentionally retain their historical
    shape.  v14 adds a duplicate-job check and rejects an already active exam
    because changing a frozen contract while the exam is being consumed would
    produce a mixed cohort.
    """

    plans = await _inspect_contracts(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        source_prompt_version=V13_PROMPT_VERSION,
        target_prompt_version=V14_PROMPT_VERSION,
    )
    if not plans:
        return plans
    for plan in plans:
        exam = await tenant_db[EXAMS_COLLECTION].find_one(
            {"exam_id": plan["exam_id"]},
            {"_id": 0, "status": 1, "publication_status": 1},
        )
        exam_status = str((exam or {}).get("status") or "").strip().lower()
        publication_status = str(
            (exam or {}).get("publication_status") or ""
        ).strip().lower()
        if exam_status in {"active", "in_progress", "published"} or publication_status in {
            "active",
            "in_progress",
            "published",
        }:
            plan["blockers"].append(
                f"exam is {exam_status or publication_status} and cannot be migrated live"
            )

        submission_ids = await tenant_db[SUBMISSIONS_COLLECTION].distinct(
            "submission_id", {"exam_id": plan["exam_id"]}
        )
        duplicate_job_count = 0
        for submission_id in submission_ids:
            count = await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
                {"exam_id": plan["exam_id"], "submission_id": submission_id}
            )
            if count != 1:
                duplicate_job_count += max(count - 1, 1)
        plan["duplicate_job_count"] = duplicate_job_count
        if duplicate_job_count:
            plan["blockers"].append(
                f"{duplicate_job_count} duplicate/missing submission job(s) prevent idempotent queueing"
            )
        plan["eligible"] = not plan["blockers"]
    return plans


async def _migrate_v13_exam_to_v14(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
) -> dict[str, Any]:
    """Apply the guarded v13 -> v14 migration.

    This deliberately uses one ``update_one`` per submission job.  A job is
    the idempotency boundary: no job is inserted, and a retry with the same
    migration id cannot create a second queue item or increment its attempt
    history twice.
    """

    exam = await tenant_db[EXAMS_COLLECTION].find_one({"exam_id": exam_id})
    if not exam:
        raise GradingContractMigrationError(f"Exam {exam_id} was not found in {db_name}")
    contract = dict(exam.get("pcr_grading_contract") or {})
    current_version = str(contract.get("prompt_version") or "")
    migration_state = dict(exam.get("pcr_grading_contract_migration") or {})
    resuming = (
        current_version == V14_PROMPT_VERSION
        and migration_state.get("status") in {"applying", "failed"}
    )
    if (
        current_version == V14_PROMPT_VERSION
        and migration_state.get("status") == "complete"
    ):
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "already_migrated",
            "migration_id": migration_state.get("migration_id"),
            "queued_job_count": int(migration_state.get("queued_job_count") or 0),
        }
    if current_version != V13_PROMPT_VERSION and not resuming:
        raise GradingContractMigrationError(
            f"Exam {exam_id} is locked to {current_version or 'no contract'}, not {V13_PROMPT_VERSION}"
        )

    if resuming:
        active_jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
            {"exam_id": exam_id, "status": {"$in": list(ACTIVE_JOB_STATUSES)}}
        )
        if active_jobs:
            raise GradingContractMigrationError(
                f"Exam {exam_id} has {active_jobs} active processing job(s) while resuming"
            )
    else:
        plans = await _inspect_v13_contracts(tenant_db, db_name=db_name, exam_id=exam_id)
        if not plans or not plans[0]["eligible"]:
            blockers = plans[0]["blockers"] if plans else ["exam no longer matches the source contract"]
            raise GradingContractMigrationError(
                f"Exam {exam_id} cannot be migrated: {'; '.join(blockers)}"
            )

    now = _utcnow()
    migration_id = str(migration_state.get("migration_id") or "").strip() or f"PCR-MIG-{uuid4().hex}"
    await tenant_db[MIGRATIONS_COLLECTION].update_one(
        {"migration_id": migration_id},
        {
            "$set": {
                "migration_id": migration_id,
                "db_name": db_name,
                "exam_id": exam_id,
                "source_prompt_version": V13_PROMPT_VERSION,
                "target_prompt_version": V14_PROMPT_VERSION,
                "requested_by": requested_by,
                "status": "applying",
                "updated_at": now,
            },
            "$setOnInsert": {"started_at": now, "contract_before": contract},
        },
        upsert=True,
    )
    try:
        migrated_contract = {
            **contract,
            "prompt_version": V14_PROMPT_VERSION,
            "pipeline_version": V14_PIPELINE_VERSION,
            "mapping_pipeline_version": V14_MAPPING_PIPELINE_VERSION,
            "required_processing_path": V14_REQUIRED_PROCESSING_PATH,
            "migrated_from": V13_PROMPT_VERSION,
            "migrated_at": now,
            "migration_id": migration_id,
        }
        if not resuming:
            updated = await tenant_db[EXAMS_COLLECTION].update_one(
                {
                    "exam_id": exam_id,
                    "paper_version_id": exam.get("paper_version_id"),
                    "prepared_document_id": exam.get("prepared_document_id"),
                    "pcr_grading_contract.prompt_version": V13_PROMPT_VERSION,
                },
                {
                    "$set": {
                        "pcr_grading_contract": migrated_contract,
                        "pcr_grading_contract_migration": {
                            "migration_id": migration_id,
                            "source_prompt_version": V13_PROMPT_VERSION,
                            "target_prompt_version": V14_PROMPT_VERSION,
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

        # Existing v13 runs are no longer valid once the contract changes.
        superseded_runs = await tenant_db[GRADING_RUNS_COLLECTION].update_many(
            {
                "exam_id": exam_id,
                "$or": [
                    {"prompt_version": V13_PROMPT_VERSION},
                    {"pipeline_version": PIPELINE_VERSION_BY_PROMPT[V13_PROMPT_VERSION]},
                    {"mapping_pipeline_version": MAPPING_PIPELINE_BY_PROMPT[V13_PROMPT_VERSION]},
                ],
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

        submission_ids = await tenant_db[SUBMISSIONS_COLLECTION].distinct(
            "submission_id", {"exam_id": exam_id}
        )
        queued_job_count = 0
        for submission_id in submission_ids:
            job = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
                {"exam_id": exam_id, "submission_id": submission_id},
                {"_id": 0, "job_id": 1, "migration_id": 1, "status": 1},
            )
            if not job:
                raise GradingContractMigrationError(
                    f"Submission {submission_id} has no durable processing job"
                )
            if job.get("migration_id") == migration_id and job.get("status") in {
                V14_CAPABILITY_QUEUED_JOB_STATUS,
                "queued",
            }:
                queued_job_count += 1
                continue
            result = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
                {
                    "job_id": job.get("job_id"),
                    "exam_id": exam_id,
                    "submission_id": submission_id,
                    "status": {"$nin": list(ACTIVE_JOB_STATUSES)},
                },
                {
                    "$set": {
                        "status": V14_CAPABILITY_QUEUED_JOB_STATUS,
                        "pipeline_version": V14_PIPELINE_VERSION,
                        "mapping_pipeline_version": V14_MAPPING_PIPELINE_VERSION,
                        "required_processing_path": V14_REQUIRED_PROCESSING_PATH,
                        "migration_id": migration_id,
                        "attempts": 0,
                        "queued_at": now,
                        "updated_at": now,
                        "reprocess_requested_at": now,
                        "reprocess_requested_by": requested_by,
                        "reprocess_reason": "grading_contract_migration",
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
                        "reprocess_history": {
                            "$each": [
                                {
                                    "requested_at": now,
                                    "requested_by": requested_by,
                                    "reason": "grading_contract_migration",
                                    "migration_id": migration_id,
                                }
                            ],
                            "$slice": -20,
                        }
                    },
                },
            )
            if result.modified_count == 1:
                queued_job_count += 1

        if queued_job_count != len(submission_ids):
            raise GradingContractMigrationError(
                f"Expected exactly one queued job per submission ({len(submission_ids)}), "
                f"but only observed {queued_job_count}"
            )

        completed_at = _utcnow()
        await tenant_db[SUBMISSIONS_COLLECTION].update_many(
            {"exam_id": exam_id},
            {"$set": {"review_state": "processing", "updated_at": completed_at, "grading_contract_migration_id": migration_id}},
        )
        completion = {
            "status": "complete",
            "completed_at": completed_at,
            "updated_at": completed_at,
            "queued_job_count": queued_job_count,
            "superseded_run_count": superseded_runs.modified_count,
        }
        await tenant_db[EXAMS_COLLECTION].update_one(
            {"exam_id": exam_id, "pcr_grading_contract.migration_id": migration_id},
            {"$set": {"pcr_grading_contract_migration": {**completion, "migration_id": migration_id, "source_prompt_version": V13_PROMPT_VERSION, "target_prompt_version": V14_PROMPT_VERSION, "requested_by": requested_by}, "updated_at": completed_at}},
        )
        await tenant_db[MIGRATIONS_COLLECTION].update_one(
            {"migration_id": migration_id}, {"$set": completion}
        )
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "migrated",
            "migration_id": migration_id,
            "queued_job_count": queued_job_count,
            "superseded_run_count": superseded_runs.modified_count,
        }
    except Exception as exc:
        failed_at = _utcnow()
        await tenant_db[MIGRATIONS_COLLECTION].update_one(
            {"migration_id": migration_id},
            {"$set": {"status": "failed", "failure_code": exc.__class__.__name__, "failure_detail": str(exc), "failed_at": failed_at, "updated_at": failed_at}},
        )
        raise


async def _inspect_v14_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect v14 cohorts before the isolated v15 worker rollout."""

    plans = await _inspect_contracts(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        source_prompt_version=V14_PROMPT_VERSION,
        target_prompt_version=V15_PROMPT_VERSION,
    )
    for plan in plans:
        exam = await tenant_db[EXAMS_COLLECTION].find_one(
            {"exam_id": plan["exam_id"]},
            {"_id": 0, "status": 1, "publication_status": 1},
        )
        exam_status = str((exam or {}).get("status") or "").strip().lower()
        publication_status = str((exam or {}).get("publication_status") or "").strip().lower()
        if exam_status in {"active", "in_progress", "published"} or publication_status in {
            "active", "in_progress", "published"
        }:
            plan["blockers"].append(
                f"exam is {exam_status or publication_status} and cannot be migrated live"
            )
        submission_ids = await tenant_db[SUBMISSIONS_COLLECTION].distinct(
            "submission_id", {"exam_id": plan["exam_id"]}
        )
        duplicate_job_count = 0
        for submission_id in submission_ids:
            count = await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
                {"exam_id": plan["exam_id"], "submission_id": submission_id}
            )
            if count != 1:
                duplicate_job_count += max(count - 1, 1)
        plan["duplicate_job_count"] = duplicate_job_count
        if duplicate_job_count:
            plan["blockers"].append(
                f"{duplicate_job_count} duplicate/missing submission job(s) prevent idempotent queueing"
            )
        plan["eligible"] = not plan["blockers"]
    return plans


async def _migrate_v14_exam_to_v15(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
) -> dict[str, Any]:
    """Apply v14 -> v15 without changing the v14 mapper/evidence contract."""

    exam = await tenant_db[EXAMS_COLLECTION].find_one({"exam_id": exam_id})
    if not exam:
        raise GradingContractMigrationError(f"Exam {exam_id} was not found in {db_name}")
    contract = dict(exam.get("pcr_grading_contract") or {})
    current_version = str(contract.get("prompt_version") or "")
    migration_state = dict(exam.get("pcr_grading_contract_migration") or {})
    resuming = current_version == V15_PROMPT_VERSION and migration_state.get("status") in {"applying", "failed"}
    if current_version == V15_PROMPT_VERSION and migration_state.get("status") == "complete":
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "already_migrated",
            "migration_id": migration_state.get("migration_id"),
            "queued_job_count": int(migration_state.get("queued_job_count") or 0),
        }
    if current_version != V14_PROMPT_VERSION and not resuming:
        raise GradingContractMigrationError(
            f"Exam {exam_id} is locked to {current_version or 'no contract'}, not {V14_PROMPT_VERSION}"
        )
    if resuming:
        active_jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
            {"exam_id": exam_id, "status": {"$in": list(ACTIVE_JOB_STATUSES)}}
        )
        if active_jobs:
            raise GradingContractMigrationError(
                f"Exam {exam_id} has {active_jobs} active processing job(s) while resuming"
            )
    else:
        plans = await _inspect_v14_contracts(tenant_db, db_name=db_name, exam_id=exam_id)
        if not plans or not plans[0]["eligible"]:
            blockers = plans[0]["blockers"] if plans else ["exam no longer matches the source contract"]
            raise GradingContractMigrationError(
                f"Exam {exam_id} cannot be migrated: {'; '.join(blockers)}"
            )

    now = _utcnow()
    migration_id = str(migration_state.get("migration_id") or "").strip() or f"PCR-MIG-{uuid4().hex}"
    await tenant_db[MIGRATIONS_COLLECTION].update_one(
        {"migration_id": migration_id},
        {
            "$set": {
                "migration_id": migration_id,
                "db_name": db_name,
                "exam_id": exam_id,
                "source_prompt_version": V14_PROMPT_VERSION,
                "target_prompt_version": V15_PROMPT_VERSION,
                "requested_by": requested_by,
                "status": "applying",
                "updated_at": now,
            },
            "$setOnInsert": {"started_at": now, "contract_before": contract},
        },
        upsert=True,
    )
    try:
        migrated_contract = {
            **contract,
            "prompt_version": V15_PROMPT_VERSION,
            "pipeline_version": V15_PIPELINE_VERSION,
            "mapping_pipeline_version": V15_MAPPING_PIPELINE_VERSION,
            "required_processing_path": V15_REQUIRED_PROCESSING_PATH,
            "migrated_from": V14_PROMPT_VERSION,
            "migrated_at": now,
            "migration_id": migration_id,
        }
        if not resuming:
            updated = await tenant_db[EXAMS_COLLECTION].update_one(
                {
                    "exam_id": exam_id,
                    "paper_version_id": exam.get("paper_version_id"),
                    "prepared_document_id": exam.get("prepared_document_id"),
                    "pcr_grading_contract.prompt_version": V14_PROMPT_VERSION,
                },
                {
                    "$set": {
                        "pcr_grading_contract": migrated_contract,
                        "pcr_grading_contract_migration": {
                            "migration_id": migration_id,
                            "source_prompt_version": V14_PROMPT_VERSION,
                            "target_prompt_version": V15_PROMPT_VERSION,
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
                "$or": [
                    {"prompt_version": V14_PROMPT_VERSION},
                    {"pipeline_version": V14_PIPELINE_VERSION},
                    {"mapping_pipeline_version": V14_MAPPING_PIPELINE_VERSION},
                ],
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

        submission_ids = await tenant_db[SUBMISSIONS_COLLECTION].distinct(
            "submission_id", {"exam_id": exam_id}
        )
        queued_job_count = 0
        for submission_id in submission_ids:
            job = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
                {"exam_id": exam_id, "submission_id": submission_id},
                {"_id": 0, "job_id": 1, "migration_id": 1, "status": 1},
            )
            if not job:
                raise GradingContractMigrationError(
                    f"Submission {submission_id} has no durable processing job"
                )
            if job.get("migration_id") == migration_id and job.get("status") in {
                V15_CAPABILITY_QUEUED_JOB_STATUS,
                "queued",
            }:
                queued_job_count += 1
                continue
            result = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
                {
                    "job_id": job.get("job_id"),
                    "exam_id": exam_id,
                    "submission_id": submission_id,
                    "status": {"$nin": list(ACTIVE_JOB_STATUSES)},
                },
                {
                    "$set": {
                        "status": V15_CAPABILITY_QUEUED_JOB_STATUS,
                        "pipeline_version": V15_PIPELINE_VERSION,
                        "mapping_pipeline_version": V15_MAPPING_PIPELINE_VERSION,
                        "required_processing_path": V15_REQUIRED_PROCESSING_PATH,
                        "migration_id": migration_id,
                        "attempts": 0,
                        "queued_at": now,
                        "updated_at": now,
                        "reprocess_requested_at": now,
                        "reprocess_requested_by": requested_by,
                        "reprocess_reason": "grading_contract_migration",
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
                        "reprocess_history": {
                            "$each": [{
                                "requested_at": now,
                                "requested_by": requested_by,
                                "reason": "grading_contract_migration",
                                "migration_id": migration_id,
                            }],
                            "$slice": -20,
                        }
                    },
                },
            )
            if result.modified_count == 1:
                queued_job_count += 1
        if queued_job_count != len(submission_ids):
            raise GradingContractMigrationError(
                f"Expected exactly one queued job per submission ({len(submission_ids)}), but only observed {queued_job_count}"
            )

        completed_at = _utcnow()
        await tenant_db[SUBMISSIONS_COLLECTION].update_many(
            {"exam_id": exam_id},
            {"$set": {
                "review_state": "processing",
                "updated_at": completed_at,
                "grading_contract_migration_id": migration_id,
            }},
        )
        completion = {
            "status": "complete",
            "completed_at": completed_at,
            "updated_at": completed_at,
            "queued_job_count": queued_job_count,
            "superseded_run_count": superseded_runs.modified_count,
        }
        await tenant_db[EXAMS_COLLECTION].update_one(
            {"exam_id": exam_id, "pcr_grading_contract.migration_id": migration_id},
            {"$set": {"pcr_grading_contract_migration": {
                **completion,
                "migration_id": migration_id,
                "source_prompt_version": V14_PROMPT_VERSION,
                "target_prompt_version": V15_PROMPT_VERSION,
                "requested_by": requested_by,
            }, "updated_at": completed_at}},
        )
        await tenant_db[MIGRATIONS_COLLECTION].update_one(
            {"migration_id": migration_id}, {"$set": completion}
        )
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "migrated",
            "migration_id": migration_id,
            "queued_job_count": queued_job_count,
            "superseded_run_count": superseded_runs.modified_count,
        }
    except Exception as exc:
        failed_at = _utcnow()
        await tenant_db[MIGRATIONS_COLLECTION].update_one(
            {"migration_id": migration_id},
            {"$set": {
                "status": "failed",
                "failure_code": exc.__class__.__name__,
                "failure_detail": str(exc),
                "failed_at": failed_at,
                "updated_at": failed_at,
            }},
        )
        raise


async def _migrate_exam_contract(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
    source_prompt_version: str,
    target_prompt_version: str,
) -> dict[str, Any]:
    """Migrate one complete unpublished cohort and queue exactly one new run per job."""

    exam = await tenant_db[EXAMS_COLLECTION].find_one({"exam_id": exam_id})
    if not exam:
        raise GradingContractMigrationError(f"Exam {exam_id} was not found in {db_name}")

    contract = dict(exam.get("pcr_grading_contract") or {})
    current_version = str(contract.get("prompt_version") or "")
    migration_state = dict(exam.get("pcr_grading_contract_migration") or {})
    if current_version == target_prompt_version and migration_state.get("status") == "complete":
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "already_migrated",
            "migration_id": migration_state.get("migration_id"),
        }
    if current_version not in {source_prompt_version, target_prompt_version}:
        raise GradingContractMigrationError(
            f"Exam {exam_id} is locked to {current_version or 'no contract'}, not "
            f"{source_prompt_version}"
        )

    plans = await _inspect_contracts(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        source_prompt_version=source_prompt_version,
        target_prompt_version=target_prompt_version,
    )
    if current_version == source_prompt_version:
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
    previous_migration_id = str(migration_state.get("migration_id") or "").strip()
    migration_state_matches_pair = (
        str(migration_state.get("source_prompt_version") or "")
        == source_prompt_version
        and str(migration_state.get("target_prompt_version") or "")
        == target_prompt_version
    )
    migration_id = (
        previous_migration_id
        if migration_state_matches_pair and previous_migration_id
        else f"PCR-MIG-{uuid4().hex}"
    )
    audit = {
        "migration_id": migration_id,
        "db_name": db_name,
        "exam_id": exam_id,
        "source_prompt_version": source_prompt_version,
        "target_prompt_version": target_prompt_version,
        "requested_by": requested_by,
        "started_at": (
            migration_state.get("started_at")
            if migration_state_matches_pair and migration_state.get("started_at")
            else now
        ),
        "updated_at": now,
        "status": "applying",
        **(
            {"previous_migration_id": previous_migration_id}
            if previous_migration_id and previous_migration_id != migration_id
            else {}
        ),
    }
    await tenant_db[MIGRATIONS_COLLECTION].update_one(
        {"migration_id": migration_id},
        {"$set": audit, "$setOnInsert": {"contract_before": contract}},
        upsert=True,
    )

    try:
        if current_version == source_prompt_version:
            target_pipeline_version = PIPELINE_VERSION_BY_PROMPT[target_prompt_version]
            migrated_contract = {
                **contract,
                "prompt_version": target_prompt_version,
                "pipeline_version": target_pipeline_version,
                "mapping_pipeline_version": MAPPING_PIPELINE_BY_PROMPT.get(
                    target_prompt_version,
                    f"grading-contract-{target_pipeline_version}",
                ),
                "required_processing_path": "full_document_visual",
                "migrated_from": source_prompt_version,
                "migrated_at": now,
                "migration_id": migration_id,
            }
            updated = await tenant_db[EXAMS_COLLECTION].update_one(
                {
                    "exam_id": exam_id,
                    "paper_version_id": exam.get("paper_version_id"),
                    "prepared_document_id": exam.get("prepared_document_id"),
                    "pcr_grading_contract.prompt_version": source_prompt_version,
                },
                {
                    "$set": {
                        "pcr_grading_contract": migrated_contract,
                        "pcr_grading_contract_migration": {
                            "migration_id": migration_id,
                            "source_prompt_version": source_prompt_version,
                            "target_prompt_version": target_prompt_version,
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
                "prompt_version": source_prompt_version,
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
        target_pipeline_version = PIPELINE_VERSION_BY_PROMPT[target_prompt_version]
        queued_status = (
            V16_CAPABILITY_QUEUED_JOB_STATUS
            if target_pipeline_version >= V16_PIPELINE_VERSION
            else V15_CAPABILITY_QUEUED_JOB_STATUS
            if target_pipeline_version >= V15_PIPELINE_VERSION
            else V14_CAPABILITY_QUEUED_JOB_STATUS
            if target_pipeline_version >= V14_PIPELINE_VERSION
            else CAPABILITY_QUEUED_JOB_STATUS
            if target_pipeline_version >= 3
            else "queued"
        )
        queued_jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].update_many(
            {"exam_id": exam_id, "status": {"$nin": list(ACTIVE_JOB_STATUSES)}},
            {
                "$set": {
                    "status": queued_status,
                    "pipeline_version": target_pipeline_version,
                    "mapping_pipeline_version": MAPPING_PIPELINE_BY_PROMPT.get(
                        target_prompt_version,
                        f"grading-contract-{target_pipeline_version}",
                    ),
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


async def inspect_v5_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect legacy v5 cohorts for the established v5-to-v6 migration."""

    return await _inspect_contracts(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        source_prompt_version=V5_PROMPT_VERSION,
        target_prompt_version=V6_PROMPT_VERSION,
    )


async def migrate_v5_exam_to_v6(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
) -> dict[str, Any]:
    return await _migrate_exam_contract(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        requested_by=requested_by,
        source_prompt_version=V5_PROMPT_VERSION,
        target_prompt_version=V6_PROMPT_VERSION,
    )


async def inspect_v11_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect cohorts that need the bounded whole-copy visual grader v12."""

    return await _inspect_contracts(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        source_prompt_version=V11_PROMPT_VERSION,
        target_prompt_version=V12_PROMPT_VERSION,
    )


async def migrate_v11_exam_to_v12(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
) -> dict[str, Any]:
    """Atomically migrate and requeue one complete unpublished v11 cohort."""

    return await _migrate_exam_contract(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        requested_by=requested_by,
        source_prompt_version=V11_PROMPT_VERSION,
        target_prompt_version=V12_PROMPT_VERSION,
    )


async def inspect_v12_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect frozen v12 cohorts eligible for the grounded visual grader v13."""

    return await _inspect_contracts(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        source_prompt_version=V12_PROMPT_VERSION,
        target_prompt_version=V13_PROMPT_VERSION,
    )


async def migrate_v12_exam_to_v13(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
) -> dict[str, Any]:
    """Atomically migrate and requeue one complete unpublished v12 cohort."""

    return await _migrate_exam_contract(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        requested_by=requested_by,
        source_prompt_version=V12_PROMPT_VERSION,
        target_prompt_version=V13_PROMPT_VERSION,
    )


async def inspect_v13_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect frozen v13 cohorts before the bounded-evidence v14 rollout."""

    return await _inspect_v13_contracts(tenant_db, db_name=db_name, exam_id=exam_id)


async def migrate_v13_exam_to_v14(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """Apply v13 -> v14 only with the exact operator confirmation token.

    There is intentionally no default for ``confirmation_token``.  Callers
    must make the destructive contract change explicit, while the CLI remains
    dry-run by default.
    """

    if confirmation_token != V13_TO_V14_CONFIRMATION_TOKEN:
        raise GradingContractMigrationError(
            f"v13 to v14 migration requires confirmation token {V13_TO_V14_CONFIRMATION_TOKEN}"
        )
    return await _migrate_v13_exam_to_v14(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        requested_by=requested_by,
    )


async def inspect_v14_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect frozen v14 cohorts before the v15 pipeline rollout."""

    return await _inspect_v14_contracts(tenant_db, db_name=db_name, exam_id=exam_id)


async def migrate_v14_exam_to_v15(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """Apply v14 -> v15 only with the exact operator confirmation token."""

    if confirmation_token != V14_TO_V15_CONFIRMATION_TOKEN:
        raise GradingContractMigrationError(
            f"v14 to v15 migration requires confirmation token {V14_TO_V15_CONFIRMATION_TOKEN}"
        )
    return await _migrate_v14_exam_to_v15(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        requested_by=requested_by,
    )


async def inspect_v15_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect frozen v15 cohorts before the holistic v16 rollout."""

    return await _inspect_contracts(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        source_prompt_version=V15_PROMPT_VERSION,
        target_prompt_version=V16_PROMPT_VERSION,
    )


async def migrate_v15_exam_to_v16(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """Apply v15 -> v16 only with the exact operator confirmation token."""

    if confirmation_token != V15_TO_V16_CONFIRMATION_TOKEN:
        raise GradingContractMigrationError(
            "v15 to v16 migration requires confirmation token "
            f"{V15_TO_V16_CONFIRMATION_TOKEN}"
        )
    return await _migrate_exam_contract(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        requested_by=requested_by,
        source_prompt_version=V15_PROMPT_VERSION,
        target_prompt_version=V16_PROMPT_VERSION,
    )
