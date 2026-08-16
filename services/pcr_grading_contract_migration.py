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
    CONTRACT_MIGRATION_PENDING_JOB_STATUS,
    V14_CAPABILITY_QUEUED_JOB_STATUS,
    V15_CAPABILITY_QUEUED_JOB_STATUS,
    V16_CAPABILITY_QUEUED_JOB_STATUS,
)


V4_PROMPT_VERSION = "pcr-full-document-visual-v4"
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
LEGACY_TO_V16_CONFIRMATION_TOKEN = "MIGRATE_PCR_LEGACY_TO_V16"
LEGACY_V16_SOURCE_PROMPT_VERSIONS = (
    V4_PROMPT_VERSION,
    V5_PROMPT_VERSION,
    V6_PROMPT_VERSION,
    V11_PROMPT_VERSION,
    V12_PROMPT_VERSION,
    V13_PROMPT_VERSION,
    V14_PROMPT_VERSION,
    V15_PROMPT_VERSION,
)
LEGACY_V16_PIPELINE_VERSIONS = (1, 2, 3, 4, 5, 6)
LEGACY_V16_MAPPING_PIPELINE_VERSIONS = (
    "full-document-visual-v2",
    "whole-copy-rubric-v3",
    "evidence-first-visual-v4",
    V14_MAPPING_PIPELINE_VERSION,
    V15_MAPPING_PIPELINE_VERSION,
)
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


def _is_explicit_objective(payload: Any) -> bool:
    """Recognize objective cohorts without treating missing legacy fields as objective."""

    item = payload if isinstance(payload, dict) else {}
    values = {
        str(item.get(field) or "").strip().lower()
        for field in ("grading_mode", "marking_mode", "assessment_mode")
    }
    return bool(values & {"objective", "omr", "mcq"})


async def _inspect_legacy_v16_exam(
    tenant_db: Any,
    *,
    db_name: str,
    exam: dict[str, Any],
    allowed_migration_id: str | None = None,
) -> dict[str, Any]:
    """Build one fail-closed whole-cohort plan for a legacy subjective exam."""

    current_exam_id = str(exam.get("exam_id") or "").strip()
    contract = dict(exam.get("pcr_grading_contract") or {})
    source_prompt_version = str(contract.get("prompt_version") or "").strip()
    blockers = await _validate_frozen_paper_version(tenant_db, exam)

    exam_type = str(exam.get("exam_type") or "").strip().lower()
    if exam_type and exam_type != "pcr":
        blockers.append(f"exam type {exam_type} is not PCR")
    if _is_explicit_objective(exam) or _is_explicit_objective(contract):
        blockers.append("objective PCR cohorts use a separate grading contract")

    exam_status = str(exam.get("status") or "").strip().lower()
    publication_status = str(exam.get("publication_status") or "").strip().lower()
    if exam_status in {"active", "in_progress", "published"}:
        blockers.append(f"exam is {exam_status} and cannot be migrated live")
    elif publication_status in {"active", "in_progress", "published"}:
        blockers.append(
            f"exam is {publication_status} and cannot be migrated live"
        )

    migration_state = dict(exam.get("pcr_grading_contract_migration") or {})
    applying_migration_id = str(migration_state.get("migration_id") or "").strip()
    if (
        migration_state.get("status") == "applying"
        and applying_migration_id != str(allowed_migration_id or "").strip()
    ):
        blockers.append(
            f"grading contract migration {applying_migration_id or 'unknown'} is already applying"
        )

    submissions = await tenant_db[SUBMISSIONS_COLLECTION].find(
        {"exam_id": current_exam_id},
        {
            "_id": 0,
            "submission_id": 1,
            "grading_mode": 1,
            "marking_mode": 1,
            "assessment_mode": 1,
            "publication_status": 1,
            "published_at": 1,
        },
    ).to_list(length=None)
    jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].find(
        {"exam_id": current_exam_id},
        {
            "_id": 0,
            "job_id": 1,
            "submission_id": 1,
            "status": 1,
            "pipeline_version": 1,
            "mapping_pipeline_version": 1,
            "migration_id": 1,
        },
    ).to_list(length=None)

    submission_ids = [
        str(item.get("submission_id") or "").strip() for item in submissions
    ]
    valid_submission_ids = {value for value in submission_ids if value}
    published_count = sum(
        1
        for item in submissions
        if str(item.get("publication_status") or "").strip().lower() == "published"
        or item.get("published_at") is not None
    )
    objective_submission_count = sum(
        1 for item in submissions if _is_explicit_objective(item)
    )
    if published_count:
        blockers.append(
            f"{published_count} published submission(s) require an explicit unpublish/regrade decision"
        )
    if objective_submission_count:
        blockers.append(
            f"{objective_submission_count} objective submission(s) cannot enter the subjective migration"
        )

    jobs_by_submission: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        submission_id = str(job.get("submission_id") or "").strip()
        jobs_by_submission.setdefault(submission_id, []).append(job)

    missing_job_count = sum(
        1
        for submission_id in submission_ids
        if not submission_id or len(jobs_by_submission.get(submission_id, [])) == 0
    )
    duplicate_job_count = sum(
        max(len(jobs_by_submission.get(submission_id, [])) - 1, 0)
        for submission_id in valid_submission_ids
    )
    orphan_job_count = sum(
        len(group)
        for submission_id, group in jobs_by_submission.items()
        if not submission_id or submission_id not in valid_submission_ids
    )
    invalid_job_id_count = sum(1 for job in jobs if not str(job.get("job_id") or "").strip())
    active_job_count = sum(
        1
        for job in jobs
        if str(job.get("status") or "").strip().lower() in ACTIVE_JOB_STATUSES
    )

    mixed_job_count = 0
    for job in jobs:
        pipeline_value = job.get("pipeline_version")
        try:
            pipeline_version = int(pipeline_value) if pipeline_value is not None else None
        except (TypeError, ValueError):
            pipeline_version = -1
        mapping_version = str(job.get("mapping_pipeline_version") or "").strip()
        if (
            pipeline_version == V16_PIPELINE_VERSION
            or mapping_version == V16_MAPPING_PIPELINE_VERSION
            or pipeline_version not in {None, *LEGACY_V16_PIPELINE_VERSIONS}
        ):
            mixed_job_count += 1

    if missing_job_count:
        blockers.append(
            f"{missing_job_count} submitted copy/copies have no durable processing job"
        )
    if duplicate_job_count:
        blockers.append(
            f"{duplicate_job_count} duplicate processing job(s) prevent exactly-once migration"
        )
    if orphan_job_count:
        blockers.append(
            f"{orphan_job_count} orphan processing job(s) do not belong to a submission"
        )
    if invalid_job_id_count:
        blockers.append(f"{invalid_job_id_count} processing job(s) have no durable job ID")
    if active_job_count:
        blockers.append(
            f"{active_job_count} processing job(s) still own an active lease or queue slot"
        )
    if mixed_job_count:
        blockers.append(
            f"{mixed_job_count} job(s) already use a mixed or unsupported grading contract"
        )

    return {
        "db_name": db_name,
        "exam_id": current_exam_id,
        "exam_name": exam.get("exam_name") or exam.get("title") or current_exam_id,
        "source_prompt_version": source_prompt_version,
        "source_prompt_versions": list(LEGACY_V16_SOURCE_PROMPT_VERSIONS),
        "target_prompt_version": V16_PROMPT_VERSION,
        "submission_count": len(submissions),
        "job_count": len(jobs),
        "published_count": published_count,
        "active_job_count": active_job_count,
        "missing_job_count": missing_job_count,
        "duplicate_job_count": duplicate_job_count,
        "orphan_job_count": orphan_job_count,
        "mixed_job_count": mixed_job_count,
        "eligible": not blockers,
        "blockers": blockers,
    }


async def inspect_legacy_contracts(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect released legacy subjective contracts for one direct v16 migration."""

    query: dict[str, Any] = {
        "pcr_grading_contract.prompt_version": {
            "$in": list(LEGACY_V16_SOURCE_PROMPT_VERSIONS)
        }
    }
    if exam_id:
        query["exam_id"] = exam_id
    exams = await tenant_db[EXAMS_COLLECTION].find(query).to_list(length=None)
    return [
        await _inspect_legacy_v16_exam(
            tenant_db,
            db_name=db_name,
            exam=exam,
        )
        for exam in exams
    ]


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


async def _reconcile_completed_v16_pending_jobs(
    tenant_db: Any,
    *,
    exam_id: str,
    migration_id: str,
    requested_by: str,
) -> dict[str, int]:
    """Drain the only crash window left after the exam fence closes."""

    if not migration_id:
        raise GradingContractMigrationError(
            f"Exam {exam_id} has a completed migration without a durable migration ID"
        )
    now = _utcnow()
    foreign_pending = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
        {
            "exam_id": exam_id,
            "status": CONTRACT_MIGRATION_PENDING_JOB_STATUS,
            "migration_id": {"$nin": [None, "", migration_id]},
        },
        {"_id": 0, "job_id": 1, "migration_id": 1},
    )
    if foreign_pending:
        raise GradingContractMigrationError(
            "Pending job "
            f"{foreign_pending.get('job_id')} belongs to a different migration"
        )
    pending_submission_ids = await tenant_db[PROCESSING_JOBS_COLLECTION].distinct(
        "submission_id",
        {
            "exam_id": exam_id,
            "status": CONTRACT_MIGRATION_PENDING_JOB_STATUS,
        },
    )
    await tenant_db[PROCESSING_JOBS_COLLECTION].update_many(
        {
            "exam_id": exam_id,
            "status": CONTRACT_MIGRATION_PENDING_JOB_STATUS,
        },
        {
            "$set": {
                "status": V16_CAPABILITY_QUEUED_JOB_STATUS,
                "pipeline_version": V16_PIPELINE_VERSION,
                "mapping_pipeline_version": V16_MAPPING_PIPELINE_VERSION,
                "required_processing_path": V16_REQUIRED_PROCESSING_PATH,
                "migration_id": migration_id,
                "attempts": 0,
                "queued_at": now,
                "updated_at": now,
                "reprocess_requested_at": now,
                "reprocess_requested_by": requested_by,
                "reprocess_reason": "grading_contract_migration_pending_drain",
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
        },
    )
    if pending_submission_ids:
        await tenant_db[SUBMISSIONS_COLLECTION].update_many(
            {
                "exam_id": exam_id,
                "submission_id": {"$in": pending_submission_ids},
            },
            {
                "$set": {
                    "review_state": "processing",
                    "updated_at": now,
                    "grading_contract_migration_id": migration_id,
                }
            },
        )

    submission_ids = await tenant_db[SUBMISSIONS_COLLECTION].distinct(
        "submission_id", {"exam_id": exam_id}
    )
    for submission_id in submission_ids:
        jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].find(
            {"exam_id": exam_id, "submission_id": submission_id}
        ).to_list(length=2)
        if len(jobs) != 1:
            raise GradingContractMigrationError(
                f"Post-fence cohort check failed for submission {submission_id}"
            )
        job = jobs[0]
        if (
            int(job.get("pipeline_version") or 0) != V16_PIPELINE_VERSION
            or str(job.get("mapping_pipeline_version") or "")
            != V16_MAPPING_PIPELINE_VERSION
            or str(job.get("status") or "")
            == CONTRACT_MIGRATION_PENDING_JOB_STATUS
        ):
            raise GradingContractMigrationError(
                f"Post-fence cohort check found a non-v16 job for submission {submission_id}"
            )
    return {
        "cohort_submission_count": len(submission_ids),
        "queued_job_count": await tenant_db[PROCESSING_JOBS_COLLECTION].count_documents(
            {
                "exam_id": exam_id,
                "migration_id": migration_id,
                "pipeline_version": V16_PIPELINE_VERSION,
                "mapping_pipeline_version": V16_MAPPING_PIPELINE_VERSION,
            }
        ),
    }


async def _migrate_legacy_exam_to_v16(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
) -> dict[str, Any]:
    """Move one complete released subjective cohort directly onto v16.

    The exam-level ``applying`` state is the reprocess fence.  The contract is
    changed with compare-and-set semantics, then each pre-existing durable job
    is independently reconciled to pipeline 7.  A retry reuses the same
    migration identity and never appends a second reprocess history entry.
    """

    exam = await tenant_db[EXAMS_COLLECTION].find_one({"exam_id": exam_id})
    if not exam:
        raise GradingContractMigrationError(f"Exam {exam_id} was not found in {db_name}")

    contract = dict(exam.get("pcr_grading_contract") or {})
    current_version = str(contract.get("prompt_version") or "").strip()
    migration_state = dict(exam.get("pcr_grading_contract_migration") or {})
    state_source = str(migration_state.get("source_prompt_version") or "").strip()
    state_target = str(migration_state.get("target_prompt_version") or "").strip()
    state_status = str(migration_state.get("status") or "").strip().lower()
    matching_state = (
        state_source in LEGACY_V16_SOURCE_PROMPT_VERSIONS
        and state_target == V16_PROMPT_VERSION
    )

    if current_version == V16_PROMPT_VERSION and matching_state and state_status == "complete":
        migration_id = str(migration_state.get("migration_id") or "").strip()
        repaired = await _reconcile_completed_v16_pending_jobs(
            tenant_db,
            exam_id=exam_id,
            migration_id=migration_id,
            requested_by=requested_by,
        )
        await tenant_db[EXAMS_COLLECTION].update_one(
            {"exam_id": exam_id, "pcr_grading_contract.migration_id": migration_id},
            {
                "$set": {
                    "pcr_grading_contract_migration.queued_job_count": repaired[
                        "queued_job_count"
                    ],
                    "pcr_grading_contract_migration.cohort_submission_count": repaired[
                        "cohort_submission_count"
                    ],
                    "updated_at": _utcnow(),
                }
            },
        )
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "already_migrated",
            "migration_id": migration_state.get("migration_id"),
            "queued_job_count": repaired["queued_job_count"],
        }
    if current_version == V16_PROMPT_VERSION:
        if not matching_state or state_status not in {"applying", "failed"}:
            raise GradingContractMigrationError(
                f"Exam {exam_id} already uses v16 and is not a resumable legacy migration"
            )
        source_prompt_version = state_source
        resuming_after_contract_change = True
    elif current_version in LEGACY_V16_SOURCE_PROMPT_VERSIONS:
        source_prompt_version = current_version
        resuming_after_contract_change = False
    else:
        raise GradingContractMigrationError(
            f"Exam {exam_id} has unsupported legacy source contract "
            f"{current_version or 'missing'}"
        )

    if _is_explicit_objective(exam) or _is_explicit_objective(contract):
        raise GradingContractMigrationError(
            f"Exam {exam_id} is objective and cannot use the subjective v16 migration"
        )

    if not resuming_after_contract_change:
        plans = await inspect_legacy_contracts(
            tenant_db,
            db_name=db_name,
            exam_id=exam_id,
        )
        if not plans or not plans[0]["eligible"]:
            blockers = plans[0]["blockers"] if plans else [
                "exam no longer matches a released legacy source contract"
            ]
            raise GradingContractMigrationError(
                f"Exam {exam_id} cannot be migrated: {'; '.join(blockers)}"
            )

    now = _utcnow()
    reusable_state = matching_state and state_status in {"applying", "failed"}
    migration_id = (
        str(migration_state.get("migration_id") or "").strip()
        if reusable_state
        else ""
    ) or f"PCR-MIG-{uuid4().hex}"
    started_at = migration_state.get("started_at") if reusable_state else None

    audit = {
        "migration_id": migration_id,
        "db_name": db_name,
        "exam_id": exam_id,
        "source_prompt_version": source_prompt_version,
        "source_prompt_versions": list(LEGACY_V16_SOURCE_PROMPT_VERSIONS),
        "target_prompt_version": V16_PROMPT_VERSION,
        "requested_by": requested_by,
        "status": "applying",
        "updated_at": now,
    }
    await tenant_db[MIGRATIONS_COLLECTION].update_one(
        {"migration_id": migration_id},
        {
            "$set": audit,
            "$setOnInsert": {
                "started_at": started_at or now,
                "contract_before": contract,
            },
        },
        upsert=True,
    )

    try:
        if not resuming_after_contract_change:
            fence_filter: dict[str, Any] = {
                "exam_id": exam_id,
                "paper_version_id": exam.get("paper_version_id"),
                "prepared_document_id": exam.get("prepared_document_id"),
                "pcr_grading_contract.prompt_version": source_prompt_version,
            }
            if reusable_state:
                fence_filter["pcr_grading_contract_migration.migration_id"] = migration_id
            else:
                fence_filter["pcr_grading_contract_migration.status"] = {"$ne": "applying"}
            fenced = await tenant_db[EXAMS_COLLECTION].update_one(
                fence_filter,
                {
                    "$set": {
                        "pcr_grading_contract_migration": {
                            "migration_id": migration_id,
                            "source_prompt_version": source_prompt_version,
                            "source_prompt_versions": list(
                                LEGACY_V16_SOURCE_PROMPT_VERSIONS
                            ),
                            "target_prompt_version": V16_PROMPT_VERSION,
                            "requested_by": requested_by,
                            "started_at": started_at or now,
                            "status": "applying",
                        },
                        "updated_at": now,
                    }
                },
            )
            if fenced.matched_count != 1:
                raise GradingContractMigrationError(
                    f"Exam {exam_id} changed concurrently; migration fence was not acquired"
                )

            # Re-read every invariant after acquiring the exam fence.  A
            # reprocess that won immediately before the fence is now visible
            # as an active job and prevents the contract change.
            fenced_exam = await tenant_db[EXAMS_COLLECTION].find_one(
                {"exam_id": exam_id}
            )
            fenced_plan = await _inspect_legacy_v16_exam(
                tenant_db,
                db_name=db_name,
                exam=fenced_exam or {},
                allowed_migration_id=migration_id,
            )
            if not fenced_plan["eligible"]:
                raise GradingContractMigrationError(
                    f"Exam {exam_id} changed while migration was fenced: "
                    f"{'; '.join(fenced_plan['blockers'])}"
                )

            migrated_contract = {
                **contract,
                "prompt_version": V16_PROMPT_VERSION,
                "pipeline_version": V16_PIPELINE_VERSION,
                "mapping_pipeline_version": V16_MAPPING_PIPELINE_VERSION,
                "required_processing_path": V16_REQUIRED_PROCESSING_PATH,
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
                    "pcr_grading_contract_migration.migration_id": migration_id,
                    "pcr_grading_contract_migration.status": "applying",
                },
                {
                    "$set": {
                        "pcr_grading_contract": migrated_contract,
                        "updated_at": now,
                    }
                },
            )
            if updated.modified_count != 1:
                raise GradingContractMigrationError(
                    f"Exam {exam_id} changed concurrently; no contract was migrated"
                )

        supersede_filter = {
            "exam_id": exam_id,
            "status": {"$ne": "superseded"},
            "$or": [
                {
                    "prompt_version": {
                        "$in": list(LEGACY_V16_SOURCE_PROMPT_VERSIONS)
                    }
                },
                {"pipeline_version": {"$in": list(LEGACY_V16_PIPELINE_VERSIONS)}},
                {
                    "mapping_pipeline_version": {
                        "$in": list(LEGACY_V16_MAPPING_PIPELINE_VERSIONS)
                    }
                },
            ],
        }
        await tenant_db[GRADING_RUNS_COLLECTION].update_many(
            supersede_filter,
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

        submissions = await tenant_db[SUBMISSIONS_COLLECTION].find(
            {"exam_id": exam_id},
            {
                "_id": 0,
                "submission_id": 1,
                "grading_mode": 1,
                "marking_mode": 1,
                "assessment_mode": 1,
                "publication_status": 1,
                "published_at": 1,
            },
        ).to_list(length=None)
        submission_ids = [
            str(item.get("submission_id") or "").strip() for item in submissions
        ]
        if any(not value for value in submission_ids):
            raise GradingContractMigrationError(
                "One or more submissions have no durable submission ID"
            )
        if any(
            _is_explicit_objective(item)
            or str(item.get("publication_status") or "").strip().lower()
            == "published"
            or item.get("published_at") is not None
            for item in submissions
        ):
            raise GradingContractMigrationError(
                "The cohort became objective or published while migration was applying"
            )

        history_entry = {
            "requested_at": now,
            "requested_by": requested_by,
            "reason": "grading_contract_migration",
            "migration_id": migration_id,
            "source_prompt_version": source_prompt_version,
            "target_prompt_version": V16_PROMPT_VERSION,
        }
        for submission_id in submission_ids:
            matching_jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].find(
                {"exam_id": exam_id, "submission_id": submission_id}
            ).to_list(length=2)
            if len(matching_jobs) != 1:
                raise GradingContractMigrationError(
                    f"Submission {submission_id} must have exactly one durable processing job"
                )
            job = matching_jobs[0]
            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                raise GradingContractMigrationError(
                    f"Submission {submission_id} has a processing job without a durable ID"
                )
            try:
                pipeline_version = int(job.get("pipeline_version"))
            except (TypeError, ValueError):
                pipeline_version = None
            mapping_version = str(job.get("mapping_pipeline_version") or "").strip()
            job_status = str(job.get("status") or "").strip().lower()
            job_migration_id = str(job.get("migration_id") or "").strip()
            already_target = (
                pipeline_version == V16_PIPELINE_VERSION
                and mapping_version == V16_MAPPING_PIPELINE_VERSION
            )
            same_migration = job_migration_id == migration_id

            if already_target:
                if not same_migration:
                    owner = job_migration_id or "no migration"
                    raise GradingContractMigrationError(
                        f"Target v16 job {job_id} belongs to {owner}, not migration "
                        f"{migration_id}"
                    )
                if job_status == "processing":
                    raise GradingContractMigrationError(
                        f"Target v16 job {job_id} became active while migration was applying"
                    )
                if same_migration and job_status not in {
                    V16_CAPABILITY_QUEUED_JOB_STATUS,
                    "queued",
                    "completed",
                }:
                    # Resume a partially applied migration without charging a
                    # second reprocess count/history entry.
                    await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
                        {
                            "job_id": job_id,
                            "migration_id": migration_id,
                            "status": job.get("status"),
                        },
                        {
                            "$set": {
                                "status": V16_CAPABILITY_QUEUED_JOB_STATUS,
                                "attempts": 0,
                                "queued_at": now,
                                "updated_at": now,
                                "reprocess_requested_at": now,
                                "reprocess_requested_by": requested_by,
                                "reprocess_reason": "grading_contract_migration_resume",
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
                        },
                    )
                continue

            if job_status in ACTIVE_JOB_STATUSES:
                raise GradingContractMigrationError(
                    f"Legacy job {job_id} became active while migration was applying"
                )
            result = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
                {
                    "job_id": job_id,
                    "exam_id": exam_id,
                    "submission_id": submission_id,
                    "status": job.get("status"),
                    "pipeline_version": job.get("pipeline_version"),
                    "mapping_pipeline_version": job.get(
                        "mapping_pipeline_version"
                    ),
                },
                {
                    "$set": {
                        "status": V16_CAPABILITY_QUEUED_JOB_STATUS,
                        "pipeline_version": V16_PIPELINE_VERSION,
                        "mapping_pipeline_version": V16_MAPPING_PIPELINE_VERSION,
                        "required_processing_path": V16_REQUIRED_PROCESSING_PATH,
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
                            "$each": [history_entry],
                            "$slice": -20,
                        }
                    },
                },
            )
            if result.modified_count != 1:
                latest = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
                    {"job_id": job_id}
                )
                if not (
                    latest
                    and int(latest.get("pipeline_version") or 0)
                    == V16_PIPELINE_VERSION
                    and str(latest.get("mapping_pipeline_version") or "")
                    == V16_MAPPING_PIPELINE_VERSION
                ):
                    raise GradingContractMigrationError(
                        f"Processing job {job_id} changed concurrently"
                    )

        # Final whole-cohort assertion.  Submissions created after the exam
        # contract CAS are valid only when their own exactly-once job already
        # carries the v16 metadata.
        final_submissions = await tenant_db[SUBMISSIONS_COLLECTION].distinct(
            "submission_id", {"exam_id": exam_id}
        )
        for submission_id in final_submissions:
            final_jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].find(
                {"exam_id": exam_id, "submission_id": submission_id}
            ).to_list(length=2)
            if len(final_jobs) != 1:
                raise GradingContractMigrationError(
                    f"Final cohort check failed for submission {submission_id}"
                )
            final_job = final_jobs[0]
            if (
                int(final_job.get("pipeline_version") or 0) != V16_PIPELINE_VERSION
                or str(final_job.get("mapping_pipeline_version") or "")
                != V16_MAPPING_PIPELINE_VERSION
            ):
                raise GradingContractMigrationError(
                    f"Final cohort check found a non-v16 job for submission {submission_id}"
                )

        migrated_submission_ids = await tenant_db[PROCESSING_JOBS_COLLECTION].distinct(
            "submission_id",
            {
                "exam_id": exam_id,
                "migration_id": migration_id,
                "status": {"$ne": "completed"},
            },
        )
        if migrated_submission_ids:
            await tenant_db[SUBMISSIONS_COLLECTION].update_many(
                {
                    "exam_id": exam_id,
                    "submission_id": {"$in": migrated_submission_ids},
                },
                {
                    "$set": {
                        "review_state": "processing",
                        "updated_at": now,
                        "grading_contract_migration_id": migration_id,
                    }
                },
            )

        # Close the ingestion fence before the final pending-job drain.  A
        # scheduler that observed ``applying`` before this CAS either gets
        # drained below or re-reads ``complete`` and reconciles its own job to
        # v16 before dispatch.
        completed_at = _utcnow()
        closed = await tenant_db[EXAMS_COLLECTION].update_one(
            {
                "exam_id": exam_id,
                "pcr_grading_contract.prompt_version": V16_PROMPT_VERSION,
                "pcr_grading_contract.migration_id": migration_id,
            },
            {
                "$set": {
                    "pcr_grading_contract_migration": {
                        "status": "complete",
                        "completed_at": completed_at,
                        "updated_at": completed_at,
                        "migration_id": migration_id,
                        "source_prompt_version": source_prompt_version,
                        "source_prompt_versions": list(
                            LEGACY_V16_SOURCE_PROMPT_VERSIONS
                        ),
                        "target_prompt_version": V16_PROMPT_VERSION,
                        "requested_by": requested_by,
                        "started_at": started_at or now,
                    },
                    "updated_at": completed_at,
                }
            },
        )
        if closed.matched_count != 1:
            raise GradingContractMigrationError(
                f"Exam {exam_id} changed concurrently while closing the migration fence"
            )

        reconciled = await _reconcile_completed_v16_pending_jobs(
            tenant_db,
            exam_id=exam_id,
            migration_id=migration_id,
            requested_by=requested_by,
        )
        queued_job_count = reconciled["queued_job_count"]
        final_submission_count = reconciled["cohort_submission_count"]
        superseded_run_count = await tenant_db[GRADING_RUNS_COLLECTION].count_documents(
            {
                "exam_id": exam_id,
                "superseded_by_migration_id": migration_id,
            }
        )
        completion = {
            "status": "complete",
            "completed_at": completed_at,
            "updated_at": completed_at,
            "queued_job_count": queued_job_count,
            "cohort_submission_count": final_submission_count,
            "superseded_run_count": superseded_run_count,
        }
        await tenant_db[EXAMS_COLLECTION].update_one(
            {
                "exam_id": exam_id,
                "pcr_grading_contract.migration_id": migration_id,
            },
            {
                "$set": {
                    "pcr_grading_contract_migration.queued_job_count": queued_job_count,
                    "pcr_grading_contract_migration.cohort_submission_count": final_submission_count,
                    "pcr_grading_contract_migration.superseded_run_count": superseded_run_count,
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
            "source_prompt_version": source_prompt_version,
            "target_prompt_version": V16_PROMPT_VERSION,
            "migration_id": migration_id,
            "queued_job_count": queued_job_count,
            "cohort_submission_count": final_submission_count,
            "superseded_run_count": superseded_run_count,
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
            {
                "exam_id": exam_id,
                "pcr_grading_contract_migration.migration_id": migration_id,
            },
            {
                "$set": {
                    "pcr_grading_contract_migration.status": "failed",
                    "pcr_grading_contract_migration.failure_code": exc.__class__.__name__,
                    "pcr_grading_contract_migration.failure_detail": str(exc),
                    "pcr_grading_contract_migration.failed_at": failed_at,
                    "updated_at": failed_at,
                }
            },
        )
        raise


async def migrate_legacy_exam_to_v16(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    requested_by: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """Apply a direct legacy subjective -> v16 cohort migration."""

    if confirmation_token != LEGACY_TO_V16_CONFIRMATION_TOKEN:
        raise GradingContractMigrationError(
            "legacy to v16 migration requires confirmation token "
            f"{LEGACY_TO_V16_CONFIRMATION_TOKEN}"
        )
    return await _migrate_legacy_exam_to_v16(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        requested_by=requested_by,
    )


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
