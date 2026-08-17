"""Durable PCR processing workflow for conducted ExamPen submissions.

Capture endpoints only own canonical ingest.  This module persists an
idempotent job after ingest, dispatches it to Celery, and records enough state
for the web UI and lifecycle coordinator to explain what is still pending.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from services.pcr_grading_contract_policy import (
    SELECTED_COPY_CONTRACT_SCOPE,
    V16_MAPPING_PIPELINE_VERSION,
    V16_PIPELINE_VERSION,
    V16_REQUIRED_PROCESSING_PATH,
    build_selected_copy_v16_override,
    effective_grading_contract,
    is_legacy_selected_copy_source,
    is_supported_worker_contract,
    selected_copy_contract_override,
)

logger = logging.getLogger(__name__)

PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
TERMINAL_JOB_STATUSES = {"completed", "blocked_for_review", "failed"}
CAPABILITY_QUEUED_JOB_STATUS = "queued_pipeline_v3"
# Pipeline 5 deliberately has its own queue capability.  A pre-v14 worker
# knows only ``queued_pipeline_v3`` and therefore cannot claim a newly migrated
# v14 job during a rolling restart.
V14_CAPABILITY_QUEUED_JOB_STATUS = "queued_pipeline_v5"
V15_CAPABILITY_QUEUED_JOB_STATUS = "queued_pipeline_v6"
V16_CAPABILITY_QUEUED_JOB_STATUS = "queued_pipeline_v7"
CONTRACT_MIGRATION_PENDING_JOB_STATUS = "grading_contract_migration_pending"
CONTRACT_MIGRATION_BLOCKING_STATUSES = frozenset({"applying", "failed"})
DISPATCHABLE_JOB_STATUSES = {
    "queued",
    "retryable_error",
    "enqueue_failed",
    CAPABILITY_QUEUED_JOB_STATUS,
    V14_CAPABILITY_QUEUED_JOB_STATUS,
    V15_CAPABILITY_QUEUED_JOB_STATUS,
    V16_CAPABILITY_QUEUED_JOB_STATUS,
}
REPROCESS_IN_FLIGHT_STATUSES = {
    "queued",
    "processing",
    CAPABILITY_QUEUED_JOB_STATUS,
    V14_CAPABILITY_QUEUED_JOB_STATUS,
    V15_CAPABILITY_QUEUED_JOB_STATUS,
    V16_CAPABILITY_QUEUED_JOB_STATUS,
    CONTRACT_MIGRATION_PENDING_JOB_STATUS,
}
CURRENT_PCR_PIPELINE_VERSION = 4
CURRENT_PCR_MAPPING_VERSION = "evidence-first-visual-v4"
# v14 is intentionally named here even while the default worker remains v13
# during rollout.  The bounded worker can opt into these values explicitly;
# old workers must continue to claim only pipeline 4 jobs.
PCR_V14_PIPELINE_VERSION = 5
PCR_V14_MAPPING_VERSION = "bounded-evidence-visual-v5"
PCR_V14_REQUIRED_PROCESSING_PATH = "full_document_visual"
PCR_V15_PIPELINE_VERSION = 6
PCR_V15_MAPPING_VERSION = "bounded-evidence-visual-v6"
PCR_V15_REQUIRED_PROCESSING_PATH = "full_document_visual"
PCR_V16_PIPELINE_VERSION = 7
PCR_V16_MAPPING_VERSION = "whole-copy-rubric-v7"
PCR_V16_REQUIRED_PROCESSING_PATH = "full_document_visual"
FULL_DOCUMENT_PROCESSING_PATH = "full_document_visual"
PROCESSING_LEASE_MINUTES = 30
PROCESSING_HEARTBEAT_SECONDS = 60
PROCESSING_MAX_AUTOMATIC_ATTEMPTS = 4
PROCESSING_RETRY_BASE_SECONDS = 60
PROCESSING_RETRY_MAX_SECONDS = 15 * 60
# A live worker renews ``updated_at`` every heartbeat. Waiting for the full
# 30-minute ownership lease after a process crash leaves the UI stuck on
# "Checking" even though no worker exists. Three missed heartbeats provide a
# conservative crash detector while the lease token still fences late writes
# from the old owner.
PROCESSING_HEARTBEAT_STALE_SECONDS = PROCESSING_HEARTBEAT_SECONDS * 3
SUPPORTED_PCR_PIPELINE_VERSIONS = frozenset({
    4,
    PCR_V14_PIPELINE_VERSION,
    PCR_V15_PIPELINE_VERSION,
    PCR_V16_PIPELINE_VERSION,
})


def _celery_broker_available() -> bool:
    """Return True only when Redis/Celery broker answers a fast PING.

    Local development often runs the API without Redis.  Calling
    ``task.delay()`` in that state can hang for tens of seconds while Celery
    retries the broker.  A one-second ping keeps reprocess responsive and lets
    the inline processor own the job instead.
    """
    try:
        import redis
        from config_async import settings

        broker_url = (
            getattr(settings, "CELERY_BROKER_URL", None)
            or getattr(settings, "REDIS_URL", None)
            or "redis://localhost:6379/0"
        )
        client = redis.from_url(
            str(broker_url),
            socket_connect_timeout=1,
            socket_timeout=1,
            retry_on_timeout=False,
        )
        try:
            return bool(client.ping())
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception as exc:
        logger.info("Celery broker unavailable for PCR dispatch: %s", exc)
        return False


class ProcessingJobBusyError(RuntimeError):
    """Raised when an operator tries to reprocess a copy already in flight.

    A reprocess must never reset a job that a worker has already claimed.  That
    would allow two workers to supersede each other's answer mapping and make
    the teacher review screen depend on a timing race.
    """


class GradingContractMigrationRequiredError(RuntimeError):
    """Raised before a per-copy rerun can mutate an unsupported exam cohort."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _queued_status_for_pipeline(pipeline_version: Any) -> str:
    """Fence each incompatible worker generation behind its own queue."""

    try:
        version = int(pipeline_version or 0)
    except (TypeError, ValueError):
        version = 0
    if version >= PCR_V16_PIPELINE_VERSION:
        return V16_CAPABILITY_QUEUED_JOB_STATUS
    if version >= PCR_V15_PIPELINE_VERSION:
        return V15_CAPABILITY_QUEUED_JOB_STATUS
    if version >= PCR_V14_PIPELINE_VERSION:
        return V14_CAPABILITY_QUEUED_JOB_STATUS
    return CAPABILITY_QUEUED_JOB_STATUS if version >= 3 else "queued"


def _retryable_status_for_pipeline(pipeline_version: Any) -> str:
    """Preserve legacy retry semantics while fencing v3 work from old workers."""

    queued_status = _queued_status_for_pipeline(pipeline_version)
    return queued_status if queued_status.startswith("queued_pipeline_") else "retryable_error"


def public_processing_status(status: Any) -> str:
    """Expose capability-fenced queued work through the stable API status."""

    normalized = str(status or "").strip().lower()
    return (
        "queued"
        if normalized in {
            CAPABILITY_QUEUED_JOB_STATUS,
            V14_CAPABILITY_QUEUED_JOB_STATUS,
            V15_CAPABILITY_QUEUED_JOB_STATUS,
            V16_CAPABILITY_QUEUED_JOB_STATUS,
            CONTRACT_MIGRATION_PENDING_JOB_STATUS,
        }
        else normalized
    )


def is_supported_pcr_grading_contract(contract: Any) -> bool:
    """Return whether this worker generation can execute a frozen contract."""

    return is_supported_worker_contract(contract)


def can_upgrade_legacy_pcr_contract_on_demand(contract: Any) -> bool:
    """Return whether one explicit unpublished copy may opt into v16."""

    return is_legacy_selected_copy_source(contract)


def _grading_contract_migration_blocks_processing(exam: Any) -> bool:
    payload = exam if isinstance(exam, dict) else {}
    migration = dict(payload.get("pcr_grading_contract_migration") or {})
    return (
        str(migration.get("status") or "").strip().lower()
        in CONTRACT_MIGRATION_BLOCKING_STATUSES
    )


def _job_id(submission_id: str) -> str:
    return f"pcr-job-{submission_id}"


def _short_error(error: Exception | str) -> str:
    message = str(error).replace("\n", " ").strip()
    return message[:800] or type(error).__name__


def _as_utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _lease_expiry(job: Dict[str, Any]) -> Optional[datetime]:
    """Return the explicit expiry, or the legacy updated-at lease boundary."""
    explicit = _as_utc(job.get("lease_expires_at"))
    if explicit is not None:
        return explicit
    updated_at = _as_utc(job.get("updated_at"))
    if updated_at is None:
        return None
    return updated_at + timedelta(minutes=PROCESSING_LEASE_MINUTES)


def _lease_is_active(job: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    if str(job.get("status") or "") != "processing":
        return False
    expiry = _lease_expiry(job)
    return expiry is not None and expiry > (now or _now())


def _owned_lease_filter(job: Dict[str, Any]) -> Dict[str, Any]:
    token = str(job.get("lease_token") or "")
    if not token:
        raise ProcessingJobBusyError("Processing job has no worker lease token")
    return {
        "job_id": str(job.get("job_id") or ""),
        "status": "processing",
        "lease_token": token,
    }


async def _heartbeat_processing_job(tenant_db: Any, job: Dict[str, Any]) -> None:
    """Renew a worker lease, failing closed if another worker fenced it out."""
    now = _now()
    expires_at = now + timedelta(minutes=PROCESSING_LEASE_MINUTES)
    result = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        _owned_lease_filter(job),
        {"$set": {"lease_expires_at": expires_at, "updated_at": now}},
    )
    if result.matched_count != 1:
        raise ProcessingJobBusyError(
            f"Processing lease for job {job.get('job_id')} is no longer owned by this worker"
        )
    job["lease_expires_at"] = expires_at
    job["updated_at"] = now


async def _run_with_lease_heartbeat(
    tenant_db: Any,
    job: Dict[str, Any],
    operation: Awaitable[Any],
) -> Any:
    """Keep the ownership fence live while OCR/mapping performs a long call."""

    async def _pulse() -> None:
        while True:
            await asyncio.sleep(PROCESSING_HEARTBEAT_SECONDS)
            await _heartbeat_processing_job(tenant_db, job)

    operation_task = asyncio.create_task(operation)
    heartbeat_task = asyncio.create_task(_pulse())
    try:
        done, _ = await asyncio.wait(
            {operation_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            error = heartbeat_task.exception()
            operation_task.cancel()
            with suppress(asyncio.CancelledError):
                await operation_task
            if error is not None:
                raise error
            raise ProcessingJobBusyError("Processing worker heartbeat stopped unexpectedly")

        result = await operation_task
        await _heartbeat_processing_job(tenant_db, job)
        return result
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task


async def ensure_indexes(tenant_db: Any) -> None:
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    await jobs.create_index("job_id", unique=True, name="uniq_processing_job")
    await jobs.create_index("submission_id", unique=True, name="uniq_processing_submission")
    await jobs.create_index([("exam_id", 1), ("status", 1)], name="idx_processing_exam_status")


async def _required_processing_path(tenant_db: Any, exam_id: str) -> str:
    """Resolve the immutable grading lane selected when the paper was finalized."""

    exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        {"paper_version_id": 1},
    )
    paper_version_id = (exam or {}).get("paper_version_id")
    if not paper_version_id:
        return "legacy_ocr_mapping"
    paper_version = await tenant_db["exampen_paper_versions"].find_one(
        {"paper_version_id": paper_version_id},
        {"paper_context": 1},
    )
    context = dict((paper_version or {}).get("paper_context") or {})
    if (
        context.get("ready")
        and str(context.get("version") or "")
        in {
            "canonical-full-document-visual-v1",
            "canonical-full-document-visual-v2",
        }
    ):
        return FULL_DOCUMENT_PROCESSING_PATH
    return "legacy_ocr_mapping"


def _is_explicit_objective(payload: Any) -> bool:
    item = payload if isinstance(payload, dict) else {}
    values = {
        str(item.get(field) or "").strip().lower()
        for field in (
            "grading_mode",
            "marking_mode",
            "assessment_mode",
            "question_type",
            "response_type",
        )
    }
    return bool(values & {"objective", "omr", "mcq"})


async def _build_on_demand_selected_copy_contract(
    tenant_db: Any,
    *,
    exam: Dict[str, Any],
    submission: Dict[str, Any],
    requested_by: str,
    requested_at: datetime,
) -> Dict[str, Any]:
    """Validate and build one audited v16 override without touching the cohort."""

    source_contract = dict(exam.get("pcr_grading_contract") or {})
    if not is_legacy_selected_copy_source(source_contract):
        version = str(source_contract.get("prompt_version") or "missing")
        raise GradingContractMigrationRequiredError(
            f"Grading contract {version} cannot be upgraded for one selected copy"
        )
    if str(exam.get("exam_type") or "").strip().lower() != "pcr":
        raise GradingContractMigrationRequiredError("Selected-copy upgrade requires a PCR exam")
    if str(exam.get("publication_status") or "").strip().lower() == "published":
        raise GradingContractMigrationRequiredError(
            "Published exams cannot be changed by selected-copy reprocessing"
        )
    if (
        _is_explicit_objective(exam)
        or _is_explicit_objective(source_contract)
        or _is_explicit_objective(submission)
    ):
        raise GradingContractMigrationRequiredError(
            "Objective answer sheets use their separate deterministic reprocess lane"
        )
    if str(submission.get("publication_status") or "").lower() == "published":
        raise GradingContractMigrationRequiredError(
            "Published results cannot be reprocessed; use the recheck workflow"
        )
    questions = await tenant_db["evalpen_questions"].find(
        {"exam_id": str(exam.get("exam_id") or "")},
        {
            "grading_mode": 1,
            "marking_mode": 1,
            "assessment_mode": 1,
            "question_type": 1,
            "response_type": 1,
        },
    ).to_list(length=2000)
    if not questions:
        raise GradingContractMigrationRequiredError(
            "The frozen question catalog is unavailable for selected-copy reprocessing"
        )
    if all(_is_explicit_objective(question) for question in questions):
        raise GradingContractMigrationRequiredError(
            "Pure Objective papers use their separate deterministic reprocess lane"
        )
    answer_page = await tenant_db["evalpen_answer_pages"].find_one(
        {"submission_id": str(submission.get("submission_id") or "")},
        {"_id": 1},
    )
    if answer_page is None:
        raise GradingContractMigrationRequiredError(
            "The original uploaded answer pages are unavailable for reprocessing"
        )

    paper_version_id = str(exam.get("paper_version_id") or "").strip()
    paper_version = await tenant_db["exampen_paper_versions"].find_one(
        {"paper_version_id": paper_version_id}
    )
    if not paper_version:
        raise GradingContractMigrationRequiredError(
            "The frozen paper version is unavailable for selected-copy reprocessing"
        )
    prepared_document_id = str(exam.get("prepared_document_id") or "").strip()
    if prepared_document_id and str(paper_version.get("document_id") or "") != prepared_document_id:
        raise GradingContractMigrationRequiredError(
            "The frozen paper version does not belong to this exam"
        )
    context = dict(paper_version.get("paper_context") or {})
    if not context.get("ready") or str(context.get("version") or "") != (
        "canonical-full-document-visual-v2"
    ):
        raise GradingContractMigrationRequiredError(
            "This legacy paper does not have the canonical visual package required by v16"
        )
    assets = dict(paper_version.get("paper_assets") or {})
    question_asset = dict(assets.get("question_paper") or {})
    if (
        not question_asset.get("asset_id")
        or not question_asset.get("storage_uri")
        or context.get("question_paper_asset_id") != question_asset.get("asset_id")
    ):
        raise GradingContractMigrationRequiredError(
            "The frozen question-paper asset is unavailable or inconsistent"
        )
    if context.get("has_teacher_solution_asset"):
        solution_asset = dict(assets.get("teacher_solution") or {})
        if (
            not solution_asset.get("asset_id")
            or not solution_asset.get("storage_uri")
            or context.get("teacher_solution_asset_id") != solution_asset.get("asset_id")
        ):
            raise GradingContractMigrationRequiredError(
                "The frozen teacher-solution asset is unavailable or inconsistent"
            )
    return build_selected_copy_v16_override(
        source_contract,
        submission_id=str(submission.get("submission_id") or ""),
        requested_by=requested_by,
        requested_at=requested_at,
    )


async def _pipeline_metadata_for_exam(
    tenant_db: Any,
    exam_id: str,
    *,
    job: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str]:
    """Resolve the immutable worker contract without downgrading v14 jobs."""

    exam_contract: Dict[str, Any] = {}
    exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id}, {"pcr_grading_contract": 1}
    )
    if exam:
        exam_contract = dict(exam.get("pcr_grading_contract") or {})
    contract, _ = effective_grading_contract(exam_contract, job)
    candidate = contract.get("pipeline_version")
    if candidate is None:
        prompt_version = str(contract.get("prompt_version") or "")
        if prompt_version.endswith("-v16"):
            candidate = PCR_V16_PIPELINE_VERSION
        elif prompt_version.endswith("-v15"):
            candidate = PCR_V15_PIPELINE_VERSION
        elif prompt_version.endswith("-v14"):
            candidate = PCR_V14_PIPELINE_VERSION
        elif prompt_version.endswith("-v13"):
            candidate = CURRENT_PCR_PIPELINE_VERSION
    if candidate is None:
        candidate = (job or {}).get("pipeline_version")
    if candidate is None:
        candidate = CURRENT_PCR_PIPELINE_VERSION
    try:
        pipeline_version = int(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported PCR pipeline version: {candidate!r}") from exc
    if pipeline_version not in SUPPORTED_PCR_PIPELINE_VERSIONS:
        # Pre-v4 jobs are legacy records.  An explicit reprocess upgrades them
        # to the current immutable v13 lane; unknown future revisions fail
        # closed instead of being silently downgraded.
        if pipeline_version in {1, 2, 3}:
            pipeline_version = CURRENT_PCR_PIPELINE_VERSION
        else:
            raise ValueError(f"Unsupported PCR pipeline version: {pipeline_version}")
    mapping_version = (
        PCR_V16_MAPPING_VERSION
        if pipeline_version == PCR_V16_PIPELINE_VERSION
        else PCR_V15_MAPPING_VERSION
        if pipeline_version == PCR_V15_PIPELINE_VERSION
        else PCR_V14_MAPPING_VERSION
        if pipeline_version == PCR_V14_PIPELINE_VERSION
        else CURRENT_PCR_MAPPING_VERSION
    )
    return pipeline_version, mapping_version


async def ensure_processing_job(
    tenant_db: Any,
    *,
    exam_id: str,
    submission_id: str,
    student_id: Optional[str] = None,
    db_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Create the exactly-once processing record for a canonical submission."""
    await ensure_indexes(tenant_db)
    required_processing_path = await _required_processing_path(tenant_db, exam_id)
    pipeline_version, mapping_pipeline_version = await _pipeline_metadata_for_exam(
        tenant_db, exam_id
    )
    migration_exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        {"_id": 0, "pcr_grading_contract_migration": 1},
    )
    migration_state = dict(
        (migration_exam or {}).get("pcr_grading_contract_migration") or {}
    )
    migration_applying = (
        str(migration_state.get("status") or "").strip().lower()
        in CONTRACT_MIGRATION_BLOCKING_STATUSES
    )
    migration_id = str(migration_state.get("migration_id") or "").strip()
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    job_id = _job_id(submission_id)
    now = _now()
    initial_status = (
        CONTRACT_MIGRATION_PENDING_JOB_STATUS
        if migration_applying
        else _queued_status_for_pipeline(pipeline_version)
    )
    set_on_insert: Dict[str, Any] = {
        "job_id": job_id,
        "submission_id": submission_id,
        "exam_id": exam_id,
        "student_id": student_id,
        "db_name": db_name,
        "pipeline_version": pipeline_version,
        "mapping_pipeline_version": mapping_pipeline_version,
        "required_processing_path": required_processing_path,
        "status": initial_status,
        "attempts": 0,
        "lease_generation": 0,
        "created_at": now,
        "updated_at": now,
        "last_error": None,
    }
    if migration_applying and migration_id:
        set_on_insert["migration_id"] = migration_id
    result = await jobs.update_one(
        {"submission_id": submission_id},
        {"$setOnInsert": set_on_insert},
        upsert=True,
    )
    job = await jobs.find_one({"submission_id": submission_id})

    # Fence the narrow race in which a submission scheduler reads the legacy
    # contract immediately before an operator migrates the exam.  The exam CAS
    # happens before this job can be dispatched, so a newly inserted job is
    # reconciled to the latest contract metadata here.
    latest_pipeline_version, latest_mapping_pipeline_version = (
        await _pipeline_metadata_for_exam(tenant_db, exam_id, job=job)
    )
    latest_migration_exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        {"_id": 0, "pcr_grading_contract_migration": 1},
    )
    latest_migration_state = dict(
        (latest_migration_exam or {}).get("pcr_grading_contract_migration") or {}
    )
    latest_migration_applying = (
        str(latest_migration_state.get("status") or "").strip().lower()
        in CONTRACT_MIGRATION_BLOCKING_STATUSES
    )
    job_status = str((job or {}).get("status") or "").strip().lower()
    should_reconcile = bool(
        job is not None
        and not latest_migration_applying
        and (
            result.upserted_id is not None
            or job_status == CONTRACT_MIGRATION_PENDING_JOB_STATUS
        )
        and (
            latest_pipeline_version != job.get("pipeline_version")
            or latest_mapping_pipeline_version
            != str(job.get("mapping_pipeline_version") or "")
            or job_status == CONTRACT_MIGRATION_PENDING_JOB_STATUS
        )
    )
    if should_reconcile:
        reconciled = await jobs.update_one(
            {
                "submission_id": submission_id,
                "status": job.get("status"),
                "pipeline_version": job.get("pipeline_version"),
                "mapping_pipeline_version": job.get("mapping_pipeline_version"),
            },
            {
                "$set": {
                    "status": _queued_status_for_pipeline(latest_pipeline_version),
                    "pipeline_version": latest_pipeline_version,
                    "mapping_pipeline_version": latest_mapping_pipeline_version,
                    "required_processing_path": await _required_processing_path(
                        tenant_db, exam_id
                    ),
                    "updated_at": _now(),
                }
            },
        )
        if reconciled.modified_count != 1:
            raise ValueError(
                f"Submission {submission_id} changed while its exam grading contract was migrated"
            )
        job = await jobs.find_one({"submission_id": submission_id})
    pipeline_version = latest_pipeline_version
    mapping_pipeline_version = latest_mapping_pipeline_version

    if (
        job is not None
        and latest_migration_applying
        and job_status == CONTRACT_MIGRATION_PENDING_JOB_STATUS
    ):
        return job, result.upserted_id is not None

    if job is not None and result.upserted_id is None:
        persisted_pipeline = job.get("pipeline_version")
        try:
            persisted_pipeline = int(persisted_pipeline)
        except (TypeError, ValueError):
            persisted_pipeline = None
        if persisted_pipeline != pipeline_version:
            raise ValueError(
                f"Submission {submission_id} is attached to pipeline "
                f"{persisted_pipeline!r}, but exam {exam_id} requires "
                f"pipeline {pipeline_version}; run the explicit migration/reprocess"
            )
        if str(job.get("mapping_pipeline_version") or "") != mapping_pipeline_version:
            raise ValueError(
                f"Submission {submission_id} has mapping pipeline "
                f"{job.get('mapping_pipeline_version')!r}, expected "
                f"{mapping_pipeline_version}; run the explicit migration/reprocess"
            )
    return job, result.upserted_id is not None


async def dispatch_processing_job(
    tenant_db: Any,
    *,
    db_name: str,
    job: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """Send a persisted job to Celery without losing it if Redis is down."""
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    status = str(job.get("status") or "queued")
    # Never move an actively claimed job back to queued, even when an operator
    # explicitly requested a retry.  The conditional update below repeats the
    # guard so it remains true if another worker claims it between this read
    # and the write.
    if status == "processing":
        return job
    if not force and status in TERMINAL_JOB_STATUSES:
        return job
    if not force and status not in DISPATCHABLE_JOB_STATUSES:
        return job

    exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": str(job.get("exam_id") or "")},
        {"_id": 0, "pcr_grading_contract_migration": 1},
    )
    if _grading_contract_migration_blocks_processing(exam):
        migration = dict((exam or {}).get("pcr_grading_contract_migration") or {})
        await jobs.update_one(
            {"job_id": job.get("job_id"), "status": status},
            {
                "$set": {
                    "status": CONTRACT_MIGRATION_PENDING_JOB_STATUS,
                    "migration_id": migration.get("migration_id"),
                    "updated_at": _now(),
                }
            },
        )
        return await jobs.find_one({"job_id": job.get("job_id")}) or job

    now = _now()
    required_processing_path = str(job.get("required_processing_path") or "").strip()
    if not required_processing_path:
        required_processing_path = await _required_processing_path(
            tenant_db,
            str(job.get("exam_id") or ""),
        )
    pipeline_version, mapping_pipeline_version = await _pipeline_metadata_for_exam(
        tenant_db,
        str(job.get("exam_id") or ""),
        job=job,
    )
    queue_filter: Dict[str, Any] = {
        "job_id": job["job_id"],
        "status": status,
    }
    if job.get("enqueue_attempted_at") is None:
        queue_filter["enqueue_attempted_at"] = {"$exists": False}
    else:
        queue_filter["enqueue_attempted_at"] = job["enqueue_attempted_at"]
    capability_status = _queued_status_for_pipeline(
        pipeline_version
    )
    queued = await jobs.update_one(
        queue_filter,
        {
            "$set": {
                "status": capability_status,
                "pipeline_version": pipeline_version,
                "mapping_pipeline_version": mapping_pipeline_version,
                "required_processing_path": required_processing_path,
                "db_name": db_name,
                "enqueue_attempted_at": now,
                "updated_at": now,
                "last_error": None,
            },
            "$unset": {"next_retry_at": ""},
        },
    )
    if not queued.matched_count:
        current = await jobs.find_one({"job_id": job["job_id"]})
        return current or job

    job_id = str(job["job_id"])
    if not _celery_broker_available():
        # Keep the job durable and dispatchable. The supervised local inline
        # processor will claim it within its next bounded polling pass. Do not
        # create an untracked API-process task here: those tasks bypass the
        # configured concurrency limit and strand a ``processing`` lease when
        # the API reloads or exits.
        await jobs.update_one(
            {"job_id": job_id, "status": capability_status},
            {
                "$set": {
                    "last_error": (
                        "Celery/Redis broker unavailable; job kept queued for "
                        "inline processor"
                    ),
                    "updated_at": _now(),
                }
            },
        )
        return await jobs.find_one({"job_id": job_id})

    try:
        from celery_app import process_exampen_pcr_submission

        # Passing the required pipeline version as an argument is deliberate.
        # A stale Celery worker with the older two-argument task signature will
        # reject this job before it can silently grade a student through the
        # retired OCR-first path. The durable reconciler can then deliver the
        # still-queued job to a current worker after deployment converges.
        process_exampen_pcr_submission.delay(
            db_name,
            job_id,
            pipeline_version,
        )
    except Exception as exc:
        logger.exception("Unable to enqueue PCR processing job %s", job_id)
        await jobs.update_one(
            # If the worker claimed the job before the broker client raised,
            # preserve the worker state rather than overwriting it with an
            # enqueue error.
            {"job_id": job_id, "status": capability_status},
            {
                "$set": {
                    "status": "enqueue_failed",
                    "last_error": _short_error(exc),
                    "updated_at": _now(),
                }
            },
        )
    return await jobs.find_one({"job_id": job_id})


async def schedule_submission_processing(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    submission_id: str,
    student_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist then dispatch a PCR job after canonical ingest succeeds."""
    job, _ = await ensure_processing_job(
        tenant_db,
        exam_id=exam_id,
        submission_id=submission_id,
        student_id=student_id,
        db_name=db_name,
    )
    return await dispatch_processing_job(tenant_db, db_name=db_name, job=job)


async def retry_processing_job(
    tenant_db: Any,
    *,
    db_name: str,
    job_id: str,
) -> Dict[str, Any]:
    """Backward-compatible alias for an audited operator reprocess."""
    return await reprocess_processing_job(
        tenant_db,
        db_name=db_name,
        job_id=job_id,
        requested_by="system-retry",
        reason="Operator requested a retry",
    )


async def reprocess_processing_job(
    tenant_db: Any,
    *,
    db_name: str,
    job_id: str,
    requested_by: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Safely rerun full-document visual marking for an existing copy.

    The canonical uploaded pages remain unchanged. The primary path first maps
    student-owned evidence without an answer key, then grades that fixed map
    against the immutable solution and criteria. OCR/segmentation is retained
    only for legacy sessions without the required canonical files. Fresh rows
    supersede old mappings, so stale marks cannot mix with the new ledger.
    """
    await ensure_indexes(tenant_db)
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    job = await jobs.find_one({"job_id": job_id})
    if job is None:
        raise ValueError(f"Processing job {job_id} not found")

    exam_id = str(job.get("exam_id") or "").strip()
    exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        {
            "_id": 0,
            "exam_id": 1,
            "exam_type": 1,
            "paper_version_id": 1,
            "prepared_document_id": 1,
            "grading_mode": 1,
            "marking_mode": 1,
            "assessment_mode": 1,
            "publication_status": 1,
            "pcr_grading_contract": 1,
            "pcr_grading_contract_migration": 1,
        },
    )
    if exam is None:
        raise ValueError(f"Exam {exam_id} not found")
    migration_state = dict(exam.get("pcr_grading_contract_migration") or {})
    if (
        str(migration_state.get("status") or "").strip().lower()
        in CONTRACT_MIGRATION_BLOCKING_STATUSES
    ):
        raise GradingContractMigrationRequiredError(
            "The whole-exam grading contract migration is incomplete. "
            "Resume it successfully before reprocessing a copy."
        )
    submission_id = str(job.get("submission_id") or "").strip()
    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id, "exam_id": exam_id}
    )
    if submission is None:
        raise ValueError(f"Canonical submission {submission_id} not found")
    if str(submission.get("publication_status") or "").strip().lower() == "published":
        raise GradingContractMigrationRequiredError(
            "Published results cannot be reprocessed; use the recheck workflow"
        )

    now = _now()
    exam_contract = dict(exam.get("pcr_grading_contract") or {})
    contract_override = selected_copy_contract_override(job)
    if not contract_override and is_legacy_selected_copy_source(exam_contract):
        contract_override = await _build_on_demand_selected_copy_contract(
            tenant_db,
            exam=exam,
            submission=submission,
            requested_by=requested_by or "unknown",
            requested_at=now,
        )
    effective_contract, contract_scope = effective_grading_contract(
        exam_contract,
        {**job, "grading_contract_override": contract_override}
        if contract_override
        else job,
    )
    if not is_supported_pcr_grading_contract(effective_contract):
        version = str(exam_contract.get("prompt_version") or "missing")
        raise GradingContractMigrationRequiredError(
            f"Grading contract {version} cannot be reprocessed by this worker"
        )

    if contract_override:
        pipeline_version = V16_PIPELINE_VERSION
        mapping_pipeline_version = V16_MAPPING_PIPELINE_VERSION
        required_processing_path = V16_REQUIRED_PROCESSING_PATH
    else:
        pipeline_version, mapping_pipeline_version = await _pipeline_metadata_for_exam(
            tenant_db,
            exam_id,
            job=job,
        )
        required_processing_path = str(
            effective_contract.get("required_processing_path")
            or job.get("required_processing_path")
            or ""
        ).strip()
        if not required_processing_path:
            required_processing_path = await _required_processing_path(
                tenant_db,
                exam_id,
            )
    current_status = str(job.get("status") or "queued").strip().lower()
    if _lease_is_active(job, now=now):
        expiry = _lease_expiry(job)
        raise ProcessingJobBusyError(
            "This answer copy is still being processed by an active worker"
            + (f" until {expiry.isoformat()}" if expiry else "")
            + ". Wait for it to finish before reprocessing."
        )
    if current_status in REPROCESS_IN_FLIGHT_STATUSES and current_status != "processing":
        raise ProcessingJobBusyError(
            "This answer copy is already queued for checking. Wait for that run to finish."
        )

    # Reclaiming is allowed only after the observed lease expires.  Include the
    # exact observed fence fields in the reset so a concurrent heartbeat wins
    # safely instead of allowing two mapping runs to overlap.
    reset_filter: Dict[str, Any] = {"job_id": job_id, "status": current_status}
    if current_status == "processing":
        if job.get("lease_token"):
            reset_filter["lease_token"] = job["lease_token"]
        if job.get("lease_expires_at") is not None:
            reset_filter["lease_expires_at"] = job["lease_expires_at"]
        elif job.get("updated_at") is not None:
            reset_filter["updated_at"] = job["updated_at"]
        logger.warning(
            "Reclaiming expired processing lease for job %s (requested_by=%s)",
            job_id,
            requested_by,
        )

    history_entry = {
        "requested_at": now,
        "requested_by": requested_by or "unknown",
        "reason": (reason or "Operator requested reprocessing").strip()[:500],
        "previous_status": current_status,
        "previous_attempts": int(job.get("attempts") or 0),
        "previous_last_error": job.get("last_error"),
        "previous_pipeline_version": job.get("pipeline_version"),
        "force_reclaim": current_status == "processing",
        "contract_scope": contract_scope,
        "selected_copy_only": contract_scope == SELECTED_COPY_CONTRACT_SCOPE,
        "source_prompt_version": (
            contract_override.get("source_prompt_version")
            if contract_override
            else str(exam_contract.get("prompt_version") or "")
        ),
        "target_prompt_version": str(effective_contract.get("prompt_version") or ""),
        "contract_override_id": (
            contract_override.get("override_id") if contract_override else None
        ),
    }
    reset_fields: Dict[str, Any] = {
        "status": _queued_status_for_pipeline(pipeline_version),
        "pipeline_version": pipeline_version,
        "required_processing_path": required_processing_path,
        "db_name": db_name,
        "last_error": None,
        "segmentation": {},
        "evaluation": {},
        "reprocess_requested_at": now,
        "reprocess_requested_by": requested_by or "unknown",
        "reprocess_reason": history_entry["reason"],
        "mapping_pipeline_version": mapping_pipeline_version,
        "attempts": 0,
        "updated_at": now,
    }
    if contract_override:
        reset_fields["grading_contract_override"] = contract_override
    reset = await jobs.update_one(
        reset_filter,
        {
            "$set": reset_fields,
            "$unset": {
                "finished_at": "",
                "started_at": "",
                "lease_token": "",
                "lease_owner": "",
                "lease_expires_at": "",
                "heartbeat_at": "",
                "next_retry_at": "",
                "failure_code": "",
            },
            "$inc": {"reprocess_count": 1},
            "$push": {"reprocess_history": {"$each": [history_entry], "$slice": -20}},
        },
    )
    if not reset.matched_count:
        latest = await jobs.find_one({"job_id": job_id})
        if latest is not None and _lease_is_active(latest):
            raise ProcessingJobBusyError(
                "This answer copy was claimed by a worker while reprocessing was requested"
            )
        raise ProcessingJobBusyError(
            "The processing job changed while reprocessing was requested; refresh and try again"
        )

    refreshed = await jobs.find_one({"job_id": job_id})
    if refreshed is None:  # Defensive: the job was deleted after the reset.
        raise ValueError(f"Processing job {job_id} not found")
    return await dispatch_processing_job(tenant_db, db_name=db_name, job=refreshed, force=True)


async def recover_stale_processing_jobs(
    tenant_db: Any,
    *,
    now: Optional[datetime] = None,
) -> int:
    """Atomically return dead worker leases to the durable retry queue.

    Lease expiry remains the hard ownership boundary, but a worker that misses
    three consecutive heartbeats is also considered dead. The update predicate
    is evaluated atomically, so a concurrent heartbeat wins and prevents a live
    worker from being reclaimed. Late writes from a reclaimed worker remain
    fenced by the removed lease token.
    """

    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    recovered_at = now or _now()
    heartbeat_stale_before = recovered_at - timedelta(
        seconds=PROCESSING_HEARTBEAT_STALE_SECONDS
    )
    stale_predicates: List[Dict[str, Any]] = [
        {"status": "processing"},
        {
            "$or": [
                {"lease_expires_at": {"$lte": recovered_at}},
                {"updated_at": {"$lt": heartbeat_stale_before}},
                {"updated_at": {"$exists": False}},
            ]
        },
    ]
    exhausted_result = await jobs.update_many(
        {
            "$and": [
                *stale_predicates,
                {"attempts": {"$gte": PROCESSING_MAX_AUTOMATIC_ATTEMPTS}},
            ]
        },
        {
            "$set": {
                "status": "failed",
                "last_error": (
                    "Processing worker heartbeat stopped and the automatic retry "
                    "budget was exhausted"
                ),
                "failure_code": "ProcessingWorkerHeartbeatExpired",
                "stale_worker_recovered_at": recovered_at,
                "finished_at": recovered_at,
                "updated_at": recovered_at,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "next_retry_at": "",
            },
        },
    )
    retryable_attempt_predicate = {
        "$or": [
            {"attempts": {"$lt": PROCESSING_MAX_AUTOMATIC_ATTEMPTS}},
            {"attempts": {"$exists": False}},
        ]
    }
    v14_retry_result = await jobs.update_many(
        {
            "$and": [
                *stale_predicates,
                retryable_attempt_predicate,
                {"pipeline_version": PCR_V14_PIPELINE_VERSION},
            ]
        },
        {
            "$set": {
                "status": V14_CAPABILITY_QUEUED_JOB_STATUS,
                "last_error": (
                    "Processing worker heartbeat stopped; queued for automatic recovery"
                ),
                "failure_code": "ProcessingWorkerHeartbeatExpired",
                "next_retry_at": recovered_at
                + timedelta(seconds=PROCESSING_RETRY_BASE_SECONDS),
                "stale_worker_recovered_at": recovered_at,
                "updated_at": recovered_at,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "finished_at": "",
            },
        },
    )
    v16_retry_result = await jobs.update_many(
        {
            "$and": [
                *stale_predicates,
                retryable_attempt_predicate,
                {"pipeline_version": PCR_V16_PIPELINE_VERSION},
            ]
        },
        {
            "$set": {
                "status": V16_CAPABILITY_QUEUED_JOB_STATUS,
                "last_error": (
                    "Processing worker heartbeat stopped; queued for automatic recovery"
                ),
                "failure_code": "ProcessingWorkerHeartbeatExpired",
                "next_retry_at": recovered_at
                + timedelta(seconds=PROCESSING_RETRY_BASE_SECONDS),
                "stale_worker_recovered_at": recovered_at,
                "updated_at": recovered_at,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "finished_at": "",
            },
        },
    )
    v15_retry_result = await jobs.update_many(
        {
            "$and": [
                *stale_predicates,
                retryable_attempt_predicate,
                {"pipeline_version": PCR_V15_PIPELINE_VERSION},
            ]
        },
        {
            "$set": {
                "status": V15_CAPABILITY_QUEUED_JOB_STATUS,
                "last_error": (
                    "Processing worker heartbeat stopped; queued for automatic recovery"
                ),
                "failure_code": "ProcessingWorkerHeartbeatExpired",
                "next_retry_at": recovered_at
                + timedelta(seconds=PROCESSING_RETRY_BASE_SECONDS),
                "stale_worker_recovered_at": recovered_at,
                "updated_at": recovered_at,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "finished_at": "",
            },
        },
    )
    current_retry_result = await jobs.update_many(
        {
            "$and": [
                *stale_predicates,
                retryable_attempt_predicate,
                {
                    "pipeline_version": {
                        "$gte": 3,
                        "$lt": PCR_V14_PIPELINE_VERSION,
                    }
                },
            ]
        },
        {
            "$set": {
                "status": CAPABILITY_QUEUED_JOB_STATUS,
                "last_error": (
                    "Processing worker heartbeat stopped; queued for automatic recovery"
                ),
                "failure_code": "ProcessingWorkerHeartbeatExpired",
                "next_retry_at": recovered_at
                + timedelta(seconds=PROCESSING_RETRY_BASE_SECONDS),
                "stale_worker_recovered_at": recovered_at,
                "updated_at": recovered_at,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "finished_at": "",
            },
        },
    )
    legacy_retry_result = await jobs.update_many(
        {
            "$and": [
                *stale_predicates,
                retryable_attempt_predicate,
            ]
        },
        {
            "$set": {
                "status": "retryable_error",
                "last_error": (
                    "Processing worker heartbeat stopped; queued for automatic recovery"
                ),
                "failure_code": "ProcessingWorkerHeartbeatExpired",
                "next_retry_at": recovered_at
                + timedelta(seconds=PROCESSING_RETRY_BASE_SECONDS),
                "stale_worker_recovered_at": recovered_at,
                "updated_at": recovered_at,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "finished_at": "",
            },
        },
    )
    # Malformed legacy attempt counters must fail closed instead of producing an
    # unbounded crash/reclaim loop. Valid numeric rows were consumed above.
    malformed_result = await jobs.update_many(
        {"$and": stale_predicates},
        {
            "$set": {
                "status": "failed",
                "last_error": (
                    "Processing worker heartbeat stopped and its retry counter "
                    "is invalid"
                ),
                "failure_code": "InvalidProcessingAttemptCounter",
                "stale_worker_recovered_at": recovered_at,
                "finished_at": recovered_at,
                "updated_at": recovered_at,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "next_retry_at": "",
            },
        },
    )
    return int(
        exhausted_result.modified_count
        + v14_retry_result.modified_count
        + v15_retry_result.modified_count
        + v16_retry_result.modified_count
        + current_retry_result.modified_count
        + legacy_retry_result.modified_count
        + malformed_result.modified_count
    )


async def reconcile_processing_jobs(
    tenant_db: Any,
    *,
    db_name: str,
    limit: int = 200,
) -> Dict[str, int]:
    """Re-dispatch durable PCR jobs that were never picked up by a worker.

    Capture must remain successful even if Redis/Celery is briefly down.  The
    job record is therefore the source of truth and this periodic reconciler
    is responsible for replaying queued/enqueue-failed work.  A stale worker
    lease is also returned to the retryable state so a crashed worker cannot
    strand an entire conducted session forever.
    """
    await ensure_indexes(tenant_db)
    now = _now()
    stale_recovered = await recover_stale_processing_jobs(
        tenant_db,
        now=now,
    )

    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    retry_after = now - timedelta(seconds=60)
    cursor = jobs.find(
        {
            "status": {"$in": list(DISPATCHABLE_JOB_STATUSES)},
            "$and": [
                {
                    "$or": [
                        {"enqueue_attempted_at": {"$exists": False}},
                        {"enqueue_attempted_at": {"$lt": retry_after}},
                    ]
                },
                {
                    "$or": [
                        {"next_retry_at": {"$exists": False}},
                        {"next_retry_at": {"$lte": now}},
                    ]
                },
            ],
        }
    ).sort("updated_at", 1)
    pending = await cursor.to_list(length=max(1, min(limit, 1000)))
    dispatched = 0
    for job in pending:
        updated = await dispatch_processing_job(
            tenant_db,
            db_name=db_name,
            job=job,
        )
        if updated.get("status") in {
            "queued",
            CAPABILITY_QUEUED_JOB_STATUS,
            V14_CAPABILITY_QUEUED_JOB_STATUS,
            V15_CAPABILITY_QUEUED_JOB_STATUS,
            V16_CAPABILITY_QUEUED_JOB_STATUS,
            "processing",
        }:
            dispatched += 1

    return {
        "stale_recovered": stale_recovered,
        "dispatched": dispatched,
        "pending": len(pending),
    }


async def _claim_job(
    tenant_db: Any,
    job_id: str,
    *,
    execution_token: Optional[str] = None,
    required_pipeline_version: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    now = _now()
    lease_token = str(execution_token or uuid.uuid4().hex)
    claim_filter: Dict[str, Any] = {
        "job_id": job_id,
        "status": {"$in": list(DISPATCHABLE_JOB_STATUSES)},
        "$and": [
            {
                "$or": [
                    {"next_retry_at": {"$exists": False}},
                    {"next_retry_at": None},
                    {"next_retry_at": {"$lte": now}},
                ]
            }
        ],
    }
    if required_pipeline_version is not None:
        required_pipeline_version = int(required_pipeline_version)
        claim_filter["pipeline_version"] = required_pipeline_version
    else:
        claim_filter["$and"].append(
            {
                "$or": [
                    {"pipeline_version": {"$in": list(SUPPORTED_PCR_PIPELINE_VERSIONS)}},
                    {"pipeline_version": {"$exists": False}},
                ]
            }
        )
    worker_pipeline_version = int(required_pipeline_version or 0)
    if not worker_pipeline_version:
        # The claim filter is restricted to supported versions; this value is
        # replaced with the actual persisted target after the atomic claim.
        worker_pipeline_version = CURRENT_PCR_PIPELINE_VERSION
    result = await jobs.update_one(
        claim_filter,
        {
            "$set": {
                "status": "processing",
                "worker_pipeline_version": worker_pipeline_version,
                "started_at": now,
                "updated_at": now,
                "lease_token": lease_token,
                "lease_expires_at": now + timedelta(minutes=PROCESSING_LEASE_MINUTES),
                "last_error": None,
            },
            "$inc": {"attempts": 1, "lease_generation": 1},
        },
    )
    if result.matched_count != 1:
        return None
    claimed = await jobs.find_one({"job_id": job_id, "lease_token": lease_token})
    if claimed is not None and required_pipeline_version is None:
        actual_pipeline_version = int(
            claimed.get("pipeline_version") or CURRENT_PCR_PIPELINE_VERSION
        )
        if actual_pipeline_version not in SUPPORTED_PCR_PIPELINE_VERSIONS:
            raise ValueError(f"Unsupported PCR pipeline version: {actual_pipeline_version}")
        await jobs.update_one(
            {"job_id": job_id, "lease_token": lease_token},
            {"$set": {"worker_pipeline_version": actual_pipeline_version}},
        )
        claimed["worker_pipeline_version"] = actual_pipeline_version
    return claimed


async def _maybe_mark_exam_ready_for_review(tenant_db: Any, exam_id: str) -> None:
    """Advance uploading when every expected copy has reached its engine boundary.

    PCR requires a terminal OCR/segmentation/evaluation job.  DCR has no
    PCR job queue, so its boundary is canonical ingest; it then becomes ready
    for the dedicated DCR evaluation route.
    """
    exam_col = tenant_db["exampen_exams"]
    exam = await exam_col.find_one({"exam_id": exam_id})
    if exam is None or exam.get("lifecycle_state") != "uploading":
        return

    absent = {str(student_id) for student_id in (exam.get("absent_student_ids") or [])}
    expected = {str(student_id) for student_id in (exam.get("roster") or []) if str(student_id)} - absent
    if not expected:
        return

    submissions = await tenant_db["evalpen_submissions"].find({"exam_id": exam_id}).to_list(length=5000)
    by_student = {str(item.get("student_id")): item for item in submissions if item.get("student_id")}
    if not expected.issubset(by_student):
        return

    exam_type = str(exam.get("exam_type") or "")
    if exam_type == "pcr":
        submission_ids = [str(by_student[student_id].get("submission_id")) for student_id in expected]
        jobs = await tenant_db[PROCESSING_JOBS_COLLECTION].find(
            {"submission_id": {"$in": submission_ids}}
        ).to_list(length=5000)
        jobs_by_submission = {str(job.get("submission_id")): job for job in jobs}
        statuses = [
            str((jobs_by_submission.get(submission_id) or {}).get("status") or "not_enqueued")
            for submission_id in submission_ids
        ]
        if not all(job_status in {"completed", "blocked_for_review"} for job_status in statuses):
            return

    now = _now()
    result = await exam_col.update_one(
        {"exam_id": exam_id, "lifecycle_state": "uploading"},
        {
            "$set": {
                "lifecycle_state": "ready_for_eval",
                "ready_for_eval_at": now,
                "processing_completed_at": now,
                "updated_at": now,
            }
        },
    )
    if result.matched_count:
        logger.info("Exam %s automatically moved to ready_for_eval", exam_id)


async def process_pcr_processing_job(
    tenant_db: Any,
    job_id: str,
    *,
    execution_token: Optional[str] = None,
    required_pipeline_version: Optional[int] = None,
) -> Dict[str, Any]:
    """Run OCR/segmentation/evaluation for one persisted PCR job.

    Exceptions reach the worker boundary, which records them through the
    lease-fenced durable retry scheduler. Deterministic failures become
    terminal; transient failures are redelivered only when ``next_retry_at`` is
    due and only within the global attempt budget.
    """
    job = await _claim_job(
        tenant_db,
        job_id,
        execution_token=execution_token,
        required_pipeline_version=required_pipeline_version,
    )
    if job is None:
        existing = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": job_id})
        return {
            "job_id": job_id,
            "status": (existing or {}).get("status", "not_found"),
            "claimed": False,
        }

    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    lease_filter = _owned_lease_filter(job)
    submission_id = str(job["submission_id"])
    submission = await tenant_db["evalpen_submissions"].find_one({"submission_id": submission_id})
    if submission is None:
        failed = await jobs.update_one(
            lease_filter,
            {
                "$set": {
                    "status": "failed",
                    "last_error": "Canonical submission not found",
                    "updated_at": _now(),
                },
                "$unset": {"lease_token": "", "lease_expires_at": ""},
            },
        )
        if failed.matched_count != 1:
            return {"job_id": job_id, "status": "lease_lost", "claimed": False}
        return {"job_id": job_id, "status": "failed", "error": "Canonical submission not found"}
    if submission.get("publication_status") == "published":
        stopped = await jobs.update_one(
            lease_filter,
            {
                "$set": {
                    "status": "failed",
                    "last_error": (
                        "Published answer copies are immutable and cannot be reprocessed"
                    ),
                    "finished_at": _now(),
                    "updated_at": _now(),
                },
                "$unset": {"lease_token": "", "lease_expires_at": ""},
            },
        )
        if stopped.matched_count != 1:
            return {"job_id": job_id, "status": "lease_lost", "claimed": False}
        return {
            "job_id": job_id,
            "status": "failed",
            "error": "Published answer copies are immutable",
        }

    exam = await tenant_db["exampen_exams"].find_one({"exam_id": submission.get("exam_id")})
    if exam is None or exam.get("exam_type") != "pcr":
        failed = await jobs.update_one(
            lease_filter,
            {
                "$set": {
                    "status": "failed",
                    "last_error": "Submission is not attached to a PCR session",
                    "updated_at": _now(),
                },
                "$unset": {"lease_token": "", "lease_expires_at": ""},
            },
        )
        if failed.matched_count != 1:
            return {"job_id": job_id, "status": "lease_lost", "claimed": False}
        return {"job_id": job_id, "status": "failed", "error": "Not a PCR session"}

    if _grading_contract_migration_blocks_processing(exam):
        migration = dict(exam.get("pcr_grading_contract_migration") or {})
        paused = await jobs.update_one(
            lease_filter,
            {
                "$set": {
                    "status": CONTRACT_MIGRATION_PENDING_JOB_STATUS,
                    "migration_id": migration.get("migration_id"),
                    "updated_at": _now(),
                },
                "$unset": {
                    "lease_token": "",
                    "lease_owner": "",
                    "lease_expires_at": "",
                    "heartbeat_at": "",
                },
            },
        )
        if paused.matched_count != 1:
            return {"job_id": job_id, "status": "lease_lost", "claimed": False}
        return {
            "job_id": job_id,
            "status": CONTRACT_MIGRATION_PENDING_JOB_STATUS,
            "claimed": False,
        }

    expected_pipeline_version, expected_mapping_version = (
        await _pipeline_metadata_for_exam(
            tenant_db,
            str(submission.get("exam_id") or ""),
            job=job,
        )
    )
    effective_contract, contract_scope = effective_grading_contract(
        exam.get("pcr_grading_contract"),
        job,
    )
    has_frozen_contract = bool(effective_contract)
    persisted_pipeline_version = int(
        job.get("pipeline_version") or CURRENT_PCR_PIPELINE_VERSION
    )
    if (
        has_frozen_contract
        and (
            not is_supported_pcr_grading_contract(effective_contract)
            or persisted_pipeline_version != expected_pipeline_version
            or str(job.get("mapping_pipeline_version") or "")
            != expected_mapping_version
        )
    ):
        failed = await jobs.update_one(
            lease_filter,
            {
                "$set": {
                    "status": "failed",
                    "failure_code": "GradingContractJobMismatch",
                    "last_error": (
                        "Processing job metadata does not match its effective frozen "
                        f"grading contract ({contract_scope})"
                    ),
                    "updated_at": _now(),
                },
                "$unset": {
                    "lease_token": "",
                    "lease_owner": "",
                    "lease_expires_at": "",
                    "heartbeat_at": "",
                },
            },
        )
        if failed.matched_count != 1:
            return {"job_id": job_id, "status": "lease_lost", "claimed": False}
        return {
            "job_id": job_id,
            "status": "failed",
            "error": "Grading contract job mismatch",
        }

    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_evaluate_async import _build_eval_core
    from api.v1.evalpen_submissions_async import _build_submission_service

    # Pure multiple-choice PCR papers use an isolated answer-sheet lane. It
    # transcribes only student selections and never receives the answer key;
    # immutable server code applies positive/negative marking afterwards.
    # Subjective, integer-answer, and mixed papers explicitly decline this lane
    # and continue through the bounded whole-copy visual grader unchanged.
    pcr_services = load_exampen("pcr.services")
    LLMGate = load_exampen("llm_gate").LLMGate
    visual_gate = LLMGate(tenant_db)
    if hasattr(visual_gate, "initialize"):
        await visual_gate.initialize()
    objective_grader = pcr_services.ObjectiveAnswerSheetGradingService(
        tenant_db,
        visual_gate,
    )
    required_processing_path = await _required_processing_path(
        tenant_db,
        str(submission.get("exam_id") or ""),
    )
    worker_pipeline_version = int(job.get("pipeline_version") or CURRENT_PCR_PIPELINE_VERSION)
    if worker_pipeline_version not in SUPPORTED_PCR_PIPELINE_VERSIONS:
        raise ValueError(f"Unsupported PCR pipeline version: {worker_pipeline_version}")
    await jobs.update_one(
        lease_filter,
        {
            "$set": {
                "required_processing_path": required_processing_path,
                "worker_pipeline_version": worker_pipeline_version,
                "updated_at": _now(),
            }
        },
    )
    try:
        await jobs.update_one(
            lease_filter,
            {
                "$set": {
                    "progress": {
                        "stage": "objective_answer_extraction",
                        "message": "Checking the Objective answer-sheet contract",
                    },
                    "updated_at": _now(),
                }
            },
        )
        document_result = await _run_with_lease_heartbeat(
            tenant_db,
            job,
            objective_grader.grade_submission(submission_id),
        )
        if not document_result.handled:
            # Construct the Subjective grader only after the Objective service
            # proves the immutable catalog is subjective, integer, or mixed.
            full_document_grader = pcr_services.FullDocumentGradingService(
                tenant_db,
                visual_gate,
            )
            await jobs.update_one(
                lease_filter,
                {
                    "$set": {
                        "progress": {
                            "stage": "full_document_visual",
                            "message": "Reading the complete Subjective answer copy",
                        },
                        "updated_at": _now(),
                    }
                },
            )
            document_result = await _run_with_lease_heartbeat(
                tenant_db,
                job,
                full_document_grader.grade_submission(submission_id),
            )
    except ProcessingJobBusyError:
        logger.warning(
            "PCR worker lost its lease for job %s during visual grading",
            job_id,
        )
        return {"job_id": job_id, "status": "lease_lost", "claimed": False}

    if document_result.handled:
        processing_path = str(
            getattr(document_result, "processing_path", "")
            or FULL_DOCUMENT_PROCESSING_PATH
        )
        now = _now()
        final_status = str(document_result.status or "blocked_for_review")
        finished = await jobs.update_one(
            lease_filter,
            {
                "$set": {
                    "status": final_status,
                    "processing_path": processing_path,
                    "document_grading_run_id": document_result.run_id,
                    "segmentation": {
                        "path": processing_path,
                        "page_count": document_result.page_count,
                        "response_count": document_result.response_count,
                        "blocked_count": document_result.blocked_count,
                        "warning_count": document_result.warning_count,
                    },
                    "evaluation": {
                        "path": processing_path,
                        "evaluated_count": document_result.evaluated_count,
                        "blocked_count": document_result.blocked_count,
                        "error_count": len(document_result.errors),
                        "remaining_ready": 0,
                        "scored_questions": document_result.evaluated_count,
                        "missing_question_count": document_result.blocked_count,
                    },
                    "review": {
                        "state": document_result.review_state,
                        "document_review_required": (
                            document_result.document_review_required
                        ),
                        "reasons": document_result.review_reasons[:20],
                    },
                    "last_error": "; ".join(document_result.errors[:10]) or None,
                    "finished_at": now,
                    "updated_at": now,
                },
                "$unset": {"lease_token": "", "lease_expires_at": ""},
            },
        )
        if finished.matched_count != 1:
            logger.warning(
                "PCR worker lost its lease for job %s before document-grade commit",
                job_id,
            )
            return {"job_id": job_id, "status": "lease_lost", "claimed": False}
        await _maybe_mark_exam_ready_for_review(
            tenant_db,
            str(submission.get("exam_id")),
        )
        return {
            "job_id": job_id,
            "status": final_status,
            "submission_id": submission_id,
            "processing_path": processing_path,
            "document_grading_run_id": document_result.run_id,
            "evaluated_count": document_result.evaluated_count,
            "blocked_count": document_result.blocked_count,
            "errors": document_result.errors,
        }

    if required_processing_path == FULL_DOCUMENT_PROCESSING_PATH:
        raise pcr_services.FullDocumentGradingError(
            "Canonical full-document grading was required for this exam, but the "
            "worker declined that path"
            + (
                f": {document_result.skipped_reason}"
                if getattr(document_result, "skipped_reason", None)
                else ""
            )
        )

    processor = await _build_submission_service(tenant_db)
    try:
        processing_result = await _run_with_lease_heartbeat(
            tenant_db,
            job,
            processor.process_submission(submission_id),
        )
    except ProcessingJobBusyError:
        logger.warning("PCR worker lost its lease for job %s during answer mapping", job_id)
        return {"job_id": job_id, "status": "lease_lost", "claimed": False}
    if processing_result.error:
        failed = await jobs.update_one(
            lease_filter,
            {
                "$set": {
                    "status": "failed",
                    "last_error": str(processing_result.error),
                    "segmentation": {
                        "page_count": processing_result.page_count,
                        "response_count": processing_result.response_count,
                    },
                    "finished_at": _now(),
                    "updated_at": _now(),
                },
                "$unset": {"lease_token": "", "lease_expires_at": ""},
            },
        )
        if failed.matched_count != 1:
            return {"job_id": job_id, "status": "lease_lost", "claimed": False}
        return {"job_id": job_id, "status": "failed", "error": processing_result.error}

    # Soft-unblock false "diagram heavy" / geometry blocks when the response
    # already has a question id and student text.  Those rows must be scored;
    # teachers can still see the warning flags on the review card.
    await _soften_false_blocking_flags(tenant_db, submission_id)

    evaluation_errors: List[str] = []
    evaluated_count = 0
    blocked_count = 0
    eval_core = await _build_eval_core(tenant_db)

    async def _evaluate_scoreable_batch() -> None:
        nonlocal evaluated_count, blocked_count
        batch = await tenant_db["evalpen_detected_responses"].find(
            {
                "submission_id": submission_id,
                "superseded_at": {"$exists": False},
                "eval_status": {
                    "$in": ["ready", "ready_with_warnings", "blocked"]
                },
            }
        ).to_list(length=2000)
        await _heartbeat_processing_job(tenant_db, job)
        for response_index, response in enumerate(batch):
            if response_index and response_index % 10 == 0:
                await _heartbeat_processing_job(tenant_db, job)
            response_id = str(response.get("response_id") or "")
            question_id = response.get("question_id")
            eval_status = str(response.get("eval_status") or "")
            detected_text = str(response.get("detected_text") or "").strip()
            is_missing = bool(response.get("is_missing_response"))

            if not response_id:
                continue

            # Unmapped text stays blocked for teacher ownership — never invent
            # a question id.  Missing-answer slots always have a question id.
            if not question_id:
                if eval_status != "blocked":
                    await tenant_db["evalpen_detected_responses"].update_one(
                        {"response_id": response_id},
                        {
                            "$set": {
                                "eval_status": "blocked",
                                "manual_review_reason": (
                                    "Question could not be safely associated "
                                    "from the submitted copy"
                                ),
                                "updated_at": _now(),
                            }
                        },
                    )
                blocked_count += 1
                continue

            # Still-blocked rows without missing-slot intent and without text
            # stay blocked (true diagram blanks / unreadable).
            if eval_status == "blocked" and not is_missing and not detected_text:
                blocked_count += 1
                continue

            # Promote blocked-but-scoreable rows so EvalCore does not skip them.
            if eval_status == "blocked":
                await tenant_db["evalpen_detected_responses"].update_one(
                    {"response_id": response_id},
                    {
                        "$set": {
                            "eval_status": "ready_with_warnings",
                            "updated_at": _now(),
                        }
                    },
                )

            try:
                result = await eval_core.evaluate_response(
                    response_id, question_id=str(question_id)
                )
            except Exception as exc:
                logger.exception(
                    "Evaluation crashed for response %s question %s",
                    response_id,
                    question_id,
                )
                evaluation_errors.append(f"{response_id}: {exc}")
                await tenant_db["evalpen_detected_responses"].update_one(
                    {"response_id": response_id},
                    {
                        "$set": {
                            "eval_status": "ready",
                            "manual_review_reason": (
                                f"Evaluation failed: {_short_error(exc)}"
                            ),
                            "updated_at": _now(),
                        }
                    },
                )
                continue
            if result.error:
                evaluation_errors.append(f"{response_id}: {result.error}")
            elif result.skipped:
                blocked_count += 1
            else:
                evaluated_count += 1

    # Pass 1+2: score every scoreable row (including not-attempted zeros).
    try:
        await _evaluate_scoreable_batch()
        await _evaluate_scoreable_batch()
    except ProcessingJobBusyError:
        logger.warning("PCR worker lost its lease for job %s during evaluation", job_id)
        return {"job_id": job_id, "status": "lease_lost", "claimed": False}

    # Pass 3: evaluate any existing paper rows and report questions whose
    # answer state is unresolved.  Missing database rows are never proof that
    # the student left a question blank.
    try:
        paper_coverage = await _ensure_full_paper_evaluations(
            tenant_db,
            eval_core=eval_core,
            submission_id=submission_id,
            exam_id=str(submission.get("exam_id") or ""),
            student_id=str(submission.get("student_id") or ""),
            lease_heartbeat=lambda: _heartbeat_processing_job(tenant_db, job),
        )
    except ProcessingJobBusyError:
        logger.warning("PCR worker lost its lease for job %s during paper coverage", job_id)
        return {"job_id": job_id, "status": "lease_lost", "claimed": False}
    evaluated_count += int(paper_coverage.get("evaluated_existing") or 0)
    evaluation_errors.extend(paper_coverage.get("errors") or [])
    missing_question_count = int(paper_coverage.get("missing_question_count") or 0)
    if missing_question_count:
        evaluation_errors.append(
            f"{missing_question_count} paper question(s) have no verified answer state"
        )

    if processing_result.response_count == 0 and evaluated_count == 0:
        evaluation_errors.append("No student responses were detected")

    remaining_ready = await tenant_db["evalpen_detected_responses"].count_documents(
        {
            "submission_id": submission_id,
            "superseded_at": {"$exists": False},
            "eval_status": {"$in": ["ready", "ready_with_warnings"]},
        }
    )
    blocked_count = await tenant_db["evalpen_detected_responses"].count_documents(
        {
            "submission_id": submission_id,
            "superseded_at": {"$exists": False},
            "eval_status": "blocked",
        }
    )

    # Terminal statuses teachers can reprocess from.  Prefer completed when
    # every paper question has a score, even if some unmapped text remains blocked.
    scored_questions = await tenant_db["evalpen_evaluations"].count_documents(
        {
            "response_id": {
                "$in": [
                    doc["response_id"]
                    for doc in await tenant_db["evalpen_detected_responses"]
                    .find(
                        {
                            "submission_id": submission_id,
                            "superseded_at": {"$exists": False},
                            "question_id": {"$exists": True, "$nin": [None, ""]},
                        },
                        {"response_id": 1},
                    )
                    .to_list(length=2000)
                ]
            }
        }
    )
    if remaining_ready > 0:
        final_status = "failed"
        if not any("waiting for evaluation" in e for e in evaluation_errors):
            evaluation_errors.append(
                f"{remaining_ready} answer(s) still waiting for evaluation"
            )
    elif missing_question_count:
        final_status = "blocked_for_review"
    elif evaluated_count == 0 and evaluation_errors:
        final_status = "failed"
    elif blocked_count and scored_questions == 0:
        final_status = "blocked_for_review"
    elif blocked_count:
        # Some unmapped evidence remains, but the paper has marks — reviewable.
        final_status = "blocked_for_review"
    else:
        final_status = "completed"
    now = _now()
    finished = await jobs.update_one(
        lease_filter,
        {
            "$set": {
                "status": final_status,
                "segmentation": {
                    "page_count": processing_result.page_count,
                    "response_count": processing_result.response_count,
                    "blocked_count": processing_result.blocked_count,
                    "warning_count": processing_result.warning_count,
                },
                "evaluation": {
                    "evaluated_count": evaluated_count,
                    "blocked_count": blocked_count,
                    "error_count": len(evaluation_errors),
                    "remaining_ready": remaining_ready,
                    "scored_questions": scored_questions,
                    "missing_question_count": missing_question_count,
                },
                "last_error": "; ".join(evaluation_errors[:10]) or None,
                "finished_at": now,
                "updated_at": now,
            },
            "$unset": {"lease_token": "", "lease_expires_at": ""},
        },
    )
    if finished.matched_count != 1:
        logger.warning("PCR worker lost its lease for job %s before final commit", job_id)
        return {"job_id": job_id, "status": "lease_lost", "claimed": False}
    await _maybe_mark_exam_ready_for_review(tenant_db, str(submission.get("exam_id")))
    return {
        "job_id": job_id,
        "status": final_status,
        "submission_id": submission_id,
        "evaluated_count": evaluated_count,
        "blocked_count": blocked_count,
        "errors": evaluation_errors,
    }


_SOFT_BLOCK_FLAG_TYPES = {
    "DIAGRAM_HEAVY_CONTENT",
    "diagram_heavy_content",
    "LOW_SEGMENTATION_CONFIDENCE",
    "low_segmentation_confidence",
}


async def _soften_false_blocking_flags(tenant_db: Any, submission_id: str) -> int:
    """Re-open blocked responses that already have Q ownership + OCR text.

    Production failure: reprocess classified sparse handwriting as diagram-heavy
    (blocking), set eval_status=blocked, and left the paper at 0/9 evaluated.
    """
    cursor = tenant_db["evalpen_detected_responses"].find(
        {
            "submission_id": submission_id,
            "superseded_at": {"$exists": False},
            "eval_status": "blocked",
            "question_id": {"$exists": True, "$nin": [None, ""]},
        }
    )
    docs = await cursor.to_list(length=2000)
    softened = 0
    for doc in docs:
        text = str(doc.get("detected_text") or "").strip()
        if not text and not doc.get("is_missing_response"):
            continue
        flags = doc.get("flags") or []
        hard_blocks = []
        for flag in flags:
            if not isinstance(flag, dict):
                continue
            if str(flag.get("severity") or "").lower() != "blocking":
                continue
            flag_type = str(flag.get("flag_type") or "")
            if flag_type not in _SOFT_BLOCK_FLAG_TYPES:
                hard_blocks.append(flag_type)
        if hard_blocks:
            continue
        # Demote soft blocking flags to warning so EvalCore will score them.
        new_flags = []
        for flag in flags:
            if not isinstance(flag, dict):
                continue
            updated = dict(flag)
            if (
                str(updated.get("severity") or "").lower() == "blocking"
                and str(updated.get("flag_type") or "") in _SOFT_BLOCK_FLAG_TYPES
            ):
                updated["severity"] = "warning"
                updated["suggested_action"] = (
                    updated.get("suggested_action")
                    or "Auto-scored with caution; review if needed"
                )
            new_flags.append(updated)
        await tenant_db["evalpen_detected_responses"].update_one(
            {"response_id": doc.get("response_id")},
            {
                "$set": {
                    "eval_status": "ready_with_warnings",
                    "flags": new_flags,
                    "manual_review_reason": None,
                    "updated_at": _now(),
                }
            },
        )
        softened += 1
    if softened:
        logger.info(
            "Softened %d false blocking flag(s) for submission %s before scoring",
            softened,
            submission_id,
        )
    return softened


async def _ensure_full_paper_evaluations(
    tenant_db: Any,
    *,
    eval_core: Any,
    submission_id: str,
    exam_id: str,
    student_id: str,
    lease_heartbeat: Optional[Callable[[], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    """Evaluate existing rows and report unresolved paper coverage.

    Absence of a response document is not evidence that the student skipped a
    question.  Only ingestion may create a ``not_attempted`` row, and only
    after document coverage has been independently verified.
    """
    evaluated_existing = 0
    errors: List[str] = []
    if not exam_id:
        return {
            "evaluated_existing": 0,
            "missing_question_count": 0,
            "missing_question_ids": [],
            "errors": errors,
        }

    questions = await tenant_db["evalpen_questions"].find(
        {"exam_id": exam_id},
        {"question_id": 1, "question_number": 1, "max_marks": 1},
    ).sort([("question_number", 1)]).to_list(length=500)

    responses = await tenant_db["evalpen_detected_responses"].find(
        {
            "submission_id": submission_id,
            "superseded_at": {"$exists": False},
        }
    ).to_list(length=2000)
    response_by_question = {
        str(doc.get("question_id")): doc
        for doc in responses
        if doc.get("question_id")
    }

    missing_question_ids: List[str] = []
    for question_index, question in enumerate(questions):
        if lease_heartbeat is not None and question_index % 10 == 0:
            await lease_heartbeat()
        question_id = str(question.get("question_id") or "").strip()
        if not question_id:
            continue
        response = response_by_question.get(question_id)
        if response is None:
            missing_question_ids.append(question_id)
            continue

        response_id = str(response.get("response_id") or "")
        existing_eval = await tenant_db["evalpen_evaluations"].find_one(
            {"response_id": response_id}
        )
        if existing_eval:
            continue
        # Response exists but was never scored — force a mark pass.
        if str(response.get("eval_status")) == "blocked" and not str(
            response.get("detected_text") or ""
        ).strip():
            continue
        try:
            if str(response.get("eval_status")) == "blocked":
                await tenant_db["evalpen_detected_responses"].update_one(
                    {"response_id": response_id},
                    {"$set": {"eval_status": "ready_with_warnings", "updated_at": _now()}},
                )
            result = await eval_core.evaluate_response(
                response_id, question_id=question_id
            )
            if result.error:
                errors.append(f"{response_id}: {result.error}")
            elif not result.skipped:
                evaluated_existing += 1
        except Exception as exc:
            errors.append(f"{response_id}: {_short_error(exc)}")

    return {
        "evaluated_existing": evaluated_existing,
        "missing_question_count": len(missing_question_ids),
        "missing_question_ids": missing_question_ids,
        "errors": errors,
    }


async def mark_processing_job_retryable_error(
    tenant_db: Any,
    job_id: str,
    error: Exception | str,
    *,
    expected_lease_token: Optional[str] = None,
) -> bool:
    """Persist a retryable failure only for the worker that owned the lease."""

    query: Dict[str, Any] = {"job_id": job_id}
    if expected_lease_token:
        query.update(
            {
                "status": "processing",
                "lease_token": str(expected_lease_token),
            }
        )
    else:
        # Legacy callers have no ownership proof. They may update an idle job,
        # but must never fence out an active worker.
        query["status"] = {"$ne": "processing"}
    job = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
        query,
        {"pipeline_version": 1},
    )
    if job is None:
        return False
    result = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        query,
        {
            "$set": {
                "status": _retryable_status_for_pipeline(
                    job.get("pipeline_version")
                ),
                "last_error": _short_error(error),
                "updated_at": _now(),
            },
            "$unset": {"lease_token": "", "lease_expires_at": ""},
        },
    )
    return result.matched_count == 1


async def record_processing_job_failure(
    tenant_db: Any,
    job_id: str,
    error: Exception | str,
    *,
    expected_lease_token: str,
) -> Dict[str, Any]:
    """Persist one worker failure and let MongoDB own bounded retry scheduling.

    Celery delivery is at-least-once, so mixing ``self.retry`` with the durable
    reconciler creates parallel retry chains. This function records exactly one
    next attempt behind the worker lease fence. Deterministic errors may opt out
    of retries with ``retryable = False`` on their exception class.
    """

    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    query: Dict[str, Any] = {
        "job_id": job_id,
        "status": "processing",
        "lease_token": str(expected_lease_token),
    }
    job = await jobs.find_one(query)
    if job is None:
        return {
            "recorded": False,
            "terminal": False,
            "status": "lease_lost",
        }

    try:
        attempts = max(1, int(job.get("attempts") or 1))
    except (TypeError, ValueError):
        attempts = PROCESSING_MAX_AUTOMATIC_ATTEMPTS
    retryable = bool(getattr(error, "retryable", True))
    terminal = not retryable or attempts >= PROCESSING_MAX_AUTOMATIC_ATTEMPTS
    now = _now()
    error_text = _short_error(error)
    failure_code = type(error).__name__

    if terminal:
        update: Dict[str, Any] = {
            "$set": {
                "status": "failed",
                "last_error": error_text,
                "failure_code": failure_code,
                "finished_at": now,
                "updated_at": now,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "next_retry_at": "",
            },
        }
        status = "failed"
        next_retry_at = None
    else:
        retry_delay = min(
            PROCESSING_RETRY_MAX_SECONDS,
            PROCESSING_RETRY_BASE_SECONDS * (2 ** (attempts - 1)),
        )
        next_retry_at = now + timedelta(seconds=retry_delay)
        update = {
            "$set": {
                "status": _retryable_status_for_pipeline(
                    job.get("pipeline_version")
                ),
                "last_error": error_text,
                "failure_code": failure_code,
                "next_retry_at": next_retry_at,
                "updated_at": now,
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "finished_at": "",
            },
        }
        status = _retryable_status_for_pipeline(job.get("pipeline_version"))

    result = await jobs.update_one(query, update)
    return {
        "recorded": result.matched_count == 1,
        "terminal": terminal,
        "status": status if result.matched_count == 1 else "lease_lost",
        "attempts": attempts,
        "next_retry_at": next_retry_at,
    }


async def mark_processing_job_failed(
    tenant_db: Any,
    job_id: str,
    error: Exception | str,
    *,
    expected_lease_token: Optional[str] = None,
) -> bool:
    """Persist terminal failure only for the worker that owned the lease."""

    query: Dict[str, Any] = {"job_id": job_id}
    if expected_lease_token:
        query.update(
            {
                "status": "processing",
                "lease_token": str(expected_lease_token),
            }
        )
    else:
        query["status"] = {"$ne": "processing"}
    result = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        query,
        {
            "$set": {
                "status": "failed",
                "last_error": _short_error(error),
                "finished_at": _now(),
                "updated_at": _now(),
            },
            "$unset": {"lease_token": "", "lease_expires_at": ""},
        },
    )
    return result.matched_count == 1
