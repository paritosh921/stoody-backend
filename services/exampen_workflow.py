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

logger = logging.getLogger(__name__)

PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
TERMINAL_JOB_STATUSES = {"completed", "blocked_for_review", "failed"}
DISPATCHABLE_JOB_STATUSES = {"queued", "retryable_error", "enqueue_failed"}
CURRENT_PCR_PIPELINE_VERSION = 3
FULL_DOCUMENT_PROCESSING_PATH = "full_document_visual"
PROCESSING_LEASE_MINUTES = 30
PROCESSING_HEARTBEAT_SECONDS = 60
PROCESSING_MAX_AUTOMATIC_ATTEMPTS = 3
PROCESSING_RETRY_BASE_SECONDS = 60
PROCESSING_RETRY_MAX_SECONDS = 15 * 60
# A live worker renews ``updated_at`` every heartbeat. Waiting for the full
# 30-minute ownership lease after a process crash leaves the UI stuck on
# "Checking" even though no worker exists. Three missed heartbeats provide a
# conservative crash detector while the lease token still fences late writes
# from the old owner.
PROCESSING_HEARTBEAT_STALE_SECONDS = PROCESSING_HEARTBEAT_SECONDS * 3


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


def _now() -> datetime:
    return datetime.now(timezone.utc)


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


def dispatchable_job_filter(*, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Return the database-authoritative eligibility predicate for workers."""

    ready_at = now or _now()
    return {
        "status": {"$in": list(DISPATCHABLE_JOB_STATUSES)},
        "$or": [
            {"next_attempt_at": {"$exists": False}},
            {"next_attempt_at": None},
            {"next_attempt_at": {"$lte": ready_at}},
        ],
    }


def _retry_delay_seconds(attempts: int) -> int:
    exponent = max(0, int(attempts) - 1)
    return min(
        PROCESSING_RETRY_MAX_SECONDS,
        PROCESSING_RETRY_BASE_SECONDS * (2**exponent),
    )


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
            "objective-answer-ledger-v1",
            "objective-answer-ledger-v2",
            "objective-answer-ledger-v3",
        }
    ):
        return FULL_DOCUMENT_PROCESSING_PATH
    return "legacy_ocr_mapping"


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
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    job_id = _job_id(submission_id)
    now = _now()
    result = await jobs.update_one(
        {"submission_id": submission_id},
        {
            "$setOnInsert": {
                "job_id": job_id,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "student_id": student_id,
                "db_name": db_name,
                "pipeline_version": CURRENT_PCR_PIPELINE_VERSION,
                "required_processing_path": required_processing_path,
                "status": "queued",
                "attempts": 0,
                "lease_generation": 0,
                "created_at": now,
                "updated_at": now,
                "last_error": None,
            }
        },
        upsert=True,
    )
    job = await jobs.find_one({"submission_id": submission_id})
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

    now = _now()
    next_attempt_at = _as_utc(job.get("next_attempt_at"))
    if not force and next_attempt_at is not None and next_attempt_at > now:
        return job
    required_processing_path = await _required_processing_path(
        tenant_db,
        str(job.get("exam_id") or ""),
    )
    queued = await jobs.update_one(
        {"job_id": job["job_id"], "status": {"$ne": "processing"}},
        {
            "$set": {
                "status": "queued",
                "pipeline_version": CURRENT_PCR_PIPELINE_VERSION,
                "required_processing_path": required_processing_path,
                "db_name": db_name,
                "enqueue_attempted_at": now,
                "updated_at": now,
                "last_error": None,
            },
            "$unset": {"next_attempt_at": ""},
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
            {"job_id": job_id, "status": "queued"},
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
            CURRENT_PCR_PIPELINE_VERSION,
        )
    except Exception as exc:
        logger.exception("Unable to enqueue PCR processing job %s", job_id)
        await jobs.update_one(
            # If the worker claimed the job before the broker client raised,
            # preserve the worker state rather than overwriting it with an
            # enqueue error.
            {"job_id": job_id, "status": "queued"},
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

    The canonical uploaded pages remain unchanged. The primary path inspects
    the immutable paper, teacher solution, and full answer copy together; the
    OCR/segmentation service is retained only for legacy sessions without the
    required canonical files. Fresh response rows supersede old mappings, so
    stale marks cannot be mixed with the new evidence ledger.
    """
    await ensure_indexes(tenant_db)
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    job = await jobs.find_one({"job_id": job_id})
    if job is None:
        raise ValueError(f"Processing job {job_id} not found")

    now = _now()
    required_processing_path = await _required_processing_path(
        tenant_db,
        str(job.get("exam_id") or ""),
    )
    current_status = str(job.get("status") or "queued")
    if _lease_is_active(job, now=now):
        expiry = _lease_expiry(job)
        raise ProcessingJobBusyError(
            "This answer copy is still being processed by an active worker"
            + (f" until {expiry.isoformat()}" if expiry else "")
            + ". Wait for it to finish before reprocessing."
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
    }
    reset = await jobs.update_one(
        reset_filter,
        {
            "$set": {
                "status": "queued",
                "pipeline_version": CURRENT_PCR_PIPELINE_VERSION,
                "required_processing_path": required_processing_path,
                "db_name": db_name,
                "last_error": None,
                "segmentation": {},
                "evaluation": {},
                "reprocess_requested_at": now,
                "reprocess_requested_by": requested_by or "unknown",
                "reprocess_reason": history_entry["reason"],
                "mapping_pipeline_version": "full-document-visual-v2",
                "attempts": 0,
                "updated_at": now,
            },
            "$unset": {
                "finished_at": "",
                "started_at": "",
                "lease_token": "",
                "lease_expires_at": "",
                "next_attempt_at": "",
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
    stale_result = await jobs.update_many(
        {
            "status": "processing",
            "$or": [
                {"lease_expires_at": {"$lte": recovered_at}},
                {"updated_at": {"$lt": heartbeat_stale_before}},
                {"updated_at": {"$exists": False}},
            ],
        },
        {
            "$set": {
                "status": "retryable_error",
                "last_error": (
                    "Processing worker heartbeat stopped; queued for automatic recovery"
                ),
                "stale_worker_recovered_at": recovered_at,
                "next_attempt_at": recovered_at,
                "updated_at": recovered_at,
            },
            "$unset": {"lease_token": "", "lease_expires_at": ""},
        },
    )
    return int(stale_result.modified_count)


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
            "$and": [
                dispatchable_job_filter(now=now),
                {
                    "$or": [
                        {"enqueue_attempted_at": {"$exists": False}},
                        {"enqueue_attempted_at": {"$lt": retry_after}},
                    ],
                },
            ]
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
        if updated.get("status") in {"queued", "processing"}:
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
        "$and": [
            {"job_id": job_id},
            dispatchable_job_filter(now=now),
        ]
    }
    if required_pipeline_version is not None:
        claim_filter["$and"].append(
            {"pipeline_version": int(required_pipeline_version)}
        )
    result = await jobs.update_one(
        claim_filter,
        {
            "$set": {
                "status": "processing",
                "worker_pipeline_version": CURRENT_PCR_PIPELINE_VERSION,
                "started_at": now,
                "updated_at": now,
                "lease_token": lease_token,
                "lease_expires_at": now + timedelta(minutes=PROCESSING_LEASE_MINUTES),
                "last_error": None,
            },
            "$unset": {"next_attempt_at": ""},
            "$inc": {"attempts": 1, "lease_generation": 1},
        },
    )
    if result.matched_count != 1:
        return None
    return await jobs.find_one({"job_id": job_id, "lease_token": lease_token})


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

    Exceptions from transient infrastructure are allowed to reach Celery so it
    can retry.  Expected content-quality failures are persisted as a terminal
    ``failed`` job for teacher/admin action rather than retried indefinitely.
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

    from api.v1._exampen_imports import load_exampen
    from api.v1.evalpen_evaluate_async import _build_eval_core
    from api.v1.evalpen_submissions_async import _build_submission_service

    # Primary PCR camera/PDF path: the model sees the immutable question
    # paper, teacher solution, and complete student copy in one visual request.
    # OCR/segmentation remains a legacy fallback only when a session predates
    # the canonical paper assets or the feature is explicitly disabled.
    pcr_services = load_exampen("pcr.services")
    LLMGate = load_exampen("llm_gate").LLMGate
    full_document_gate = LLMGate(tenant_db)
    if hasattr(full_document_gate, "initialize"):
        await full_document_gate.initialize()
    full_document_grader = pcr_services.FullDocumentGradingService(
        tenant_db,
        full_document_gate,
    )
    required_processing_path = await _required_processing_path(
        tenant_db,
        str(submission.get("exam_id") or ""),
    )
    await jobs.update_one(
        lease_filter,
        {
            "$set": {
                "required_processing_path": required_processing_path,
                "worker_pipeline_version": CURRENT_PCR_PIPELINE_VERSION,
                "updated_at": _now(),
            }
        },
    )
    try:
        document_result = await _run_with_lease_heartbeat(
            tenant_db,
            job,
            full_document_grader.grade_submission(submission_id),
        )
    except ProcessingJobBusyError:
        logger.warning(
            "PCR worker lost its lease for job %s during full-document grading",
            job_id,
        )
        return {"job_id": job_id, "status": "lease_lost", "claimed": False}

    if document_result.handled:
        now = _now()
        final_status = str(document_result.status or "blocked_for_review")
        processing_path = str(
            document_result.processing_path or "full_document_visual"
        )
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
                    # A handled grading result is an operational success even
                    # when individual answers require teacher review.  Keep
                    # those diagnostics for audit, but never expose them as a
                    # worker failure through ``last_error``.
                    "diagnostics": {
                        "errors": document_result.errors[:50],
                        "recorded_at": now,
                    },
                    "last_error": None,
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
    """Persist bounded retry state owned by MongoDB, not worker memory."""

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
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    current = await jobs.find_one(query, {"attempts": 1})
    if current is None:
        return False
    attempts = max(0, int(current.get("attempts") or 0))
    now = _now()
    terminal = attempts >= PROCESSING_MAX_AUTOMATIC_ATTEMPTS
    status_value = "failed" if terminal else "retryable_error"
    update: Dict[str, Any] = {
        "$set": {
            "status": status_value,
            "last_error": _short_error(error),
            "failure_code": (
                type(error).__name__
                if isinstance(error, Exception)
                else "worker_error"
            ),
            "updated_at": now,
        },
        "$unset": {"lease_token": "", "lease_expires_at": ""},
        "$push": {
            "failure_history": {
                "$each": [
                    {
                        "attempt": attempts,
                        "status": status_value,
                        "error": _short_error(error),
                        "failed_at": now,
                    }
                ],
                "$slice": -20,
            }
        },
    }
    if terminal:
        update["$set"]["finished_at"] = now
        update["$unset"]["next_attempt_at"] = ""
    else:
        update["$set"]["next_attempt_at"] = now + timedelta(
            seconds=_retry_delay_seconds(attempts)
        )
        update["$unset"]["finished_at"] = ""
    result = await jobs.update_one(
        query,
        update,
    )
    return result.matched_count == 1


async def release_processing_job_for_restart(
    tenant_db: Any,
    job_id: str,
    *,
    expected_lease_token: str,
) -> bool:
    """Release a cancelled worker lease without consuming failure budget."""

    now = _now()
    result = await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        {
            "job_id": job_id,
            "status": "processing",
            "lease_token": str(expected_lease_token),
        },
        {
            "$set": {
                "status": "retryable_error",
                "last_error": "Processing worker restarted; queued for recovery",
                "next_attempt_at": now,
                "updated_at": now,
            },
            "$unset": {"lease_token": "", "lease_expires_at": ""},
        },
    )
    return result.matched_count == 1


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
                "failure_code": (
                    type(error).__name__
                    if isinstance(error, Exception)
                    else "worker_error"
                ),
                "finished_at": _now(),
                "updated_at": _now(),
            },
            "$unset": {
                "lease_token": "",
                "lease_expires_at": "",
                "next_attempt_at": "",
            },
        },
    )
    return result.matched_count == 1
