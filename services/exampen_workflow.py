"""Durable PCR processing workflow for conducted ExamPen submissions.

Capture endpoints only own canonical ingest.  This module persists an
idempotent job after ingest, dispatches it to Celery, and records enough state
for the web UI and lifecycle coordinator to explain what is still pending.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
TERMINAL_JOB_STATUSES = {"completed", "blocked_for_review", "failed"}
DISPATCHABLE_JOB_STATUSES = {"queued", "retryable_error", "enqueue_failed"}
PROCESSING_LEASE_MINUTES = 30


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _job_id(submission_id: str) -> str:
    return f"pcr-job-{submission_id}"


def _short_error(error: Exception | str) -> str:
    message = str(error).replace("\n", " ").strip()
    return message[:800] or type(error).__name__


async def ensure_indexes(tenant_db: Any) -> None:
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    await jobs.create_index("job_id", unique=True, name="uniq_processing_job")
    await jobs.create_index("submission_id", unique=True, name="uniq_processing_submission")
    await jobs.create_index([("exam_id", 1), ("status", 1)], name="idx_processing_exam_status")


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
                "pipeline_version": 1,
                "status": "queued",
                "attempts": 0,
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
    if not force and status in TERMINAL_JOB_STATUSES | {"processing"}:
        return job

    now = _now()
    await jobs.update_one(
        {"job_id": job["job_id"]},
        {
            "$set": {
                "status": "queued",
                "db_name": db_name,
                "enqueue_attempted_at": now,
                "updated_at": now,
                "last_error": None,
            }
        },
    )
    try:
        from celery_app import process_exampen_pcr_submission

        process_exampen_pcr_submission.delay(db_name, job["job_id"])
    except Exception as exc:
        logger.exception("Unable to enqueue PCR processing job %s", job["job_id"])
        await jobs.update_one(
            {"job_id": job["job_id"]},
            {
                "$set": {
                    "status": "enqueue_failed",
                    "last_error": _short_error(exc),
                    "updated_at": _now(),
                }
            },
        )
    return await jobs.find_one({"job_id": job["job_id"]})


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
    """Reset a failed/pending job and send it to Celery again."""
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    job = await jobs.find_one({"job_id": job_id})
    if job is None:
        raise ValueError(f"Processing job {job_id} not found")
    if str(job.get("status")) == "processing":
        return job
    return await dispatch_processing_job(tenant_db, db_name=db_name, job=job, force=True)


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
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    now = _now()
    stale_before = now - timedelta(minutes=PROCESSING_LEASE_MINUTES)
    stale_result = await jobs.update_many(
        {
            "status": "processing",
            "updated_at": {"$lt": stale_before},
        },
        {
            "$set": {
                "status": "retryable_error",
                "last_error": "Processing worker lease expired; queued for recovery",
                "updated_at": now,
            }
        },
    )

    retry_after = now - timedelta(seconds=60)
    cursor = jobs.find(
        {
            "status": {"$in": list(DISPATCHABLE_JOB_STATUSES)},
            "$or": [
                {"enqueue_attempted_at": {"$exists": False}},
                {"enqueue_attempted_at": {"$lt": retry_after}},
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
        if updated.get("status") in {"queued", "processing"}:
            dispatched += 1

    return {
        "stale_recovered": int(stale_result.modified_count),
        "dispatched": dispatched,
        "pending": len(pending),
    }


async def _claim_job(tenant_db: Any, job_id: str) -> Optional[Dict[str, Any]]:
    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    now = _now()
    result = await jobs.update_one(
        {"job_id": job_id, "status": {"$in": list(DISPATCHABLE_JOB_STATUSES)}},
        {
            "$set": {
                "status": "processing",
                "started_at": now,
                "updated_at": now,
                "last_error": None,
            },
            "$inc": {"attempts": 1},
        },
    )
    if result.matched_count != 1:
        return None
    return await jobs.find_one({"job_id": job_id})


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


async def process_pcr_processing_job(tenant_db: Any, job_id: str) -> Dict[str, Any]:
    """Run OCR/segmentation/evaluation for one persisted PCR job.

    Exceptions from transient infrastructure are allowed to reach Celery so it
    can retry.  Expected content-quality failures are persisted as a terminal
    ``failed`` job for teacher/admin action rather than retried indefinitely.
    """
    job = await _claim_job(tenant_db, job_id)
    if job is None:
        existing = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": job_id})
        return {
            "job_id": job_id,
            "status": (existing or {}).get("status", "not_found"),
            "claimed": False,
        }

    jobs = tenant_db[PROCESSING_JOBS_COLLECTION]
    submission_id = str(job["submission_id"])
    submission = await tenant_db["evalpen_submissions"].find_one({"submission_id": submission_id})
    if submission is None:
        await jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "last_error": "Canonical submission not found", "updated_at": _now()}},
        )
        return {"job_id": job_id, "status": "failed", "error": "Canonical submission not found"}

    exam = await tenant_db["exampen_exams"].find_one({"exam_id": submission.get("exam_id")})
    if exam is None or exam.get("exam_type") != "pcr":
        await jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "last_error": "Submission is not attached to a PCR session", "updated_at": _now()}},
        )
        return {"job_id": job_id, "status": "failed", "error": "Not a PCR session"}

    from api.v1.evalpen_evaluate_async import _build_eval_core
    from api.v1.evalpen_submissions_async import _build_submission_service

    processor = await _build_submission_service(tenant_db)
    processing_result = await processor.process_submission(submission_id)
    if processing_result.error:
        await jobs.update_one(
            {"job_id": job_id},
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
                }
            },
        )
        return {"job_id": job_id, "status": "failed", "error": processing_result.error}

    responses = await tenant_db["evalpen_detected_responses"].find(
        {
            "submission_id": submission_id,
            "superseded_at": {"$exists": False},
            "eval_status": {"$in": ["ready", "ready_with_warnings"]},
        }
    ).to_list(length=2000)
    blocked_count = await tenant_db["evalpen_detected_responses"].count_documents(
        {
            "submission_id": submission_id,
            "superseded_at": {"$exists": False},
            "eval_status": "blocked",
        }
    )

    evaluation_errors: List[str] = []
    evaluated_count = 0
    if responses:
        eval_core = await _build_eval_core(tenant_db)
        for response in responses:
            response_id = str(response.get("response_id") or "")
            question_id = response.get("question_id")
            if not response_id or not question_id:
                # A copy without reliable question association must not make
                # the whole PCR job look like an infrastructure failure.  It
                # is a normal teacher-review outcome: retain the OCR text,
                # mark the response blocked, and let the staff UI explain why.
                await tenant_db["evalpen_detected_responses"].update_one(
                    {"response_id": response_id},
                    {
                        "$set": {
                            "eval_status": "blocked",
                            "manual_review_reason": "Question could not be safely associated from the submitted copy",
                            "updated_at": _now(),
                        }
                    },
                )
                blocked_count += 1
                continue
            result = await eval_core.evaluate_response(response_id, question_id=str(question_id))
            if result.error:
                evaluation_errors.append(f"{response_id}: {result.error}")
            elif result.skipped:
                blocked_count += 1
            else:
                evaluated_count += 1

    if processing_result.response_count == 0:
        evaluation_errors.append("No student responses were detected")

    final_status = "failed" if evaluation_errors else ("blocked_for_review" if blocked_count else "completed")
    now = _now()
    await jobs.update_one(
        {"job_id": job_id},
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
                },
                "last_error": "; ".join(evaluation_errors[:10]) or None,
                "finished_at": now,
                "updated_at": now,
            }
        },
    )
    await _maybe_mark_exam_ready_for_review(tenant_db, str(submission.get("exam_id")))
    return {
        "job_id": job_id,
        "status": final_status,
        "submission_id": submission_id,
        "evaluated_count": evaluated_count,
        "blocked_count": blocked_count,
        "errors": evaluation_errors,
    }


async def mark_processing_job_retryable_error(tenant_db: Any, job_id: str, error: Exception | str) -> None:
    """Persist a retryable worker failure before Celery schedules its retry."""
    await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "retryable_error",
                "last_error": _short_error(error),
                "updated_at": _now(),
            }
        },
    )


async def mark_processing_job_failed(tenant_db: Any, job_id: str, error: Exception | str) -> None:
    """Persist a terminal worker failure after retry exhaustion."""
    await tenant_db[PROCESSING_JOBS_COLLECTION].update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "failed",
                "last_error": _short_error(error),
                "finished_at": _now(),
                "updated_at": _now(),
            }
        },
    )
