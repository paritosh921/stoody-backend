"""Audited removal of a wrongly assigned ExamPen answer copy.

This is a lifecycle command, not a score override.  It removes the canonical
copy and every active result derived from it so the same student/exam pair can
accept a replacement upload.  Exam and paper records are deliberately outside
this ownership boundary.

The deployment cannot assume MongoDB transactions are available for every
tenant.  Cleanup is therefore archive-first, idempotent, and ordered with the
canonical submission deleted last.  A failed intermediate delete can be
retried while the submission still exists.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from core.upload_security.storage import PrivateUploadStorage
from services.exampen_workflow import TERMINAL_JOB_STATUSES
from utils.s3_storage import PrivateObjectStorageError, delete_private_object


logger = logging.getLogger(__name__)

DELETION_AUDIT_COLLECTION = "evalpen_submission_deletion_audit"
PRIVATE_STUDENT_COPY_S3_PREFIX = "private/exampen/student-answer-copies"

# These collections contain state owned by one submitted answer copy.  Paper,
# exam, question, solution, marking-plan, and taxonomy records are exam-owned
# and must survive deleting one student's copy.
SUBMISSION_DERIVED_COLLECTIONS = (
    "evalpen_detected_responses",
    "evalpen_document_grading_runs",
    "evalpen_objective_grading_runs",
    "evalpen_recheck_requests",
)


class SubmissionCopyBusyError(RuntimeError):
    """Raised when an active grader could still write results for the copy."""


class SubmissionCopyDeleteError(RuntimeError):
    """Raised when active copy state could not be removed completely."""


def _without_mongo_id(document: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in document.items() if key != "_id"}


def _unique_nonempty(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return result


def _evaluation_audit_summary(document: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the awarded-score facts without archiving large LLM payloads."""

    return {
        key: document.get(key)
        for key in (
            "evaluation_id",
            "response_id",
            "question_id",
            "marks_awarded",
            "total_marks",
            "max_marks",
            "eval_status",
            "review_status",
            "grading_run_id",
            "created_at",
            "updated_at",
        )
        if document.get(key) is not None
    }


async def _cancel_unclaimed_processing_job(
    tenant_db: Any,
    submission_id: str,
    *,
    actor_id: str,
    reason: str,
    now: datetime,
) -> Dict[str, Any] | None:
    """Fence a queued/retryable job before deletion.

    A processing worker may already be inside a model request and can write
    after the API call returns.  Such a job is never force-deleted here.  For
    every non-terminal status, an exact-status compare-and-set wins against a
    worker claim; losing that race returns a conflict instead of risking stale
    marks on a later replacement submission.
    """

    jobs = tenant_db["exampen_processing_jobs"]
    job = await jobs.find_one({"submission_id": submission_id})
    if job is None:
        return None

    current_status = str(job.get("status") or "").strip()
    if current_status == "processing":
        raise SubmissionCopyBusyError(
            "This copy is currently being checked. Wait for the active check "
            "to finish, then delete the copy."
        )

    if current_status not in TERMINAL_JOB_STATUSES and current_status != "cancelled":
        identity_filter: Dict[str, Any]
        if job.get("_id") is not None:
            identity_filter = {"_id": job["_id"], "status": job.get("status")}
        else:
            identity_filter = {
                "submission_id": submission_id,
                "status": job.get("status"),
            }
        result = await jobs.update_one(
            identity_filter,
            {
                "$set": {
                    "status": "cancelled",
                    "cancelled_at": now,
                    "cancelled_by": actor_id,
                    "cancellation_reason": reason,
                    "updated_at": now,
                },
                "$unset": {
                    "lease_token": "",
                    "lease_expires_at": "",
                    "next_retry_at": "",
                },
            },
        )
        if result.matched_count != 1:
            raise SubmissionCopyBusyError(
                "The copy started checking while deletion was requested. Wait "
                "for that check to finish, then try Delete copy again."
            )

    return job


async def _cleanup_storage_paths(storage_paths: Iterable[str]) -> Dict[str, List[str]]:
    """Best-effort physical cleanup constrained to private answer-copy roots."""

    deleted: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []
    local_storage = PrivateUploadStorage()

    for storage_path in _unique_nonempty(storage_paths):
        try:
            if storage_path.startswith("s3://"):
                # Only the dedicated answer-copy prefix is eligible.  Old or
                # externally managed S3 references stay recorded for an
                # operator rather than widening a destructive key boundary.
                if f"/{PRIVATE_STUDENT_COPY_S3_PREFIX}/" not in storage_path:
                    skipped.append(storage_path)
                    continue
                await delete_private_object(
                    storage_path,
                    allowed_key_prefix=PRIVATE_STUDENT_COPY_S3_PREFIX,
                )
                deleted.append(storage_path)
                continue

            if await local_storage.delete_released_path(storage_path):
                deleted.append(storage_path)
            else:
                skipped.append(storage_path)
        except (PrivateObjectStorageError, OSError, ValueError):
            logger.exception("Could not remove deleted answer-copy asset %s", storage_path)
            failed.append(storage_path)

    return {"deleted": deleted, "failed": failed, "skipped": skipped}


async def delete_submission_copy(
    tenant_db: Any,
    submission: Dict[str, Any],
    *,
    actor_id: str,
    actor_role: str,
    reason_code: str,
    reason_note: str | None = None,
) -> Dict[str, Any]:
    """Archive and remove one complete answer-copy lifecycle.

    The caller owns authorization and the submission-level review lease.
    This function performs no LLM calls and creates no replacement job.
    """

    submission_id = str(submission.get("submission_id") or "").strip()
    exam_id = str(submission.get("exam_id") or "").strip()
    student_id = str(submission.get("student_id") or "").strip()
    if not submission_id or not exam_id or not student_id:
        raise ValueError("Submission is missing its canonical identity")

    now = datetime.now(timezone.utc)
    clean_actor_id = str(actor_id or "unknown")
    clean_reason = str(reason_code or "wrong_student")
    clean_note = str(reason_note or "").strip()[:500] or None

    job_snapshot = await _cancel_unclaimed_processing_job(
        tenant_db,
        submission_id,
        actor_id=clean_actor_id,
        reason=clean_reason,
        now=now,
    )

    page_docs = await tenant_db["evalpen_answer_pages"].find(
        {"submission_id": submission_id}
    ).to_list(length=500)
    camera_docs = await tenant_db["exampen_camera_uploads"].find(
        {
            "$or": [
                {"submission_id": submission_id},
                {"exam_id": exam_id, "student_id": student_id},
            ]
        }
    ).to_list(length=500)
    upload_attempts = await tenant_db["exampen_student_copy_uploads"].find(
        {"exam_id": exam_id, "student_id": student_id}
    ).to_list(length=100)
    response_docs = await tenant_db["evalpen_detected_responses"].find(
        {"submission_id": submission_id}
    ).to_list(length=5000)
    response_ids = _unique_nonempty(
        response.get("response_id") for response in response_docs
    )
    # Legacy per-response evaluations do not carry submission_id.  Resolve
    # ownership through their canonical response_id as well as supporting the
    # newer full-document records that do persist submission_id.
    evaluation_query: Dict[str, Any] = {"submission_id": submission_id}
    if response_ids:
        evaluation_query = {
            "$or": [
                {"submission_id": submission_id},
                {"response_id": {"$in": response_ids}},
            ]
        }
    evaluation_docs = await tenant_db["evalpen_evaluations"].find(
        evaluation_query
    ).to_list(length=5000)

    storage_paths = _unique_nonempty(
        [page.get("raw_image_ref") for page in page_docs]
        + [page.get("storage_path") for page in camera_docs]
        + [
            page.get("storage_path")
            for attempt in upload_attempts
            for page in (attempt.get("pages") or [])
            if isinstance(page, dict)
        ]
    )

    deletion_id = f"COPY-DELETE-{uuid.uuid4().hex}"
    audit_doc: Dict[str, Any] = {
        "deletion_id": deletion_id,
        "event": "answer_copy_deleted",
        "status": "archived",
        "submission_id": submission_id,
        "exam_id": exam_id,
        "student_id": student_id,
        "actor_id": clean_actor_id,
        "actor_role": str(actor_role or "unknown"),
        "reason_code": clean_reason,
        "reason_note": clean_note,
        "deleted_at": now,
        "submission_snapshot": _without_mongo_id(submission),
        "page_snapshot": [
            {
                key: page.get(key)
                for key in (
                    "page_id",
                    "page_number",
                    "content_hash",
                    "asset_sha256",
                    "content_type",
                    "file_size_bytes",
                    "raw_image_ref",
                )
                if page.get(key) is not None
            }
            for page in page_docs
        ],
        "evaluation_snapshot": [
            _evaluation_audit_summary(document) for document in evaluation_docs
        ],
        "processing_job_snapshot": (
            _without_mongo_id(job_snapshot) if job_snapshot is not None else None
        ),
        "storage_paths": storage_paths,
        "deleted_counts": {},
    }
    await tenant_db[DELETION_AUDIT_COLLECTION].insert_one(audit_doc)

    deleted_counts: Dict[str, int] = {}
    try:
        evaluation_result = await tenant_db["evalpen_evaluations"].delete_many(
            evaluation_query
        )
        deleted_counts["evalpen_evaluations"] = int(
            evaluation_result.deleted_count
        )

        for collection_name in SUBMISSION_DERIVED_COLLECTIONS:
            result = await tenant_db[collection_name].delete_many(
                {"submission_id": submission_id}
            )
            deleted_counts[collection_name] = int(result.deleted_count)

        # DCR's canonical key predates submission_id and is owned by the
        # exam/student/question tuple.  The exam/student pair is one-copy-only,
        # so clearing that pair is the correct lifecycle boundary.
        dcr_result = await tenant_db["exampen_dcr_results"].delete_many(
            {"exam_id": exam_id, "student_id": student_id}
        )
        deleted_counts["exampen_dcr_results"] = int(dcr_result.deleted_count)

        job_result = await tenant_db["exampen_processing_jobs"].delete_many(
            {"submission_id": submission_id}
        )
        deleted_counts["exampen_processing_jobs"] = int(job_result.deleted_count)

        page_result = await tenant_db["evalpen_answer_pages"].delete_many(
            {"submission_id": submission_id}
        )
        deleted_counts["evalpen_answer_pages"] = int(page_result.deleted_count)

        camera_result = await tenant_db["exampen_camera_uploads"].delete_many(
            {
                "$or": [
                    {"submission_id": submission_id},
                    {"exam_id": exam_id, "student_id": student_id},
                ]
            }
        )
        deleted_counts["exampen_camera_uploads"] = int(camera_result.deleted_count)

        upload_result = await tenant_db["exampen_student_copy_uploads"].delete_many(
            {"exam_id": exam_id, "student_id": student_id}
        )
        deleted_counts["exampen_student_copy_uploads"] = int(upload_result.deleted_count)

        # Canonical identity goes last.  Until this succeeds, a replacement
        # upload remains blocked and retrying this command is safe.
        submission_result = await tenant_db["evalpen_submissions"].delete_one(
            {"submission_id": submission_id}
        )
        if submission_result.deleted_count != 1:
            raise SubmissionCopyDeleteError(
                "The answer copy changed during deletion; active state was not finalized"
            )
        deleted_counts["evalpen_submissions"] = 1
    except Exception as exc:
        await tenant_db[DELETION_AUDIT_COLLECTION].update_one(
            {"deletion_id": deletion_id},
            {
                "$set": {
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc),
                    "failure": str(exc)[:1000],
                    "deleted_counts": deleted_counts,
                }
            },
        )
        raise

    storage_cleanup = await _cleanup_storage_paths(storage_paths)
    try:
        await tenant_db[DELETION_AUDIT_COLLECTION].update_one(
            {"deletion_id": deletion_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc),
                    "deleted_counts": deleted_counts,
                    "storage_cleanup": storage_cleanup,
                }
            },
        )
    except Exception:
        # The active lifecycle is already gone.  Do not report a failed user
        # action solely because the final audit annotation could not be saved.
        logger.exception("Could not finalize deletion audit %s", deletion_id)

    return {
        "deletion_id": deletion_id,
        "submission_id": submission_id,
        "exam_id": exam_id,
        "student_id": student_id,
        "status": "deleted",
        "deleted_counts": deleted_counts,
        "storage_cleanup_pending": bool(storage_cleanup["failed"]),
    }
