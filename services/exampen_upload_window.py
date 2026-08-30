"""Answer-copy upload-window coordination for conducted PCR exams.

The exam lifecycle describes capture/review progression, while this module
owns the narrower question of whether a new photographed/scanned answer copy
may become canonical.  A short-lived reservation fences the final ingest
against the teacher closing uploads and starting an Economy batch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from pymongo import ReturnDocument


ANSWER_COPY_UPLOAD_OPEN = "open"
ANSWER_COPY_UPLOAD_CLOSED = "closed"
ANSWER_COPY_UPLOAD_STATES = {
    ANSWER_COPY_UPLOAD_OPEN,
    ANSWER_COPY_UPLOAD_CLOSED,
}
ANSWER_COPY_UPLOAD_LIFECYCLES = {
    "in_progress",
    "collection_closed",
    "uploading",
}
UPLOAD_RESERVATION_LEASE = timedelta(minutes=15)


def answer_copy_upload_state(exam: Optional[Dict[str, Any]]) -> str:
    """Return the explicit state, with a safe legacy-session fallback."""

    exam = exam or {}
    explicit = str(exam.get("answer_copy_upload_state") or "").strip().lower()
    if explicit in ANSWER_COPY_UPLOAD_STATES:
        return explicit
    lifecycle = str(exam.get("lifecycle_state") or "draft").strip().lower()
    return (
        ANSWER_COPY_UPLOAD_OPEN
        if lifecycle in ANSWER_COPY_UPLOAD_LIFECYCLES
        else ANSWER_COPY_UPLOAD_CLOSED
    )


def answer_copy_upload_is_open(exam: Optional[Dict[str, Any]]) -> bool:
    exam = exam or {}
    lifecycle = str(exam.get("lifecycle_state") or "draft").strip().lower()
    return (
        lifecycle in ANSWER_COPY_UPLOAD_LIFECYCLES
        and answer_copy_upload_state(exam) == ANSWER_COPY_UPLOAD_OPEN
    )


async def _discard_expired_reservations(
    tenant_db: Any,
    *,
    exam_id: str,
    now: Optional[datetime] = None,
) -> None:
    current = now or datetime.now(timezone.utc)
    await tenant_db["exampen_exams"].update_one(
        {"exam_id": exam_id},
        {
            "$pull": {
                "answer_copy_upload_reservations": {
                    "expires_at": {"$lte": current},
                }
            }
        },
    )


async def reserve_answer_copy_ingest(
    tenant_db: Any,
    *,
    exam_id: str,
    actor_id: str,
    reservation_token: Optional[str] = None,
) -> str:
    """Reserve the right to make one already-scanned copy canonical.

    The push and the close operation both update the exam document, so MongoDB
    serializes the race without relying on process-local locks.
    """

    token = str(reservation_token or uuid.uuid4().hex).strip()
    if not token:
        token = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await _discard_expired_reservations(tenant_db, exam_id=exam_id, now=now)
    result = await tenant_db["exampen_exams"].update_one(
        {
            "exam_id": exam_id,
            "lifecycle_state": {"$in": sorted(ANSWER_COPY_UPLOAD_LIFECYCLES)},
            "$or": [
                {"answer_copy_upload_state": ANSWER_COPY_UPLOAD_OPEN},
                {"answer_copy_upload_state": {"$exists": False}},
            ],
        },
        {
            "$push": {
                "answer_copy_upload_reservations": {
                    "token": token,
                    "actor_id": str(actor_id or "unknown"),
                    "reserved_at": now,
                    "expires_at": now + UPLOAD_RESERVATION_LEASE,
                }
            },
            "$set": {"updated_at": now},
        },
    )
    if result.matched_count == 1:
        return token

    exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        {"lifecycle_state": 1, "answer_copy_upload_state": 1},
    )
    if exam is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            "Answer-copy uploads are closed for this exam. Refresh the exam "
            "before trying again."
        ),
    )


async def release_answer_copy_ingest(
    tenant_db: Any,
    *,
    exam_id: str,
    reservation_token: Optional[str],
) -> None:
    token = str(reservation_token or "").strip()
    if not token:
        return
    await tenant_db["exampen_exams"].update_one(
        {"exam_id": exam_id},
        {
            "$pull": {"answer_copy_upload_reservations": {"token": token}},
            "$set": {"updated_at": datetime.now(timezone.utc)},
        },
    )


async def close_answer_copy_uploads(
    tenant_db: Any,
    *,
    exam_id: str,
    actor_id: str,
) -> Dict[str, Any]:
    """Close uploads and advance collection_closed to uploading atomically."""

    now = datetime.now(timezone.utc)
    await _discard_expired_reservations(tenant_db, exam_id=exam_id, now=now)
    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    lifecycle = str(exam.get("lifecycle_state") or "draft")
    if lifecycle not in {"collection_closed", "uploading"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Close the live exam collection before starting economy checking",
        )

    # The legacy camera flow saves individual pages before the invigilator
    # confirms that the student's copy is complete.  Do not freeze a Batch
    # snapshot while any of those draft pages are still waiting to become a
    # canonical submission.
    incomplete_camera_page = await tenant_db["exampen_camera_uploads"].find_one(
        {
            "exam_id": exam_id,
            "$or": [
                {"submission_id": {"$exists": False}},
                {"submission_id": None},
                {"submission_id": ""},
            ],
        },
        {"student_id": 1, "page_number": 1},
    )
    if incomplete_camera_page is not None:
        student_id = str(incomplete_camera_page.get("student_id") or "a student")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"The photographed answer copy for {student_id} is not finalized. "
                "Complete or remove its draft pages before starting economy checking."
            ),
        )

    target_lifecycle = "uploading" if lifecycle == "collection_closed" else lifecycle
    updated = await tenant_db["exampen_exams"].find_one_and_update(
        {
            "exam_id": exam_id,
            "lifecycle_state": lifecycle,
            "$or": [
                {"answer_copy_upload_reservations": {"$exists": False}},
                {"answer_copy_upload_reservations": {"$size": 0}},
            ],
        },
        {
            "$set": {
                "answer_copy_upload_state": ANSWER_COPY_UPLOAD_CLOSED,
                "answer_copy_uploads_closed_at": now,
                "answer_copy_uploads_closed_by": str(actor_id or "unknown"),
                "lifecycle_state": target_lifecycle,
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "An answer copy is still being finalized. Wait for the upload "
                "to finish, then start economy checking again."
            ),
        )
    return updated


async def reopen_answer_copy_uploads(
    tenant_db: Any,
    *,
    exam_id: str,
    actor_id: str,
) -> Dict[str, Any]:
    """Explicitly reopen late-copy collection after a terminal/no Batch."""

    now = datetime.now(timezone.utc)
    exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        {"lifecycle_state": 1},
    )
    if exam is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    lifecycle = str(exam.get("lifecycle_state") or "draft")
    if lifecycle not in {"collection_closed", "uploading", "ready_for_eval"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Uploads can be reopened only after collection has closed",
        )
    # A completed first batch may already have moved the session into teacher
    # review. Reopening is the explicit, audited exception that returns it to
    # collection for a late copy and its later follow-up batch.
    target_lifecycle = "uploading" if lifecycle == "ready_for_eval" else lifecycle
    updated = await tenant_db["exampen_exams"].find_one_and_update(
        {
            "exam_id": exam_id,
            "lifecycle_state": lifecycle,
        },
        {
            "$set": {
                "answer_copy_upload_state": ANSWER_COPY_UPLOAD_OPEN,
                "lifecycle_state": target_lifecycle,
                "answer_copy_uploads_reopened_at": now,
                "answer_copy_uploads_reopened_by": str(actor_id or "unknown"),
                "updated_at": now,
            },
            "$unset": {
                "answer_copy_uploads_closed_at": "",
                "answer_copy_uploads_closed_by": "",
            },
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The exam changed while uploads were being reopened. Refresh and try again.",
        )
    return updated


__all__ = [
    "ANSWER_COPY_UPLOAD_CLOSED",
    "ANSWER_COPY_UPLOAD_OPEN",
    "answer_copy_upload_is_open",
    "answer_copy_upload_state",
    "close_answer_copy_uploads",
    "release_answer_copy_ingest",
    "reopen_answer_copy_uploads",
    "reserve_answer_copy_ingest",
]
