"""Append-only exam chat messaging endpoints.

Routes are mounted at ``/api/v1/exampen/chat``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
)
from exampen.dcr.storage.chat_repo import ChatRepo

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db


async def _publish_event(request: Request, subject: str, data: dict) -> None:
    nats = getattr(request.app.state, "exampen_nats", None)
    if nats is None or not nats.is_connected:
        return
    try:
        await nats.publish(subject, data)
    except Exception:
        logger.warning("NATS publish to %s failed (non-fatal)", subject)


def _resolve_thread_parties(
    user: ExamPenUser, other_user_id: str,
) -> tuple[str, str]:
    """Determine teacher_id and student_id from the current user and the other party.

    If the current user is a student, they are the student party.
    Otherwise (evaluator, hod, etc.) they are the teacher party.
    """
    if "student" in user.exampen_roles:
        return other_user_id, user.user_id  # teacher, student
    return user.user_id, other_user_id  # teacher, student


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SendMessageBody(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/threads/{exam_id}/{other_user_id}", status_code=status.HTTP_201_CREATED)
async def send_message(
    exam_id: str,
    other_user_id: str,
    body: SendMessageBody,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Send a message in a teacher-student thread for an exam.

    Append-only: messages are never updated or deleted.
    """
    if not body.text.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Message text is empty")

    teacher_id, student_id = _resolve_thread_parties(user, other_user_id)

    db = await _get_tenant_db(request, user)
    repo = ChatRepo(db)
    doc = await repo.append_message(user.tenant_id, {
        "exam_id": exam_id,
        "teacher_id": teacher_id,
        "student_id": student_id,
        "sender_id": user.user_id,
        "text": body.text.strip(),
    })

    await _publish_event(request, "exampen.chat.message", {
        "exam_id": exam_id,
        "sender_id": user.user_id,
        "thread": f"{teacher_id}:{student_id}",
    })
    return doc


@router.get("/threads/{exam_id}/{other_user_id}")
async def get_thread(
    exam_id: str,
    other_user_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Fetch all messages in a teacher-student thread for an exam.

    Returns messages sorted chronologically (oldest first).
    """
    teacher_id, student_id = _resolve_thread_parties(user, other_user_id)

    db = await _get_tenant_db(request, user)
    repo = ChatRepo(db)
    messages = await repo.get_thread(exam_id, teacher_id, student_id, user.tenant_id)
    return {
        "exam_id": exam_id,
        "teacher_id": teacher_id,
        "student_id": student_id,
        "messages": messages,
    }


@router.post("/threads/{exam_id}/{other_user_id}/read")
async def mark_thread_read(
    exam_id: str,
    other_user_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, str]:
    """Mark all messages in the thread as read for the current user.

    Stores a last-read timestamp in a separate collection so that the
    append-only chat messages are never mutated.
    """
    teacher_id, student_id = _resolve_thread_parties(user, other_user_id)

    db = await _get_tenant_db(request, user)
    now = datetime.now(timezone.utc)

    # Upsert into a read-receipts collection (separate from chat messages)
    coll = db["exampen_chat_read_receipts"]
    await coll.find_one_and_update(
        {
            "exam_id": exam_id,
            "teacher_id": teacher_id,
            "student_id": student_id,
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
        },
        {
            "$set": {"read_at": now},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return {"status": "ok"}
