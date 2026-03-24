"""Chat message routes — append messages, list threads, read receipts.

Endpoints (all under ``/api/v1/chat``):
  POST /threads/{exam_id}/{other_user_id}      — Append a message
  GET  /threads/{exam_id}/{other_user_id}       — Get thread messages
  POST /threads/{exam_id}/{other_user_id}/read  — Mark thread as read
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.logging import get_logger

from src.domain.message_rules import (
    check_rbac,
    check_sender_role,
    is_teacher_role,
    validate_message,
)
from src.adapters.exam_enrollment import ExamEnrollmentAdapter
from src.domain.thread_logic import (
    ThreadKey,
    can_participate,
    resolve_other_user_id,
)
from src.storage.message_repo import MessageRepo

_log = get_logger(__name__)

router = APIRouter()


# -- Request / Response models (match chat.openapi.yaml) -------------------


class SendChatMessageRequest(BaseModel):
    """POST request body for appending a message."""

    content: str
    attachment_uri: str | None = None


class ChatMessageResponse(BaseModel):
    """A single chat message."""

    message_id: str
    sender_id: str
    recipient_id: str
    exam_id: str
    content: str
    attachment_uri: str | None = None
    sent_at: str
    read_at: str | None = None


class ThreadListResponse(BaseModel):
    """Wrapper for a list of messages in a thread."""

    items: list[ChatMessageResponse]


class ReadReceiptResponse(BaseModel):
    """Read receipt confirmation."""

    exam_id: str
    other_user_id: str
    read_at: str


# -- Helpers ---------------------------------------------------------------


def _get_repo(request: Request) -> MessageRepo:
    return request.app.state.message_repo


def _get_enrollment(request: Request) -> ExamEnrollmentAdapter:
    return request.app.state.enrollment


async def _enforce_rbac(
    request: Request, user: ExamPenUser, sender_role: str,
    exam_id: str, other_user_id: str,
) -> None:
    """Shared RBAC enforcement for all endpoints (append, read, mark-read)."""
    enrollment = _get_enrollment(request)
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    teacher_ids = await enrollment.get_teacher_ids(exam_id, token)
    student_ids = await enrollment.get_student_ids(exam_id, user.user_id, token)
    rbac_result = check_rbac(
        sender_role=sender_role,
        sender_id=user.user_id,
        recipient_id=other_user_id,
        teacher_ids=teacher_ids,
        student_ids=student_ids,
    )
    if not rbac_result.valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=rbac_result.error,
        )


# -- Endpoints -------------------------------------------------------------


@router.post(
    "/threads/{exam_id}/{other_user_id}",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def append_message(
    exam_id: str,
    other_user_id: str,
    body: SendChatMessageRequest,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Append a new message to the thread.

    The sender is the authenticated user; the recipient is
    ``other_user_id`` from the URL path.
    """
    sender_role = _effective_role(user)

    # RBAC: only teachers and students can send messages
    role_result = check_sender_role(sender_role)
    if not role_result.valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=role_result.error,
        )

    # Content validation
    validation = validate_message(
        sender_id=user.user_id,
        recipient_id=other_user_id,
        exam_id=exam_id,
        content=body.content,
    )
    if not validation.valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=validation.error,
        )

    # RBAC: verify sender may message this specific recipient
    await _enforce_rbac(request, user, sender_role, exam_id, other_user_id)

    # Participation check: user must belong to this thread
    teacher_id, student_id = resolve_other_user_id(
        user.user_id, sender_role, other_user_id,
    )
    thread_key = ThreadKey(
        exam_id=exam_id,
        teacher_id=teacher_id,
        student_id=student_id,
    )
    if not can_participate(user.user_id, sender_role, thread_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a participant in this thread",
        )

    repo = _get_repo(request)
    return await repo.append_message(
        sender_id=user.user_id,
        recipient_id=other_user_id,
        exam_id=exam_id,
        content=body.content,
        tenant_id=user.tenant_id,
        attachment_uri=body.attachment_uri,
    )


@router.get(
    "/threads/{exam_id}/{other_user_id}",
    response_model=ThreadListResponse,
)
async def get_thread(
    exam_id: str,
    other_user_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Get all messages in a specific thread."""
    sender_role = _effective_role(user)

    teacher_id, student_id = resolve_other_user_id(
        user.user_id, sender_role, other_user_id,
    )
    thread_key = ThreadKey(
        exam_id=exam_id,
        teacher_id=teacher_id,
        student_id=student_id,
    )
    if not can_participate(user.user_id, sender_role, thread_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a participant in this thread",
        )

    # RBAC: verify caller is enrolled in this exam
    await _enforce_rbac(request, user, sender_role, exam_id, other_user_id)

    repo = _get_repo(request)
    messages = await repo.get_thread(
        exam_id=exam_id,
        teacher_id=teacher_id,
        student_id=student_id,
        tenant_id=user.tenant_id,
    )
    return {"items": messages}


@router.post(
    "/threads/{exam_id}/{other_user_id}/read",
    response_model=ReadReceiptResponse,
)
async def mark_thread_read(
    exam_id: str,
    other_user_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Mark the thread as read for the current user.

    Appends a read receipt — this is NOT an update to any existing row.
    """
    sender_role = _effective_role(user)

    teacher_id, student_id = resolve_other_user_id(
        user.user_id, sender_role, other_user_id,
    )
    thread_key = ThreadKey(
        exam_id=exam_id,
        teacher_id=teacher_id,
        student_id=student_id,
    )
    if not can_participate(user.user_id, sender_role, thread_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a participant in this thread",
        )

    # RBAC: verify caller is enrolled in this exam
    await _enforce_rbac(request, user, sender_role, exam_id, other_user_id)

    repo = _get_repo(request)
    return await repo.append_read_receipt(
        exam_id=exam_id,
        reader_id=user.user_id,
        other_user_id=other_user_id,
        tenant_id=user.tenant_id,
    )


# -- Internal helpers ------------------------------------------------------


def _effective_role(user: ExamPenUser) -> str:
    """Pick the most relevant ExamPen role for chat RBAC.

    The ``exampen_roles`` list may contain several roles. For chat
    purposes, prefer teacher-like roles over student.
    """
    for role in user.exampen_roles:
        if is_teacher_role(role):
            return role
    if "student" in user.exampen_roles:
        return "student"
    # Fallback — will be rejected by check_sender_role
    return user.exampen_roles[0] if user.exampen_roles else "unknown"
