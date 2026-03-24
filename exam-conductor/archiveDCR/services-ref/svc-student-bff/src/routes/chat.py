"""Chat proxy routes — relay to svc-chat (append-only messaging).

Endpoints:
  GET  /student/exams/{exam_id}/chat/{teacher_id} — Thread messages
  POST /student/exams/{exam_id}/chat/{teacher_id} — Send message relay

Only students may send messages. Parents have read-only chat access
to their linked children's threads.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from exampen_common.logging import get_logger

from src.middleware.auth import (
    StudentBFFIdentity,
    require_own_data,
    require_student_or_parent,
)
from src.routes.models import (
    ErrorBody,
    MessageListResponse,
    MessageResponse,
    SendMessageRequest,
)

_log = get_logger(__name__)

router = APIRouter()


def _extract_token(request: Request) -> str:
    """Extract raw bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


@router.get(
    "/exams/{exam_id}/chat/{teacher_id}",
    response_model=MessageListResponse,
)
async def get_chat_thread(
    exam_id: str,
    teacher_id: str,
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
    student_id: str | None = Query(None, description="Required for parent"),
) -> dict[str, Any]:
    """Fetch chat thread between student and teacher for an exam."""
    effective_sid = require_own_data(identity, student_id)
    token = _extract_token(request)
    chat_client = request.app.state.chat_client

    messages = await chat_client.get_thread(
        exam_id=exam_id,
        student_id=effective_sid,
        teacher_id=teacher_id,
        token=token,
    )
    return {"items": messages}


@router.post(
    "/exams/{exam_id}/chat/{teacher_id}",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        403: {"model": ErrorBody},
        502: {"model": ErrorBody},
    },
)
async def send_message(
    exam_id: str,
    teacher_id: str,
    body: SendMessageRequest,
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
) -> dict[str, Any]:
    """Send a message to the teacher — relays to svc-chat.

    Only students may send messages.  Parents have read-only access
    to their linked children's threads.
    """
    if identity.role != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students may send chat messages",
        )

    token = _extract_token(request)
    chat_client = request.app.state.chat_client

    payload: dict[str, Any] = {
        "sender_id": identity.user.user_id,
        "content": body.content,
    }
    if body.attachment_uri:
        payload["attachment_uri"] = body.attachment_uri

    result = await chat_client.send_message(
        exam_id=exam_id,
        student_id=identity.user.user_id,
        teacher_id=teacher_id,
        payload=payload,
        token=token,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to relay message to chat service",
        )
    return result
