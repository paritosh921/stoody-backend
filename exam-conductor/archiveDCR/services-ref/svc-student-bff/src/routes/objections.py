"""Objection routes — relay filing to svc-review, read status.

Endpoints:
  POST /student/exams/{exam_id}/objections  — File objection (relay)
  GET  /student/objections                  — List own objections
  GET  /student/objections/{objection_id}   — Status + resolution detail
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
    CreateObjectionRequest,
    ErrorBody,
    ObjectionListResponse,
    StudentObjection,
)

_log = get_logger(__name__)

router = APIRouter()


def _extract_token(request: Request) -> str:
    """Extract raw bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


@router.post(
    "/exams/{exam_id}/objections",
    response_model=StudentObjection,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorBody},
        502: {"model": ErrorBody},
    },
)
async def file_objection(
    exam_id: str,
    body: CreateObjectionRequest,
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
) -> dict[str, Any]:
    """File an objection — relays to svc-review.

    Only students may file objections (not parents).
    """
    if identity.role != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students may file objections",
        )

    token = _extract_token(request)
    review_client = request.app.state.review_client

    payload = {
        "exam_id": exam_id,
        "student_id": identity.user.user_id,
        "question_id": body.question_id,
        "objection_text": body.objection_text,
    }
    result = await review_client.file_objection(payload, token)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to relay objection to review service",
        )
    return result


@router.get(
    "/objections",
    response_model=ObjectionListResponse,
)
async def list_objections(
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
    student_id: str | None = Query(None, description="Required for parent"),
    exam_id: str | None = Query(None),
) -> dict[str, Any]:
    """List objections filed by the current student (or parent's child)."""
    effective_sid = require_own_data(identity, student_id)
    token = _extract_token(request)
    review_client = request.app.state.review_client

    items = await review_client.list_objections(
        token=token,
        exam_id=exam_id,
        student_id=effective_sid,
    )
    return {"items": items}


@router.get(
    "/objections/{objection_id}",
    response_model=StudentObjection,
    responses={404: {"model": ErrorBody}},
)
async def get_objection_detail(
    objection_id: str,
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
) -> dict[str, Any]:
    """Get objection status and resolution detail."""
    token = _extract_token(request)
    review_client = request.app.state.review_client

    result = await review_client.get_objection(objection_id, token)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Objection {objection_id} not found",
        )

    # Verify the student has access to this objection
    obj_student = result.get("student_id", "")
    if obj_student not in identity.allowed_student_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this objection",
        )
    return result
