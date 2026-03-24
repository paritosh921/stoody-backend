"""Revocation routes — manage ExamPen-side token revocations.

Endpoints:
  POST   /revocations       — revoke a token JTI
  GET    /revocations/{jti} — check revocation status
  DELETE /revocations/{jti} — un-revoke a token JTI

All endpoints require a valid Stoody bearer token (principal+ role).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from exampen_common.auth import get_current_user, ExamPenUser
from exampen_common.logging import get_logger

from src.domain.role_mapper import has_minimum_role

_log = get_logger(__name__)

router = APIRouter()


# -- Request / Response models (match auth.openapi.yaml) -------------------


class RevocationRequest(BaseModel):
    """POST /revocations request body."""

    jti: str
    subject_user_id: str | None = None
    reason: str = Field(..., min_length=5)
    expires_at: datetime | None = None


class RevocationStatus(BaseModel):
    """Revocation status response."""

    jti: str
    revoked: bool
    revoked_at: str | None = None
    reason: str | None = None


class ErrorResponse(BaseModel):
    """Standard error body."""

    code: str
    message: str


# -- Helpers ---------------------------------------------------------------


async def _require_principal(request: Request) -> ExamPenUser:
    """Extract bearer token and ensure actor has at least principal role."""
    user = await get_current_user(request)
    if not has_minimum_role(user.exampen_roles, "principal"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requires principal or higher role",
        )
    return user


# -- Endpoints -------------------------------------------------------------


@router.post(
    "/revocations",
    response_model=RevocationStatus,
    status_code=status.HTTP_202_ACCEPTED,
    responses={401: {"model": ErrorResponse}},
)
async def create_revocation(
    body: RevocationRequest,
    request: Request,
) -> dict[str, Any]:
    """Revoke a normalized token/session inside ExamPen."""
    actor = await _require_principal(request)
    repo = request.app.state.revocation_repo
    result = await repo.revoke(
        jti=body.jti,
        tenant_id=actor.tenant_id,
        reason=body.reason,
        revoked_by=actor.user_id,
        subject_user_id=body.subject_user_id,
        expires_at=body.expires_at,
    )
    return result


@router.get(
    "/revocations/{jti}",
    response_model=RevocationStatus,
    responses={401: {"model": ErrorResponse}},
)
async def get_revocation(jti: str, request: Request) -> dict[str, Any]:
    """Check whether a token JTI is revoked in ExamPen."""
    await _require_principal(request)
    repo = request.app.state.revocation_repo
    return await repo.is_revoked(jti)


@router.delete(
    "/revocations/{jti}",
    response_model=RevocationStatus,
    responses={401: {"model": ErrorResponse}},
)
async def delete_revocation(jti: str, request: Request) -> dict[str, Any]:
    """Un-revoke a previously revoked token JTI."""
    await _require_principal(request)
    repo = request.app.state.revocation_repo
    deleted = await repo.delete(jti)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No revocation found for jti={jti}",
        )
    return {"jti": jti, "revoked": False}
