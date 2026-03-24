"""Plagiarism flag endpoints — list, detail, teacher verdict.

Routes are mounted at ``/api/v1/exampen/plagiarism``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator

from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
    require_exampen_role,
)
from exampen.dcr.storage.plagiarism_repo import PlagiarismRepo

logger = logging.getLogger(__name__)
router = APIRouter()

_VALID_VERDICTS = frozenset({"confirmed", "dismissed", "inconclusive"})


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


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class VerdictBody(BaseModel):
    verdict: str
    reason: str

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        if v not in _VALID_VERDICTS:
            raise ValueError(
                f"verdict must be one of: {', '.join(sorted(_VALID_VERDICTS))}"
            )
        return v

    @field_validator("reason")
    @classmethod
    def reason_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("reason must be at least 5 characters")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/exams/{exam_id}/flags")
async def list_flags(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("evaluator", "hod", "principal", "super_admin")
    ),
) -> dict[str, Any]:
    """List all plagiarism flags for an exam.

    Requires evaluator+ role.
    """
    db = await _get_tenant_db(request, user)
    repo = PlagiarismRepo(db)
    flags = await repo.list_by_exam(exam_id, user.tenant_id)
    return {"exam_id": exam_id, "flags": flags}


@router.get("/flags/{flag_id}")
async def get_flag(
    flag_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("evaluator", "hod", "principal", "super_admin")
    ),
) -> dict[str, Any]:
    """Return detail for a single plagiarism flag."""
    db = await _get_tenant_db(request, user)
    repo = PlagiarismRepo(db)
    flag = await repo.get_by_id(flag_id, user.tenant_id)
    if flag is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Plagiarism flag not found")
    return flag


@router.patch("/flags/{flag_id}/verdict")
async def update_verdict(
    flag_id: str,
    body: VerdictBody,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("evaluator", "hod", "principal")
    ),
) -> dict[str, Any]:
    """Set or update the teacher's verdict on a plagiarism flag.

    Valid verdicts: confirmed, dismissed, inconclusive.
    """
    db = await _get_tenant_db(request, user)
    repo = PlagiarismRepo(db)

    # Verify flag exists
    flag = await repo.get_by_id(flag_id, user.tenant_id)
    if flag is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Plagiarism flag not found")

    updated = await repo.update_verdict(
        flag_id, user.tenant_id, body.verdict, body.reason.strip(),
    )
    if not updated:
        raise HTTPException(status.HTTP_409_CONFLICT, "Failed to update verdict")

    await _publish_event(request, "exampen.plagiarism.verdict", {
        "flag_id": flag_id,
        "verdict": body.verdict,
        "actor_id": user.user_id,
        "exam_id": flag.get("exam_id"),
    })

    # Re-fetch to return updated state
    return await repo.get_by_id(flag_id, user.tenant_id) or {"status": "updated"}
