"""Invigilator and evaluator assignment endpoints.

Matches ``api/exam-orch.openapi.yaml`` path:
    POST  /api/v1/exams/{exam_id}/invigilators
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.db import rls_session

from src.domain.rbac import require_role
from src.storage.assignment_repo import AssignmentRepo
from src.storage.exam_repo import ExamRepo

router = APIRouter(tags=["invigilators"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class AssignmentBody(BaseModel):
    invigilator_ids: list[str]
    evaluator_ids: list[str]
    double_blind: bool = False


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("")
async def assign_staff(
    exam_id: str,
    body: AssignmentBody,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Assign invigilators and evaluators to an exam."""
    require_role(user, "super_admin", "principal", "hod")
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        # Verify exam exists
        exam_repo = ExamRepo(session)
        exam = await exam_repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(status_code=404, detail="Exam not found")

        repo = AssignmentRepo(session)
        result = await repo.upsert_assignments(
            exam_id=exam_id,
            tenant_id=user.tenant_id,
            invigilator_ids=body.invigilator_ids,
            evaluator_ids=body.evaluator_ids,
            double_blind=body.double_blind,
        )
        return result

    raise HTTPException(status_code=500, detail="session error")  # pragma: no cover
