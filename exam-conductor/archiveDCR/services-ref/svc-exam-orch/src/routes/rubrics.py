"""Rubric and question-region management endpoints.

These are exam configuration endpoints owned by svc-exam-orch.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.db import rls_session

from src.domain.rbac import require_role
from src.storage.exam_repo import ExamRepo

router = APIRouter(tags=["rubrics"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class RubricStepBody(BaseModel):
    name: str
    marks: float


class RubricQuestionBody(BaseModel):
    question_number: int
    max_marks: float
    answer_type: str = "text"
    steps: list[RubricStepBody] = Field(default_factory=list)


class SaveRubricBody(BaseModel):
    questions: list[RubricQuestionBody]
    confidence_threshold: float = 0.85


class QuestionRegionBody(BaseModel):
    question_number: int
    x_pct: float
    y_pct: float
    width_pct: float
    height_pct: float


class SaveRegionsBody(BaseModel):
    regions: list[QuestionRegionBody]


# ---------------------------------------------------------------------------
# Rubric endpoints
# ---------------------------------------------------------------------------


@router.get("/{exam_id}/rubric")
async def get_rubric(
    exam_id: str, request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    require_role(user, "super_admin", "principal", "hod", "evaluator")
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        exam = await repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(404, "Exam not found")
        return exam.get("rubric", {"questions": [], "confidence_threshold": 0.85})
    raise HTTPException(500, "session error")


@router.put("/{exam_id}/rubric")
async def save_rubric(
    exam_id: str, body: SaveRubricBody, request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    require_role(user, "super_admin", "principal", "hod", "evaluator")
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        exam = await repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(404, "Exam not found")
        await repo.update_rubric(exam_id, body.model_dump())
        return {"ok": True}
    raise HTTPException(500, "session error")


# ---------------------------------------------------------------------------
# Question region endpoints
# ---------------------------------------------------------------------------


@router.get("/{exam_id}/regions")
async def get_regions(
    exam_id: str, request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    require_role(user, "super_admin", "principal", "hod", "evaluator")
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        exam = await repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(404, "Exam not found")
        return {"regions": exam.get("question_regions", [])}
    raise HTTPException(500, "session error")


@router.put("/{exam_id}/regions")
async def save_regions(
    exam_id: str, body: SaveRegionsBody, request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    require_role(user, "super_admin", "principal", "hod", "evaluator")
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        exam = await repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(404, "Exam not found")
        await repo.update_regions(exam_id, [r.model_dump() for r in body.regions])
        return {"ok": True}
    raise HTTPException(500, "session error")
