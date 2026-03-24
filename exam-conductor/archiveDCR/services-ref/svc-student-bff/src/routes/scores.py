"""Score viewing routes — read-only aggregation from svc-score-engine.

Endpoints:
  GET /student/exams/{exam_id}/score       — Score summary
  GET /student/exams/{exam_id}/questions    — Question-wise breakdown
  GET /student/exams/{exam_id}/questions/{qid}/answer — Answer + AI analysis
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
from src.routes.models import AnswerInsight, ErrorBody, StudentScoreView

_log = get_logger(__name__)

router = APIRouter()


def _extract_token(request: Request) -> str:
    """Extract raw bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


@router.get(
    "/exams/{exam_id}/score",
    response_model=StudentScoreView,
    responses={404: {"model": ErrorBody}},
)
async def get_score_summary(
    exam_id: str,
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
    student_id: str | None = Query(None, description="Required for parent"),
) -> dict[str, Any]:
    """Score summary: total, percentage, percentile, pass/fail."""
    effective_sid = require_own_data(identity, student_id)
    token = _extract_token(request)
    score_client = request.app.state.score_client

    summary = await score_client.get_score_summary(
        exam_id=exam_id,
        student_id=effective_sid,
        token=token,
    )
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Score not found for exam {exam_id}",
        )
    return summary


@router.get(
    "/exams/{exam_id}/questions",
    responses={404: {"model": ErrorBody}},
)
async def get_question_breakdown(
    exam_id: str,
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
    student_id: str | None = Query(None, description="Required for parent"),
) -> dict[str, Any]:
    """Per-question score breakdown with AI confidence and miss indicators."""
    effective_sid = require_own_data(identity, student_id)
    token = _extract_token(request)
    score_client = request.app.state.score_client

    questions = await score_client.get_question_breakdown(
        exam_id=exam_id,
        student_id=effective_sid,
        token=token,
    )
    return {"items": questions}


@router.get(
    "/exams/{exam_id}/questions/{question_id}/answer",
    response_model=AnswerInsight,
    responses={404: {"model": ErrorBody}},
)
async def get_answer_insight(
    exam_id: str,
    question_id: str,
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
    student_id: str | None = Query(None, description="Required for parent"),
) -> dict[str, Any]:
    """Answer image + AI analysis (recognized text, steps, feedback)."""
    effective_sid = require_own_data(identity, student_id)
    token = _extract_token(request)
    score_client = request.app.state.score_client

    insight = await score_client.get_answer_insight(
        exam_id=exam_id,
        student_id=effective_sid,
        question_id=question_id,
        token=token,
    )
    if insight is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Answer not found for question {question_id}",
        )
    return insight
