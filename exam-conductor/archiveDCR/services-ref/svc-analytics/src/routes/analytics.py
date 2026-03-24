"""Analytics API routes — leaderboard, class stats, student performance.

Endpoints (all under ``/api/v1/analytics``):
  GET /exams/{exam_id}/leaderboard            — Leaderboard rows
  GET /exams/{exam_id}/class-stats            — Class-level statistics
  GET /students/{student_id}/performance      — Cross-exam student trend
  GET /exams/{exam_id}/student/{student_id}   — Per-student exam perf
  GET /exams/{exam_id}/questions              — Question difficulty
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.logging import get_logger

from src.domain.class_stats import (
    compute_class_stats,
    compute_question_difficulty,
    QuestionResponse,
)
from src.storage.analytics_repo import AnalyticsRepo

_log = get_logger(__name__)

router = APIRouter()


# -- Response models (match analytics.openapi.yaml) -------------------------


class LeaderboardRowResponse(BaseModel):
    rank: int
    student_id: str
    student_name: str | None = None
    score: float
    percentile: float


class LeaderboardResponse(BaseModel):
    items: list[LeaderboardRowResponse]


class QuestionDifficultyResponse(BaseModel):
    question_id: str
    avg_score: float


class ClassStatsResponse(BaseModel):
    mean: float
    median: float
    std_dev: float
    pass_rate: float
    question_difficulty: list[QuestionDifficultyResponse] | None = None


class ExamHistoryEntry(BaseModel):
    exam_id: str
    score: float
    percentile: float


class StudentPerformanceResponse(BaseModel):
    student_id: str
    history: list[ExamHistoryEntry]
    strengths: list[str] | None = None
    weaknesses: list[str] | None = None


class StudentExamResponse(BaseModel):
    student_id: str
    exam_id: str
    score: float
    percentile: float | None = None


class QuestionAnalysisEntry(BaseModel):
    question_id: str
    avg_score: float
    pct_attempted: float
    pct_correct: float


class QuestionAnalysisResponse(BaseModel):
    exam_id: str
    items: list[QuestionAnalysisEntry]


# -- Helpers ---------------------------------------------------------------


def _get_repo(request: Request) -> AnalyticsRepo:
    return request.app.state.analytics_repo


# -- Endpoints -------------------------------------------------------------


@router.get(
    "/exams/{exam_id}/leaderboard",
    response_model=LeaderboardResponse,
)
async def get_leaderboard(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Get leaderboard rows for one exam."""
    repo = _get_repo(request)
    rows = await repo.get_leaderboard(
        exam_id=exam_id,
        tenant_id=user.tenant_id,
    )
    return {"items": rows}


@router.get(
    "/exams/{exam_id}/class-stats",
    response_model=ClassStatsResponse,
)
async def get_class_stats(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Get class-level statistics for one exam."""
    repo = _get_repo(request)
    all_scores = await repo.get_exam_scores(
        exam_id=exam_id,
        tenant_id=user.tenant_id,
    )

    if not all_scores:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No scores found for exam {exam_id}",
        )

    from src.config import DEFAULT_PASS_THRESHOLD

    scores = [s["total_score"] for s in all_scores]
    stats = compute_class_stats(
        scores, pass_threshold=DEFAULT_PASS_THRESHOLD,
    )

    return {
        "mean": stats.mean,
        "median": stats.median,
        "std_dev": stats.std_dev,
        "pass_rate": stats.pass_pct,
    }


@router.get(
    "/students/{student_id}/performance",
    response_model=StudentPerformanceResponse,
)
async def get_student_performance(
    student_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Get longitudinal performance for one student."""
    repo = _get_repo(request)
    history = await repo.get_student_history(
        student_id=student_id,
        tenant_id=user.tenant_id,
    )

    return {
        "student_id": student_id,
        "history": history,
    }


@router.get(
    "/exams/{exam_id}/student/{student_id}",
    response_model=StudentExamResponse,
)
async def get_student_exam_performance(
    exam_id: str,
    student_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Get per-student performance for a single exam."""
    repo = _get_repo(request)
    result = await repo.get_student_exam_score(
        exam_id=exam_id,
        student_id=student_id,
        tenant_id=user.tenant_id,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No score found for student {student_id} "
                f"in exam {exam_id}"
            ),
        )

    return {
        "student_id": student_id,
        "exam_id": exam_id,
        "score": result["score"],
        "percentile": result["percentile"],
    }


@router.get(
    "/exams/{exam_id}/questions",
    response_model=QuestionAnalysisResponse,
)
async def get_question_analysis(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Get question-wise difficulty analysis for an exam."""
    repo = _get_repo(request)
    responses = await repo.get_question_responses(
        exam_id=exam_id,
        tenant_id=user.tenant_id,
    )

    if not responses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No question responses found for exam {exam_id}"
            ),
        )

    total_students_data = await repo.get_exam_scores(
        exam_id=exam_id,
        tenant_id=user.tenant_id,
    )
    total_students = len(total_students_data)

    qr_list = [
        QuestionResponse(
            question_id=r["question_id"],
            score=r["score"],
            max_score=r["max_score"],
            attempted=r["attempted"],
        )
        for r in responses
    ]

    difficulty = compute_question_difficulty(qr_list, total_students)

    return {
        "exam_id": exam_id,
        "items": [
            {
                "question_id": d.question_id,
                "avg_score": d.avg_score,
                "pct_attempted": d.pct_attempted,
                "pct_correct": d.pct_correct,
            }
            for d in difficulty
        ],
    }
