"""Historical performance routes — read-only from svc-analytics.

Endpoints:
  GET /student/performance/history    — Score history across exams
  GET /student/performance/trends     — Trend data for charts
  GET /student/performance/strengths  — AI-generated strength/weakness
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
from src.routes.models import ErrorBody, PerformanceView, StrengthsView, TrendData

_log = get_logger(__name__)

router = APIRouter()


def _extract_token(request: Request) -> str:
    """Extract raw bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


@router.get(
    "/history",
    response_model=PerformanceView,
)
async def get_performance_history(
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
    student_id: str | None = Query(None, description="Required for parent"),
) -> dict[str, Any]:
    """Score history across exams with strengths and weaknesses."""
    effective_sid = require_own_data(identity, student_id)
    token = _extract_token(request)
    analytics = request.app.state.analytics_client

    history = await analytics.get_score_history(effective_sid, token)
    strengths_data = await analytics.get_strengths(effective_sid, token)

    strengths = []
    weaknesses = []
    if strengths_data:
        strengths = strengths_data.get("strengths", [])
        weaknesses = strengths_data.get("weaknesses", [])

    return {
        "history": history,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


@router.get(
    "/trends",
    response_model=TrendData,
)
async def get_performance_trends(
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
    student_id: str | None = Query(None, description="Required for parent"),
) -> dict[str, Any]:
    """Trend data for performance charts (scores + percentiles over time)."""
    effective_sid = require_own_data(identity, student_id)
    token = _extract_token(request)
    analytics = request.app.state.analytics_client

    trends = await analytics.get_trends(effective_sid, token)
    if trends is None:
        return {"history": []}

    return {"history": trends.get("history", trends.get("items", []))}


@router.get(
    "/strengths",
    response_model=StrengthsView,
    responses={404: {"model": ErrorBody}},
)
async def get_strengths(
    request: Request,
    identity: StudentBFFIdentity = Depends(require_student_or_parent),
    student_id: str | None = Query(None, description="Required for parent"),
) -> dict[str, Any]:
    """AI-generated strength/weakness analysis based on historical exams."""
    effective_sid = require_own_data(identity, student_id)
    token = _extract_token(request)
    analytics = request.app.state.analytics_client

    data = await analytics.get_strengths(effective_sid, token)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No performance data available yet",
        )

    return {
        "strengths": data.get("strengths", []),
        "weaknesses": data.get("weaknesses", []),
    }
