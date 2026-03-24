"""Analytics proxy — leaderboards, class stats, question analysis.

Endpoints:
- GET /teacher/exams/{id}/leaderboard   — Proxy to svc-analytics
- GET /teacher/exams/{id}/class-stats   — Proxy
- GET /teacher/exams/{id}/questions     — Question-wise analysis proxy
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from exampen_common.auth import ExamPenUser

from src.adapters import analytics_client
from src.adapters.http_client import BackingClients
from src.middleware.auth import require_teacher

router = APIRouter(tags=["analytics"])


def _get_clients(request: Request) -> BackingClients:
    return request.app.state.clients


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


@router.get("/teacher/exams/{exam_id}/leaderboard")
async def get_leaderboard(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Proxy leaderboard from svc-analytics."""
    clients = _get_clients(request)
    token = _get_token(request)
    data = await analytics_client.get_leaderboard(clients, token, exam_id)
    if data is None:
        raise HTTPException(status_code=502, detail="Analytics service unavailable")
    return data


@router.get("/teacher/exams/{exam_id}/class-stats")
async def get_class_stats(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Proxy class statistics from svc-analytics."""
    clients = _get_clients(request)
    token = _get_token(request)
    data = await analytics_client.get_class_stats(clients, token, exam_id)
    if data is None:
        raise HTTPException(status_code=502, detail="Analytics service unavailable")
    return data


@router.get("/teacher/exams/{exam_id}/questions")
async def get_question_analysis(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Proxy question-wise analysis from svc-analytics."""
    clients = _get_clients(request)
    token = _get_token(request)
    data = await analytics_client.get_question_analysis(clients, token, exam_id)
    if data is None:
        raise HTTPException(status_code=502, detail="Analytics service unavailable")
    return data
