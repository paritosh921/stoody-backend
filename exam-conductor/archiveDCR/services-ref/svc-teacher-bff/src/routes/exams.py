"""Teacher exam management views — proxy to svc-exam-orch.

Endpoints:
- GET /teacher/exams        — List exams for the authenticated teacher
- GET /teacher/exams/{id}   — Exam detail with roster
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from exampen_common.auth import ExamPenUser

from src.adapters import exam_client
from src.adapters.http_client import BackingClients
from src.middleware.auth import require_teacher

router = APIRouter(tags=["exams"])


def _get_clients(request: Request) -> BackingClients:
    return request.app.state.clients


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


@router.get("/teacher/exams")
async def list_exams(
    request: Request,
    subject_id: str | None = None,
    class_id: str | None = None,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """List exams visible to the authenticated teacher."""
    clients = _get_clients(request)
    token = _get_token(request)
    items = await exam_client.list_exams(
        clients, token, subject_id=subject_id, class_id=class_id,
    )
    return {"items": items}


@router.get("/teacher/exams/{exam_id}")
async def get_exam_detail(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Exam detail with roster."""
    clients = _get_clients(request)
    token = _get_token(request)
    data = await exam_client.get_exam_detail(clients, token, exam_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Exam not found")
    return data
