"""Teacher exam management relay routes — proxies writes to svc-exam-orch.

Endpoints:
- POST   /teacher/exams                          — Create exam
- PUT    /teacher/exams/{id}/rubric               — Save rubric
- GET    /teacher/exams/{id}/rubric               — Get rubric
- PUT    /teacher/exams/{id}/regions              — Save question regions
- GET    /teacher/exams/{id}/regions              — Get question regions
- POST   /teacher/exams/{id}/invigilators         — Assign invigilators
- POST   /teacher/exams/{id}/evaluators           — Assign evaluators
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from exampen_common.auth import ExamPenUser

from src.adapters import exam_client
from src.adapters.http_client import BackingClients
from src.middleware.auth import require_teacher

router = APIRouter(tags=["exam-management"])


def _clients(r: Request) -> BackingClients:
    return r.app.state.clients


def _token(r: Request) -> str:
    return r.headers.get("Authorization", "").removeprefix("Bearer ").strip()


@router.post("/teacher/exams")
async def create_exam(
    request: Request, body: dict[str, Any],
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    return await exam_client.create_exam(_clients(request), _token(request), body) or {}


@router.put("/teacher/exams/{exam_id}/rubric")
async def save_rubric(
    request: Request, exam_id: str, body: dict[str, Any],
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    return await exam_client.save_rubric(_clients(request), _token(request), exam_id, body) or {}


@router.get("/teacher/exams/{exam_id}/rubric")
async def get_rubric(
    request: Request, exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    return await exam_client.get_rubric(_clients(request), _token(request), exam_id) or {}


@router.put("/teacher/exams/{exam_id}/regions")
async def save_regions(
    request: Request, exam_id: str, body: dict[str, Any],
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    return await exam_client.save_question_regions(_clients(request), _token(request), exam_id, body) or {}


@router.get("/teacher/exams/{exam_id}/regions")
async def get_regions(
    request: Request, exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    return await exam_client.get_question_regions(_clients(request), _token(request), exam_id) or {}


@router.post("/teacher/exams/{exam_id}/invigilators")
async def assign_staff(
    request: Request, exam_id: str, body: dict[str, Any],
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Relay to svc-exam-orch single assignment endpoint (invigilators + evaluators)."""
    return await exam_client.assign_staff(_clients(request), _token(request), exam_id, body) or {}
