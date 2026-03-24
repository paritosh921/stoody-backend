"""Objection inbox proxy — aggregates answer image + AI + score + rubric.

Endpoints:
- GET  /teacher/exams/{id}/objections   — List objections for an exam
- GET  /teacher/objections/{id}         — Detail with rich context
- POST /teacher/objections/{id}/resolve — Resolve relay
- POST /teacher/objections/{id}/escalate — Escalate relay
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from exampen_common.auth import ExamPenUser

from src.adapters import review_client
from src.adapters.http_client import BackingClients
from src.middleware.auth import require_teacher

router = APIRouter(tags=["objections"])


def _get_clients(request: Request) -> BackingClients:
    return request.app.state.clients


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


class ResolveBody(BaseModel):
    verdict: str  # "approved" | "rejected"
    new_score: float | None = None
    reason: str


class EscalateBody(BaseModel):
    target_role: str  # "hod" | "senior_evaluator"
    reason: str


@router.get("/teacher/exams/{exam_id}/objections")
async def list_objections(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """List objections for an exam (proxies to svc-review)."""
    clients = _get_clients(request)
    token = _get_token(request)
    items = await review_client.list_objections(clients, token, exam_id)
    return {"items": items}


@router.get("/teacher/objections/{objection_id}")
async def get_objection_detail(
    request: Request,
    objection_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Objection detail with context (answer image + AI + score + rubric).

    Aggregates from svc-review which itself enriches with score and
    doc-assembly data.
    """
    clients = _get_clients(request)
    token = _get_token(request)
    data = await review_client.get_objection_detail(clients, token, objection_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Objection not found")
    return data


@router.post("/teacher/objections/{objection_id}/resolve")
async def resolve_objection(
    request: Request,
    objection_id: str,
    body: ResolveBody,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Relay objection resolution to svc-review."""
    clients = _get_clients(request)
    token = _get_token(request)
    return await review_client.relay_resolve(
        clients, token, objection_id, payload=body.model_dump(),
    )


@router.post("/teacher/objections/{objection_id}/escalate")
async def escalate_objection(
    request: Request,
    objection_id: str,
    body: EscalateBody,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Relay objection escalation to svc-review."""
    clients = _get_clients(request)
    token = _get_token(request)
    return await review_client.relay_escalate(
        clients, token, objection_id, payload=body.model_dump(),
    )
