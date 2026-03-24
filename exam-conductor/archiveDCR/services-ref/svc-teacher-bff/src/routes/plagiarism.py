"""Plagiarism review proxy — flags and teacher verdicts.

Endpoints:
- GET   /teacher/exams/{id}/plagiarism      — List flags (proxy)
- PATCH /teacher/plagiarism/{flag_id}/verdict — Verdict relay
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from exampen_common.auth import ExamPenUser

from src.adapters import plagiarism_client
from src.adapters.http_client import BackingClients
from src.middleware.auth import require_teacher

router = APIRouter(tags=["plagiarism"])


def _get_clients(request: Request) -> BackingClients:
    return request.app.state.clients


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


class VerdictBody(BaseModel):
    verdict: str  # "confirmed" | "dismissed"
    reason: str


@router.get("/teacher/exams/{exam_id}/plagiarism")
async def list_plagiarism_flags(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """List plagiarism flags for an exam (proxy to svc-plagiarism)."""
    clients = _get_clients(request)
    token = _get_token(request)
    items = await plagiarism_client.list_flags(clients, token, exam_id)
    return {"items": items}


@router.patch("/teacher/plagiarism/{flag_id}/verdict")
async def submit_verdict(
    request: Request,
    flag_id: str,
    body: VerdictBody,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Relay a teacher verdict to svc-plagiarism."""
    clients = _get_clients(request)
    token = _get_token(request)
    return await plagiarism_client.relay_verdict(
        clients, token, flag_id, payload=body.model_dump(),
    )
