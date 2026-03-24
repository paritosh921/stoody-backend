"""Teacher workflow endpoints — bulk approve, step-level marking, export.

These are higher-level actions that relay to backing services
(svc-score-engine, svc-analytics).  Like all teacher-bff endpoints
they perform ZERO direct database writes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from exampen_common.auth import ExamPenUser

from src.adapters import score_client, analytics_client
from src.adapters.http_client import BackingClients
from src.middleware.auth import require_teacher

router = APIRouter(tags=["workflows"])


def _get_clients(request: Request) -> BackingClients:
    return request.app.state.clients


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


# ---- Request schemas -------------------------------------------------------


class BulkApproveBody(BaseModel):
    """Approve (or reject) multiple student scores at once.

    Each entry maps a student_id to an approval decision.
    If svc-score-engine does not support native bulk approve, this
    endpoint iterates and sends individual overrides.
    """
    decisions: list[dict[str, Any]] = Field(
        ...,
        description=(
            "List of {student_id, approved: bool, reason?: str} dicts"
        ),
    )


class StepMarkingBody(BaseModel):
    """Step-level score adjustment for a single question."""
    steps: list[dict[str, Any]] = Field(
        ...,
        description=(
            "List of {step_index, awarded_marks, comment?} dicts"
        ),
    )


class ExportTriggerBody(BaseModel):
    """Optional parameters for the export trigger."""
    format: str = Field("pdf", description="Export format: pdf | csv | xlsx")
    include_answer_sheets: bool = False


# ---- Endpoints -------------------------------------------------------------


@router.post("/teacher/exams/{exam_id}/scores/bulk-approve")
async def bulk_approve_scores(
    request: Request,
    exam_id: str,
    body: BulkApproveBody,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Bulk-approve (or reject) student scores.

    Relays to svc-score-engine.  If score-engine does not expose a
    native bulk endpoint, each decision is sent as an individual
    override request in parallel.
    """
    clients = _get_clients(request)
    token = _get_token(request)
    return await score_client.relay_bulk_approve(
        clients, token, exam_id, body.decisions,
    )


@router.patch(
    "/teacher/exams/{exam_id}/students/{student_id}"
    "/questions/{question_id}/steps",
)
async def step_level_marking(
    request: Request,
    exam_id: str,
    student_id: str,
    question_id: str,
    body: StepMarkingBody,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Apply step-level marking for a single question.

    Relays to svc-score-engine step-marking endpoint.
    """
    clients = _get_clients(request)
    token = _get_token(request)
    return await score_client.relay_step_marking(
        clients, token, exam_id, student_id, question_id,
        steps=body.steps,
    )


@router.post("/teacher/exams/{exam_id}/export")
async def trigger_export(
    request: Request,
    exam_id: str,
    body: ExportTriggerBody | None = None,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Build an export from analytics data (server-side CSV/PDF).

    Since svc-analytics only exposes read endpoints (no dedicated export),
    the BFF fetches class-stats + leaderboard and assembles the export itself.
    """
    clients = _get_clients(request)
    token = _get_token(request)
    fmt = body.format if body else "csv"

    # Fetch the data from analytics read endpoints
    stats = await analytics_client.get_class_stats(clients, token, exam_id)
    leaderboard = await analytics_client.get_leaderboard(clients, token, exam_id)

    if fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["rank", "student_id", "student_name", "score", "percentile"])
        for row in (leaderboard or {}).get("items", []):
            writer.writerow([
                row.get("rank", ""),
                row.get("student_id", ""),
                row.get("student_name", ""),
                row.get("score", ""),
                row.get("percentile", ""),
            ])
        return {"format": "csv", "content": buf.getvalue(), "stats": stats}
    else:
        return {"format": fmt, "stats": stats, "leaderboard": leaderboard}
