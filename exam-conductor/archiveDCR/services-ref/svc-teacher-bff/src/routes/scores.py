"""Score review aggregation — the core value of the teacher BFF.

Endpoints:
- GET   /teacher/exams/{id}/scores                             — Class overview
- GET   /teacher/exams/{id}/students/{sid}                     — Student drill-down
- PATCH /teacher/exams/{id}/students/{sid}/questions/{qid}     — Override relay
- POST  /teacher/exams/{id}/scores/finalize                    — Finalize relay
- POST  /teacher/exams/{id}/scores/publish                     — Publish relay

The class overview aggregates data from svc-score-engine, svc-doc-assembly
(miss indicators), and svc-plagiarism into a single response.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from exampen_common.auth import ExamPenUser

from src.adapters import doc_assembly_client, plagiarism_client, score_client
from src.adapters.http_client import BackingClients
from src.middleware.auth import require_teacher

router = APIRouter(tags=["scores"])


def _get_clients(request: Request) -> BackingClients:
    return request.app.state.clients


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


# ---- Request schemas -------------------------------------------------------

class ScoreOverrideBody(BaseModel):
    question_id: str
    new_score: float
    reason: str


# ---- Endpoints -------------------------------------------------------------

@router.get("/teacher/exams/{exam_id}/scores")
async def class_score_overview(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Class score overview: aggregates scores + AI confidence + miss
    indicators + plagiarism flags into one response.

    Fires parallel requests to three backing services and merges
    the results by student_id.
    """
    clients = _get_clients(request)
    token = _get_token(request)

    scores_task = score_client.get_exam_scores(clients, token, exam_id)
    miss_task = doc_assembly_client.get_miss_indicators(clients, token, exam_id)
    plag_task = plagiarism_client.list_flags(clients, token, exam_id)

    scores, miss_data, plag_flags = await asyncio.gather(
        scores_task, miss_task, plag_task,
    )

    # Index miss counts by student_id
    miss_counts: dict[str, int] = {}
    if miss_data and "cells" in miss_data:
        for cell in miss_data["cells"]:
            sid = cell.get("student_id", "")
            if cell.get("state", "").startswith("miss_"):
                miss_counts[sid] = miss_counts.get(sid, 0) + 1

    # Index plagiarism flag counts by student_id
    plag_counts: dict[str, int] = {}
    for flag in plag_flags:
        for key in ("student_a_id", "student_b_id"):
            sid = flag.get(key, "")
            if sid:
                plag_counts[sid] = plag_counts.get(sid, 0) + 1

    # Merge into unified rows
    rows: list[dict[str, Any]] = []
    for score_row in scores:
        sid = score_row.get("student_id", "")
        rows.append({
            **score_row,
            "miss_indicator_count": miss_counts.get(sid, 0),
            "plagiarism_flag_count": plag_counts.get(sid, 0),
        })

    return {"rows": rows}


@router.get("/teacher/exams/{exam_id}/students/{student_id}")
async def student_drill_down(
    request: Request,
    exam_id: str,
    student_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Per-student drill-down: per-question breakdown with AI analysis."""
    clients = _get_clients(request)
    token = _get_token(request)

    detail_task = score_client.get_student_detail(
        clients, token, exam_id, student_id,
    )
    pages_task = doc_assembly_client.get_answer_pages(
        clients, token, exam_id, student_id,
    )

    detail, pages = await asyncio.gather(detail_task, pages_task)

    if detail is None:
        raise HTTPException(status_code=404, detail="Student scores not found")

    detail["answer_pages"] = pages
    return detail


@router.patch(
    "/teacher/exams/{exam_id}/students/{student_id}/questions/{question_id}",
)
async def score_override(
    request: Request,
    exam_id: str,
    student_id: str,
    question_id: str,
    body: ScoreOverrideBody,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Relay a score override to svc-score-engine."""
    clients = _get_clients(request)
    token = _get_token(request)
    return await score_client.relay_score_override(
        clients, token, exam_id, student_id, question_id,
        payload=body.model_dump(),
    )


@router.post("/teacher/exams/{exam_id}/scores/finalize")
async def finalize_scores(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Relay score finalization to svc-score-engine."""
    clients = _get_clients(request)
    token = _get_token(request)
    return await score_client.relay_finalize(clients, token, exam_id)


@router.post("/teacher/exams/{exam_id}/scores/publish")
async def publish_scores(
    request: Request,
    exam_id: str,
    user: ExamPenUser = Depends(require_teacher),
) -> dict[str, Any]:
    """Relay score publication to svc-score-engine."""
    clients = _get_clients(request)
    token = _get_token(request)
    return await score_client.relay_publish(clients, token, exam_id)
