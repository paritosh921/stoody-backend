"""HTTP routes for score queries and teacher overrides.

Matches the OpenAPI contract at ``api/score-engine.openapi.yaml``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from exampen_common.auth import ExamPenUser, get_current_user
from src.domain.override_logic import (
    build_override_event,
    validate_override,
)
from src.domain.score_fsm import ScoreState, is_mutable
from src.events.score_publisher import publish_score_updated
from src.storage.db import get_session
from src.storage.score_event_store import (
    append_event,
    get_current_scores,
    get_event_history,
    get_exam_overview,
)

router = APIRouter(tags=["scores"])

# ---------------------------------------------------------------------------
# RBAC helpers
# ---------------------------------------------------------------------------

_EVALUATOR_ROLES = {"teacher", "evaluator", "hod", "principal", "super_admin"}


def _require_evaluator(user: ExamPenUser) -> None:
    """Raise 403 if the user does not hold an evaluator-level role."""
    if not _EVALUATOR_ROLES.intersection(user.exampen_roles):
        raise HTTPException(
            status_code=403,
            detail="Evaluator role required for this action",
        )


# ---- Request / Response schemas ----------------------------------------------

class ScoreOverrideRequest(BaseModel):
    new_score: float
    reason: str

    @field_validator("reason")
    @classmethod
    def reason_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("reason must be at least 5 characters")
        return v


class StepScoreOut(BaseModel):
    label: str
    awarded: float
    max: float


class QuestionScoreOut(BaseModel):
    question_id: str
    ai_score: float | None = None
    current_score: float
    max_score: float | None = None
    confidence: float | None = None
    override_reason: str | None = None
    step_scores: list[StepScoreOut] | None = None


class StudentScoreDetailOut(BaseModel):
    exam_id: str
    student_id: str
    total_score: float
    max_score: float | None = None
    lifecycle_state: str
    questions: list[QuestionScoreOut]


class StudentOverviewRow(BaseModel):
    student_id: str
    total_score: float
    question_count: int
    lifecycle_state: str


class ScoreHistoryItemOut(BaseModel):
    event_id: str
    event_type: str
    old_value: float | None
    new_value: float
    actor_id: str
    reason: str | None = None
    created_at: str


# ---- Endpoints ---------------------------------------------------------------


@router.get("/scores/{exam_id}/overview")
async def exam_score_overview(
    exam_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: ExamPenUser = Depends(get_current_user),
) -> dict[str, list[StudentOverviewRow]]:
    """Class-level score overview: per-student totals from the materialised view.

    Requires evaluator+ role.
    """
    _require_evaluator(current_user)

    rows = await get_exam_overview(session, exam_id)
    items = [
        StudentOverviewRow(
            student_id=r["student_id"],
            total_score=float(r["total_score"]),
            question_count=int(r["question_count"]),
            lifecycle_state=r["lifecycle_state"],
        )
        for r in rows
    ]
    return {"items": items}


@router.get("/scores/{exam_id}/students/{student_id}")
async def get_student_scores(
    exam_id: str,
    student_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: ExamPenUser = Depends(get_current_user),
) -> StudentScoreDetailOut:
    # Students may only view their own scores.
    is_evaluator = bool(_EVALUATOR_ROLES.intersection(current_user.exampen_roles))
    if not is_evaluator and current_user.user_id != student_id:
        raise HTTPException(
            status_code=403,
            detail="Students may only view their own scores",
        )

    rows = await get_current_scores(session, exam_id, student_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No scores found")

    questions: list[QuestionScoreOut] = []
    total = 0.0
    state = "ai_draft"
    for r in rows:
        if r["question_id"] == "__exam__":
            state = r["lifecycle_state"]
            continue
        questions.append(
            QuestionScoreOut(
                question_id=r["question_id"],
                current_score=r["current_score"],
            )
        )
        total += r["current_score"]

    return StudentScoreDetailOut(
        exam_id=exam_id,
        student_id=student_id,
        total_score=total,
        lifecycle_state=state,
        questions=questions,
    )


@router.get("/scores/{exam_id}/students/{student_id}/history")
async def get_student_history(
    exam_id: str,
    student_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    # Students may only view their own history.
    is_evaluator = bool(_EVALUATOR_ROLES.intersection(current_user.exampen_roles))
    if not is_evaluator and current_user.user_id != student_id:
        raise HTTPException(
            status_code=403,
            detail="Students may only view their own score history",
        )

    events = await get_event_history(session, exam_id, student_id)
    items = [
        ScoreHistoryItemOut(
            event_id=e["event_id"],
            event_type=e["event_type"],
            old_value=e["old_value"],
            new_value=e["new_value"],
            actor_id=e["actor_id"],
            reason=e["reason"],
            created_at=str(e["created_at"]),
        )
        for e in events
    ]
    return {"items": items}


@router.patch(
    "/scores/{exam_id}/students/{student_id}/questions/{question_id}"
)
async def apply_override(
    exam_id: str,
    student_id: str,
    question_id: str,
    body: ScoreOverrideRequest,
    session: AsyncSession = Depends(get_session),
    current_user: ExamPenUser = Depends(get_current_user),
) -> QuestionScoreOut:
    _require_evaluator(current_user)

    # Bind teacher_id from the authenticated user — never trust the body.
    teacher_id = current_user.user_id

    # Fetch current score.
    rows = await get_current_scores(session, exam_id, student_id)
    current_row = next(
        (r for r in rows if r["question_id"] == question_id), None
    )
    if current_row is None:
        raise HTTPException(404, "Question score not found")

    old_value = current_row["current_score"]
    current_state = ScoreState(current_row["lifecycle_state"])

    if not is_mutable(current_state):
        raise HTTPException(
            409, f"Score is in '{current_state.value}' state and cannot be overridden"
        )

    # Domain validation (pure, no I/O).
    vr = validate_override(old_value, body.new_score, body.reason)
    if not vr.valid:
        raise HTTPException(422, detail="; ".join(vr.errors))

    override = build_override_event(
        old_value=old_value,
        new_value=body.new_score,
        teacher_id=teacher_id,
        reason=body.reason,
        timestamp=datetime.now(timezone.utc),
    )

    event_id = await append_event(
        session,
        exam_id=exam_id,
        student_id=student_id,
        question_id=question_id,
        event_type="override_applied",
        old_value=override.old_value,
        new_value=override.new_value,
        actor_id=override.teacher_id,
        reason=override.reason,
    )
    await session.commit()

    # Publish AFTER commit.
    await publish_score_updated(
        exam_id=exam_id,
        student_id=student_id,
        question_id=question_id,
        lifecycle_state="teacher_reviewed",
        total_score=override.new_value,
        previous_total_score=override.old_value,
        reason="override_applied",
    )

    return QuestionScoreOut(
        question_id=question_id,
        ai_score=old_value,
        current_score=override.new_value,
    )
