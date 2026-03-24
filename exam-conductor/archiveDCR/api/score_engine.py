"""Score query and teacher-override endpoints.

Routes are mounted at ``/api/v1/exampen/scores``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, field_validator

from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
    require_exampen_role,
)
from exampen.dcr.domain.override_logic import (
    build_override_event,
    validate_override,
)
from exampen.dcr.domain.score_fsm import ScoreState, is_mutable
from exampen.dcr.storage.score_event_store import ScoreEventStore

logger = logging.getLogger(__name__)
router = APIRouter()

# Evaluator-level roles (re-used in RBAC checks)
_EVALUATOR_ROLES = frozenset({
    "teacher", "evaluator", "hod", "principal", "super_admin",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db


def _require_evaluator(user: ExamPenUser) -> None:
    if not _EVALUATOR_ROLES.intersection(user.exampen_roles):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Evaluator role required for this action",
        )


async def _publish_event(request: Request, subject: str, data: dict) -> None:
    nats = getattr(request.app.state, "exampen_nats", None)
    if nats is None or not nats.is_connected:
        return
    try:
        await nats.publish(subject, data)
    except Exception:
        logger.warning("NATS publish to %s failed (non-fatal)", subject)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ScoreOverrideBody(BaseModel):
    new_score: float
    reason: str

    @field_validator("reason")
    @classmethod
    def reason_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("reason must be at least 5 characters")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{exam_id}/overview")
async def exam_score_overview(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Class-level score overview: per-student totals from the materialised view.

    Requires evaluator+ role.
    """
    _require_evaluator(user)
    db = await _get_tenant_db(request, user)
    store = ScoreEventStore(db)
    rows = await store.get_exam_overview(exam_id, user.tenant_id)
    return {"items": rows}


@router.get("/{exam_id}/students/{student_id}")
async def get_student_scores(
    exam_id: str,
    student_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Detailed scores for a single student.

    Students may only view their own scores.
    """
    is_evaluator = bool(_EVALUATOR_ROLES.intersection(user.exampen_roles))
    if not is_evaluator and user.user_id != student_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Students may only view their own scores",
        )

    db = await _get_tenant_db(request, user)
    store = ScoreEventStore(db)
    rows = await store.get_current_scores(exam_id, student_id, user.tenant_id)
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No scores found")

    questions = []
    total = 0.0
    for r in rows:
        questions.append({
            "question_id": r.get("question_id", ""),
            "score": r.get("score", 0.0),
            "max_marks": r.get("max_marks"),
            "state": r.get("state"),
        })
        total += r.get("score", 0.0)

    return {
        "exam_id": exam_id,
        "student_id": student_id,
        "total_score": total,
        "questions": questions,
    }


@router.get("/{exam_id}/students/{student_id}/history")
async def get_student_history(
    exam_id: str,
    student_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Event history for a student's scores in an exam.

    Students may only view their own history.
    """
    is_evaluator = bool(_EVALUATOR_ROLES.intersection(user.exampen_roles))
    if not is_evaluator and user.user_id != student_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Students may only view their own score history",
        )

    db = await _get_tenant_db(request, user)
    store = ScoreEventStore(db)
    events = await store.get_event_history(exam_id, student_id, user.tenant_id)
    return {"items": events}


@router.patch("/{exam_id}/students/{student_id}/questions/{question_id}")
async def apply_override(
    exam_id: str,
    student_id: str,
    question_id: str,
    body: ScoreOverrideBody,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Apply a teacher score override on a single question.

    Requires evaluator+ role.  Score must be in a mutable state.
    """
    _require_evaluator(user)
    teacher_id = user.user_id

    db = await _get_tenant_db(request, user)
    store = ScoreEventStore(db)

    # Fetch current score for the question
    rows = await store.get_current_scores(exam_id, student_id, user.tenant_id)
    current_row = next(
        (r for r in rows if r.get("question_id") == question_id), None
    )
    if current_row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Question score not found")

    old_value = current_row.get("score", 0.0)
    current_state_str = current_row.get("state", "ai_draft")

    try:
        current_state = ScoreState(current_state_str)
    except ValueError:
        current_state = ScoreState.AI_DRAFT

    if not is_mutable(current_state):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Score is in '{current_state.value}' state and cannot be overridden",
        )

    # Domain validation
    vr = validate_override(
        old_value, body.new_score, body.reason,
        max_marks=current_row.get("max_marks"),
    )
    if not vr.valid:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="; ".join(vr.errors),
        )

    override = build_override_event(
        old_value=old_value,
        new_value=body.new_score,
        teacher_id=teacher_id,
        reason=body.reason,
        timestamp=datetime.now(timezone.utc),
    )

    # Append event and update materialised view
    await store.append_event(user.tenant_id, {
        "exam_id": exam_id,
        "student_id": student_id,
        "question_id": question_id,
        "event_type": "override_applied",
        "score": override.new_value,
        "old_value": override.old_value,
        "evaluator_id": teacher_id,
        "reason": override.reason,
        "state": "teacher_reviewed",
    })

    await _publish_event(request, "exampen.score.updated", {
        "exam_id": exam_id,
        "student_id": student_id,
        "question_id": question_id,
        "new_score": override.new_value,
        "old_score": override.old_value,
        "actor_id": teacher_id,
    })

    return {
        "question_id": question_id,
        "ai_score": old_value,
        "current_score": override.new_value,
    }
