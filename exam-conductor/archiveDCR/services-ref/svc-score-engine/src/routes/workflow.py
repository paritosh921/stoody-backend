"""Workflow endpoints: finalize, publish, lock.

These advance the exam-level score lifecycle via the FSM.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from exampen_common.auth import ExamPenUser, get_current_user
from src.domain.score_fsm import ScoreState, ScoreTransitionError, transition
from src.events.score_publisher import publish_score_updated
from src.storage.db import get_session
from src.storage.score_event_store import (
    append_event,
    get_exam_lifecycle_state,
)

router = APIRouter(tags=["workflow"])

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

class FinalizeRequest(BaseModel):
    objection_window_days: int | None = None


class PublishRequest(BaseModel):
    objection_window_days: int = 7


class LockRequest(BaseModel):
    pass


class WorkflowStateResponse(BaseModel):
    exam_id: str
    lifecycle_state: str
    changed_at: str


# ---- Helpers -----------------------------------------------------------------

async def _advance(
    session: AsyncSession,
    exam_id: str,
    actor_id: str,
    target: ScoreState,
    reason: str,
) -> WorkflowStateResponse:
    """Validate FSM transition, append event, commit, publish."""
    raw_state = await get_exam_lifecycle_state(session, exam_id)
    current = ScoreState(raw_state) if raw_state else ScoreState.AI_DRAFT

    try:
        result = transition(current, target)
    except ScoreTransitionError as exc:
        raise HTTPException(409, str(exc)) from exc

    now = datetime.now(timezone.utc)

    await append_event(
        session,
        exam_id=exam_id,
        student_id="__all__",
        question_id="__exam__",
        event_type=reason,
        old_value=None,
        new_value=0.0,
        actor_id=actor_id,
        reason=reason,
    )
    await session.commit()

    await publish_score_updated(
        exam_id=exam_id,
        student_id="__all__",
        question_id=None,
        lifecycle_state=result.new_state.value,
        total_score=0.0,
        previous_total_score=None,
        reason=reason,
    )

    return WorkflowStateResponse(
        exam_id=exam_id,
        lifecycle_state=result.new_state.value,
        changed_at=now.isoformat(),
    )


# ---- Endpoints ---------------------------------------------------------------


@router.get("/scores/{exam_id}/workflow-state")
async def get_workflow_state(
    exam_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: ExamPenUser = Depends(get_current_user),
) -> WorkflowStateResponse:
    """Return the current score lifecycle state for an exam.

    Used by svc-review to check whether the objection window is open.
    """
    raw_state = await get_exam_lifecycle_state(session, exam_id)
    state = raw_state if raw_state else ScoreState.AI_DRAFT.value
    return WorkflowStateResponse(
        exam_id=exam_id,
        lifecycle_state=state,
        changed_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/scores/{exam_id}/finalize")
async def finalize_scores(
    exam_id: str,
    body: FinalizeRequest,
    session: AsyncSession = Depends(get_session),
    current_user: ExamPenUser = Depends(get_current_user),
) -> WorkflowStateResponse:
    _require_evaluator(current_user)
    return await _advance(
        session, exam_id, current_user.user_id, ScoreState.FINALIZED, "finalized"
    )


@router.post("/scores/{exam_id}/publish")
async def publish_scores(
    exam_id: str,
    body: PublishRequest,
    session: AsyncSession = Depends(get_session),
    current_user: ExamPenUser = Depends(get_current_user),
) -> WorkflowStateResponse:
    _require_evaluator(current_user)
    return await _advance(
        session, exam_id, current_user.user_id, ScoreState.PUBLISHED, "published"
    )


@router.post("/scores/{exam_id}/lock")
async def lock_scores(
    exam_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: ExamPenUser = Depends(get_current_user),
) -> WorkflowStateResponse:
    _require_evaluator(current_user)
    return await _advance(
        session, exam_id, current_user.user_id, ScoreState.LOCKED, "objection_rescored"
    )
