"""Score workflow endpoints: finalize, publish, lock, and workflow-state query.

Split from score_engine.py to stay under 300 lines per file.
Routes are mounted at ``/api/v1/exampen/scores`` (same prefix as score_engine).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
)
from exampen.dcr.domain.score_fsm import (
    ScoreState,
    ScoreTransitionError,
    transition as score_transition,
)
from exampen.dcr.storage.score_event_store import ScoreEventStore

logger = logging.getLogger(__name__)
router = APIRouter()

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


async def _advance_workflow(
    request: Request,
    user: ExamPenUser,
    exam_id: str,
    target: ScoreState,
    reason: str,
) -> dict[str, Any]:
    """Validate FSM transition, append event, publish."""
    db = await _get_tenant_db(request, user)
    store = ScoreEventStore(db)

    # Determine current lifecycle state from a sentinel row
    rows = await store.get_current_scores(exam_id, "__all__", user.tenant_id)
    current_state_str = None
    for r in rows:
        if r.get("question_id") == "__exam__":
            current_state_str = r.get("state")
            break

    current = ScoreState(current_state_str) if current_state_str else ScoreState.AI_DRAFT

    try:
        result = score_transition(current, target)
    except ScoreTransitionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    now = datetime.now(timezone.utc)

    await store.append_event(user.tenant_id, {
        "exam_id": exam_id,
        "student_id": "__all__",
        "question_id": "__exam__",
        "event_type": reason,
        "score": 0.0,
        "state": result.new_state.value,
        "evaluator_id": user.user_id,
        "reason": reason,
    })

    await _publish_event(request, "exampen.score.updated", {
        "exam_id": exam_id,
        "lifecycle_state": result.new_state.value,
        "reason": reason,
        "actor_id": user.user_id,
    })

    return {
        "exam_id": exam_id,
        "lifecycle_state": result.new_state.value,
        "changed_at": now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class FinalizeBody(BaseModel):
    objection_window_days: int | None = None


class PublishBody(BaseModel):
    objection_window_days: int = 7


class LockBody(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{exam_id}/workflow-state")
async def get_workflow_state(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Return the current score lifecycle state for an exam."""
    db = await _get_tenant_db(request, user)
    store = ScoreEventStore(db)

    rows = await store.get_current_scores(exam_id, "__all__", user.tenant_id)
    state = ScoreState.AI_DRAFT.value
    for r in rows:
        if r.get("question_id") == "__exam__":
            state = r.get("state", ScoreState.AI_DRAFT.value)
            break

    return {
        "exam_id": exam_id,
        "lifecycle_state": state,
        "changed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/{exam_id}/finalize")
async def finalize_scores(
    exam_id: str,
    body: FinalizeBody,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Finalize scores — all teacher reviews are complete."""
    _require_evaluator(user)
    return await _advance_workflow(
        request, user, exam_id, ScoreState.FINALIZED, "finalized",
    )


@router.post("/{exam_id}/publish")
async def publish_scores(
    exam_id: str,
    body: PublishBody,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Publish scores — visible to students, objection window opens."""
    _require_evaluator(user)
    return await _advance_workflow(
        request, user, exam_id, ScoreState.PUBLISHED, "published",
    )


@router.post("/{exam_id}/lock")
async def lock_scores(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Lock scores — final, no further changes."""
    _require_evaluator(user)
    return await _advance_workflow(
        request, user, exam_id, ScoreState.LOCKED, "locked",
    )
