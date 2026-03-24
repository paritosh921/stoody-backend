"""Objection lifecycle endpoints — file, list, detail, assign, resolve, escalate.

Routes are mounted at ``/api/v1/exampen/objections``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from exampen.dcr.core.auth_bridge import (
    ExamPenUser, get_exampen_user, require_exampen_role,
)
from exampen.dcr.domain.objection_fsm import (
    InvalidTransitionError, ObjectionEvent, ObjectionState,
    transition as obj_transition,
)
from exampen.dcr.domain.objection_rules import (
    EscalationError, EscalationPayload, FilingContext, FilingError,
    ResolutionError, ResolutionPayload, validate_escalation,
    validate_filing, validate_resolution,
)
from exampen.dcr.domain.score_fsm import ScoreState
from exampen.dcr.storage.objection_repo import ObjectionRepo
from exampen.dcr.storage.score_event_store import ScoreEventStore

logger = logging.getLogger(__name__)
router = APIRouter()


async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db

def _primary_role(user: ExamPenUser) -> str:
    return user.exampen_roles[0] if user.exampen_roles else "unknown"

async def _publish(request: Request, subject: str, data: dict) -> None:
    nats = getattr(request.app.state, "exampen_nats", None)
    if nats is None or not nats.is_connected:
        return
    try:
        await nats.publish(subject, data)
    except Exception:
        logger.warning("NATS publish to %s failed (non-fatal)", subject)

async def _is_objection_window_open(store: ScoreEventStore, exam_id: str, tenant_id: str) -> bool:
    rows = await store.get_current_scores(exam_id, "__all__", tenant_id)
    for r in rows:
        if r.get("question_id") == "__exam__":
            return r.get("state", "") in {ScoreState.PUBLISHED.value, ScoreState.OBJECTION_WINDOW.value}
    return False


# -- Schemas ----------------------------------------------------------------

class CreateObjectionBody(BaseModel):
    exam_id: str
    question_id: str
    objection_text: str

class AssignObjectionBody(BaseModel):
    assigned_to: str

class ResolveObjectionBody(BaseModel):
    resolution: str  # "approved" or "rejected"
    reason: str
    new_score: float | None = None

class EscalateObjectionBody(BaseModel):
    escalated_to: str
    reason: str


# -- Endpoints --------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED)
async def file_objection(
    body: CreateObjectionBody, request: Request,
    user: ExamPenUser = Depends(require_exampen_role("student")),
) -> dict[str, Any]:
    """File a new objection. Student-only, during objection window."""
    db = await _get_tenant_db(request, user)
    obj_repo = ObjectionRepo(db)
    score_store = ScoreEventStore(db)

    duplicate = await obj_repo.exists_for_question(
        student_id=user.user_id, exam_id=body.exam_id,
        question_id=body.question_id, tenant_id=user.tenant_id,
    )
    window_open = await _is_objection_window_open(score_store, body.exam_id, user.tenant_id)

    ctx = FilingContext(
        role=_primary_role(user), objection_window_open=window_open,
        existing_objection_for_question=duplicate, objection_text=body.objection_text,
    )
    try:
        validate_filing(ctx)
    except FilingError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message) from exc

    result = await obj_repo.create(user.tenant_id, {
        "exam_id": body.exam_id, "student_id": user.user_id,
        "question_id": body.question_id, "objection_text": body.objection_text,
    })
    await _publish(request, "exampen.objection.filed", {
        "objection_id": result["_id"], "exam_id": body.exam_id,
        "student_id": user.user_id, "question_id": body.question_id,
    })
    return result

@router.get("")
async def list_objections(
    request: Request, user: ExamPenUser = Depends(get_exampen_user),
    exam_id: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
) -> dict[str, Any]:
    """List objections visible to the current actor."""
    db = await _get_tenant_db(request, user)
    repo = ObjectionRepo(db)
    if exam_id:
        filters = {"state": status_filter} if status_filter else None
        items = await repo.list_by_exam(exam_id, user.tenant_id, filters=filters)
    else:
        query: dict[str, Any] = {"tenant_id": user.tenant_id}
        if status_filter:
            query["state"] = status_filter
        items = await db["exampen_objections"].find(query).sort("created_at", -1).limit(200).to_list(length=200)
    return {"items": items}

@router.get("/{objection_id}")
async def get_objection(
    objection_id: str, request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Return objection detail."""
    db = await _get_tenant_db(request, user)
    obj = await ObjectionRepo(db).get_by_id(objection_id, user.tenant_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Objection not found")
    return obj

@router.post("/{objection_id}/assign")
async def assign_objection(
    objection_id: str, body: AssignObjectionBody, request: Request,
    user: ExamPenUser = Depends(require_exampen_role("hod", "principal")),
) -> dict[str, Any]:
    """Assign an objection to an evaluator. HOD / principal only."""
    db = await _get_tenant_db(request, user)
    repo = ObjectionRepo(db)
    obj = await repo.get_by_id(objection_id, user.tenant_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Objection not found")
    try:
        tr = obj_transition(ObjectionState(obj["state"]), ObjectionEvent.ASSIGN)
    except InvalidTransitionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    result = await repo.transition_state(
        objection_id, user.tenant_id, from_state=tr.old_state.value,
        to_state=tr.new_state.value, data={"assigned_to": body.assigned_to},
    )
    if result is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Concurrent state change")
    await _publish(request, "exampen.objection.assigned", {
        "objection_id": objection_id, "assigned_to": body.assigned_to, "actor_id": user.user_id,
    })
    return result

@router.post("/{objection_id}/resolve")
async def resolve_objection(
    objection_id: str, body: ResolveObjectionBody, request: Request,
    user: ExamPenUser = Depends(require_exampen_role("evaluator", "hod")),
) -> dict[str, Any]:
    """Resolve an objection -- approve (triggers re-score) or reject."""
    db = await _get_tenant_db(request, user)
    repo = ObjectionRepo(db)
    obj = await repo.get_by_id(objection_id, user.tenant_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Objection not found")
    payload = ResolutionPayload(resolution=body.resolution, reason=body.reason, new_score=body.new_score)
    try:
        validate_resolution(payload)
    except ResolutionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message) from exc
    try:
        tr = obj_transition(ObjectionState(obj["state"]), ObjectionEvent.RESOLVE)
    except InvalidTransitionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    result = await repo.transition_state(
        objection_id, user.tenant_id, from_state=tr.old_state.value, to_state=tr.new_state.value,
        data={"resolution": body.resolution, "resolution_reason": body.reason, "score_delta": body.new_score},
    )
    if result is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Concurrent state change")
    await _publish(request, "exampen.objection.resolved", {
        "objection_id": objection_id, "resolution": body.resolution, "actor_id": user.user_id,
    })
    return result

@router.post("/{objection_id}/escalate")
async def escalate_objection(
    objection_id: str, body: EscalateObjectionBody, request: Request,
    user: ExamPenUser = Depends(require_exampen_role("evaluator")),
) -> dict[str, Any]:
    """Escalate an objection to HOD or senior evaluator."""
    db = await _get_tenant_db(request, user)
    repo = ObjectionRepo(db)
    obj = await repo.get_by_id(objection_id, user.tenant_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Objection not found")
    esc = EscalationPayload(escalated_to=body.escalated_to, reason=body.reason)
    try:
        validate_escalation(esc)
    except EscalationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message) from exc
    try:
        tr = obj_transition(ObjectionState(obj["state"]), ObjectionEvent.ESCALATE)
    except InvalidTransitionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    result = await repo.transition_state(
        objection_id, user.tenant_id, from_state=tr.old_state.value, to_state=tr.new_state.value,
        data={"assigned_to": body.escalated_to, "escalation_reason": body.reason},
    )
    if result is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Concurrent state change")
    await _publish(request, "exampen.objection.escalated", {
        "objection_id": objection_id, "escalated_to": body.escalated_to, "actor_id": user.user_id,
    })
    return result
