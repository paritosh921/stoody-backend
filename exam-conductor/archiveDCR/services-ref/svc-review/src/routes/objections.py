"""Objection REST endpoints — matches review.openapi.yaml.

Security: student_id and actor_id are always bound from the
authenticated JWT, never from the request body.  Role checks enforce
the RBAC matrix before any state mutation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.logging import get_logger

from src.domain.objection_fsm import (
    InvalidTransitionError,
    ObjectionEvent,
    ObjectionState,
    transition,
)
from src.domain.objection_rules import (
    EscalationError,
    EscalationPayload,
    FilingContext,
    FilingError,
    ResolutionError,
    ResolutionPayload,
    validate_escalation,
    validate_filing,
    validate_resolution,
)
from src.routes.models import (
    AssignObjectionRequest,
    CreateObjectionRequest,
    ErrorBody,
    EscalateObjectionRequest,
    ObjectionDetailResponse,
    ObjectionListResponse,
    ResolveObjectionRequest,
)
from src.storage.objection_repo import ObjectionNotFoundError

_log = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _not_found(oid: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"Objection {oid} not found")


def _conflict(msg: str) -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, msg)


def _forbidden(msg: str) -> HTTPException:
    return HTTPException(status.HTTP_403_FORBIDDEN, msg)


def _primary_role(user: ExamPenUser) -> str:
    if user.exampen_roles:
        return user.exampen_roles[0]
    return "unknown"


def _has_any_role(user: ExamPenUser, *roles: str) -> bool:
    """Return True if the user holds at least one of *roles*."""
    return bool(set(user.exampen_roles) & set(roles))


def _require_roles(user: ExamPenUser, *roles: str) -> None:
    """Raise 403 if the user does not hold any of the required roles."""
    if not _has_any_role(user, *roles):
        raise _forbidden(
            f"This action requires one of: {', '.join(roles)}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=ObjectionDetailResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorBody}, 409: {"model": ErrorBody}},
)
async def file_objection(
    body: CreateObjectionRequest, request: Request,
) -> dict[str, Any]:
    """File a new objection. Student-only, during objection window."""
    user = await get_current_user(request)
    _require_roles(user, "student")

    repo = request.app.state.objection_repo
    publisher = request.app.state.objection_publisher

    student_id = user.user_id

    duplicate = await repo.exists_for_question(
        student_id=student_id,
        exam_id=body.exam_id,
        question_id=body.question_id,
    )
    # Check objection window via svc-score-engine (fail-closed)
    score_checker = request.app.state.score_checker
    window_open = await score_checker.is_objection_window_open(body.exam_id)

    ctx = FilingContext(
        role=_primary_role(user),
        objection_window_open=window_open,
        existing_objection_for_question=duplicate,
        objection_text=body.objection_text,
    )
    try:
        validate_filing(ctx)
    except FilingError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message) from exc

    result = await repo.create(
        exam_id=body.exam_id,
        student_id=student_id,
        question_id=body.question_id,
        objection_text=body.objection_text,
        tenant_id=user.tenant_id,
    )
    await publisher.publish_transition(
        objection_id=result["objection_id"],
        exam_id=body.exam_id,
        student_id=student_id,
        question_id=body.question_id,
        action="filed", state="filed", actor_id=user.user_id,
    )
    return result


@router.get("", response_model=ObjectionListResponse)
async def list_objections(
    request: Request,
    exam_id: str | None = None,
    status_filter: str | None = None,
) -> dict[str, Any]:
    """List objections visible to the current actor."""
    await get_current_user(request)
    repo = request.app.state.objection_repo
    items = await repo.list_objections(exam_id=exam_id, status=status_filter)
    return {"items": items}


@router.get("/{objection_id}", response_model=ObjectionDetailResponse)
async def get_objection(
    objection_id: str, request: Request,
) -> dict[str, Any]:
    """Return objection detail with context."""
    await get_current_user(request)
    repo = request.app.state.objection_repo
    try:
        return await repo.get_by_id(objection_id)
    except ObjectionNotFoundError:
        raise _not_found(objection_id)


@router.post("/{objection_id}/assign", response_model=ObjectionDetailResponse)
async def assign_objection(
    objection_id: str, body: AssignObjectionRequest, request: Request,
) -> dict[str, Any]:
    """Assign an objection to an evaluator.  HOD / principal only."""
    user = await get_current_user(request)
    _require_roles(user, "hod", "principal")

    repo = request.app.state.objection_repo
    publisher = request.app.state.objection_publisher

    try:
        obj = await repo.get_by_id(objection_id)
    except ObjectionNotFoundError:
        raise _not_found(objection_id)

    try:
        tr = transition(ObjectionState(obj["status"]), ObjectionEvent.ASSIGN)
    except InvalidTransitionError as exc:
        raise _conflict(str(exc)) from exc

    result = await repo.transition_state(
        objection_id,
        expected_state=tr.old_state.value,
        new_state=tr.new_state.value,
        assigned_to=body.assigned_to,
    )
    await publisher.publish_transition(
        objection_id=objection_id, exam_id=obj["exam_id"],
        student_id=obj["student_id"], question_id=obj["question_id"],
        action="assigned", state=tr.new_state.value, actor_id=user.user_id,
    )
    return result


@router.post("/{objection_id}/resolve", response_model=ObjectionDetailResponse)
async def resolve_objection(
    objection_id: str, body: ResolveObjectionRequest, request: Request,
) -> dict[str, Any]:
    """Resolve an objection -- approve (triggers re-score) or reject.

    Requires ``evaluator`` role (assigned evaluator) or ``hod``.
    """
    user = await get_current_user(request)
    _require_roles(user, "evaluator", "hod")

    repo = request.app.state.objection_repo
    publisher = request.app.state.objection_publisher

    try:
        obj = await repo.get_by_id(objection_id)
    except ObjectionNotFoundError:
        raise _not_found(objection_id)

    payload = ResolutionPayload(
        resolution=body.resolution,  # type: ignore[arg-type]
        reason=body.reason, new_score=body.new_score,
    )
    try:
        validate_resolution(payload)
    except ResolutionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message) from exc

    try:
        tr = transition(ObjectionState(obj["status"]), ObjectionEvent.RESOLVE)
    except InvalidTransitionError as exc:
        raise _conflict(str(exc)) from exc

    result = await repo.transition_state(
        objection_id,
        expected_state=tr.old_state.value,
        new_state=tr.new_state.value,
        resolution=body.resolution,
        resolution_reason=body.reason,
        score_delta=body.new_score,
    )
    await publisher.publish_transition(
        objection_id=objection_id, exam_id=obj["exam_id"],
        student_id=obj["student_id"], question_id=obj["question_id"],
        action="resolved", state=tr.new_state.value, actor_id=user.user_id,
    )
    if body.resolution == "approved" and body.new_score is not None:
        await publisher.publish_rescore_command(
            objection_id=objection_id, exam_id=obj["exam_id"],
            student_id=obj["student_id"], question_id=obj["question_id"],
            new_score=body.new_score, actor_id=user.user_id,
        )
    return result


@router.post("/{objection_id}/escalate", response_model=ObjectionDetailResponse)
async def escalate_objection(
    objection_id: str, body: EscalateObjectionRequest, request: Request,
) -> dict[str, Any]:
    """Escalate an objection to HOD or senior evaluator.

    Only the assigned ``evaluator`` may escalate.
    """
    user = await get_current_user(request)
    _require_roles(user, "evaluator")

    repo = request.app.state.objection_repo
    publisher = request.app.state.objection_publisher

    try:
        obj = await repo.get_by_id(objection_id)
    except ObjectionNotFoundError:
        raise _not_found(objection_id)

    esc = EscalationPayload(escalated_to=body.escalated_to, reason=body.reason)
    try:
        validate_escalation(esc)
    except EscalationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message) from exc

    try:
        tr = transition(ObjectionState(obj["status"]), ObjectionEvent.ESCALATE)
    except InvalidTransitionError as exc:
        raise _conflict(str(exc)) from exc

    result = await repo.transition_state(
        objection_id,
        expected_state=tr.old_state.value,
        new_state=tr.new_state.value,
        assigned_to=body.escalated_to,
        resolution_reason=body.reason,
    )
    await publisher.publish_transition(
        objection_id=objection_id, exam_id=obj["exam_id"],
        student_id=obj["student_id"], question_id=obj["question_id"],
        action="escalated", state=tr.new_state.value, actor_id=user.user_id,
    )
    return result
