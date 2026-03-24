"""Exam CRUD, lifecycle transitions, rubric, regions, bindings, and staff assignments.

Routes are mounted at ``/api/v1/exampen/exams``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from exampen.dcr.core.auth_bridge import (
    ExamPenUser, get_exampen_user, require_exampen_role,
)
from exampen.dcr.domain.exam_fsm import InvalidTransition, transition as fsm_transition
from exampen.dcr.domain.rbac import require_transition_role
from exampen.dcr.storage.exam_repo import ExamRepo
from exampen.dcr.storage.assignment_repo import AssignmentRepo
from exampen.dcr.storage.binding_repo import BindingRepo, BindingConflictError

logger = logging.getLogger(__name__)
router = APIRouter()


async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db


async def _publish(request: Request, subject: str, data: dict) -> None:
    nats = getattr(request.app.state, "exampen_nats", None)
    if nats is None or not nats.is_connected:
        return
    try:
        await nats.publish(subject, data)
    except Exception:
        logger.warning("NATS publish to %s failed (non-fatal)", subject)


# -- Schemas ----------------------------------------------------------------

class CreateExamBody(BaseModel):
    title: str
    subject_id: str
    class_id: str
    section_id: str
    scheduled_at: datetime
    duration_min: int
    question_count: int
    total_marks: float
    negative_marking: bool = False
    variants: list[str] = Field(default_factory=list)

class PatchExamBody(BaseModel):
    scheduled_at: datetime | None = None
    duration_min: int | None = None
    objection_window_days: int | None = None
    late_entry_cutoff_min: int | None = None

class LifecycleBody(BaseModel):
    to_state: str
    reason: str | None = None

class BindingBody(BaseModel):
    pen_mac: str
    student_id: str

class AssignStaffBody(BaseModel):
    invigilator_ids: list[str] = Field(default_factory=list)
    evaluator_ids: list[str] = Field(default_factory=list)


# -- Exam CRUD --------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_exam(
    body: CreateExamBody, request: Request,
    user: ExamPenUser = Depends(require_exampen_role("principal", "hod", "evaluator")),
) -> dict[str, Any]:
    """Create a new exam definition."""
    db = await _get_tenant_db(request, user)
    data = body.model_dump()
    data["created_by"] = user.user_id
    exam = await ExamRepo(db).create(user.tenant_id, data)
    await _publish(request, "exampen.exam.created", {"exam_id": exam["_id"], "tenant_id": user.tenant_id})
    return exam

@router.get("")
async def list_exams(
    request: Request, user: ExamPenUser = Depends(get_exampen_user),
    state: str | None = Query(None), subject_id: str | None = Query(None),
    skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """List exams visible to the current user."""
    db = await _get_tenant_db(request, user)
    filters: dict[str, Any] = {}
    if state:
        filters["state"] = state
    if subject_id:
        filters["subject_id"] = subject_id
    items = await ExamRepo(db).list_exams(user.tenant_id, filters=filters, limit=limit, skip=skip)
    return {"items": items}

@router.get("/{exam_id}")
async def get_exam(
    exam_id: str, request: Request, user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Return detailed exam configuration."""
    db = await _get_tenant_db(request, user)
    exam = await ExamRepo(db).get_by_id(exam_id, user.tenant_id)
    if exam is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exam not found")
    return exam

@router.patch("/{exam_id}")
async def patch_exam(
    exam_id: str, body: PatchExamBody, request: Request,
    user: ExamPenUser = Depends(require_exampen_role("principal", "hod", "evaluator")),
) -> dict[str, Any]:
    """Update mutable exam fields (only in ``created`` state)."""
    db = await _get_tenant_db(request, user)
    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No fields to update")
    updated = await ExamRepo(db).update(exam_id, user.tenant_id, data)
    if updated is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Exam not found or not in 'created' state")
    return updated


# -- FSM lifecycle transition -----------------------------------------------

@router.post("/{exam_id}/lifecycle")
async def apply_lifecycle(
    exam_id: str, body: LifecycleBody, request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Apply an FSM lifecycle transition with CAS check."""
    db = await _get_tenant_db(request, user)
    repo = ExamRepo(db)
    exam = await repo.get_by_id(exam_id, user.tenant_id)
    if exam is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exam not found")

    assign_repo = AssignmentRepo(db)
    assignments = await assign_repo.list_by_exam(exam_id, user.tenant_id)
    invig_ids = frozenset(a["user_id"] for a in assignments if "invigilator" in a.get("roles", []))
    require_transition_role(user, body.to_state, invig_ids)

    try:
        result = fsm_transition(exam["state"], body.to_state)
    except InvalidTransition as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    updated = await repo.transition_state(exam_id, user.tenant_id, result.from_state.value, result.to_state.value)
    if updated is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Concurrent state change detected")

    await _publish(request, "exampen.exam.lifecycle", {
        "exam_id": exam_id, "from_state": result.from_state.value,
        "to_state": result.to_state.value, "actor_id": user.user_id, "reason": body.reason,
    })
    return {
        "exam_id": exam_id, "from_state": result.from_state.value,
        "to_state": result.to_state.value, "changed_at": updated["updated_at"],
    }


# -- Rubric -----------------------------------------------------------------

@router.get("/{exam_id}/rubric")
async def get_rubric(
    exam_id: str, request: Request, user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Return the rubric definition for an exam."""
    db = await _get_tenant_db(request, user)
    exam = await ExamRepo(db).get_by_id(exam_id, user.tenant_id)
    if exam is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exam not found")
    return {"exam_id": exam_id, "rubric": exam.get("rubric")}

@router.put("/{exam_id}/rubric")
async def put_rubric(
    exam_id: str, rubric: dict[str, Any], request: Request,
    user: ExamPenUser = Depends(require_exampen_role("principal", "hod", "evaluator")),
) -> dict[str, str]:
    """Replace the rubric definition on an exam."""
    db = await _get_tenant_db(request, user)
    await ExamRepo(db).update_rubric(exam_id, user.tenant_id, rubric)
    return {"status": "ok"}


# -- Regions ----------------------------------------------------------------

@router.get("/{exam_id}/regions")
async def get_regions(
    exam_id: str, request: Request, user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Return the question regions for an exam."""
    db = await _get_tenant_db(request, user)
    exam = await ExamRepo(db).get_by_id(exam_id, user.tenant_id)
    if exam is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exam not found")
    return {"exam_id": exam_id, "regions": exam.get("regions", [])}

@router.put("/{exam_id}/regions")
async def put_regions(
    exam_id: str, regions: list[dict[str, Any]], request: Request,
    user: ExamPenUser = Depends(require_exampen_role("principal", "hod", "evaluator")),
) -> dict[str, str]:
    """Replace the question regions on an exam."""
    db = await _get_tenant_db(request, user)
    await ExamRepo(db).update_regions(exam_id, user.tenant_id, regions)
    return {"status": "ok"}


# -- Pen-student bindings ---------------------------------------------------

@router.post("/{exam_id}/bindings", status_code=status.HTTP_201_CREATED)
async def create_binding(
    exam_id: str, body: BindingBody, request: Request,
    user: ExamPenUser = Depends(require_exampen_role("principal", "hod", "invigilator")),
) -> dict[str, Any]:
    """Bind a pen to a student for an exam."""
    db = await _get_tenant_db(request, user)
    try:
        binding = await BindingRepo(db).create(
            exam_id=exam_id, pen_mac=body.pen_mac, tenant_id=user.tenant_id,
            data={"student_id": body.student_id, "created_by": user.user_id},
        )
    except BindingConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await _publish(request, "exampen.binding.created", {
        "exam_id": exam_id, "pen_mac": body.pen_mac,
        "student_id": body.student_id, "tenant_id": user.tenant_id,
    })
    return binding


# -- Staff assignments ------------------------------------------------------

@router.post("/{exam_id}/invigilators", status_code=status.HTTP_201_CREATED)
async def assign_staff(
    exam_id: str, body: AssignStaffBody, request: Request,
    user: ExamPenUser = Depends(require_exampen_role("principal", "hod")),
) -> dict[str, str]:
    """Assign invigilators and evaluators to an exam."""
    db = await _get_tenant_db(request, user)
    await AssignmentRepo(db).upsert(exam_id, user.tenant_id, body.invigilator_ids, body.evaluator_ids)
    await _publish(request, "exampen.staff.assigned", {"exam_id": exam_id, "tenant_id": user.tenant_id})
    return {"status": "ok"}
