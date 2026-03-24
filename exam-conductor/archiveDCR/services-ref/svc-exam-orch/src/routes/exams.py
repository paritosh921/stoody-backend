"""Exam CRUD and lifecycle transition endpoints.

Matches ``api/exam-orch.openapi.yaml`` paths:
    POST   /api/v1/exams
    GET    /api/v1/exams
    GET    /api/v1/exams/{exam_id}
    PATCH  /api/v1/exams/{exam_id}
    POST   /api/v1/exams/{exam_id}/transitions
    GET    /api/v1/exams/{exam_id}/roster
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.db import rls_session

from src.domain.exam_fsm import InvalidTransition, transition
from src.domain.rbac import require_minimum_role, require_role, require_transition_role
from src.events.lifecycle_publisher import publish_lifecycle_event
from src.storage.assignment_repo import AssignmentRepo
from src.storage.exam_repo import ExamRepo

router = APIRouter(tags=["exams"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


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


class LifecycleTransitionBody(BaseModel):
    to_state: str
    actor_id: str
    reason: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_exam(
    body: CreateExamBody,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new exam definition."""
    require_role(user, "super_admin", "principal", "hod", "evaluator")
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        data = body.model_dump()
        data["tenant_id"] = user.tenant_id
        data["created_by"] = user.user_id
        return await repo.create(data)
    raise HTTPException(status_code=500, detail="session error")  # pragma: no cover


@router.get("")
async def list_exams(
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
    state: str | None = Query(None),
    subject_id: str | None = Query(None),
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
) -> dict[str, Any]:
    """List exams visible to the current actor."""
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        items = await repo.list_exams(
            state=state,
            subject_id=subject_id,
            from_date=from_date,
            to_date=to_date,
        )
        return {"items": items}
    raise HTTPException(status_code=500, detail="session error")  # pragma: no cover


@router.get("/{exam_id}")
async def get_exam(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Return detailed exam configuration."""
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        exam = await repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(status_code=404, detail="Exam not found")
        return exam
    raise HTTPException(status_code=500, detail="session error")  # pragma: no cover


@router.patch("/{exam_id}")
async def patch_exam(
    exam_id: str,
    body: PatchExamBody,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Update mutable exam fields (only allowed in ``created`` state)."""
    require_role(user, "super_admin", "principal", "hod", "evaluator")
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        data = body.model_dump(exclude_none=True)
        updated = await repo.update(exam_id, data)
        if updated is None:
            raise HTTPException(
                status_code=409,
                detail="Exam not found or not in 'created' state",
            )
        return updated
    raise HTTPException(status_code=500, detail="session error")  # pragma: no cover


@router.post("/{exam_id}/transitions")
async def apply_transition(
    exam_id: str,
    body: LifecycleTransitionBody,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Apply an FSM transition with row-level locking.

    Publishes a NATS ``exam.lifecycle`` event AFTER the DB commit.
    """
    sf = request.app.state.session_factory

    # Phase 1: validate + commit inside DB session
    from_state: str | None = None
    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        exam = await repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(status_code=404, detail="Exam not found")

        # RBAC: load assigned invigilators for this exam
        assign_repo = AssignmentRepo(session)
        assignments = await assign_repo.list_by_exam(exam_id)
        invig_ids = frozenset(
            a["user_id"] for a in assignments if a["role"] == "invigilator"
        )
        require_transition_role(user, body.to_state, invig_ids)

        current = exam["state"]
        try:
            result = transition(current, body.to_state)
        except InvalidTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        updated = await repo.transition_state(
            exam_id, result.from_state.value, result.to_state.value,
        )
        if updated is None:
            raise HTTPException(
                status_code=409, detail="Concurrent state change detected",
            )
        from_state = result.from_state.value
        to_state = result.to_state.value
        changed_at = updated["updated_at"]

    # Phase 2: publish NATS event AFTER commit
    nats = getattr(request.app.state, "nats_client", None)
    if nats is not None and from_state is not None:
        await publish_lifecycle_event(
            nats,
            exam_id=exam_id,
            from_state=from_state,
            to_state=to_state,
            actor_id=body.actor_id,
            reason=body.reason,
        )

    return {
        "exam_id": exam_id,
        "from_state": from_state,
        "to_state": to_state,
        "changed_at": changed_at,
    }


@router.get("/{exam_id}/roster")
async def get_roster(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the Stoody-backed student roster for the exam."""
    sf = request.app.state.session_factory

    async for session in rls_session(sf, user.tenant_id):
        repo = ExamRepo(session)
        exam = await repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(status_code=404, detail="Exam not found")

    stoody = request.app.state.stoody_client
    students = await stoody.get_students(exam["class_id"], exam["section_id"])

    return {
        "exam_id": exam_id,
        "students": [
            {
                "student_id": s.get("student_id", s.get("_id", "")),
                "name": s.get("name", ""),
                "roll": s.get("roll", ""),
                "section_id": s.get("section_id", exam.get("section_id", "")),
            }
            for s in students
        ],
    }
