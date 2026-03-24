"""Pen-student binding endpoints.

Matches ``api/exam-orch.openapi.yaml`` paths:
    POST  /api/v1/exams/{exam_id}/bindings
    GET   /api/v1/exams/{exam_id}/bindings
    POST  /api/v1/exams/{exam_id}/bindings/{pen_mac}/confirm
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.db import rls_session

from src.domain.binding_logic import (
    BindingValidationError,
    ExistingBinding,
    validate_binding_confirmation,
    validate_new_binding,
)
from src.domain.rbac import has_any_role
from src.storage.assignment_repo import AssignmentRepo
from src.storage.binding_repo import BindingRepo
from src.storage.exam_repo import ExamRepo

router = APIRouter(tags=["bindings"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CreateBindingBody(BaseModel):
    pen_mac: str
    student_id: str
    source: str  # registration_scan | manual_register | server_sync
    request_id: str | None = None


class ConfirmBindingBody(BaseModel):
    status: str  # confirmed | rejected
    rejection_reason: str | None = None


# ---------------------------------------------------------------------------
# RBAC helpers
# ---------------------------------------------------------------------------


def _require_assigned_invigilator(
    user: ExamPenUser,
    assigned_invigilator_ids: frozenset[str],
) -> None:
    """Raise 403 unless user is an invigilator assigned to the exam."""
    # Super admin / principal / hod can always manage bindings
    if has_any_role(user, frozenset({"super_admin", "principal", "hod"})):
        return
    if (
        has_any_role(user, frozenset({"invigilator"}))
        and user.user_id in assigned_invigilator_ids
    ):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Forbidden: only assigned invigilators may manage bindings",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_binding(
    exam_id: str,
    body: CreateBindingBody,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a provisional pen-student binding."""
    sf = request.app.state.session_factory
    stoody = request.app.state.stoody_client

    async for session in rls_session(sf, user.tenant_id):
        # Verify exam exists
        exam_repo = ExamRepo(session)
        exam = await exam_repo.get_by_id(exam_id)
        if exam is None:
            raise HTTPException(status_code=404, detail="Exam not found")

        # RBAC: only assigned invigilators (or above) may create bindings
        assign_repo = AssignmentRepo(session)
        assignments = await assign_repo.list_by_exam(exam_id)
        invig_ids = frozenset(
            a["user_id"] for a in assignments if a["role"] == "invigilator"
        )
        _require_assigned_invigilator(user, invig_ids)

        # Fetch roster for validation
        students = await stoody.get_students(
            exam["class_id"], exam["section_id"],
        )
        roster_ids = frozenset(
            s.get("student_id", s.get("_id", "")) for s in students
        )

        # Load existing bindings
        binding_repo = BindingRepo(session)
        existing_rows = await binding_repo.list_by_exam(exam_id)
        existing = [
            ExistingBinding(
                pen_mac=b["pen_mac"],
                student_id=b["student_id"],
                status=b["status"],
            )
            for b in existing_rows
        ]

        # Domain validation (ZERO I/O)
        try:
            validate_new_binding(
                pen_mac=body.pen_mac,
                student_id=body.student_id,
                roster_student_ids=roster_ids,
                existing_bindings=existing,
            )
        except BindingValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

        # Find student metadata
        student_meta = next(
            (
                s for s in students
                if s.get("student_id", s.get("_id", "")) == body.student_id
            ),
            {},
        )

        record = await binding_repo.create({
            "exam_id": exam_id,
            "tenant_id": user.tenant_id,
            "pen_mac": body.pen_mac,
            "student_id": body.student_id,
            "student_name": student_meta.get("name"),
            "student_roll": student_meta.get("roll"),
            "source": body.source,
        })
        return record

    raise HTTPException(status_code=500, detail="session error")  # pragma: no cover


@router.get("")
async def list_bindings(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """List all pen bindings for the exam."""
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        repo = BindingRepo(session)
        items = await repo.list_by_exam(exam_id)
        return {"exam_id": exam_id, "items": items}
    raise HTTPException(status_code=500, detail="session error")  # pragma: no cover


@router.post("/{pen_mac}/confirm")
async def confirm_binding(
    exam_id: str,
    pen_mac: str,
    body: ConfirmBindingBody,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Confirm or reject a provisional binding."""
    sf = request.app.state.session_factory
    async for session in rls_session(sf, user.tenant_id):
        # RBAC: only assigned invigilators (or above) may confirm
        assign_repo = AssignmentRepo(session)
        assignments = await assign_repo.list_by_exam(exam_id)
        invig_ids = frozenset(
            a["user_id"] for a in assignments if a["role"] == "invigilator"
        )
        _require_assigned_invigilator(user, invig_ids)

        repo = BindingRepo(session)
        current = await repo.get_by_pen(exam_id, pen_mac)
        if current is None:
            raise HTTPException(status_code=404, detail="Binding not found")

        try:
            validate_binding_confirmation(current["status"], body.status)
        except BindingValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

        updated = await repo.confirm_or_reject(
            exam_id, pen_mac, body.status, body.rejection_reason,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Binding not found")
        return updated

    raise HTTPException(status_code=500, detail="session error")  # pragma: no cover
