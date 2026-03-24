"""Exam domain models: exam definitions, lifecycle, variants, rubric, question regions."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import (
    BindingSource,
    BindingStatus,
    ExamState,
)


class ExamSummary(BaseModel):
    """Lightweight exam record for list views."""

    exam_id: UUID
    subject_id: str
    class_id: str
    title: Optional[str] = None
    scheduled_at: datetime
    duration_min: Optional[int] = None
    state: ExamState


class ExamDetail(ExamSummary):
    """Full exam configuration including rubric metadata."""

    section_id: str
    total_marks: float
    question_count: int
    late_entry_cutoff_min: Optional[int] = None
    objection_window_days: Optional[int] = None
    variants: Optional[list[str]] = None
    created_by: str


class CreateExamRequest(BaseModel):
    """Request body for creating a new exam."""

    title: str
    subject_id: str
    class_id: str
    section_id: str
    scheduled_at: datetime
    duration_min: int
    question_count: int
    total_marks: float
    negative_marking: Optional[bool] = None
    variants: Optional[list[str]] = None


class PatchExamRequest(BaseModel):
    """Partial update for mutable exam fields (before lock)."""

    scheduled_at: Optional[datetime] = None
    duration_min: Optional[int] = None
    objection_window_days: Optional[int] = None
    late_entry_cutoff_min: Optional[int] = None


class LifecycleTransitionRequest(BaseModel):
    """Request to transition an exam to a new lifecycle state."""

    to_state: ExamState
    actor_id: str
    reason: Optional[str] = None


class LifecycleTransitionResult(BaseModel):
    """Result of applying a lifecycle state transition."""

    exam_id: UUID
    from_state: str
    to_state: str
    changed_at: datetime


class AssignmentRequest(BaseModel):
    """Assign invigilators and evaluators to an exam."""

    invigilator_ids: list[str]
    evaluator_ids: list[str]
    double_blind: Optional[bool] = None


class StudentRef(BaseModel):
    """Reference to a student from the Stoody roster."""

    student_id: str
    name: str
    roll: Optional[str] = None
    section_id: Optional[str] = None


class CreateBindingRequest(BaseModel):
    """Request to bind a pen to a student for an exam."""

    pen_mac: str
    student_id: str
    source: BindingSource
    request_id: Optional[str] = None


class ConfirmBindingRequest(BaseModel):
    """Confirm or reject a provisional pen-student binding."""

    status: BindingStatus
    rejection_reason: Optional[str] = None


class BindingRecord(BaseModel):
    """Server-side pen-student binding record."""

    exam_id: UUID
    pen_mac: str
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    student_roll: Optional[str] = None
    status: BindingStatus
    source: BindingSource
    bound_at: datetime
    server_confirmed_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
