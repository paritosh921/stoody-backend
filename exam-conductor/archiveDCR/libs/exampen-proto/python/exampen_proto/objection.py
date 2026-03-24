"""Objection models: filing, resolution, escalation, events."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import ObjectionResolution, ObjectionStatus


class ObjectionSummary(BaseModel):
    """Lightweight objection record for list views."""

    objection_id: UUID
    exam_id: UUID
    student_id: str
    question_id: str
    status: ObjectionStatus
    filed_at: datetime


class ObjectionDetail(ObjectionSummary):
    """Full objection with text and resolution detail."""

    objection_text: str
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    resolution_reason: Optional[str] = None
    score_delta: Optional[float] = None


class CreateObjectionRequest(BaseModel):
    """Request to file a new objection."""

    exam_id: UUID
    student_id: str
    question_id: str
    objection_text: str


class ResolveObjectionRequest(BaseModel):
    """Request to resolve an objection."""

    actor_id: str
    resolution: ObjectionResolution
    reason: str
    new_score: Optional[float] = None


class EscalateObjectionRequest(BaseModel):
    """Request to escalate an objection to a senior reviewer."""

    actor_id: str
    escalated_to: str
    reason: str


class ObjectionEvent(BaseModel):
    """NATS event: objection lifecycle transition."""

    event_id: str
    event_type: str = "objection"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    objection_id: UUID
    student_id: str
    question_id: str
    action: ObjectionStatus
    state: ObjectionStatus
    actor_id: Optional[str] = None
