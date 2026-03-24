"""Score domain models: score projections, overrides, audit history, workflow."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import ScoreEventType, ScoreLifecycleState


class StepScore(BaseModel):
    """Score for a single step within a question."""

    label: str
    awarded: float
    max: float


class QuestionScore(BaseModel):
    """Score projection for one question."""

    question_id: str
    ai_score: float
    current_score: float
    max_score: float
    confidence: float
    override_reason: Optional[str] = None
    step_scores: Optional[list[StepScore]] = None


class StudentScoreDetail(BaseModel):
    """Full score projection for one student in an exam."""

    exam_id: UUID
    student_id: str
    total_score: float
    max_score: Optional[float] = None
    lifecycle_state: ScoreLifecycleState
    published_at: Optional[datetime] = None
    objection_window_closes_at: Optional[datetime] = None
    questions: list[QuestionScore]


class ScoreOverrideRequest(BaseModel):
    """Teacher override for a single question score."""

    teacher_id: str
    new_score: float
    reason: str


class ScoreHistoryItem(BaseModel):
    """Single entry in the score audit event stream."""

    event_id: str
    event_type: ScoreEventType
    old_value: float
    new_value: float
    actor_id: str
    reason: Optional[str] = None
    created_at: datetime


class FinalizeRequest(BaseModel):
    """Request to finalize reviewed scores for an exam."""

    actor_id: str


class PublishRequest(BaseModel):
    """Request to publish finalized scores and open objection window."""

    actor_id: str
    objection_window_days: int


class WorkflowStateResponse(BaseModel):
    """Result of a score workflow state transition."""

    exam_id: UUID
    lifecycle_state: ScoreLifecycleState
    changed_at: datetime


class ScoreUpdatedEvent(BaseModel):
    """NATS event: score projection changed."""

    event_id: str
    event_type: str = "score.updated"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    student_id: str
    question_id: Optional[str] = None
    lifecycle_state: ScoreLifecycleState
    total_score: float
    previous_total_score: Optional[float] = None
    reason: ScoreEventType
