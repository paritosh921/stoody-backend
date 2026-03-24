"""AI pipeline models: recognition results, confidence, step breakdowns."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import AISourceType


class QuestionResult(BaseModel):
    """AI recognition result for a single question."""

    question_id: str
    recognized_text: str
    confidence: float
    step_breakdown: Optional[list[str]] = None


class AIResultEvent(BaseModel):
    """NATS event: AI recognition complete for a student's exam."""

    event_id: str
    event_type: str = "ai.result"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    student_id: str
    model_version: str
    source_type: Optional[AISourceType] = None
    question_results: list[QuestionResult]


class AnswerInsight(BaseModel):
    """Student-facing answer detail with AI analysis."""

    question_id: str
    answer_image_uri: str
    recognized_text: str
    confidence: float
    step_breakdown: Optional[list[str]] = None
    feedback: Optional[str] = None
