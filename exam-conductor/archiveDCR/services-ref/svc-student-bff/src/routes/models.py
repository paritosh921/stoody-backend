"""Pydantic request/response models — matches student-bff.openapi.yaml.

These are thin DTOs for HTTP serialization. No business logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# -- Score models ----------------------------------------------------------


class QuestionScore(BaseModel):
    """Per-question score in a breakdown."""

    question_id: str
    marks_obtained: float
    max_marks: float
    ai_confidence: float | None = None
    miss_indicator: str | None = None


class StudentScoreView(BaseModel):
    """Score summary for one exam."""

    exam_id: str
    total_score: float
    percentage: float
    percentile: float
    pass_fail: str | None = None
    questions: list[QuestionScore] = Field(default_factory=list)


class AnswerInsight(BaseModel):
    """Answer image + AI analysis for one question."""

    question_id: str
    answer_image_uri: str
    recognized_text: str
    confidence: float
    step_breakdown: list[str] = Field(default_factory=list)
    feedback: str | None = None


# -- Objection models ------------------------------------------------------


class CreateObjectionRequest(BaseModel):
    """POST /student/exams/{exam_id}/objections request body."""

    exam_id: str
    question_id: str
    objection_text: str = Field(..., min_length=5, max_length=2000)


class StudentObjection(BaseModel):
    """Objection visible to the student."""

    objection_id: str
    exam_id: str
    question_id: str
    status: str
    objection_text: str | None = None
    resolution_reason: str | None = None
    new_score: float | None = None


class ObjectionListResponse(BaseModel):
    """Wrapped list of objections."""

    items: list[StudentObjection] = Field(default_factory=list)


# -- Performance models ----------------------------------------------------


class HistoryEntry(BaseModel):
    """One exam's score in the history."""

    exam_id: str
    score: float
    percentile: float


class PerformanceView(BaseModel):
    """Historical performance aggregate."""

    history: list[HistoryEntry] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class TrendData(BaseModel):
    """Trend data for charts."""

    history: list[HistoryEntry] = Field(default_factory=list)


class StrengthsView(BaseModel):
    """AI-generated strength/weakness summary."""

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


# -- Chat models -----------------------------------------------------------


class MessageResponse(BaseModel):
    """A single chat message."""

    message_id: str
    sender_id: str
    content: str
    attachment_uri: str | None = None
    sent_at: str
    read_at: str | None = None


class MessageListResponse(BaseModel):
    """Wrapped list of messages."""

    items: list[MessageResponse] = Field(default_factory=list)


class SendMessageRequest(BaseModel):
    """POST /student/exams/{exam_id}/chat/{teacher_id} request body."""

    content: str = Field(..., min_length=1, max_length=5000)
    attachment_uri: str | None = None


# -- Error model -----------------------------------------------------------


class ErrorBody(BaseModel):
    """Standard error response."""

    code: str
    message: str
