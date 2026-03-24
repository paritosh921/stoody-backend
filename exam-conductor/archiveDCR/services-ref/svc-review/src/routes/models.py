"""Pydantic request/response models for objection routes.

Aligned with review.openapi.yaml.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreateObjectionRequest(BaseModel):
    """POST /objections request body.

    ``student_id`` is derived from the authenticated user — never from
    the request body — to prevent identity spoofing.
    """

    exam_id: str
    question_id: str
    objection_text: str = Field(..., min_length=10)


class AssignObjectionRequest(BaseModel):
    """POST /objections/{id}/assign request body.

    ``actor_id`` is derived from the authenticated user.
    """

    assigned_to: str


class ResolveObjectionRequest(BaseModel):
    """POST /objections/{id}/resolve request body.

    ``actor_id`` is derived from the authenticated user.
    """

    resolution: str  # "approved" | "rejected"
    reason: str = Field(..., min_length=5)
    new_score: float | None = None


class EscalateObjectionRequest(BaseModel):
    """POST /objections/{id}/escalate request body.

    ``actor_id`` is derived from the authenticated user.
    """

    escalated_to: str
    reason: str


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ObjectionDetailResponse(BaseModel):
    """Full objection detail."""

    objection_id: str
    exam_id: str
    student_id: str
    question_id: str
    objection_text: str | None = None
    status: str
    filed_at: str
    assigned_to: str | None = None
    resolution: str | None = None
    resolution_reason: str | None = None
    score_delta: float | None = None


class ObjectionSummaryResponse(BaseModel):
    """Compact objection list item."""

    objection_id: str
    exam_id: str
    student_id: str
    question_id: str
    status: str
    filed_at: str


class ObjectionListResponse(BaseModel):
    """Paginated list wrapper."""

    items: list[ObjectionSummaryResponse]


class ErrorBody(BaseModel):
    """Standard error body."""

    code: str
    message: str
