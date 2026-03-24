"""NATS event envelope models — re-exports from domain modules.

Each event model is defined in its domain module. This module provides a
single namespace for importing all event types plus the ExamLifecycleEvent
which only exists as an event (no REST counterpart).
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .ai import AIResultEvent as AIResultEvent
from .objection import ObjectionEvent as ObjectionEvent
from .page import CopyReadyEvent as CopyReadyEvent
from .page import PageReadyEvent as PageReadyEvent
from .plagiarism import PlagiarismCheckEvent as PlagiarismCheckEvent
from .plagiarism import PlagiarismResultEvent as PlagiarismResultEvent
from .score import ScoreUpdatedEvent as ScoreUpdatedEvent
from .stroke import StrokeProcessedEvent as StrokeProcessedEvent
from .stroke import StrokeRawEvent as StrokeRawEvent


class ExamLifecycleEvent(BaseModel):
    """NATS event: exam lifecycle state transition."""

    event_id: str
    event_type: str = "exam.lifecycle"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    from_state: str
    to_state: str
    actor_id: str
    reason: Optional[str] = None
