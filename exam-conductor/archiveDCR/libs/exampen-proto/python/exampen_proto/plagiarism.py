"""Plagiarism models: flags, evidence, verdicts, check/result events."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import PlagiarismCheckTrigger, PlagiarismSeverity, TeacherVerdict


class MatchingSegment(BaseModel):
    """Pair of matching text segments between two students."""

    student_a_text: str
    student_b_text: str


class Evidence(BaseModel):
    """Plagiarism detection evidence for a flag."""

    matching_segments: list[MatchingSegment]
    temporal_correlation_score: Optional[float] = None
    seating_proximity_score: Optional[float] = None


class FlagSummary(BaseModel):
    """Lightweight plagiarism flag for list views."""

    flag_id: UUID
    exam_id: UUID
    student_a_id: str
    student_b_id: str
    question_id: str
    composite_score: float
    severity: PlagiarismSeverity
    teacher_verdict: Optional[TeacherVerdict] = None


class FlagDetail(FlagSummary):
    """Full plagiarism flag with evidence and verdict detail."""

    evidence: Evidence
    verdict_reason: Optional[str] = None
    verdict_by: Optional[str] = None
    verdict_at: Optional[datetime] = None


class VerdictRequest(BaseModel):
    """Teacher verdict on a plagiarism flag."""

    teacher_id: str
    verdict: TeacherVerdict
    reason: str


class PlagiarismFlagEvent(BaseModel):
    """Single flag entry within a plagiarism.result event."""

    flag_id: UUID
    student_a_id: str
    student_b_id: str
    question_id: str
    composite_score: float
    severity: PlagiarismSeverity


class PlagiarismCheckEvent(BaseModel):
    """NATS event: plagiarism check requested."""

    event_id: str
    event_type: str = "plagiarism.check"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    student_count: int
    question_count: int
    trigger: PlagiarismCheckTrigger


class PlagiarismResultEvent(BaseModel):
    """NATS event: plagiarism check completed with flags."""

    event_id: str
    event_type: str = "plagiarism.result"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    flags: list[PlagiarismFlagEvent]
