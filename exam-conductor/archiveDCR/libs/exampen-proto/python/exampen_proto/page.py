"""Page and copy-upload models: page images, miss indicators, copy pages."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import AuthoritativeSource, CopyAuthoritativeCandidate, MissIndicatorState


class PageReadyEvent(BaseModel):
    """NATS event: rendered page image ready for AI pipeline."""

    event_id: str
    event_type: str = "page.ready"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    student_id: str
    page_id: str
    page_number: int
    image_uri: str
    vector_uri: Optional[str] = None
    authoritative_source: AuthoritativeSource
    question_ids: Optional[list[str]] = None


class CopyReadyEvent(BaseModel):
    """NATS event: photographed copy page ingested."""

    event_id: str
    event_type: str = "copy.ready"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    student_id: str
    page_number: int
    copy_image_uri: str
    authoritative_candidate: Optional[CopyAuthoritativeCandidate] = None


class CopyUploadResult(BaseModel):
    """Result of uploading a photographed answer page."""

    exam_id: UUID
    student_id: str
    page_number: int
    copy_image_uri: str
    data_source: str = "copy_image"


class CopyPage(BaseModel):
    """A single copy page record."""

    page_number: int
    copy_image_uri: str
    authoritative_source: Optional[AuthoritativeSource] = None


class MissIndicatorCell(BaseModel):
    """Single cell in the miss indicator matrix."""

    student_id: str
    question_id: str
    state: MissIndicatorState


class MissIndicatorMatrix(BaseModel):
    """Student-by-question miss indicator matrix for a teacher view."""

    exam_id: UUID
    students: list[str]
    questions: list[str]
    cells: list[MissIndicatorCell]
