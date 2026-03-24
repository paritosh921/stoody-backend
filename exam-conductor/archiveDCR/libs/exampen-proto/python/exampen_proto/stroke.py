"""Stroke domain models: raw chunks, processed strokes, upload status."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import ChunkBindingStatus, UploadPath


class StrokeChunkUploadRequest(BaseModel):
    """A single chunk of raw stroke data uploaded from the hub."""

    exam_id: UUID
    pen_mac: str
    chunk_index: int
    total_chunks: int
    payload_base64: str
    checksum_crc32: str
    upload_path: UploadPath
    idempotency_key: str
    binding_status: Optional[ChunkBindingStatus] = None


class IngestAck(BaseModel):
    """Server acknowledgement for a received stroke chunk."""

    exam_id: UUID
    pen_mac: str
    chunk_index: int
    accepted: bool
    deduplicated: bool
    next_expected_chunk: int
    pen_upload_complete: Optional[bool] = None


class PenUploadStatus(BaseModel):
    """Per-pen upload reconciliation state."""

    pen_mac: str
    acked_chunks: list[int]
    total_chunks: int
    complete: bool


class ExamUploadStatus(BaseModel):
    """Upload progress for all pens in an exam."""

    exam_id: UUID
    pens: list[PenUploadStatus]


class PageAssignment(BaseModel):
    """Mapping of a stroke segment to a question on a page."""

    page_number: int
    question_id: str
    point_count: int


class StrokeRawEvent(BaseModel):
    """NATS event: raw stroke chunk ingested from hub."""

    event_id: str
    event_type: str = "stroke.raw"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    pen_mac: str
    chunk_index: int
    total_chunks: int
    payload_base64: str
    checksum_crc32: str
    upload_path: UploadPath
    binding_status: Optional[ChunkBindingStatus] = None


class StrokeProcessedEvent(BaseModel):
    """NATS event: stroke data normalized, deduplicated, and committed."""

    event_id: str
    event_type: str = "stroke.processed"
    event_version: str = "1.0.0"
    occurred_at: datetime
    exam_id: UUID
    pen_mac: str
    student_id: Optional[str] = None
    normalized_stroke_uri: str
    page_assignments: list[PageAssignment]
