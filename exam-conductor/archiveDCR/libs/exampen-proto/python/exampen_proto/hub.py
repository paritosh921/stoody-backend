"""Hub models: dongle state, pen sync, session, WebSocket envelope."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import DongleStatus, PenSyncStatus, UploadStatus, WebSocketEventType


class SessionSummary(BaseModel):
    """Backend view of an exam session for the invigilator console."""

    exam_id: UUID
    state: str
    timer_remaining_sec: int
    upload_status: UploadStatus
    backend_seen_at: Optional[datetime] = None


class PenSyncRow(BaseModel):
    """Per-pen BLE sync progress row."""

    pen_mac: str
    student_id: Optional[str] = None
    sync_status: PenSyncStatus
    bytes_received: Optional[int] = None
    total_chunks: Optional[int] = None


class DongleRow(BaseModel):
    """BLE dongle health and capacity state."""

    dongle_mac: str
    status: DongleStatus
    connected_pens: int
    capacity: Optional[int] = None


class WebSocketEnvelope(BaseModel):
    """WebSocket message envelope for invigilator console updates."""

    event_type: WebSocketEventType
    payload: dict[str, Any]
