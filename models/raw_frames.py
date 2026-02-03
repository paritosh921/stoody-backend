"""
Raw Frame Models for BLE Smart Pen Integration

This module defines the Pydantic models for raw frame data coming from
BLE smart pens via the Pi hubs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StrokePoint(BaseModel):
    """A single point in a stroke with coordinates and pressure."""
    x: int
    y: int
    pressure: int
    timestamp: int


class Stroke(BaseModel):
    """A completed stroke from the smart pen."""
    pen_id: str
    session_id: str
    page_no: Optional[int] = None
    book_type: Optional[str] = None
    start_ts: float
    end_ts: float
    points: List[StrokePoint] = Field(default_factory=list)


class RawFrameCanonical(BaseModel):
    """Canonical representation of a raw frame after pen ID resolution."""
    pen_id: str
    session_id: str
    hub_id: str
    seq: int
    payload: bytes
    ts_edge: float = Field(description="Timestamp from edge device (Pi hub)")
    ts_ingress: float = Field(description="Timestamp when frame was received by backend")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Allow bytes type
        arbitrary_types_allowed = True


class RawFrameIn(BaseModel):
    """Input model for a single raw frame from the Pi hub."""
    pen_mac: str = Field(description="MAC address of the BLE pen")
    hub_id: str = Field(description="Identifier of the Pi hub")
    session_id: str = Field(default="", description="Session identifier")
    seq: int = Field(description="Sequence number for ordering")
    payload: str = Field(description="Base64-encoded payload data")
    ts_edge: Optional[float] = Field(default=None, description="Timestamp from edge device")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_canonical(self, pen_id: str) -> RawFrameCanonical:
        """Convert to canonical form after pen ID resolution."""
        import base64
        
        return RawFrameCanonical(
            pen_id=pen_id,
            session_id=self.session_id,
            hub_id=self.hub_id,
            seq=self.seq,
            payload=base64.b64decode(self.payload) if self.payload else b"",
            ts_edge=self.ts_edge or time.time(),
            ts_ingress=time.time(),
            metadata={
                "pen_mac": self.pen_mac,
                **self.metadata
            }
        )


class RawFrameBatchIn(BaseModel):
    """Input model for a batch of raw frames."""
    batch_id: str = Field(description="Unique identifier for this batch")
    frames: List[RawFrameIn] = Field(default_factory=list)
