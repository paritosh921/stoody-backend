from __future__ import annotations

import time
import base64
import binascii
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


@dataclass(slots=True)
class StrokePoint:
    x: float
    y: float
    pressure: float = 0
    timestamp: float = 0


@dataclass(slots=True)
class Stroke:
    pen_id: str
    session_id: str
    page_no: Optional[int]
    book_type: Optional[str]
    start_ts: float
    end_ts: float
    points: List[StrokePoint] = field(default_factory=list)


@dataclass(slots=True)
class RawFrameCanonical:
    pen_id: str
    session_id: str
    seq: int
    payload: bytes
    ts_edge: float = field(default_factory=time.time)
    ts_ingress: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hub_id: Optional[str] = None


class RawFrameIn(BaseModel):
    hub_id: Optional[str] = None
    pen_mac: str
    session_id: str
    seq: int
    payload_hex: Optional[str] = None
    payload_b64: Optional[str] = None
    ts_edge: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("pen_mac")
    @classmethod
    def _normalize_pen_mac(cls, value: str) -> str:
        mac = (value or "").strip().upper()
        if not mac:
            raise ValueError("pen_mac is required")
        return mac

    @model_validator(mode="after")
    def _require_payload(self) -> "RawFrameIn":
        if not self.payload_hex and not self.payload_b64:
            raise ValueError("payload_hex or payload_b64 is required")
        return self

    def payload_bytes(self) -> bytes:
        if self.payload_hex:
            try:
                return bytes.fromhex(self.payload_hex)
            except ValueError as exc:
                raise ValueError("payload_hex is not valid hexadecimal") from exc
        try:
            return base64.b64decode(self.payload_b64 or "", validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("payload_b64 is not valid base64") from exc

    def to_canonical(self, pen_id: str) -> RawFrameCanonical:
        metadata = dict(self.metadata)
        metadata.setdefault("pen_mac", self.pen_mac)
        return RawFrameCanonical(
            pen_id=pen_id,
            session_id=self.session_id,
            seq=self.seq,
            payload=self.payload_bytes(),
            ts_edge=self.ts_edge,
            metadata=metadata,
            hub_id=self.hub_id,
        )


class RawFrameBatchIn(BaseModel):
    hub_id: Optional[str] = None
    batch_id: Optional[str] = None
    frames: List[RawFrameIn]
    frame_stats: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_batch_hub_id(self) -> "RawFrameBatchIn":
        if self.hub_id:
            for frame in self.frames:
                if frame.hub_id is None:
                    frame.hub_id = self.hub_id
        return self
