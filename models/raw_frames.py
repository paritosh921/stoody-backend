from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
