from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..models.raw_frames import RawFrameCanonical, Stroke, StrokePoint

# Book type codes from BLE smart pen
# SIZING KEY:
#   S* (0x00-0x04): Small books - A5 equivalent (592 x 840)
#   M* (0x05-0x09): Medium books - A5 equivalent (592 x 840)
#   L* (0x0A-0x0E): Large books - A4 equivalent (842 x 1190)
# Second letter indicates ruling: S=Square, N=Normal, M=Mixed, L=Lines, W=Wide
BOOK_TYPES = {
    0x00: "SS",
    0x01: "SN",
    0x02: "SM",
    0x03: "SL",
    0x04: "SW",
    0x05: "MS",
    0x06: "MN",
    0x07: "MM",
    0x08: "ML",
    0x09: "MW",
    0x0A: "LS",  # A4 size
    0x0B: "LN",  # A4 size
    0x0C: "LM",  # A4 size
    0x0D: "LL",  # A4 size
    0x0E: "LW",  # A4 size
}

SYNC_MARKER = b"\x5a\x5b"
COORD_STRUCT = struct.Struct("<BBHHHBBL")


@dataclass
class _StrokeBuffer:
    points: List[StrokePoint]
    page_no: Optional[int]
    book_type: Optional[str]
    start_time: float

    def add_point(
        self, x: int, y: int, pressure: int, timestamp_val: int, book_type: str, page_no: Optional[int]
    ) -> None:
        self.points.append(
            StrokePoint(x=x, y=y, pressure=pressure, timestamp=timestamp_val)
        )
        if self.page_no is None and page_no is not None:
            self.page_no = page_no
        if self.book_type is None:
            self.book_type = book_type


@dataclass
class StrokeEngineState:
    current: Optional[_StrokeBuffer]
    current_page: Optional[int]
    current_book_type: Optional[str]


class StrokeEngineAdapter:
    """Wrapper around the legacy coordinate parser + stroke tracker."""

    def create_state(self, _: str) -> StrokeEngineState:
        return StrokeEngineState(current=None, current_page=None, current_book_type=None)

    def process_frame(
        self, state: StrokeEngineState, frame: RawFrameCanonical
    ) -> Tuple[List[Stroke], StrokeEngineState]:
        coordinates = self._parse_coordinates(frame.payload)
        completed: List[Stroke] = []

        for coord in coordinates:
            page_no = (
                coord.page_no if coord.page_no is not None and coord.page_no <= 511 else None
            )
            book_type_name = coord.book_type_name

            if page_no is not None and page_no != state.current_page:
                state.current_page = page_no
            if book_type_name is not None:
                state.current_book_type = book_type_name

            if coord.pen_state == "down":
                if state.current is None:
                    state.current = _StrokeBuffer(
                        points=[],
                        page_no=page_no,
                        book_type=book_type_name,
                        start_time=time.time(),
                    )
                state.current.add_point(
                    coord.x, coord.y, coord.pressure, coord.timestamp, book_type_name, page_no
                )
            elif coord.pen_state == "up":
                if state.current and coord.pressure > 0:
                    state.current.add_point(
                        coord.x, coord.y, coord.pressure, coord.timestamp, book_type_name, page_no
                    )
                if state.current and state.current.points:
                    completed.append(
                        Stroke(
                            pen_id=frame.pen_id,
                            session_id=frame.session_id,
                            page_no=state.current.page_no,
                            book_type=state.current.book_type,
                            start_ts=state.current.start_time,
                            end_ts=time.time(),
                            points=state.current.points.copy(),
                        )
                    )
                state.current = None

        return completed, state

    def _parse_coordinates(self, payload: bytes) -> List["Coordinate"]:
        frames: List[Coordinate] = []

        if len(payload) == COORD_STRUCT.size:
            frames.append(Coordinate.from_bytes(payload))
            return frames

        idx = 0
        while idx < len(payload) - 1:
            if payload[idx : idx + 2] == SYNC_MARKER:
                coord_start = idx + 2 + 12  # skip 12 bytes of header between sync + data
                coord_end = coord_start + COORD_STRUCT.size

                if coord_end <= len(payload):
                    coord_bytes = payload[coord_start:coord_end]
                    frames.append(Coordinate.from_bytes(coord_bytes))
                    idx = coord_end
                    continue

            idx += 1

        return frames


class Coordinate:
    """Represents a parsed 14-byte coordinate frame."""

    __slots__ = (
        "book_type",
        "book_seq",
        "page_no",
        "x",
        "y",
        "pressure",
        "pen_prop",
        "timestamp",
    )

    def __init__(
        self,
        book_type: int,
        book_seq: int,
        page_no: int,
        x: int,
        y: int,
        pressure: int,
        pen_prop: int,
        timestamp: int,
    ) -> None:
        self.book_type = book_type
        self.book_seq = book_seq
        self.page_no = page_no
        self.x = x
        self.y = y
        self.pressure = pressure
        self.pen_prop = pen_prop
        self.timestamp = timestamp

    @property
    def pen_state(self) -> str:
        if self.pen_prop == 2 or self.pressure == 0:
            return "up"
        return "down"

    @property
    def book_type_name(self) -> Optional[str]:
        return BOOK_TYPES.get(self.book_type)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Coordinate":
        if len(data) != COORD_STRUCT.size:
            raise ValueError(f"Coordinate frame must be {COORD_STRUCT.size} bytes")

        unpacked = COORD_STRUCT.unpack(data)
        return cls(*unpacked)

    @property
    def dict(self) -> dict:
        return {
            "bookType": self.book_type_name,
            "pageNo": self.page_no,
            "x": self.x,
            "y": self.y,
            "pressure": self.pressure,
            "penState": self.pen_state,
            "timestamp": self.timestamp,
        }

