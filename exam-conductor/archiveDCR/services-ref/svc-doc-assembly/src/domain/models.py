"""Domain models for doc-assembly. ZERO I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MissAutoState(str, Enum):
    """Auto-detected miss indicator states.

    Computed by svc-doc-assembly; never manually edited.
    """

    ANSWERED = "answered"
    MISS_NO_STROKES = "miss_no_strokes"
    MISS_SYNC_FAILURE = "miss_sync_failure"
    MISS_PEN_INACTIVE = "miss_pen_inactive"


class MissOverrideState(str, Enum):
    """Teacher-set override states.

    Written by svc-score-engine on teacher action.
    Display logic: show override_state if non-NULL, else auto_state.
    """

    NOT_ATTEMPTED_CONFIRMED = "not_attempted_confirmed"
    ANSWERED_CONFIRMED = "answered_confirmed"


@dataclass(frozen=True, slots=True)
class CanonicalPoint:
    """A single point in the canonical stroke model.

    Coordinates are in page-space millimetres.
    6-element point: x, y, pressure, tilt_x, tilt_y, timestamp_ms.
    """

    x: float
    y: float
    pressure: float
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    timestamp_ms: int = 0


@dataclass(frozen=True, slots=True)
class Stroke:
    """A single stroke with canonical points and rendering metadata."""

    stroke_id: str
    points: list[CanonicalPoint]
    color: str = "#000000"
    base_width: float = 0.4  # mm


@dataclass(frozen=True, slots=True)
class QuestionRegion:
    """Rectangular region on a page associated with a question."""

    question_id: str
    x_min: float  # mm
    y_min: float  # mm
    x_max: float  # mm
    y_max: float  # mm


@dataclass(frozen=True, slots=True)
class SyncMetadata:
    """Per-pen sync status for miss indicator detection."""

    pen_mac: str
    sync_complete: bool
    pen_connected: bool
    strokes_expected: bool


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Per-question miss indicator outcome."""

    question_id: str
    auto_state: MissAutoState
    override_state: Optional[MissOverrideState] = None

    @property
    def display_state(self) -> str:
        """Return override if set, else auto_state."""
        if self.override_state is not None:
            return self.override_state.value
        return self.auto_state.value


@dataclass(frozen=True, slots=True)
class PageDocument:
    """Assembled page: SVG content + question results + metadata."""

    exam_id: str
    student_id: str
    page_number: int
    svg_content: str
    question_results: list[QuestionResult] = field(default_factory=list)
    page_width_mm: float = 210.0  # A4 default
    page_height_mm: float = 297.0
