"""
PCR Boundary Detector — Detect double-line answer delimiters.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md section 4.1
Failure mode:   PCR-01 (boundary detection failure -> flags + review queue)
Test ID:        U-SEG-01

BLE pen path: stroke geometry analysis
Camera path:  Canny + HoughLinesP

Boundary detection parameters from spec:
    slope:               within +/- 10 degrees of horizontal
    min line length:     > 40% of page width
    Y-gap between pair:  2-15 mm
    temporal proximity:  both lines drawn within ~3 seconds  (pen only)
    horizontal overlap:  > 70%
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .response_models import DetectedBoundary, PageOCR


# ---------------------------------------------------------------------------
# Configuration — exact values from spec section 4.1
# ---------------------------------------------------------------------------

SLOPE_THRESHOLD_DEG: float = 10.0
"""Maximum absolute slope deviation from horizontal (degrees)."""

MIN_LINE_LENGTH_RATIO: float = 0.4
"""Minimum line length as a fraction of page width."""

Y_GAP_MIN_MM: float = 2.0
"""Minimum vertical gap between paired delimiter lines (mm)."""

Y_GAP_MAX_MM: float = 15.0
"""Maximum vertical gap between paired delimiter lines (mm)."""

TEMPORAL_PROXIMITY_SEC: float = 3.0
"""Maximum time between the two strokes of a delimiter pair (pen path, seconds)."""

HORIZONTAL_OVERLAP_RATIO: float = 0.70
"""Minimum horizontal overlap between paired lines (fraction)."""


# ---------------------------------------------------------------------------
# Stroke / Line primitives
# ---------------------------------------------------------------------------


class StrokeLine(BaseModel):
    """A single candidate horizontal line extracted from strokes or an image."""

    x_start: float = Field(..., description="Start X in mm")
    y_start: float = Field(..., description="Start Y in mm")
    x_end: float = Field(..., description="End X in mm")
    y_end: float = Field(..., description="End Y in mm")
    timestamp: float | None = Field(
        None,
        description="Stroke timestamp in seconds (pen path only)",
    )
    page_number: int = Field(..., ge=1)

    @property
    def length(self) -> float:
        dx = self.x_end - self.x_start
        dy = self.y_end - self.y_start
        return math.sqrt(dx * dx + dy * dy)

    @property
    def slope_degrees(self) -> float:
        dx = self.x_end - self.x_start
        dy = self.y_end - self.y_start
        if dx == 0:
            return 90.0
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    @property
    def y_midpoint(self) -> float:
        return (self.y_start + self.y_end) / 2.0

    @property
    def x_min(self) -> float:
        return min(self.x_start, self.x_end)

    @property
    def x_max(self) -> float:
        return max(self.x_start, self.x_end)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_horizontal(line: StrokeLine) -> bool:
    """Check if a line is within the slope threshold of horizontal."""
    return line.slope_degrees <= SLOPE_THRESHOLD_DEG


def _is_long_enough(line: StrokeLine, page_width_mm: float) -> bool:
    """Check if a line meets the minimum length constraint."""
    return line.length > (MIN_LINE_LENGTH_RATIO * page_width_mm)


def _horizontal_overlap_ratio(a: StrokeLine, b: StrokeLine) -> float:
    """Compute the fraction of overlap in the X dimension relative to the
    shorter line."""
    overlap_start = max(a.x_min, b.x_min)
    overlap_end = min(a.x_max, b.x_max)
    overlap_length = max(0.0, overlap_end - overlap_start)
    shorter = min(a.length, b.length)
    if shorter <= 0:
        return 0.0
    return overlap_length / shorter


def _y_gap(a: StrokeLine, b: StrokeLine) -> float:
    """Absolute vertical gap between midpoints of two lines."""
    return abs(a.y_midpoint - b.y_midpoint)


def _temporal_gap(a: StrokeLine, b: StrokeLine) -> float | None:
    """Time gap between two strokes.  Returns None if timestamps unavailable."""
    if a.timestamp is None or b.timestamp is None:
        return None
    return abs(a.timestamp - b.timestamp)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


@dataclass
class BoundaryDetectionContext:
    """Intermediate state for boundary detection on a single page."""

    page_number: int
    page_width_mm: float
    candidate_lines: list[StrokeLine] = field(default_factory=list)


def _filter_candidate_lines(
    lines: list[StrokeLine],
    page_width_mm: float,
) -> list[StrokeLine]:
    """Keep only lines that are approximately horizontal and long enough."""
    return [
        line
        for line in lines
        if _is_horizontal(line) and _is_long_enough(line, page_width_mm)
    ]


def _pair_lines(
    candidates: list[StrokeLine],
    *,
    is_pen_path: bool,
) -> list[tuple[StrokeLine, StrokeLine]]:
    """Find pairs of candidate lines that satisfy the double-line delimiter
    constraints.

    Pairing rules (spec 4.1):
    - Y-gap between 2-15 mm
    - Horizontal overlap > 70 %
    - Temporal proximity < ~3 s  (pen path only)

    Returns pairs sorted by the Y midpoint of the upper line.
    """
    # Sort by Y midpoint for efficient pairing
    sorted_lines = sorted(candidates, key=lambda ln: ln.y_midpoint)
    used: set[int] = set()
    pairs: list[tuple[StrokeLine, StrokeLine]] = []

    for i, top_line in enumerate(sorted_lines):
        if i in used:
            continue
        for j in range(i + 1, len(sorted_lines)):
            if j in used:
                continue
            bottom_line = sorted_lines[j]

            gap = _y_gap(top_line, bottom_line)
            if gap < Y_GAP_MIN_MM or gap > Y_GAP_MAX_MM:
                # If gap exceeds max, no later candidate can pair either
                if gap > Y_GAP_MAX_MM:
                    break
                continue

            if _horizontal_overlap_ratio(top_line, bottom_line) < HORIZONTAL_OVERLAP_RATIO:
                continue

            if is_pen_path:
                t_gap = _temporal_gap(top_line, bottom_line)
                if t_gap is not None and t_gap > TEMPORAL_PROXIMITY_SEC:
                    continue

            pairs.append((top_line, bottom_line))
            used.add(i)
            used.add(j)
            break

    return pairs


def _pair_to_boundary(
    top: StrokeLine,
    bottom: StrokeLine,
    detection_method: str,
) -> DetectedBoundary:
    """Convert a paired line tuple into a DetectedBoundary model."""
    # Confidence is a heuristic combining overlap quality and gap tightness
    overlap = _horizontal_overlap_ratio(top, bottom)
    gap = _y_gap(top, bottom)
    # Best confidence when gap is near the midpoint of the allowed range
    gap_mid = (Y_GAP_MIN_MM + Y_GAP_MAX_MM) / 2.0
    gap_score = 1.0 - min(abs(gap - gap_mid) / gap_mid, 1.0)
    confidence = round((overlap + gap_score) / 2.0, 4)

    return DetectedBoundary(
        y_top=min(top.y_midpoint, bottom.y_midpoint),
        y_bottom=max(top.y_midpoint, bottom.y_midpoint),
        page_number=top.page_number,
        confidence=confidence,
        detection_method=detection_method,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_boundaries_pen(
    pages: list[PageOCR],
    stroke_lines: list[StrokeLine],
) -> list[DetectedBoundary]:
    """Detect answer boundaries from BLE pen stroke geometry.

    Args:
        pages: Ordered PageOCR objects (used for page dimensions).
        stroke_lines: Pre-extracted candidate lines from raw stroke data.

    Returns:
        Detected boundaries sorted by (page_number, y_top).
    """
    page_widths: dict[int, float] = {p.page_number: p.page_width_mm for p in pages}
    boundaries: list[DetectedBoundary] = []

    # Group lines by page
    lines_by_page: dict[int, list[StrokeLine]] = {}
    for line in stroke_lines:
        lines_by_page.setdefault(line.page_number, []).append(line)

    for page_num in sorted(lines_by_page):
        page_width = page_widths.get(page_num)
        if page_width is None or page_width <= 0:
            continue
        candidates = _filter_candidate_lines(
            lines_by_page[page_num], page_width
        )
        pairs = _pair_lines(candidates, is_pen_path=True)
        for top, bottom in pairs:
            boundaries.append(
                _pair_to_boundary(top, bottom, "stroke_geometry")
            )

    boundaries.sort(key=lambda b: (b.page_number, b.y_top))
    return boundaries


def detect_boundaries_camera(
    pages: list[PageOCR],
    hough_lines: list[StrokeLine],
) -> list[DetectedBoundary]:
    """Detect answer boundaries from camera/scan images via Hough transform
    output.

    Pre-processing (done upstream, not here):
        Canny edge detection -> HoughLinesP with
        minLineLength = 0.4 * image_width

    This function receives the output lines already converted to mm-space.

    Args:
        pages: Ordered PageOCR objects (used for page dimensions).
        hough_lines: Lines from HoughLinesP converted to page-space mm.

    Returns:
        Detected boundaries sorted by (page_number, y_top).

    Spec: PCR_EVAL_ENGINE_SPEC 4.1 camera path
    """
    page_widths: dict[int, float] = {p.page_number: p.page_width_mm for p in pages}
    boundaries: list[DetectedBoundary] = []

    lines_by_page: dict[int, list[StrokeLine]] = {}
    for line in hough_lines:
        lines_by_page.setdefault(line.page_number, []).append(line)

    for page_num in sorted(lines_by_page):
        page_width = page_widths.get(page_num)
        if page_width is None or page_width <= 0:
            continue
        candidates = _filter_candidate_lines(
            lines_by_page[page_num], page_width
        )
        # Camera path: no temporal proximity constraint
        pairs = _pair_lines(candidates, is_pen_path=False)
        for top, bottom in pairs:
            boundaries.append(
                _pair_to_boundary(top, bottom, "hough_transform")
            )

    boundaries.sort(key=lambda b: (b.page_number, b.y_top))
    return boundaries
