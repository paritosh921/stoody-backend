"""Coordinate normalization: raw pen units to normalized mm coordinates.

ZERO I/O -- this module must never import asyncio, aiohttp, sqlalchemy,
nats, or any I/O library.

Scale constants (from PEN_TO_CANVAS_TO_DB_REFERENCE.md section 7):
    - Pen resolution: 10 pen-units per mm
    - Canvas resolution: 4 px per mm
    - Y-axis: pen origin = bottom-left, canvas origin = top-left

Book sizes (physical dimensions in mm):
    - A4 portrait: 210 x 297
    - A5 portrait: 148 x 210
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PEN_UNITS_PER_MM: float = 10.0
CANVAS_PX_PER_MM: float = 4.0

# Book type -> (width_mm, height_mm)
# Mappings follow the agent convention (section 7.1 of PEN reference):
#   MS/MN/MM -> A5 portrait, LS/LN/LM -> A4 portrait, LL/LW -> A3 landscape
BOOK_DIMENSIONS_MM: dict[str, tuple[float, float]] = {
    # A5 portrait variants
    "MS": (148.0, 210.0),
    "MN": (148.0, 210.0),
    "MM": (148.0, 210.0),
    # A4 portrait variants
    "LS": (210.0, 297.0),
    "LN": (210.0, 297.0),
    "LM": (210.0, 297.0),
    # A3 landscape variants
    "LL": (420.0, 297.0),
    "LW": (420.0, 297.0),
}

# Default when book type is unknown — A4 portrait
DEFAULT_BOOK_TYPE = "LS"
DEFAULT_DIMENSIONS_MM: tuple[float, float] = (210.0, 297.0)


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """A single point in normalized mm coordinates."""

    x_mm: float
    y_mm: float
    pressure: float
    timestamp: int


def _get_dimensions(book_type: str) -> tuple[float, float]:
    """Resolve physical dimensions in mm for a given book type."""
    return BOOK_DIMENSIONS_MM.get(book_type, DEFAULT_DIMENSIONS_MM)


def _pen_to_mm(pen_value: float) -> float:
    """Convert raw pen units to millimetres."""
    return pen_value / PEN_UNITS_PER_MM


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the range [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def normalize_point(
    raw_x: float,
    raw_y: float,
    pressure: float,
    timestamp: int,
    book_type: str,
) -> NormalizedPoint:
    """Normalize a single raw pen point to mm coordinates.

    - Converts pen units to mm (divide by 10).
    - Inverts Y axis (pen = bottom-left, target = top-left).
    - Clamps to book physical bounds.
    """
    width_mm, height_mm = _get_dimensions(book_type)

    x_mm = _clamp(_pen_to_mm(raw_x), 0.0, width_mm)
    # Y inversion: pen bottom-left -> top-left
    y_mm = _clamp(height_mm - _pen_to_mm(raw_y), 0.0, height_mm)

    return NormalizedPoint(
        x_mm=round(x_mm, 3),
        y_mm=round(y_mm, 3),
        pressure=round(_clamp(pressure, 0.0, 1.0), 4),
        timestamp=timestamp,
    )


def normalize_coordinates(
    raw_points: list[dict[str, Any]],
    book_type: str,
) -> list[dict[str, Any]]:
    """Normalize a list of raw pen points to mm coordinates.

    Each raw point dict is expected to have keys:
    ``x``, ``y``, ``pressure``, ``timestamp``.

    Returns a list of dicts with keys:
    ``x_mm``, ``y_mm``, ``pressure``, ``timestamp``.
    """
    results: list[dict[str, Any]] = []
    for pt in raw_points:
        np = normalize_point(
            raw_x=float(pt["x"]),
            raw_y=float(pt["y"]),
            pressure=float(pt.get("pressure", 0.5)),
            timestamp=int(pt.get("timestamp", 0)),
            book_type=book_type,
        )
        results.append({
            "x_mm": np.x_mm,
            "y_mm": np.y_mm,
            "pressure": np.pressure,
            "timestamp": np.timestamp,
        })
    return results


def mm_to_canvas_px(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Convert mm coordinates to canvas pixel coordinates."""
    return x_mm * CANVAS_PX_PER_MM, y_mm * CANVAS_PX_PER_MM


def compute_bbox_mm(
    points: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Compute axis-aligned bounding box in mm from normalized points.

    Returns ``None`` if *points* is empty.
    """
    if not points:
        return None

    xs = [p["x_mm"] for p in points]
    ys = [p["y_mm"] for p in points]
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }
