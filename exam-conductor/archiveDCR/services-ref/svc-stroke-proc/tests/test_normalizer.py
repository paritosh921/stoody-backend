"""Unit tests for domain/normalizer.py — coordinate normalization.

Test IDs: U-SPROC-10 through U-SPROC-21
Markers: unit (ZERO I/O)
"""

from __future__ import annotations

import pytest

from src.domain.normalizer import (
    BOOK_DIMENSIONS_MM,
    CANVAS_PX_PER_MM,
    DEFAULT_DIMENSIONS_MM,
    PEN_UNITS_PER_MM,
    compute_bbox_mm,
    mm_to_canvas_px,
    normalize_coordinates,
    normalize_point,
)


# ---------------------------------------------------------------------------
# U-SPROC-10: Scale constants match spec
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scale_constants():
    assert PEN_UNITS_PER_MM == 10.0
    assert CANVAS_PX_PER_MM == 4.0


# ---------------------------------------------------------------------------
# U-SPROC-11: A4 dimensions registered
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a4_dimensions():
    assert BOOK_DIMENSIONS_MM["LS"] == (210.0, 297.0)
    assert BOOK_DIMENSIONS_MM["LN"] == (210.0, 297.0)
    assert BOOK_DIMENSIONS_MM["LM"] == (210.0, 297.0)


# ---------------------------------------------------------------------------
# U-SPROC-12: A5 dimensions registered
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a5_dimensions():
    assert BOOK_DIMENSIONS_MM["MS"] == (148.0, 210.0)
    assert BOOK_DIMENSIONS_MM["MN"] == (148.0, 210.0)
    assert BOOK_DIMENSIONS_MM["MM"] == (148.0, 210.0)


# ---------------------------------------------------------------------------
# U-SPROC-13: Default dimensions for unknown book type
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_dimensions():
    assert DEFAULT_DIMENSIONS_MM == (210.0, 297.0)


# ---------------------------------------------------------------------------
# U-SPROC-14: Basic DPI transform — pen units to mm
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_point_dpi():
    # 1050 pen units = 105.0 mm in x direction
    pt = normalize_point(
        raw_x=1050, raw_y=0, pressure=0.5, timestamp=100, book_type="LS"
    )
    assert pt.x_mm == 105.0
    # Y at 0 pen units (bottom of page) -> 297.0 mm (top of canvas)
    assert pt.y_mm == 297.0


# ---------------------------------------------------------------------------
# U-SPROC-15: Y-axis inversion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_y_inversion():
    # A4: height = 297 mm = 2970 pen units
    # pen y=2970 (top of page in pen coords) -> 0 mm (top of canvas)
    pt = normalize_point(
        raw_x=0, raw_y=2970, pressure=0.5, timestamp=0, book_type="LS"
    )
    assert pt.y_mm == 0.0

    # pen y=0 (bottom) -> 297.0 mm (bottom of canvas)
    pt2 = normalize_point(
        raw_x=0, raw_y=0, pressure=0.5, timestamp=0, book_type="LS"
    )
    assert pt2.y_mm == 297.0


# ---------------------------------------------------------------------------
# U-SPROC-16: Clamping — X exceeds page width
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clamp_x_exceeds():
    # A4 width = 210 mm = 2100 pen units; pass 2500
    pt = normalize_point(
        raw_x=2500, raw_y=1000, pressure=0.5, timestamp=0, book_type="LS"
    )
    assert pt.x_mm == 210.0


# ---------------------------------------------------------------------------
# U-SPROC-17: Clamping — negative values
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clamp_negative():
    pt = normalize_point(
        raw_x=-10, raw_y=-10, pressure=-0.2, timestamp=0, book_type="LS"
    )
    assert pt.x_mm == 0.0
    # negative raw_y means pen_to_mm < 0, so height - neg = > height, clamped
    assert pt.y_mm == 297.0
    assert pt.pressure == 0.0


# ---------------------------------------------------------------------------
# U-SPROC-18: Pressure clamping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pressure_clamped():
    pt = normalize_point(
        raw_x=0, raw_y=0, pressure=1.5, timestamp=0, book_type="LS"
    )
    assert pt.pressure == 1.0


# ---------------------------------------------------------------------------
# U-SPROC-19: A5 book type dimensions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_a5():
    # A5: 148 x 210 mm = 1480 x 2100 pen units
    pt = normalize_point(
        raw_x=1480, raw_y=2100, pressure=0.5, timestamp=0, book_type="MS"
    )
    assert pt.x_mm == 148.0
    assert pt.y_mm == 0.0


# ---------------------------------------------------------------------------
# U-SPROC-20: normalize_coordinates batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_coordinates_batch():
    raw = [
        {"x": 1050, "y": 1485, "pressure": 0.7, "timestamp": 1},
        {"x": 500, "y": 500, "pressure": 0.3, "timestamp": 2},
    ]
    result = normalize_coordinates(raw, "LS")
    assert len(result) == 2
    assert result[0]["x_mm"] == 105.0
    assert result[0]["y_mm"] == 148.5
    assert result[0]["timestamp"] == 1
    assert result[1]["x_mm"] == 50.0


# ---------------------------------------------------------------------------
# U-SPROC-21: compute_bbox_mm
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_bbox():
    points = [
        {"x_mm": 10.0, "y_mm": 20.0},
        {"x_mm": 50.0, "y_mm": 5.0},
        {"x_mm": 30.0, "y_mm": 40.0},
    ]
    bbox = compute_bbox_mm(points)
    assert bbox is not None
    assert bbox["min_x"] == 10.0
    assert bbox["min_y"] == 5.0
    assert bbox["max_x"] == 50.0
    assert bbox["max_y"] == 40.0


@pytest.mark.unit
def test_compute_bbox_empty():
    assert compute_bbox_mm([]) is None


# ---------------------------------------------------------------------------
# U-SPROC-22: mm_to_canvas_px conversion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mm_to_canvas_px():
    px_x, px_y = mm_to_canvas_px(105.0, 148.5)
    assert px_x == 420.0
    assert px_y == 594.0
