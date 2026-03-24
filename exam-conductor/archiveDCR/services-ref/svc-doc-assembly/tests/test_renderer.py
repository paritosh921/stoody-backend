"""Unit tests for SVG rendering. Domain-only, ZERO I/O.

Test IDs: U-DA-R01 through U-DA-R08.
"""

from __future__ import annotations

import pytest

from src.domain.models import CanonicalPoint, Stroke
from src.domain.renderer import (
    render_page_svg,
    render_stroke_svg,
    render_stroke_svg_pressure_segments,
    _pressure_width,
    _build_path_data,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_points(coords: list[tuple[float, float, float]]) -> list[CanonicalPoint]:
    """Build CanonicalPoint list from (x, y, pressure) tuples."""
    return [CanonicalPoint(x=x, y=y, pressure=p) for x, y, p in coords]


# ── U-DA-R01: pressure width scaling ────────────────────────────────

@pytest.mark.unit
class TestPressureWidth:
    def test_full_pressure_returns_base_width(self):
        assert _pressure_width(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_pressure_returns_thirty_percent(self):
        assert _pressure_width(1.0, 0.0) == pytest.approx(0.3)

    def test_mid_pressure(self):
        assert _pressure_width(2.0, 0.5) == pytest.approx(2.0 * 0.65)

    def test_clamped_above_one(self):
        assert _pressure_width(1.0, 1.5) == pytest.approx(1.0)

    def test_clamped_below_zero(self):
        assert _pressure_width(1.0, -0.5) == pytest.approx(0.3)


# ── U-DA-R02: empty input ───────────────────────────────────────────

@pytest.mark.unit
class TestEmptyInput:
    def test_empty_points_returns_empty_string(self):
        assert render_stroke_svg([], "#000", 0.5) == ""

    def test_empty_strokes_returns_valid_svg(self):
        svg = render_page_svg([], 100.0, 100.0)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "<path" not in svg


# ── U-DA-R03: single-point stroke ───────────────────────────────────

@pytest.mark.unit
class TestSinglePoint:
    def test_single_point_produces_dot(self):
        pts = _make_points([(10.0, 20.0, 0.8)])
        result = render_stroke_svg(pts, "#ff0000", 0.5)
        assert "<path" in result
        assert 'stroke="#ff0000"' in result
        assert "M 10.000,20.000" in result


# ── U-DA-R04: two-point stroke (straight line) ──────────────────────

@pytest.mark.unit
class TestTwoPointStroke:
    def test_two_points_line(self):
        pts = _make_points([(0.0, 0.0, 1.0), (10.0, 10.0, 1.0)])
        result = render_stroke_svg(pts, "#000000", 0.4)
        assert "M 0.000,0.000" in result
        assert "L 10.000,10.000" in result


# ── U-DA-R05: multi-point stroke (bezier curves) ────────────────────

@pytest.mark.unit
class TestMultiPointStroke:
    def test_three_points_uses_quadratic_bezier(self):
        pts = _make_points([
            (0.0, 0.0, 1.0),
            (5.0, 10.0, 0.5),
            (10.0, 0.0, 1.0),
        ])
        result = render_stroke_svg(pts, "#000", 0.4)
        assert "Q " in result

    def test_many_points_smooth_path(self):
        pts = _make_points([
            (0.0, 0.0, 1.0),
            (2.0, 4.0, 0.8),
            (5.0, 3.0, 0.6),
            (8.0, 7.0, 0.9),
            (10.0, 5.0, 1.0),
        ])
        result = render_stroke_svg(pts, "#000", 0.4)
        assert result.count("Q ") >= 2


# ── U-DA-R06: pressure-segment rendering ────────────────────────────

@pytest.mark.unit
class TestPressureSegments:
    def test_segments_produce_multiple_paths(self):
        pts = _make_points([
            (0, 0, 0.2),
            (1, 1, 0.4),
            (2, 2, 0.8),
            (3, 3, 1.0),
            (4, 4, 0.3),
            (5, 5, 0.9),
        ])
        paths = render_stroke_svg_pressure_segments(pts, "#000", 0.5, segment_size=3)
        assert len(paths) >= 2
        for p in paths:
            assert "<path" in p

    def test_short_stroke_single_segment(self):
        pts = _make_points([(0, 0, 0.5)])
        paths = render_stroke_svg_pressure_segments(pts, "#000", 0.5)
        assert len(paths) == 1


# ── U-DA-R07: full page SVG document ────────────────────────────────

@pytest.mark.unit
class TestPageSvg:
    def test_page_svg_structure(self):
        strokes = [
            Stroke(
                stroke_id="s1",
                points=_make_points([(5, 5, 0.7), (10, 10, 0.9)]),
                color="#0000ff",
                base_width=0.5,
            ),
        ]
        svg = render_page_svg(strokes, 210.0, 297.0)

        assert '<?xml version="1.0"' in svg
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg
        assert 'width="210.0mm"' in svg
        assert 'height="297.0mm"' in svg
        assert 'viewBox="0 0 210.0 297.0"' in svg
        assert '<rect' in svg
        assert '<path' in svg
        assert 'stroke="#0000ff"' in svg
        assert "</svg>" in svg

    def test_page_multiple_strokes(self):
        strokes = [
            Stroke(stroke_id="s1", points=_make_points([(0, 0, 1.0), (5, 5, 1.0)])),
            Stroke(stroke_id="s2", points=_make_points([(10, 10, 0.5), (15, 15, 0.5)])),
        ]
        svg = render_page_svg(strokes)
        assert svg.count("<path") == 2


# ── U-DA-R08: XSS safety in color ───────────────────────────────────

@pytest.mark.unit
class TestColorSafety:
    def test_angle_brackets_escaped(self):
        pts = _make_points([(0, 0, 1.0), (1, 1, 1.0)])
        result = render_stroke_svg(pts, '<script>alert("xss")</script>', 0.5)
        assert "<script>" not in result
        assert "&lt;" in result
