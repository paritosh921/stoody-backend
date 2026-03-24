"""Stroke-to-SVG rendering. ZERO I/O — pure transformation logic.

Converts canonical stroke points (page-space mm) into SVG path elements
with pressure-varying width support.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from src.domain.models import CanonicalPoint, Stroke


def _pressure_width(base_width: float, pressure: float) -> float:
    """Scale stroke width by pressure. Pressure range [0, 1]."""
    clamped = max(0.0, min(1.0, pressure))
    # Min 30% of base width at zero pressure, full at pressure=1
    return base_width * (0.3 + 0.7 * clamped)


def _build_path_data(points: list[CanonicalPoint]) -> str:
    """Build SVG path 'd' attribute from canonical points.

    Uses quadratic Bezier curves between midpoints for smooth strokes.
    Single-point strokes render as a dot (tiny circle via arc).
    """
    if not points:
        return ""

    if len(points) == 1:
        p = points[0]
        return f"M {p.x:.3f},{p.y:.3f} l 0.01,0"

    parts: list[str] = []
    p0 = points[0]
    parts.append(f"M {p0.x:.3f},{p0.y:.3f}")

    if len(points) == 2:
        p1 = points[1]
        parts.append(f"L {p1.x:.3f},{p1.y:.3f}")
        return " ".join(parts)

    # Quadratic Bezier through midpoints for smooth curves
    for i in range(1, len(points) - 1):
        curr = points[i]
        nxt = points[i + 1]
        mid_x = (curr.x + nxt.x) / 2
        mid_y = (curr.y + nxt.y) / 2
        parts.append(f"Q {curr.x:.3f},{curr.y:.3f} {mid_x:.3f},{mid_y:.3f}")

    # End at the last point
    last = points[-1]
    parts.append(f"L {last.x:.3f},{last.y:.3f}")

    return " ".join(parts)


def render_stroke_svg(
    points: list[CanonicalPoint],
    color: str = "#000000",
    width: float = 0.4,
) -> str:
    """Render a single stroke as an SVG <path> element.

    For pressure-varying width, the stroke is split into segments
    with per-segment width derived from average pressure.
    """
    if not points:
        return ""

    path_data = _build_path_data(points)
    if not path_data:
        return ""

    safe_color = escape(color)

    # Compute average pressure for overall stroke width
    avg_pressure = sum(p.pressure for p in points) / len(points)
    stroke_width = _pressure_width(width, avg_pressure)

    return (
        f'<path d="{path_data}" '
        f'stroke="{safe_color}" '
        f'stroke-width="{stroke_width:.3f}" '
        f'fill="none" '
        f'stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )


def render_stroke_svg_pressure_segments(
    points: list[CanonicalPoint],
    color: str = "#000000",
    base_width: float = 0.4,
    segment_size: int = 4,
) -> list[str]:
    """Render a stroke as multiple SVG paths with per-segment pressure width.

    Splits the stroke into overlapping segments. Each segment's width is
    the average pressure of its points. Produces more realistic output
    at the cost of more SVG elements.
    """
    if len(points) < 2:
        return [render_stroke_svg(points, color, base_width)]

    safe_color = escape(color)
    paths: list[str] = []

    for start in range(0, len(points) - 1, max(1, segment_size - 1)):
        end = min(start + segment_size, len(points))
        seg_points = points[start:end]

        seg_path = _build_path_data(seg_points)
        if not seg_path:
            continue

        avg_p = sum(p.pressure for p in seg_points) / len(seg_points)
        sw = _pressure_width(base_width, avg_p)

        paths.append(
            f'<path d="{seg_path}" '
            f'stroke="{safe_color}" '
            f'stroke-width="{sw:.3f}" '
            f'fill="none" '
            f'stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )

    return paths


def render_page_svg(
    strokes: list[Stroke],
    page_width: float = 210.0,
    page_height: float = 297.0,
) -> str:
    """Render a complete SVG document from a list of strokes.

    Dimensions are in millimetres (matching canonical point space).
    """
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{page_width:.1f}mm" height="{page_height:.1f}mm" '
        f'viewBox="0 0 {page_width:.1f} {page_height:.1f}">',
        f'<rect width="{page_width:.1f}" height="{page_height:.1f}" '
        f'fill="white"/>',
    ]

    for stroke in strokes:
        path_el = render_stroke_svg(
            stroke.points,
            stroke.color,
            stroke.base_width,
        )
        if path_el:
            lines.append(f"  {path_el}")

    lines.append("</svg>")
    return "\n".join(lines)
