"""Assign strokes to question regions based on spatial overlap.

ZERO I/O -- this module must never import asyncio, aiohttp, sqlalchemy,
nats, or any I/O library.

Question regions are axis-aligned rectangles defined in mm coordinates.
A stroke's bounding box is compared against each region.  The region
with the highest overlap area wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QuestionRegion:
    """Rectangular region on a page that belongs to a question.

    All coordinates are in mm from the top-left corner.
    """

    question_id: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def area(self) -> float:
        return max(0.0, self.max_x - self.min_x) * max(
            0.0, self.max_y - self.min_y
        )


def _overlap_area(
    bbox: dict[str, float],
    region: QuestionRegion,
) -> float:
    """Compute the intersection area (mm^2) between bbox and region."""
    x_overlap = max(
        0.0,
        min(bbox["max_x"], region.max_x) - max(bbox["min_x"], region.min_x),
    )
    y_overlap = max(
        0.0,
        min(bbox["max_y"], region.max_y) - max(bbox["min_y"], region.min_y),
    )
    return x_overlap * y_overlap


def assign_to_question(
    stroke_bbox: dict[str, float] | None,
    question_regions: list[QuestionRegion],
) -> str | None:
    """Determine which question a stroke belongs to.

    Parameters
    ----------
    stroke_bbox:
        Bounding box dict with keys ``min_x``, ``min_y``, ``max_x``,
        ``max_y`` in mm coordinates.  ``None`` means no assignment.
    question_regions:
        Ordered list of question regions for the current page.

    Returns
    -------
    The ``question_id`` of the best-matching region, or ``None`` when
    the stroke does not overlap any region (e.g. margin writing).
    """
    if stroke_bbox is None or not question_regions:
        return None

    best_id: str | None = None
    best_area: float = 0.0

    for region in question_regions:
        area = _overlap_area(stroke_bbox, region)
        if area > best_area:
            best_area = area
            best_id = region.question_id

    return best_id


def assign_strokes_to_questions(
    strokes: list[dict[str, Any]],
    question_regions: list[QuestionRegion],
) -> list[dict[str, Any]]:
    """Enrich each stroke dict with a ``question_id`` field.

    Each stroke dict is expected to have a ``bbox`` key (or ``None``).
    The ``question_id`` is set to ``None`` when no region matches.

    Returns the same list, mutated in-place for efficiency.
    """
    for stroke in strokes:
        bbox = stroke.get("bbox")
        stroke["question_id"] = assign_to_question(bbox, question_regions)
    return strokes


def build_page_assignments(
    strokes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the ``page_assignments`` array for a stroke.processed event.

    Groups strokes by ``(page_number, question_id)`` and counts points.

    Returns
    -------
    List of dicts matching the ``stroke.processed`` schema's
    ``page_assignments`` items:
    ``{ "page_number": int, "question_id": str, "point_count": int }``.
    """
    counts: dict[tuple[int, str], int] = {}

    for stroke in strokes:
        page = stroke.get("page_number", 0)
        qid = stroke.get("question_id")
        if qid is None:
            qid = "unassigned"
        key = (page, qid)
        point_count = len(stroke.get("normalized_points", []))
        counts[key] = counts.get(key, 0) + point_count

    return [
        {
            "page_number": page,
            "question_id": qid,
            "point_count": pcount,
        }
        for (page, qid), pcount in sorted(counts.items())
    ]
