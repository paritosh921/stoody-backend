"""
PCR Content Classifier — Classify response content type.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md section 4.4
Failure mode:   PCR-03 (diagram-heavy answer auto-scored incorrectly ->
                classification blocks or downweights unsafe auto-eval)
Test ID:        U-CCLS-01

Thresholds from spec:
    TEXT_ONLY:      > 85% text coverage
    MIXED:          40-85% text with figure content
    DIAGRAM_HEAVY:  < 40% text
    TABLE_PRESENT:  grid or tabular structure detected

Diagram prorating:
    scoreable_marks = max_marks * (1 - diagram_weight)
"""

from __future__ import annotations

import uuid

from .flag_registry import FLAG_REGISTRY, FlagDefinition
from .response_models import (
    BoundingBox,
    ContentType,
    DetectedResponse,
    Flag,
    FlagSeverity,
    FlagType,
    TextBlock,
)


# ---------------------------------------------------------------------------
# Thresholds — exact values from PCR_EVAL_ENGINE_SPEC 4.4
# ---------------------------------------------------------------------------

TEXT_ONLY_THRESHOLD: float = 0.85
"""Text coverage > 85% -> TEXT_ONLY."""

MIXED_LOWER_THRESHOLD: float = 0.40
"""Text coverage 40-85% -> MIXED."""

# Anything below 40% -> DIAGRAM_HEAVY


# ---------------------------------------------------------------------------
# Table Detection Heuristics
# ---------------------------------------------------------------------------

# Simple heuristics for grid/tabular structure.  Full table detection is
# done upstream (image analysis); here we check text-block alignment patterns.

_TABLE_ALIGNMENT_TOLERANCE_MM: float = 2.0
"""Two text blocks are considered column-aligned if their x_min values
differ by less than this threshold."""

_MIN_TABLE_ROWS: int = 3
"""Minimum row count to classify as TABLE_PRESENT."""

_MIN_TABLE_COLS: int = 2
"""Minimum column count to classify as TABLE_PRESENT."""


def _detect_table_structure(blocks: list[TextBlock]) -> bool:
    """Lightweight heuristic for grid/tabular structure using text block
    alignment.

    Groups blocks by approximate x_min to detect columns, then checks for
    enough rows in at least _MIN_TABLE_COLS columns.
    """
    if len(blocks) < _MIN_TABLE_ROWS * _MIN_TABLE_COLS:
        return False

    # Cluster blocks by x_min
    columns: list[list[TextBlock]] = []
    sorted_blocks = sorted(blocks, key=lambda b: b.bbox.x_min)

    for block in sorted_blocks:
        placed = False
        for col in columns:
            ref_x = col[0].bbox.x_min
            if abs(block.bbox.x_min - ref_x) < _TABLE_ALIGNMENT_TOLERANCE_MM:
                col.append(block)
                placed = True
                break
        if not placed:
            columns.append([block])

    # Count columns with enough rows
    qualifying_cols = sum(
        1 for col in columns if len(col) >= _MIN_TABLE_ROWS
    )
    return qualifying_cols >= _MIN_TABLE_COLS


# ---------------------------------------------------------------------------
# Text Coverage Computation
# ---------------------------------------------------------------------------


def compute_text_coverage(
    text_blocks: list[TextBlock],
    response_bbox: BoundingBox,
) -> float:
    """Compute the fraction of a response region covered by text blocks.

    Uses a simple area ratio: sum of text block areas / response area.
    Clamped to [0, 1].
    """
    response_area = response_bbox.area
    if response_area <= 0:
        return 0.0

    text_area = 0.0
    for block in text_blocks:
        # Intersect block bbox with response bbox
        ix_min = max(block.bbox.x_min, response_bbox.x_min)
        iy_min = max(block.bbox.y_min, response_bbox.y_min)
        ix_max = min(block.bbox.x_max, response_bbox.x_max)
        iy_max = min(block.bbox.y_max, response_bbox.y_max)
        if ix_max > ix_min and iy_max > iy_min:
            text_area += (ix_max - ix_min) * (iy_max - iy_min)

    ratio = text_area / response_area
    return min(max(ratio, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _make_flag(
    flag_type: FlagType,
    response_id: str | None,
    metadata: dict | None = None,
) -> Flag:
    """Create a Flag from the registry definition."""
    defn: FlagDefinition = FLAG_REGISTRY[flag_type]
    return Flag(
        flag_id=f"FLG-{uuid.uuid4().hex[:8]}",
        response_id=response_id,
        source=defn.source,
        flag_type=defn.flag_type,
        severity=defn.severity,
        reason=defn.description,
        suggested_action=defn.suggested_action,
        metadata=metadata or {},
    )


def classify_content(
    text_blocks: list[TextBlock],
    response_bbox: BoundingBox,
    response_id: str | None = None,
    *,
    expects_diagram: bool = False,
) -> tuple[ContentType, float, list[Flag]]:
    """Classify the content type of a response region.

    Args:
        text_blocks: Text blocks within the response region.
        response_bbox: Bounding box of the response region.
        response_id: Optional response ID for flag association.
        expects_diagram: Whether the question expects a diagram (from
            evalpen_questions.expects_diagram).

    Returns:
        (content_type, text_coverage_ratio, flags)
    """
    flags: list[Flag] = []

    text_coverage = compute_text_coverage(text_blocks, response_bbox)

    # Check for table structure first (TABLE_PRESENT overrides area-based
    # classification per spec)
    if _detect_table_structure(text_blocks):
        flags.append(
            _make_flag(FlagType.TABLE_DETECTED, response_id)
        )
        return ContentType.TABLE_PRESENT, text_coverage, flags

    # Area-based classification
    if text_coverage > TEXT_ONLY_THRESHOLD:
        content_type = ContentType.TEXT_ONLY
    elif text_coverage >= MIXED_LOWER_THRESHOLD:
        content_type = ContentType.MIXED
        flags.append(
            _make_flag(
                FlagType.DIAGRAM_PRESENT,
                response_id,
                {"text_coverage": round(text_coverage, 4)},
            )
        )
    else:
        content_type = ContentType.DIAGRAM_HEAVY
        flags.append(
            _make_flag(
                FlagType.DIAGRAM_HEAVY_CONTENT,
                response_id,
                {"text_coverage": round(text_coverage, 4)},
            )
        )

    # Check for expected diagram that is missing
    if expects_diagram and content_type == ContentType.TEXT_ONLY:
        flags.append(
            _make_flag(FlagType.EXPECTED_DIAGRAM_MISSING, response_id)
        )

    return content_type, text_coverage, flags


def compute_scoreable_marks(
    max_marks: float,
    diagram_weight: float,
) -> float:
    """Compute prorated scoreable marks when diagram content is excluded.

    Formula from spec 4.4:
        scoreable_marks = max_marks * (1 - diagram_weight)

    Args:
        max_marks: Maximum marks for the question.
        diagram_weight: Fraction of marks attributable to diagram (0-1).

    Returns:
        Scoreable marks after diagram exclusion.
    """
    if diagram_weight < 0.0 or diagram_weight > 1.0:
        raise ValueError(
            f"diagram_weight must be in [0, 1], got {diagram_weight}"
        )
    return max_marks * (1.0 - diagram_weight)
