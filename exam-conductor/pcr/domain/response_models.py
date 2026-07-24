"""
PCR Domain Models — Pydantic models for segmentation, classification, and flagging.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md sections 4, 6
Test IDs: U-SEG-01, U-SEG-02, U-SEG-03, U-CCLS-01, U-CLUB-01
Failure modes: PCR-01, PCR-02, PCR-03, PCR-04
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ContentType(str, enum.Enum):
    """Content classification for a detected response.

    Thresholds (PCR_EVAL_ENGINE_SPEC 4.4):
        TEXT_ONLY:      > 85 % text coverage
        MIXED:          40-85 % text with figure content
        DIAGRAM_HEAVY:  < 40 % text
        TABLE_PRESENT:  grid / tabular structure detected
    """

    TEXT_ONLY = "TEXT_ONLY"
    MIXED = "MIXED"
    DIAGRAM_HEAVY = "DIAGRAM_HEAVY"
    TABLE_PRESENT = "TABLE_PRESENT"


class FlagSeverity(str, enum.Enum):
    """Severity levels per PCR_EVAL_ENGINE_SPEC 6.3.

    blocking  -> eval_status = blocked
    warning   -> evaluated_with_warnings
    info      -> evaluated
    """

    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


class FlagType(str, enum.Enum):
    """All 18 flag types from PCR_EVAL_ENGINE_SPEC 6.2."""

    # segmenter (4)
    NO_QUESTION_MARKER = "no_question_marker"
    NO_BOUNDARY_DETECTED = "no_boundary_detected"
    BOUNDARY_ONLY_NO_MARKER = "boundary_only_no_marker"
    LOW_SEGMENTATION_CONFIDENCE = "low_segmentation_confidence"

    # content_classifier (4)
    DIAGRAM_PRESENT = "diagram_present"
    DIAGRAM_HEAVY_CONTENT = "diagram_heavy_content"
    TABLE_DETECTED = "table_detected"
    EXPECTED_DIAGRAM_MISSING = "expected_diagram_missing"

    # clubbed_detector (4)
    CLUBBED_MULTIPLE_MARKERS = "clubbed_multiple_markers"
    CLUBBED_LENGTH_ANOMALY = "clubbed_length_anomaly"
    CLUBBED_MISSING_QUESTION = "clubbed_missing_question"
    CLUBBED_TOPIC_DISCONTINUITY = "clubbed_topic_discontinuity"

    # ocr (2)
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    OCR_REJECTED = "ocr_rejected"

    # eval (2)
    PARTIAL_EVAL_DIAGRAM_EXCLUDED = "partial_eval_diagram_excluded"
    LLM_SCORE_DIVERGENCE = "llm_score_divergence"

    # llm_gate (2)
    BUDGET_WARNING_80PCT = "budget_warning_80pct"
    BUDGET_EXHAUSTED = "budget_exhausted"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in page-space millimetres."""

    x_min: float = Field(..., description="Left edge in mm")
    y_min: float = Field(..., description="Top edge in mm")
    x_max: float = Field(..., description="Right edge in mm")
    y_max: float = Field(..., description="Bottom edge in mm")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


# ---------------------------------------------------------------------------
# OCR Primitives
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """A single recognized text block from OCR/HWR."""

    text: str = Field(..., description="Recognized text content")
    bbox: BoundingBox = Field(..., description="Bounding box in page-space mm")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Recognition confidence 0-1"
    )
    source: str = Field(
        ...,
        description="Origin: 'pen' (stroke vectors) or 'camera' (image scan)",
    )


class PageOCR(BaseModel):
    """Unified OCR output for a single page — the shared input to the
    segmentation pipeline.

    Spec: PCR_EVAL_ENGINE_SPEC section 3, 'Unified PageOCR'.
    """

    page_number: int = Field(..., ge=1, description="1-based page index")
    page_width_mm: float = Field(..., gt=0, description="Page width in mm")
    page_height_mm: float = Field(..., gt=0, description="Page height in mm")
    text_blocks: list[TextBlock] = Field(
        default_factory=list, description="Recognized text blocks on this page"
    )
    image_width_px: int | None = Field(
        None, description="Image width in pixels (camera path only)"
    )
    image_height_px: int | None = Field(
        None, description="Image height in pixels (camera path only)"
    )
    source: str = Field(
        ..., description="Capture source: 'pen' or 'camera'"
    )
    mean_ocr_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Average OCR confidence across all text blocks",
    )


# ---------------------------------------------------------------------------
# Boundary / Marker Primitives
# ---------------------------------------------------------------------------


class DetectedBoundary(BaseModel):
    """A pair of horizontal lines that form a response delimiter.

    Spec: PCR_EVAL_ENGINE_SPEC 4.1
    """

    y_top: float = Field(..., description="Y-coordinate of top line (mm)")
    y_bottom: float = Field(
        ..., description="Y-coordinate of bottom line (mm)"
    )
    page_number: int = Field(..., ge=1)
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence"
    )
    detection_method: str = Field(
        ..., description="'stroke_geometry' or 'hough_transform'"
    )


class QMarker(BaseModel):
    """A parsed question marker.

    Spec: PCR_EVAL_ENGINE_SPEC 4.2
    """

    question_number: int = Field(..., ge=1, description="Captured \\1")
    sub_part: str | None = Field(
        None, description="Captured \\2 (letter, roman, uppercase)"
    )
    raw_text: str = Field(..., description="Original OCR text that matched")
    page_number: int = Field(..., ge=1)
    y_position: float = Field(
        ..., description="Y-position of the marker on the page (mm)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="OCR confidence of the marker block"
    )


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


class Flag(BaseModel):
    """A flag raised during segmentation / classification / evaluation.

    Shape matches PCR_EVAL_ENGINE_SPEC 6.1 exactly.
    """

    flag_id: str = Field(..., description="Unique flag ID, e.g. FLG-001")
    response_id: str | None = Field(
        None, description="Associated response ID if available"
    )
    source: str = Field(
        ...,
        description="Subsystem: segmenter | content_classifier | clubbed_detector | ocr | eval | llm_gate",
    )
    flag_type: FlagType
    severity: FlagSeverity
    reason: str
    suggested_action: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detected Response
# ---------------------------------------------------------------------------


class SourcePageRef(BaseModel):
    """A reference to a page range contributing to a detected response."""

    page_number: int = Field(..., ge=1)
    x_start: float = Field(
        0.0,
        description="Start X in mm on this page (0 = left)",
    )
    y_start: float = Field(
        ..., description="Start Y in mm on this page (0 = top)"
    )
    x_end: float | None = Field(
        None,
        description="End X in mm on this page; absent means full page width",
    )
    y_end: float = Field(
        ..., description="End Y in mm on this page (page_height = bottom)"
    )
    region_id: str | None = Field(
        None,
        description="Stable visual evidence-region identifier",
    )
    evidence_kind: str | None = Field(
        None,
        description="handwriting | mathematics | diagram | table | graph | label | mixed",
    )
    continuation_group: str | None = Field(
        None,
        description="Links disconnected regions belonging to one continued answer",
    )
    evidence: str | None = Field(
        None,
        description="Short model description used for visual evidence audit",
    )
    mapping_confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Question-ownership confidence for this region",
    )


class DetectedResponse(BaseModel):
    """A segmented student response for a single question (or fragment).

    Produced by the segmenter, consumed by the evaluation core and the
    LLM gate.

    Spec: PCR_EVAL_ENGINE_SPEC sections 4, 7.2
    """

    response_id: str = Field(..., description="Unique response ID")
    question_number: int | None = Field(
        None,
        description="Parsed question number from Q marker, None if unassociated",
    )
    sub_part: str | None = Field(None, description="Sub-part from Q marker")
    detected_text: str = Field(
        ..., description="Full concatenated text of the response"
    )
    source_pages: list[SourcePageRef] = Field(
        ..., description="Page spans this response covers"
    )
    content_type: ContentType = Field(
        ContentType.TEXT_ONLY,
        description="Classification of content in this response",
    )
    text_coverage_ratio: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of response area covered by text blocks",
    )
    segmentation_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall segmentation confidence"
    )
    ocr_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean OCR confidence across contributing text blocks",
    )
    flags: list[Flag] = Field(
        default_factory=list, description="Flags raised for this response"
    )
    word_count: int = Field(0, ge=0, description="Word count of detected_text")
    is_continuation: bool = Field(
        False,
        description="True if this response continues from a previous page",
    )


# ---------------------------------------------------------------------------
# Pipeline Output
# ---------------------------------------------------------------------------


class SegmentationResult(BaseModel):
    """Complete output of the segmentation pipeline for a submission.

    Contains all detected responses, all flags (including those not tied
    to a specific response), and summary statistics.
    """

    responses: list[DetectedResponse] = Field(default_factory=list)
    flags: list[Flag] = Field(
        default_factory=list,
        description="Global flags not tied to a single response",
    )
    page_count: int = Field(..., ge=0)
    total_boundaries_detected: int = Field(0, ge=0)
    total_markers_detected: int = Field(0, ge=0)
    has_blocking_flags: bool = Field(False)
