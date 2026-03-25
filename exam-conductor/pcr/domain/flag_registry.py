"""
PCR Flag Registry — All 18 flag types with severity and descriptions.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md section 6.2
Failure modes:  PCR-01 (boundary/marker failure -> flags + review queue)
                PCR-02 (clubbed undetected -> multiple heuristics)
                PCR-03 (diagram-heavy auto-scored -> classification blocks)
Test IDs:       U-SEG-01, U-SEG-02, U-SEG-03, U-CCLS-01, U-CLUB-01
"""

from __future__ import annotations

from dataclasses import dataclass

from .response_models import FlagSeverity, FlagType


@dataclass(frozen=True)
class FlagDefinition:
    """Immutable definition for a single flag type."""

    flag_type: FlagType
    severity: FlagSeverity
    source: str
    description: str
    suggested_action: str


# ---------------------------------------------------------------------------
# Complete registry — 18 flag types, matching PCR_EVAL_ENGINE_SPEC 6.2 exactly
# ---------------------------------------------------------------------------

FLAG_REGISTRY: dict[FlagType, FlagDefinition] = {
    # ---- segmenter (4) ----
    FlagType.NO_QUESTION_MARKER: FlagDefinition(
        flag_type=FlagType.NO_QUESTION_MARKER,
        severity=FlagSeverity.WARNING,
        source="segmenter",
        description="No Q marker detected in the response region",
        suggested_action="review association",
    ),
    FlagType.NO_BOUNDARY_DETECTED: FlagDefinition(
        flag_type=FlagType.NO_BOUNDARY_DETECTED,
        severity=FlagSeverity.WARNING,
        source="segmenter",
        description="No double-line boundary detected between responses",
        suggested_action="review segmentation",
    ),
    FlagType.BOUNDARY_ONLY_NO_MARKER: FlagDefinition(
        flag_type=FlagType.BOUNDARY_ONLY_NO_MARKER,
        severity=FlagSeverity.WARNING,
        source="segmenter",
        description=(
            "Boundary delimiter detected but no Q marker in the segment"
        ),
        suggested_action="review question assignment",
    ),
    FlagType.LOW_SEGMENTATION_CONFIDENCE: FlagDefinition(
        flag_type=FlagType.LOW_SEGMENTATION_CONFIDENCE,
        severity=FlagSeverity.WARNING,
        source="segmenter",
        description="Segmentation confidence below acceptable threshold",
        suggested_action="review segmentation",
    ),
    # ---- content_classifier (4) ----
    FlagType.DIAGRAM_PRESENT: FlagDefinition(
        flag_type=FlagType.DIAGRAM_PRESENT,
        severity=FlagSeverity.INFO,
        source="content_classifier",
        description="Diagram or figure content detected alongside text",
        suggested_action="evaluate text portion, note diagram",
    ),
    FlagType.DIAGRAM_HEAVY_CONTENT: FlagDefinition(
        flag_type=FlagType.DIAGRAM_HEAVY_CONTENT,
        severity=FlagSeverity.BLOCKING,
        source="content_classifier",
        description=(
            "Less than 40% text coverage — auto-evaluation blocked"
        ),
        suggested_action="route to manual review",
    ),
    FlagType.TABLE_DETECTED: FlagDefinition(
        flag_type=FlagType.TABLE_DETECTED,
        severity=FlagSeverity.INFO,
        source="content_classifier",
        description="Grid or tabular structure detected",
        suggested_action="route to table template or flag for review",
    ),
    FlagType.EXPECTED_DIAGRAM_MISSING: FlagDefinition(
        flag_type=FlagType.EXPECTED_DIAGRAM_MISSING,
        severity=FlagSeverity.WARNING,
        source="content_classifier",
        description=(
            "Question expects a diagram but none detected in response"
        ),
        suggested_action="review answer completeness",
    ),
    # ---- clubbed_detector (4) ----
    FlagType.CLUBBED_MULTIPLE_MARKERS: FlagDefinition(
        flag_type=FlagType.CLUBBED_MULTIPLE_MARKERS,
        severity=FlagSeverity.BLOCKING,
        source="clubbed_detector",
        description=(
            "More than one Q marker detected in a single segment (H1)"
        ),
        suggested_action="split segment and re-associate",
    ),
    FlagType.CLUBBED_LENGTH_ANOMALY: FlagDefinition(
        flag_type=FlagType.CLUBBED_LENGTH_ANOMALY,
        severity=FlagSeverity.WARNING,
        source="clubbed_detector",
        description=(
            "Word count exceeds expected maximum * 2.5 (H2)"
        ),
        suggested_action="review for multiple answers clubbed together",
    ),
    FlagType.CLUBBED_MISSING_QUESTION: FlagDefinition(
        flag_type=FlagType.CLUBBED_MISSING_QUESTION,
        severity=FlagSeverity.WARNING,
        source="clubbed_detector",
        description=(
            "Manifest question not represented in any segment (H3)"
        ),
        suggested_action="review for clubbed or missing response",
    ),
    FlagType.CLUBBED_TOPIC_DISCONTINUITY: FlagDefinition(
        flag_type=FlagType.CLUBBED_TOPIC_DISCONTINUITY,
        severity=FlagSeverity.WARNING,
        source="clubbed_detector",
        description=(
            "LLM-assisted topic discontinuity detected in segment (H4)"
        ),
        suggested_action="review for potential clubbed responses",
    ),
    # ---- ocr (2) ----
    FlagType.LOW_OCR_CONFIDENCE: FlagDefinition(
        flag_type=FlagType.LOW_OCR_CONFIDENCE,
        severity=FlagSeverity.WARNING,
        source="ocr",
        description="OCR confidence below acceptable threshold",
        suggested_action="review recognized text",
    ),
    FlagType.OCR_REJECTED: FlagDefinition(
        flag_type=FlagType.OCR_REJECTED,
        severity=FlagSeverity.BLOCKING,
        source="ocr",
        description=(
            "OCR confidence too low to produce usable text — rejected"
        ),
        suggested_action="re-capture or manual transcription",
    ),
    # ---- eval (2) ----
    FlagType.PARTIAL_EVAL_DIAGRAM_EXCLUDED: FlagDefinition(
        flag_type=FlagType.PARTIAL_EVAL_DIAGRAM_EXCLUDED,
        severity=FlagSeverity.INFO,
        source="eval",
        description=(
            "Diagram portion excluded from auto-evaluation; "
            "scoreable marks prorated"
        ),
        suggested_action="review diagram portion manually if needed",
    ),
    FlagType.LLM_SCORE_DIVERGENCE: FlagDefinition(
        flag_type=FlagType.LLM_SCORE_DIVERGENCE,
        severity=FlagSeverity.WARNING,
        source="eval",
        description="LLM score diverges significantly from expected range",
        suggested_action="review evaluation result",
    ),
    # ---- llm_gate (2) ----
    FlagType.BUDGET_WARNING_80PCT: FlagDefinition(
        flag_type=FlagType.BUDGET_WARNING_80PCT,
        severity=FlagSeverity.WARNING,
        source="llm_gate",
        description="LLM token budget usage has exceeded 80%",
        suggested_action="monitor budget; consider deferring non-critical calls",
    ),
    FlagType.BUDGET_EXHAUSTED: FlagDefinition(
        flag_type=FlagType.BUDGET_EXHAUSTED,
        severity=FlagSeverity.BLOCKING,
        source="llm_gate",
        description="LLM token budget exhausted — no further calls allowed",
        suggested_action="wait for budget reset or increase budget",
    ),
}

# Compile-time assertion: exactly 18 flags, matching the spec
assert len(FLAG_REGISTRY) == 18, (
    f"FLAG_REGISTRY must contain exactly 18 flag types per spec; "
    f"found {len(FLAG_REGISTRY)}"
)


def get_flag_definition(flag_type: FlagType) -> FlagDefinition:
    """Look up the canonical definition for a flag type.

    Raises KeyError if the flag type is not registered (should never happen
    since FlagType is an enum over the same 18 values).
    """
    return FLAG_REGISTRY[flag_type]
