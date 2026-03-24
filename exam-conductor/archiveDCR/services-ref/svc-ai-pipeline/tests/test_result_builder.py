"""Unit tests for result builder — result assembly, model version inclusion.

Test IDs: U-AI-RES-01 through U-AI-RES-07
Validation level: L3 (unit, domain, no I/O)
"""

import pytest

from src.domain.classifier import ContentType
from src.domain.hwr_engine import CharConfidence, HWRResult
from src.domain.result_builder import (
    AIResult,
    QuestionResult,
    build_question_result,
    build_result,
)
from src.domain.step_detector import StepBoundary, StepResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hwr(text: str = "x = 5", confidence: float = 0.92) -> HWRResult:
    return HWRResult(
        recognized_text=text,
        confidence=confidence,
        per_character_confidence=[
            CharConfidence(char=c, confidence=0.9, flagged=False)
            for c in text
        ],
        language="en",
        flagged_for_review=False,
        flagged_characters=[],
    )


def _make_steps(steps: list[str] | None = None) -> StepResult:
    steps = steps or ["x = 5"]
    return StepResult(
        steps=steps,
        step_count=len(steps),
        step_boundaries=[
            StepBoundary(start=0, end=len(s), label="simplification")
            for s in steps
        ],
    )


# ---------------------------------------------------------------------------
# U-AI-RES-01: build_question_result produces correct QuestionResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_question_result_basic():
    """U-AI-RES-01: QuestionResult includes text, confidence, steps, content_type."""
    hwr = _make_hwr("y = 10", 0.88)
    steps = _make_steps(["y = 10"])
    qr = build_question_result(hwr, steps, ContentType.FORMULA, "q-001")

    assert isinstance(qr, QuestionResult)
    assert qr.question_id == "q-001"
    assert qr.recognized_text == "y = 10"
    assert abs(qr.confidence - 0.88) < 1e-4
    assert qr.step_breakdown == ["y = 10"]
    assert qr.content_type == "formula"


# ---------------------------------------------------------------------------
# U-AI-RES-02: build_result includes model_version
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_result_model_version():
    """U-AI-RES-02: AIResult always includes model_version."""
    qr = build_question_result(
        _make_hwr(), _make_steps(), ContentType.TEXT, "q-001",
    )
    result = build_result([qr], "exam-1", "student-1", "hwr-v2.3")

    assert isinstance(result, AIResult)
    assert result.model_version == "hwr-v2.3"
    assert result.exam_id == "exam-1"
    assert result.student_id == "student-1"


# ---------------------------------------------------------------------------
# U-AI-RES-03: Event fields match schema
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_event_fields_match_schema():
    """U-AI-RES-03: AIResult has all required fields from ai.result.schema.json."""
    result = build_result([], "exam-1", "student-1", "v1.0")

    assert result.event_type == "ai.result"
    assert result.event_version == "1.0.0"
    assert result.event_id  # non-empty UUID
    assert result.occurred_at  # non-empty ISO datetime


# ---------------------------------------------------------------------------
# U-AI-RES-04: source_type defaults to strokes
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_source_type_default():
    """U-AI-RES-04: source_type defaults to 'strokes'."""
    result = build_result([], "exam-1", "student-1", "v1.0")
    assert result.source_type == "strokes"


# ---------------------------------------------------------------------------
# U-AI-RES-05: source_type can be overridden
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_source_type_override():
    """U-AI-RES-05: source_type can be set to 'copy_image'."""
    result = build_result(
        [], "exam-1", "student-1", "v1.0", source_type="copy_image",
    )
    assert result.source_type == "copy_image"


# ---------------------------------------------------------------------------
# U-AI-RES-06: Multiple question results assembled
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_multiple_question_results():
    """U-AI-RES-06: multiple QuestionResults are included in the AIResult."""
    qr1 = build_question_result(
        _make_hwr("a"), _make_steps(["a"]), ContentType.TEXT, "q-1",
    )
    qr2 = build_question_result(
        _make_hwr("b"), _make_steps(["b"]), ContentType.FORMULA, "q-2",
    )
    result = build_result([qr1, qr2], "exam-1", "student-1", "v1.0")
    assert len(result.question_results) == 2
    assert result.question_results[0].question_id == "q-1"
    assert result.question_results[1].question_id == "q-2"


# ---------------------------------------------------------------------------
# U-AI-RES-07: Flagged question includes flagged_for_review
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_flagged_question():
    """U-AI-RES-07: flagged HWR result propagates to QuestionResult."""
    hwr = HWRResult(
        recognized_text="5",
        confidence=0.60,
        per_character_confidence=[
            CharConfidence(char="5", confidence=0.60, flagged=True),
        ],
        language="en",
        flagged_for_review=True,
        flagged_characters=[0],
    )
    steps = _make_steps(["5"])
    qr = build_question_result(hwr, steps, ContentType.TEXT, "q-1")
    assert qr.flagged_for_review is True
