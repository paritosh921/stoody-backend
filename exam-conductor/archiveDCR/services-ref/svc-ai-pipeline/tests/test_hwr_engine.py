"""Unit tests for HWR engine — confidence scoring, threshold flagging.

Test IDs: U-AI-HWR-01 through U-AI-HWR-07
Validation level: L3 (unit, domain, no I/O)
"""

import pytest

from src.domain.hwr_engine import (
    ANSWER_FLAG_RATIO,
    DEFAULT_CONFIDENCE_THRESHOLD,
    CharConfidence,
    HWRResult,
    build_per_char_confidence,
    recognize_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inference_fn(chars: list[str], confidences: list[float]):
    """Return a canned inference callable."""
    def _fn(_data: bytes) -> dict:
        return {"chars": chars, "confidences": confidences}
    return _fn


# ---------------------------------------------------------------------------
# U-AI-HWR-01: Basic recognition returns correct text
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_recognize_text_basic():
    """U-AI-HWR-01: recognized_text is the concatenation of chars."""
    fn = _make_inference_fn(["H", "i"], [0.99, 0.97])
    result = recognize_text(b"image", "en", fn)
    assert isinstance(result, HWRResult)
    assert result.recognized_text == "Hi"
    assert result.language == "en"


# ---------------------------------------------------------------------------
# U-AI-HWR-02: Aggregate confidence is mean of per-char confidences
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_aggregate_confidence():
    """U-AI-HWR-02: confidence is the mean of per-character scores."""
    fn = _make_inference_fn(["A", "B"], [0.80, 1.00])
    result = recognize_text(b"img", "en", fn)
    assert abs(result.confidence - 0.90) < 1e-6


# ---------------------------------------------------------------------------
# U-AI-HWR-03: Characters below threshold are flagged
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_below_threshold_characters_flagged():
    """U-AI-HWR-03: indices of chars below 0.85 appear in flagged_characters."""
    fn = _make_inference_fn(["X", "Y", "Z"], [0.90, 0.70, 0.95])
    result = recognize_text(b"img", "en", fn, threshold=0.85)
    assert 1 in result.flagged_characters
    assert 0 not in result.flagged_characters
    assert 2 not in result.flagged_characters


# ---------------------------------------------------------------------------
# U-AI-HWR-04: Answer flagged when >30% chars are below threshold
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_answer_flagged_when_many_below_threshold():
    """U-AI-HWR-04: flagged_for_review=True when >30% are below threshold."""
    # 3 out of 5 = 60% below threshold -> flagged
    fn = _make_inference_fn(
        ["A", "B", "C", "D", "E"],
        [0.50, 0.60, 0.70, 0.95, 0.99],
    )
    result = recognize_text(b"img", "en", fn, threshold=0.85)
    assert result.flagged_for_review is True


# ---------------------------------------------------------------------------
# U-AI-HWR-05: Answer NOT flagged when few chars below threshold
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_answer_not_flagged_when_few_below():
    """U-AI-HWR-05: flagged_for_review=False when <=30% are below threshold."""
    # 1 out of 5 = 20% below threshold -> not flagged
    fn = _make_inference_fn(
        ["A", "B", "C", "D", "E"],
        [0.70, 0.90, 0.95, 0.92, 0.88],
    )
    result = recognize_text(b"img", "en", fn, threshold=0.85)
    assert result.flagged_for_review is False


# ---------------------------------------------------------------------------
# U-AI-HWR-06: Empty input returns empty result
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_input():
    """U-AI-HWR-06: empty chars list produces empty recognized text."""
    fn = _make_inference_fn([], [])
    result = recognize_text(b"img", "en", fn)
    assert result.recognized_text == ""
    assert result.confidence == 0.0
    assert result.flagged_for_review is False


# ---------------------------------------------------------------------------
# U-AI-HWR-07: Mismatched chars/confidences raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_mismatched_lengths_raises():
    """U-AI-HWR-07: build_per_char_confidence raises if lengths differ."""
    with pytest.raises(ValueError, match="chars length"):
        build_per_char_confidence(["A", "B"], [0.9])


# ---------------------------------------------------------------------------
# U-AI-HWR-08: Per-char flagged field is correct
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_per_char_flagged_field():
    """U-AI-HWR-08: CharConfidence.flagged reflects threshold correctly."""
    per_char = build_per_char_confidence(
        ["A", "B"], [0.90, 0.80], threshold=0.85,
    )
    assert per_char[0].flagged is False
    assert per_char[1].flagged is True


# ---------------------------------------------------------------------------
# U-AI-HWR-09: Custom threshold is respected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_custom_threshold():
    """U-AI-HWR-09: a stricter threshold flags more characters."""
    fn = _make_inference_fn(["A", "B"], [0.90, 0.92])
    result = recognize_text(b"img", "en", fn, threshold=0.95)
    # Both below 0.95 -> both flagged
    assert len(result.flagged_characters) == 2
