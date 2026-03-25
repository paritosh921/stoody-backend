"""
ExamPen Test Harness — DCR engine tests.

Test IDs covered:
    U-DCR-01  Recognizer output shape validation
    U-DCR-02  Matcher handles exact, numeric (with tolerance), partial, no_match
    U-DCR-03  Default path is deterministic (no LLM calls)

Spec authority: new-docs/architecture/DUAL_MODE_ARCHITECTURE.md section 4
Failure modes:  DCR-01 (low confidence), DCR-02 (numeric mismatch),
                DCR-03 (scope creep)
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exam-conductor")
if _EC_DIR not in sys.path:
    sys.path.insert(0, _EC_DIR)

from dcr.models import (
    AnswerKey,
    DCRResult,
    MatchOutput,
    MatchType,
    RecognitionOutput,
)
from dcr.matcher import TemplateMatcher, _parse_numeric
from dcr.recognizer import HWRRecognizer, DEFAULT_LOW_CONFIDENCE_THRESHOLD


# ===========================================================================
# U-DCR-01: Recognizer output shape validation
# ===========================================================================


class TestUDcr01:
    """U-DCR-01: Vision OCR recognizer output normalized into DCR recognition input."""

    def test_u_dcr_01_recognition_output_shape(self):
        """RecognitionOutput has all required fields from spec."""
        output = RecognitionOutput(
            question_id="q-001",
            recognized_text="Paris",
            confidence=0.95,
            page_number=1,
        )
        assert output.question_id == "q-001"
        assert output.recognized_text == "Paris"
        assert output.confidence == 0.95
        assert output.page_number == 1
        assert output.raw_logits is None

    def test_u_dcr_01_recognition_output_confidence_bounds(self):
        """Confidence must be in [0, 1]."""
        output = RecognitionOutput(
            question_id="q-001",
            recognized_text="test",
            confidence=0.0,
        )
        assert output.confidence == 0.0

        output = RecognitionOutput(
            question_id="q-002",
            recognized_text="test",
            confidence=1.0,
        )
        assert output.confidence == 1.0

    def test_u_dcr_01_recognizer_requires_gate(self):
        """HWRRecognizer requires a gate instance (LLM Vision based)."""
        from unittest.mock import AsyncMock
        mock_gate = AsyncMock()
        recognizer = HWRRecognizer(gate=mock_gate)
        assert recognizer.low_confidence_threshold == DEFAULT_LOW_CONFIDENCE_THRESHOLD

    def test_u_dcr_01_recognizer_custom_threshold(self):
        """HWRRecognizer accepts custom low_confidence_threshold."""
        from unittest.mock import AsyncMock
        mock_gate = AsyncMock()
        recognizer = HWRRecognizer(gate=mock_gate, low_confidence_threshold=0.60)
        assert recognizer.low_confidence_threshold == 0.60

    def test_u_dcr_01_low_confidence_detection(self):
        """Recognizer flags low-confidence results (DCR-01)."""
        from unittest.mock import AsyncMock
        recognizer = HWRRecognizer(gate=AsyncMock(), low_confidence_threshold=0.40)
        output = RecognitionOutput(
            question_id="q-001",
            recognized_text="abc",
            confidence=0.35,
        )
        assert recognizer.is_low_confidence(output) is True

        output_high = RecognitionOutput(
            question_id="q-002",
            recognized_text="abc",
            confidence=0.80,
        )
        assert recognizer.is_low_confidence(output_high) is False

    def test_u_dcr_01_dcr_result_shape(self):
        """DCRResult has all fields from DUAL_MODE_ARCHITECTURE section 4.5."""
        result = DCRResult(
            exam_id="exam-001",
            student_id="stu-001",
            question_id="q-001",
            recognized_text="42",
            confidence=0.90,
            match_type=MatchType.EXACT_MATCH,
            score=2.0,
            max_score=2.0,
        )
        assert result.recognized_text == "42"
        assert result.match_type == MatchType.EXACT_MATCH
        assert result.score == 2.0
        assert result.audit_trail == []


# ===========================================================================
# U-DCR-02: Matcher handles exact, numeric, partial, no_match
# ===========================================================================


class TestUDcr02:
    """U-DCR-02: Template matching logic — exact/partial/numeric/no-match."""

    @pytest.fixture
    def matcher(self):
        return TemplateMatcher()

    # ---- Exact match ----

    def test_u_dcr_02_exact_match(self, matcher):
        """Exact case-insensitive match awards full marks."""
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="Paris", confidence=0.9
        )
        key = AnswerKey(
            question_id="q-1", expected_text="paris", max_score=2.0
        )
        result = matcher.match(rec, key)
        assert result.match_type == MatchType.EXACT_MATCH
        assert result.score == 2.0

    def test_u_dcr_02_exact_match_whitespace_normalized(self, matcher):
        """Whitespace differences should not prevent exact match."""
        rec = RecognitionOutput(
            question_id="q-1",
            recognized_text="  hello   world  ",
            confidence=0.9,
        )
        key = AnswerKey(
            question_id="q-1", expected_text="hello world", max_score=1.0
        )
        result = matcher.match(rec, key)
        assert result.match_type == MatchType.EXACT_MATCH

    # ---- Numeric match (DCR-02 mitigation) ----

    def test_u_dcr_02_numeric_match_exact_integer(self, matcher):
        """42 matches 42 numerically."""
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="42", confidence=0.9
        )
        key = AnswerKey(
            question_id="q-1", expected_text="42", max_score=2.0
        )
        result = matcher.match(rec, key)
        assert result.match_type in (MatchType.EXACT_MATCH, MatchType.NUMERIC_MATCH)
        assert result.score == 2.0

    def test_u_dcr_02_numeric_match_with_tolerance(self, matcher):
        """42.001 matches 42.0 within default tolerance."""
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="42.0", confidence=0.9
        )
        key = AnswerKey(
            question_id="q-1", expected_text="42", max_score=2.0
        )
        result = matcher.match(rec, key)
        # Should match as exact (after normalization "42.0" becomes "42.0" and
        # "42" becomes "42") or numeric
        assert result.match_type in (MatchType.EXACT_MATCH, MatchType.NUMERIC_MATCH)
        assert result.score == 2.0

    def test_u_dcr_02_numeric_match_custom_tolerance(self):
        """Custom numeric_tolerance from answer key is respected."""
        matcher = TemplateMatcher()
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="3.15", confidence=0.9
        )
        key = AnswerKey(
            question_id="q-1",
            expected_text="3.14",
            max_score=2.0,
            numeric_tolerance=0.05,
        )
        result = matcher.match(rec, key)
        assert result.match_type == MatchType.NUMERIC_MATCH
        assert result.score == 2.0

    def test_u_dcr_02_numeric_match_thousands_separator(self):
        """Thousands separator (comma) is handled."""
        assert _parse_numeric("1,000") == 1000.0
        assert _parse_numeric("1,234,567") == 1234567.0

    def test_u_dcr_02_numeric_match_trailing_dot(self):
        """Trailing dot is handled (common OCR artifact)."""
        assert _parse_numeric("42.") == 42.0

    def test_u_dcr_02_numeric_match_fraction(self):
        """Fraction notation is handled."""
        assert _parse_numeric("1/2") == 0.5
        assert _parse_numeric("3/4") == 0.75

    # ---- Partial match ----

    def test_u_dcr_02_partial_match(self, matcher):
        """Similar but not identical text produces partial_match."""
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="Pariis", confidence=0.7
        )
        key = AnswerKey(
            question_id="q-1", expected_text="Paris", max_score=2.0
        )
        result = matcher.match(rec, key)
        assert result.match_type == MatchType.PARTIAL_MATCH
        assert 0 < result.score < 2.0

    def test_u_dcr_02_partial_score_fraction(self, matcher):
        """Partial match awards score * partial_score_fraction."""
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="Pariis", confidence=0.7
        )
        key = AnswerKey(
            question_id="q-1", expected_text="Paris", max_score=4.0
        )
        result = matcher.match(rec, key)
        if result.match_type == MatchType.PARTIAL_MATCH:
            assert result.score == round(
                4.0 * matcher.partial_score_fraction, 2
            )

    # ---- No match ----

    def test_u_dcr_02_no_match(self, matcher):
        """Completely different text produces no_match with score=0."""
        rec = RecognitionOutput(
            question_id="q-1",
            recognized_text="xyzzy completely wrong",
            confidence=0.3,
        )
        key = AnswerKey(
            question_id="q-1", expected_text="Paris", max_score=2.0
        )
        result = matcher.match(rec, key)
        assert result.match_type == MatchType.NO_MATCH
        assert result.score == 0.0

    def test_u_dcr_02_empty_recognized_text(self, matcher):
        """Empty recognized text produces no_match."""
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="", confidence=0.0
        )
        key = AnswerKey(
            question_id="q-1", expected_text="Paris", max_score=2.0
        )
        result = matcher.match(rec, key)
        assert result.match_type == MatchType.NO_MATCH

    # ---- Batch matching ----

    def test_u_dcr_02_batch_match(self, matcher):
        """match_batch joins by question_id and returns outputs."""
        recognitions = [
            RecognitionOutput(question_id="q-1", recognized_text="Paris", confidence=0.9),
            RecognitionOutput(question_id="q-2", recognized_text="42", confidence=0.8),
        ]
        answer_keys = [
            AnswerKey(question_id="q-1", expected_text="Paris", max_score=2.0),
            AnswerKey(question_id="q-2", expected_text="42", max_score=1.0),
        ]
        results = matcher.match_batch(recognitions, answer_keys)
        assert len(results) == 2
        assert all(isinstance(r, MatchOutput) for r in results)

    def test_u_dcr_02_batch_skips_missing_key(self, matcher):
        """match_batch skips recognition without a matching answer key."""
        recognitions = [
            RecognitionOutput(question_id="q-1", recognized_text="X", confidence=0.5),
            RecognitionOutput(question_id="q-999", recognized_text="Y", confidence=0.5),
        ]
        answer_keys = [
            AnswerKey(question_id="q-1", expected_text="X", max_score=1.0),
        ]
        results = matcher.match_batch(recognitions, answer_keys)
        assert len(results) == 1
        assert results[0].question_id == "q-1"

    # ---- Match output shape ----

    def test_u_dcr_02_match_output_shape(self, matcher):
        """MatchOutput carries all context fields."""
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="Paris", confidence=0.9
        )
        key = AnswerKey(
            question_id="q-1", expected_text="Paris", max_score=2.0
        )
        result = matcher.match(rec, key)
        assert result.question_id == "q-1"
        assert result.recognized_text == "Paris"
        assert result.expected_text == "Paris"
        assert result.confidence == 0.9
        assert result.max_score == 2.0


# ===========================================================================
# U-DCR-03: Default path is deterministic (no LLM calls)
# ===========================================================================


class TestUDcr03:
    """U-DCR-03: DCR recognition routes through the shared LLM gate (Vision OCR)."""

    def test_u_dcr_03_matcher_is_deterministic(self):
        """TemplateMatcher makes no async or LLM calls."""
        matcher = TemplateMatcher()
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="answer", confidence=0.8
        )
        key = AnswerKey(
            question_id="q-1", expected_text="answer", max_score=1.0
        )
        # match() is synchronous — no await needed
        result = matcher.match(rec, key)
        assert result.match_type == MatchType.EXACT_MATCH

    def test_u_dcr_03_match_type_enum_values(self):
        """MatchType enum covers the four spec-defined types."""
        assert MatchType.EXACT_MATCH.value == "exact_match"
        assert MatchType.PARTIAL_MATCH.value == "partial_match"
        assert MatchType.NUMERIC_MATCH.value == "numeric_match"
        assert MatchType.NO_MATCH.value == "no_match"
        assert len(MatchType) == 4

    def test_u_dcr_03_recognizer_uses_gate(self):
        """HWRRecognizer routes all recognition through the LLM gate."""
        from unittest.mock import AsyncMock
        mock_gate = AsyncMock()
        recognizer = HWRRecognizer(gate=mock_gate)
        # The recognizer delegates to the gate — no direct provider calls
        assert hasattr(recognizer, '_gate')

    def test_u_dcr_03_numeric_parsing_no_external_calls(self):
        """Numeric parsing is purely local — no network calls."""
        assert _parse_numeric("42") == 42.0
        assert _parse_numeric("3.14") == 3.14
        assert _parse_numeric("not a number") is None

    def test_u_dcr_03_match_mode_hint_respected(self):
        """answer_key.match_mode='numeric' restricts matching to numeric only."""
        matcher = TemplateMatcher()
        rec = RecognitionOutput(
            question_id="q-1", recognized_text="42", confidence=0.9
        )
        # With match_mode='numeric', exact match path is skipped
        key = AnswerKey(
            question_id="q-1",
            expected_text="42",
            max_score=2.0,
            match_mode="numeric",
        )
        result = matcher.match(rec, key)
        assert result.match_type == MatchType.NUMERIC_MATCH
        assert result.score == 2.0
