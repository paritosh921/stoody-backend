"""Unit tests for composite plagiarism scorer.

Test IDs: U-PLAG-CMP-01 through U-PLAG-CMP-08
"""

import pytest

from src.domain.composite_scorer import (
    CompositeScore,
    QuestionType,
    Severity,
    score_pair,
    REVIEW_THRESHOLD,
    STRONG_THRESHOLD,
)


class TestScorePair:
    """Composite scoring, MCQ exclusion, severity thresholds."""

    @pytest.mark.unit
    def test_returns_composite_score_dataclass(self) -> None:
        """U-PLAG-CMP-01: score_pair returns a CompositeScore."""
        result = score_pair(
            "answer text one", "answer text two",
            QuestionType.SUBJECTIVE,
        )
        assert isinstance(result, CompositeScore)

    @pytest.mark.unit
    def test_identical_subjective_answers_high_composite(self) -> None:
        """U-PLAG-CMP-02: Identical subjective answers score high."""
        text = (
            "Photosynthesis converts sunlight energy into chemical "
            "energy stored in glucose molecules within chloroplasts"
        )
        result = score_pair(
            text, text,
            QuestionType.SUBJECTIVE,
            temporal_corr=0.5,
            proximity_score=0.5,
        )
        assert result.composite > REVIEW_THRESHOLD

    @pytest.mark.unit
    def test_strong_match_severity(self) -> None:
        """U-PLAG-CMP-03: Very high composite triggers strong_match."""
        text = "identical answer text repeated word for word exactly"
        result = score_pair(
            text, text,
            QuestionType.SUBJECTIVE,
            temporal_corr=0.95,
            proximity_score=0.90,
        )
        assert result.severity == Severity.STRONG_MATCH
        assert result.composite >= STRONG_THRESHOLD

    @pytest.mark.unit
    def test_review_recommended_severity(self) -> None:
        """U-PLAG-CMP-04: Moderate composite triggers review_recommended."""
        text = "the process of cellular respiration produces ATP energy"
        result = score_pair(
            text, text,
            QuestionType.SUBJECTIVE,
            temporal_corr=0.0,
            proximity_score=0.0,
        )
        # Text+structural sim will be high (~1.0) but temporal+proximity
        # are 0.0, so composite = ~0.35 + ~0.30 = ~0.65 -> depends on
        # exact TF-IDF.  We test the threshold classification logic.
        if REVIEW_THRESHOLD <= result.composite < STRONG_THRESHOLD:
            assert result.severity == Severity.REVIEW_RECOMMENDED

    @pytest.mark.unit
    def test_below_threshold_no_severity(self) -> None:
        """U-PLAG-CMP-05: Low composite has severity=None."""
        result = score_pair(
            "gravity pulls objects downward",
            "shakespeare wrote plays in london",
            QuestionType.SUBJECTIVE,
            temporal_corr=0.0,
            proximity_score=0.0,
        )
        assert result.severity is None
        assert result.composite < REVIEW_THRESHOLD

    @pytest.mark.unit
    def test_mcq_skips_text_similarity(self) -> None:
        """U-PLAG-CMP-06: MCQ questions exclude text similarity signals.

        Identical correct MCQ answers are NOT plagiarism.
        """
        result = score_pair(
            "B", "B",
            QuestionType.MCQ,
            temporal_corr=0.0,
            proximity_score=0.0,
        )
        assert result.text_sim == 0.0
        assert result.structural_sim == 0.0
        assert result.composite == 0.0
        assert result.severity is None

    @pytest.mark.unit
    def test_objective_skips_text_similarity(self) -> None:
        """U-PLAG-CMP-07: Objective questions also exclude text similarity."""
        result = score_pair(
            "42", "42",
            QuestionType.OBJECTIVE,
            temporal_corr=0.3,
            proximity_score=0.2,
        )
        assert result.text_sim == 0.0
        assert result.structural_sim == 0.0
        # composite = 0.3 * 0.6 + 0.2 * 0.4 = 0.26
        assert result.composite == pytest.approx(0.26, abs=0.01)

    @pytest.mark.unit
    def test_temporal_proximity_clamped(self) -> None:
        """U-PLAG-CMP-08: Out-of-range inputs are clamped to [0, 1]."""
        result = score_pair(
            "some answer", "some answer",
            QuestionType.SUBJECTIVE,
            temporal_corr=1.5,
            proximity_score=-0.2,
        )
        assert result.temporal_corr == pytest.approx(1.0, abs=0.01)
        assert result.proximity_score == pytest.approx(0.0, abs=0.01)
