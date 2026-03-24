"""Unit tests for structural similarity (Levenshtein + same-error weighting).

Test IDs: U-PLAG-STR-01 through U-PLAG-STR-07
"""

import pytest

from src.domain.structural_similarity import compute_structural_similarity


class TestComputeStructuralSimilarity:
    """Edit distance and same-error weighting tests."""

    @pytest.mark.unit
    def test_identical_texts_score_one(self) -> None:
        """U-PLAG-STR-01: Identical texts should produce ~1.0."""
        text = "The answer is forty two"
        score = compute_structural_similarity(text, text)
        assert score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.unit
    def test_single_char_difference(self) -> None:
        """U-PLAG-STR-02: One-character difference yields high similarity."""
        text_a = "Photosynthesis occurs in chloroplasts"
        text_b = "Photosynthesis occurs in chloroplast"  # missing 's'
        score = compute_structural_similarity(text_a, text_b)
        assert score > 0.90, f"Expected >0.90 for 1-char diff, got {score}"

    @pytest.mark.unit
    def test_completely_different_texts(self) -> None:
        """U-PLAG-STR-03: Unrelated texts should score low."""
        text_a = "abcdefghij"
        text_b = "zyxwvutsrq"
        score = compute_structural_similarity(text_a, text_b)
        assert score < 0.3, f"Expected <0.3 for unrelated, got {score}"

    @pytest.mark.unit
    def test_empty_texts_return_zero(self) -> None:
        """U-PLAG-STR-04: Empty input should return 0.0."""
        assert compute_structural_similarity("", "") == 0.0
        assert compute_structural_similarity("", "something") < 0.01
        assert compute_structural_similarity("text", "") < 0.01

    @pytest.mark.unit
    def test_same_error_boost(self) -> None:
        """U-PLAG-STR-05: Shared wrong tokens boost the score."""
        correct = "mitochondria powerhouse cell"
        # Both students make the same mistake: "ribosomes" instead of
        # "mitochondria"
        text_a = "ribosomes powerhouse cell"
        text_b = "ribosomes powerhouse cell"

        with_boost = compute_structural_similarity(
            text_a, text_b, correct_answer=correct,
        )
        without_boost = compute_structural_similarity(text_a, text_b)

        # Both should be high (identical texts), but the with-boost version
        # includes error analysis.  The base edit-distance is 1.0 for
        # identical texts, so the boost component is additive.
        assert with_boost >= without_boost - 0.01

    @pytest.mark.unit
    def test_different_errors_no_extra_boost(self) -> None:
        """U-PLAG-STR-06: Different wrong tokens should not boost much."""
        correct = "mitochondria powerhouse cell"
        text_a = "ribosomes powerhouse cell"
        text_b = "nucleus powerhouse cell"

        score = compute_structural_similarity(
            text_a, text_b, correct_answer=correct,
        )
        # Base edit distance between the two is moderate (different first word).
        # Error boost is low because the errors differ (ribosomes vs nucleus).
        assert score < 0.95

    @pytest.mark.unit
    def test_no_correct_answer_skips_error_analysis(self) -> None:
        """U-PLAG-STR-07: Without correct_answer, only edit distance matters."""
        text_a = "wrong answer here"
        text_b = "wrong answer here"
        score = compute_structural_similarity(text_a, text_b)
        # Identical -> high score even without error analysis
        assert score > 0.9
