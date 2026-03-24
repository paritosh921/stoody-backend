"""Unit tests for TF-IDF cosine text similarity.

Test IDs: U-PLAG-TXT-01 through U-PLAG-TXT-06
"""

import pytest

from src.domain.text_similarity import compute_tfidf_similarity


class TestComputeTfidfSimilarity:
    """TF-IDF cosine similarity: known similar pair >0.85."""

    @pytest.mark.unit
    def test_identical_texts_score_one(self) -> None:
        """U-PLAG-TXT-01: Identical texts should produce similarity 1.0."""
        text = "The mitochondria is the powerhouse of the cell"
        assert compute_tfidf_similarity(text, text) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_known_similar_pair_above_085(self) -> None:
        """U-PLAG-TXT-02: Two nearly identical answers must score >0.85."""
        text_a = (
            "Photosynthesis converts sunlight energy into chemical energy "
            "stored in glucose molecules within chloroplasts"
        )
        text_b = (
            "Photosynthesis converts sunlight energy into chemical energy "
            "stored in glucose molecules in the chloroplasts"
        )
        score = compute_tfidf_similarity(text_a, text_b)
        assert score > 0.85, f"Expected >0.85 for near-identical, got {score}"

    @pytest.mark.unit
    def test_completely_different_texts_low_score(self) -> None:
        """U-PLAG-TXT-03: Unrelated texts should score near 0.0."""
        text_a = "Gravity pulls objects toward earth at acceleration"
        text_b = "Shakespeare wrote Romeo Juliet during Elizabethan era"
        score = compute_tfidf_similarity(text_a, text_b)
        assert score < 0.2, f"Expected <0.2 for unrelated, got {score}"

    @pytest.mark.unit
    def test_empty_text_returns_zero(self) -> None:
        """U-PLAG-TXT-04: Empty input should return 0.0."""
        assert compute_tfidf_similarity("", "some text here") == 0.0
        assert compute_tfidf_similarity("hello world", "") == 0.0
        assert compute_tfidf_similarity("", "") == 0.0

    @pytest.mark.unit
    def test_stopword_only_texts_return_zero(self) -> None:
        """U-PLAG-TXT-05: Texts with only stopwords should return 0.0."""
        assert compute_tfidf_similarity("the a an is", "the a an is") == 0.0

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        """U-PLAG-TXT-06: Comparison should be case-insensitive."""
        text_a = "PHOTOSYNTHESIS CONVERTS SUNLIGHT"
        text_b = "photosynthesis converts sunlight"
        score = compute_tfidf_similarity(text_a, text_b)
        assert score == pytest.approx(1.0)
