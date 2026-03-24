"""U-SCR-11 .. U-SCR-18: Rubric evaluation unit tests.

Domain-only -- no DB, no network.
"""

import pytest

from src.domain.rubric_eval import (
    QuestionScore,
    Rubric,
    RubricStep,
    StepScore,
    evaluate,
)


def _make_rubric(
    *,
    steps: list[RubricStep] | None = None,
    negative: bool = False,
    neg_factor: float = 0.0,
) -> Rubric:
    if steps is None:
        steps = [
            RubricStep(label="Step A", max_marks=2.0, keywords=["alpha"]),
            RubricStep(label="Step B", max_marks=1.0, keywords=["beta"]),
            RubricStep(label="Step C", max_marks=1.0, keywords=["gamma"]),
        ]
    return Rubric(
        question_id="q1",
        version=1,
        steps=steps,
        negative_marking=negative,
        negative_factor=neg_factor,
    )


class TestBasicEvaluation:
    """Step marking: 2 + 1 + 1 = 4 marks when all steps match."""

    def test_all_steps_match(self) -> None:
        ai = {
            "question_id": "q1",
            "recognized_text": "full answer",
            "confidence": 0.95,
            "step_breakdown": ["alpha result", "beta calc", "gamma final"],
        }
        rubric = _make_rubric()
        result = evaluate(ai, rubric)

        assert result.total_marks == 4.0
        assert result.max_marks == 4.0
        assert result.confidence == 0.95
        assert result.rubric_version == 1
        assert len(result.step_scores) == 3
        assert all(s.matched for s in result.step_scores)

    def test_partial_match(self) -> None:
        """Only first two steps matched -> 2 + 1 = 3."""
        ai = {
            "question_id": "q1",
            "recognized_text": "partial",
            "confidence": 0.80,
            "step_breakdown": ["alpha done", "beta done"],
        }
        result = evaluate(ai, _make_rubric())
        assert result.total_marks == 3.0
        assert result.step_scores[2].matched is False

    def test_no_match(self) -> None:
        ai = {
            "question_id": "q1",
            "recognized_text": "nothing",
            "confidence": 0.40,
            "step_breakdown": ["irrelevant text"],
        }
        result = evaluate(ai, _make_rubric())
        assert result.total_marks == 0.0

    def test_empty_step_breakdown(self) -> None:
        ai = {
            "question_id": "q1",
            "recognized_text": "empty",
            "confidence": 0.10,
            "step_breakdown": [],
        }
        result = evaluate(ai, _make_rubric())
        assert result.total_marks == 0.0


class TestNegativeMarking:
    def test_deduction_applied(self) -> None:
        """With neg factor 0.25, unmatched step C deducts 0.25 * 1 = 0.25."""
        ai = {
            "question_id": "q1",
            "recognized_text": "partial",
            "confidence": 0.70,
            "step_breakdown": ["alpha found", "beta found"],
        }
        rubric = _make_rubric(negative=True, neg_factor=0.25)
        result = evaluate(ai, rubric)

        # 2 + 1 + (-0.25) = 2.75
        assert result.total_marks == 2.75
        assert result.step_scores[2].awarded == -0.25

    def test_total_never_below_zero(self) -> None:
        """All wrong with negative marking still clamps to 0."""
        ai = {
            "question_id": "q1",
            "recognized_text": "wrong",
            "confidence": 0.10,
            "step_breakdown": ["xyz"],
        }
        rubric = _make_rubric(negative=True, neg_factor=1.0)
        result = evaluate(ai, rubric)
        assert result.total_marks == 0.0


class TestEdgeCases:
    def test_confidence_passthrough(self) -> None:
        ai = {"question_id": "q1", "confidence": 0.42, "step_breakdown": []}
        result = evaluate(ai, _make_rubric())
        assert result.confidence == 0.42

    def test_missing_confidence_defaults_zero(self) -> None:
        ai = {"question_id": "q1", "step_breakdown": []}
        result = evaluate(ai, _make_rubric())
        assert result.confidence == 0.0

    def test_rubric_version_recorded(self) -> None:
        rubric = Rubric(
            question_id="q9",
            version=7,
            steps=[RubricStep(label="only", max_marks=5, keywords=["x"])],
        )
        ai = {"question_id": "q9", "confidence": 0.5, "step_breakdown": ["x"]}
        result = evaluate(ai, rubric)
        assert result.rubric_version == 7

    def test_case_insensitive_matching(self) -> None:
        rubric = Rubric(
            question_id="q1",
            version=1,
            steps=[RubricStep(label="A", max_marks=2, keywords=["Alpha"])],
        )
        ai = {"question_id": "q1", "confidence": 0.9, "step_breakdown": ["ALPHA result"]}
        result = evaluate(ai, rubric)
        assert result.step_scores[0].matched is True
