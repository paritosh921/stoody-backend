"""Unit tests for exam domain models — validation logic, no I/O.

Test IDs: U-ORCH-MOD-01 through U-ORCH-MOD-08.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.domain.exam_models import (
    ExamDefinition,
    ExamValidationError,
    QuestionRegion,
    RubricDefinition,
    RubricItem,
    validate_regions_match_exam,
    validate_rubric_matches_exam,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_exam(**overrides: object) -> ExamDefinition:
    defaults = {
        "title": "Math Final",
        "subject_id": "math-101",
        "class_id": "cls-10",
        "section_id": "sec-a",
        "scheduled_at": datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc),
        "duration_min": 90,
        "question_count": 3,
        "total_marks": 30.0,
    }
    defaults.update(overrides)
    return ExamDefinition(**defaults)  # type: ignore[arg-type]


def _make_rubric_items(count: int, marks_each: float) -> list[RubricItem]:
    return [
        RubricItem(question_number=i, max_marks=marks_each)
        for i in range(1, count + 1)
    ]


# ---------------------------------------------------------------------------
# U-ORCH-MOD-01: ExamDefinition validation
# ---------------------------------------------------------------------------


class TestExamDefinition:
    def test_valid_exam(self) -> None:
        exam = _make_exam()
        assert exam.title == "Math Final"
        assert exam.question_count == 3

    def test_blank_title_rejected(self) -> None:
        with pytest.raises(ExamValidationError, match="title"):
            _make_exam(title="   ")

    def test_zero_duration_rejected(self) -> None:
        with pytest.raises(ExamValidationError, match="duration_min"):
            _make_exam(duration_min=0)

    def test_negative_marks_rejected(self) -> None:
        with pytest.raises(ExamValidationError, match="total_marks"):
            _make_exam(total_marks=-5)

    def test_zero_questions_rejected(self) -> None:
        with pytest.raises(ExamValidationError, match="question_count"):
            _make_exam(question_count=0)

    def test_variants_default_empty(self) -> None:
        exam = _make_exam()
        assert exam.variants == []


# ---------------------------------------------------------------------------
# U-ORCH-MOD-02: RubricItem validation
# ---------------------------------------------------------------------------


class TestRubricItem:
    def test_valid_item(self) -> None:
        item = RubricItem(question_number=1, max_marks=10.0)
        assert item.max_marks == 10.0

    def test_zero_marks_rejected(self) -> None:
        with pytest.raises(ExamValidationError, match="max_marks"):
            RubricItem(question_number=1, max_marks=0)

    def test_step_breakdown_sum_mismatch(self) -> None:
        with pytest.raises(ExamValidationError, match="step_breakdown"):
            RubricItem(
                question_number=1,
                max_marks=10.0,
                step_breakdown=[3.0, 3.0, 3.0],  # sum=9 != 10
            )

    def test_step_breakdown_sum_match(self) -> None:
        item = RubricItem(
            question_number=1,
            max_marks=10.0,
            step_breakdown=[4.0, 3.0, 3.0],
        )
        assert sum(item.step_breakdown) == 10.0

    def test_invalid_answer_type(self) -> None:
        with pytest.raises(ExamValidationError, match="answer type"):
            RubricItem(
                question_number=1,
                max_marks=5.0,
                expected_answer_type="drawing",
            )

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ExamValidationError, match="confidence"):
            RubricItem(
                question_number=1,
                max_marks=5.0,
                confidence_threshold=1.5,
            )


# ---------------------------------------------------------------------------
# U-ORCH-MOD-03: QuestionRegion validation
# ---------------------------------------------------------------------------


class TestQuestionRegion:
    def test_valid_region(self) -> None:
        r = QuestionRegion(question_number=1, x=10, y=20, width=100, height=50)
        assert r.page == 1

    def test_zero_width_rejected(self) -> None:
        with pytest.raises(ExamValidationError, match="dimensions"):
            QuestionRegion(question_number=1, x=0, y=0, width=0, height=50)

    def test_negative_coords_rejected(self) -> None:
        with pytest.raises(ExamValidationError, match="coordinates"):
            QuestionRegion(question_number=1, x=-1, y=0, width=10, height=10)

    def test_zero_question_number(self) -> None:
        with pytest.raises(ExamValidationError, match="question_number"):
            QuestionRegion(question_number=0, x=0, y=0, width=10, height=10)


# ---------------------------------------------------------------------------
# U-ORCH-MOD-04: validate_rubric_matches_exam
# ---------------------------------------------------------------------------


class TestRubricExamCross:
    def test_matching_rubric(self) -> None:
        exam = _make_exam(question_count=3, total_marks=30.0)
        rubric = RubricDefinition(items=_make_rubric_items(3, 10.0))
        validate_rubric_matches_exam(exam, rubric)  # should not raise

    def test_wrong_count(self) -> None:
        exam = _make_exam(question_count=3, total_marks=30.0)
        rubric = RubricDefinition(items=_make_rubric_items(2, 15.0))
        with pytest.raises(ExamValidationError, match="items"):
            validate_rubric_matches_exam(exam, rubric)

    def test_wrong_total(self) -> None:
        exam = _make_exam(question_count=3, total_marks=30.0)
        rubric = RubricDefinition(items=_make_rubric_items(3, 8.0))
        with pytest.raises(ExamValidationError, match="total"):
            validate_rubric_matches_exam(exam, rubric)

    def test_non_contiguous_numbers(self) -> None:
        exam = _make_exam(question_count=2, total_marks=20.0)
        items = [
            RubricItem(question_number=1, max_marks=10.0),
            RubricItem(question_number=3, max_marks=10.0),
        ]
        rubric = RubricDefinition(items=items)
        with pytest.raises(ExamValidationError, match="contiguous"):
            validate_rubric_matches_exam(exam, rubric)


# ---------------------------------------------------------------------------
# U-ORCH-MOD-05: validate_regions_match_exam
# ---------------------------------------------------------------------------


class TestRegionExamCross:
    def test_matching_regions(self) -> None:
        exam = _make_exam(question_count=2, total_marks=20.0)
        regions = [
            QuestionRegion(question_number=1, x=0, y=0, width=100, height=50),
            QuestionRegion(question_number=2, x=0, y=60, width=100, height=50),
        ]
        validate_regions_match_exam(exam, regions)

    def test_wrong_region_count(self) -> None:
        exam = _make_exam(question_count=2, total_marks=20.0)
        regions = [
            QuestionRegion(question_number=1, x=0, y=0, width=100, height=50),
        ]
        with pytest.raises(ExamValidationError, match="regions"):
            validate_regions_match_exam(exam, regions)
