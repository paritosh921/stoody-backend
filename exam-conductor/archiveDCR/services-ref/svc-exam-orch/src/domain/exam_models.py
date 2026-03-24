"""Exam, rubric, and question-region domain models — ZERO I/O, pure logic.

All validation is synchronous and side-effect-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class ExamValidationError(Exception):
    """Raised when exam data fails a domain invariant."""


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuestionRegion:
    """Bounding box of a question on the answer sheet."""

    question_number: int
    x: float
    y: float
    width: float
    height: float
    page: int = 1

    def __post_init__(self) -> None:
        if self.question_number < 1:
            raise ExamValidationError("question_number must be >= 1")
        if self.width <= 0 or self.height <= 0:
            raise ExamValidationError("Region dimensions must be positive")
        if self.x < 0 or self.y < 0:
            raise ExamValidationError("Region coordinates must be >= 0")


@dataclass(frozen=True, slots=True)
class RubricItem:
    """Marks breakdown for a single question."""

    question_number: int
    max_marks: float
    step_breakdown: list[float] = field(default_factory=list)
    expected_answer_type: str = "text"
    confidence_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.max_marks <= 0:
            raise ExamValidationError(
                f"Q{self.question_number}: max_marks must be positive"
            )
        if self.step_breakdown:
            step_total = sum(self.step_breakdown)
            if abs(step_total - self.max_marks) > 0.01:
                raise ExamValidationError(
                    f"Q{self.question_number}: step_breakdown sum "
                    f"({step_total}) != max_marks ({self.max_marks})"
                )
        if self.expected_answer_type not in (
            "text", "formula", "diagram", "mixed",
        ):
            raise ExamValidationError(
                f"Q{self.question_number}: invalid answer type "
                f"'{self.expected_answer_type}'"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ExamValidationError(
                f"Q{self.question_number}: confidence_threshold "
                f"must be in [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class RubricDefinition:
    """Complete rubric for an exam."""

    items: list[RubricItem]

    def total_marks(self) -> float:
        return sum(item.max_marks for item in self.items)


@dataclass(frozen=True, slots=True)
class ExamDefinition:
    """Immutable snapshot of an exam's core configuration."""

    title: str
    subject_id: str
    class_id: str
    section_id: str
    scheduled_at: datetime
    duration_min: int
    question_count: int
    total_marks: float
    negative_marking: bool = False
    variants: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.duration_min <= 0:
            raise ExamValidationError("duration_min must be positive")
        if self.question_count <= 0:
            raise ExamValidationError("question_count must be positive")
        if self.total_marks <= 0:
            raise ExamValidationError("total_marks must be positive")
        if not self.title.strip():
            raise ExamValidationError("title must not be blank")


# ---------------------------------------------------------------------------
# Cross-model validation helpers
# ---------------------------------------------------------------------------


def validate_rubric_matches_exam(
    exam: ExamDefinition,
    rubric: RubricDefinition,
) -> None:
    """Ensure rubric question count and total marks match the exam.

    Raises
    ------
    ExamValidationError
        On mismatch.
    """
    if len(rubric.items) != exam.question_count:
        raise ExamValidationError(
            f"Rubric has {len(rubric.items)} items but exam expects "
            f"{exam.question_count} questions"
        )

    rubric_total = rubric.total_marks()
    if abs(rubric_total - exam.total_marks) > 0.01:
        raise ExamValidationError(
            f"Rubric total ({rubric_total}) != exam total_marks "
            f"({exam.total_marks})"
        )

    # Ensure contiguous question numbering starting at 1
    numbers = sorted(item.question_number for item in rubric.items)
    expected = list(range(1, exam.question_count + 1))
    if numbers != expected:
        raise ExamValidationError(
            "Rubric question numbers must be contiguous 1..N"
        )


def validate_regions_match_exam(
    exam: ExamDefinition,
    regions: list[QuestionRegion],
) -> None:
    """Ensure regions cover every question exactly once.

    Raises
    ------
    ExamValidationError
        On mismatch.
    """
    if len(regions) != exam.question_count:
        raise ExamValidationError(
            f"Got {len(regions)} regions but exam has "
            f"{exam.question_count} questions"
        )

    numbers = sorted(r.question_number for r in regions)
    expected = list(range(1, exam.question_count + 1))
    if numbers != expected:
        raise ExamValidationError(
            "Region question numbers must be contiguous 1..N"
        )
