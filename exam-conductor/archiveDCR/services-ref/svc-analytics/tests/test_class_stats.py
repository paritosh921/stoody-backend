"""Unit tests for class statistics — domain logic only, ZERO I/O.

Test IDs: U-ANA-CS-01 through U-ANA-CS-10
"""

from __future__ import annotations

import math

import pytest

from src.domain.class_stats import (
    ClassStats,
    QuestionDifficulty,
    QuestionResponse,
    compute_class_stats,
    compute_question_difficulty,
)


# -- U-ANA-CS-01: Empty input ---------------------------------------------


def test_empty_input() -> None:
    """U-ANA-CS-01: Empty scores → all zeros with count=0."""
    stats = compute_class_stats([])
    assert stats == ClassStats(
        mean=0.0,
        median=0.0,
        std_dev=0.0,
        pass_pct=0.0,
        min_score=0.0,
        max_score=0.0,
        count=0,
    )


# -- U-ANA-CS-02: Single score --------------------------------------------


def test_single_score() -> None:
    """U-ANA-CS-02: Single score — mean=median=score, std_dev=0."""
    stats = compute_class_stats([75.0])
    assert stats.mean == 75.0
    assert stats.median == 75.0
    assert stats.std_dev == 0.0
    assert stats.count == 1


# -- U-ANA-CS-03: Mean calculation ----------------------------------------


def test_mean_calculation() -> None:
    """U-ANA-CS-03: Mean of [40, 50, 60, 70, 80] = 60."""
    scores = [40.0, 50.0, 60.0, 70.0, 80.0]
    stats = compute_class_stats(scores)
    assert stats.mean == 60.0


# -- U-ANA-CS-04: Median odd count ----------------------------------------


def test_median_odd_count() -> None:
    """U-ANA-CS-04: Median of 5 sorted values is the middle one."""
    scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_class_stats(scores)
    assert stats.median == 30.0


# -- U-ANA-CS-05: Median even count ---------------------------------------


def test_median_even_count() -> None:
    """U-ANA-CS-05: Median of 4 values is average of two middle."""
    scores = [10.0, 20.0, 30.0, 40.0]
    stats = compute_class_stats(scores)
    assert stats.median == 25.0


# -- U-ANA-CS-06: Standard deviation --------------------------------------


def test_std_dev() -> None:
    """U-ANA-CS-06: Population std dev of [2, 4, 4, 4, 5, 5, 7, 9].

    Mean = 5.0, variance = 4.0, std_dev = 2.0
    """
    scores = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    stats = compute_class_stats(scores, pass_threshold=0.0)
    assert stats.mean == 5.0
    assert stats.std_dev == 2.0


# -- U-ANA-CS-07: Pass percentage -----------------------------------------


def test_pass_percentage() -> None:
    """U-ANA-CS-07: 3 out of 5 students pass threshold=40."""
    scores = [20.0, 30.0, 40.0, 50.0, 60.0]
    stats = compute_class_stats(scores, pass_threshold=40.0)
    assert stats.pass_pct == 60.0


def test_pass_percentage_all_pass() -> None:
    """U-ANA-CS-07b: All students pass → 100%."""
    scores = [80.0, 90.0, 100.0]
    stats = compute_class_stats(scores, pass_threshold=40.0)
    assert stats.pass_pct == 100.0


def test_pass_percentage_none_pass() -> None:
    """U-ANA-CS-07c: No students pass → 0%."""
    scores = [10.0, 20.0, 30.0]
    stats = compute_class_stats(scores, pass_threshold=40.0)
    assert stats.pass_pct == 0.0


# -- U-ANA-CS-08: Min and max ---------------------------------------------


def test_min_max() -> None:
    """U-ANA-CS-08: Min and max are correct."""
    scores = [55.0, 23.0, 97.0, 41.0, 68.0]
    stats = compute_class_stats(scores)
    assert stats.min_score == 23.0
    assert stats.max_score == 97.0


# -- U-ANA-CS-09: Question difficulty — basic ------------------------------


def test_question_difficulty_basic() -> None:
    """U-ANA-CS-09: Basic question difficulty with 3 students, 2 questions."""
    responses = [
        # Question Q1: 3 attempted, 1 got full marks, avg = (5+3+5)/3
        QuestionResponse("Q1", score=5.0, max_score=5.0, attempted=True),
        QuestionResponse("Q1", score=3.0, max_score=5.0, attempted=True),
        QuestionResponse("Q1", score=5.0, max_score=5.0, attempted=True),
        # Question Q2: 2 attempted, 0 got full marks, avg = (2+1)/2
        QuestionResponse("Q2", score=2.0, max_score=5.0, attempted=True),
        QuestionResponse("Q2", score=1.0, max_score=5.0, attempted=True),
        QuestionResponse("Q2", score=0.0, max_score=5.0, attempted=False),
    ]
    result = compute_question_difficulty(responses, total_students=3)

    assert len(result) == 2

    q1 = result[0]
    assert q1.question_id == "Q1"
    assert q1.avg_score == pytest.approx(4.33, abs=0.01)
    assert q1.pct_attempted == pytest.approx(100.0)
    assert q1.pct_correct == pytest.approx(66.67, abs=0.01)

    q2 = result[1]
    assert q2.question_id == "Q2"
    assert q2.avg_score == pytest.approx(1.5, abs=0.01)
    assert q2.pct_attempted == pytest.approx(66.67, abs=0.01)
    assert q2.pct_correct == 0.0


# -- U-ANA-CS-10: Question difficulty — zero students ---------------------


def test_question_difficulty_zero_students() -> None:
    """U-ANA-CS-10: Zero total students → empty result."""
    result = compute_question_difficulty([], total_students=0)
    assert result == []


def test_question_difficulty_no_attempts() -> None:
    """U-ANA-CS-10b: All not attempted → avg=0, pct_correct=0."""
    responses = [
        QuestionResponse("Q1", score=0.0, max_score=5.0, attempted=False),
        QuestionResponse("Q1", score=0.0, max_score=5.0, attempted=False),
    ]
    result = compute_question_difficulty(responses, total_students=2)
    assert len(result) == 1
    assert result[0].avg_score == 0.0
    assert result[0].pct_attempted == 0.0
    assert result[0].pct_correct == 0.0
