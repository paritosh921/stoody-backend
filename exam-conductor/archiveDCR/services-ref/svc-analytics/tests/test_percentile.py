"""Unit tests for percentile computation — domain logic only, ZERO I/O.

Test IDs: U-ANA-PCTL-01 through U-ANA-PCTL-09
"""

from __future__ import annotations

import pytest

from src.domain.percentile import (
    StudentScore,
    compute_percentiles,
    percentile_for_score,
)


# -- Helpers ---------------------------------------------------------------


def _make_scores(
    values: list[float],
    prefix: str = "student_",
) -> list[StudentScore]:
    """Create StudentScore list from raw values."""
    return [
        StudentScore(student_id=f"{prefix}{i}", score=v)
        for i, v in enumerate(values)
    ]


# -- U-ANA-PCTL-01: Empty input -------------------------------------------


def test_empty_input() -> None:
    """U-ANA-PCTL-01: Empty list returns empty dict."""
    assert compute_percentiles([]) == {}


# -- U-ANA-PCTL-02: Single student ----------------------------------------


def test_single_student() -> None:
    """U-ANA-PCTL-02: Single student gets percentile 100.0."""
    scores = [StudentScore(student_id="s1", score=75.0)]
    result = compute_percentiles(scores)
    assert result == {"s1": 100.0}


# -- U-ANA-PCTL-03: Two students ------------------------------------------


def test_two_students() -> None:
    """U-ANA-PCTL-03: Two students — lower gets 0, higher gets 100."""
    scores = [
        StudentScore(student_id="low", score=30.0),
        StudentScore(student_id="high", score=90.0),
    ]
    result = compute_percentiles(scores)
    assert result["low"] == 0.0
    assert result["high"] == 100.0


# -- U-ANA-PCTL-04: 40 students with known scores -------------------------


def test_40_students_known_scores() -> None:
    """U-ANA-PCTL-04: 40 students, scores 1-40 — verify percentiles.

    With N=40 students and unique scores 1..40:
    percentile(score=k) = (k-1) / 39 * 100

    Student with score=1 → 0.0
    Student with score=40 → 100.0
    Student with score=20 → (19/39)*100 ≈ 48.72
    Student with score=21 → (20/39)*100 ≈ 51.28
    """
    values = [float(i) for i in range(1, 41)]  # 1.0 through 40.0
    scores = _make_scores(values)
    result = compute_percentiles(scores)

    assert len(result) == 40

    # Lowest score: percentile 0
    assert result["student_0"] == 0.0

    # Highest score: percentile 100
    assert result["student_39"] == 100.0

    # Score=20 (index 19): (19/39)*100 ≈ 48.72
    assert result["student_19"] == pytest.approx(48.72, abs=0.01)

    # Score=21 (index 20): (20/39)*100 ≈ 51.28
    assert result["student_20"] == pytest.approx(51.28, abs=0.01)

    # Monotonically increasing
    prev_pct = -1.0
    for i in range(40):
        pct = result[f"student_{i}"]
        assert pct >= prev_pct
        prev_pct = pct


# -- U-ANA-PCTL-05: All same score ----------------------------------------


def test_all_same_score() -> None:
    """U-ANA-PCTL-05: All students have the same score → same percentile.

    With N=5 students all scoring 80: below_count=0 for all.
    percentile = 0/4 * 100 = 0.0 for all.
    """
    scores = _make_scores([80.0] * 5)
    result = compute_percentiles(scores)
    assert len(result) == 5
    for pct in result.values():
        assert pct == 0.0


# -- U-ANA-PCTL-06: Ties with distinct scores -----------------------------


def test_ties_mixed_with_distinct() -> None:
    """U-ANA-PCTL-06: Some ties, some distinct — tied students match."""
    scores = [
        StudentScore(student_id="a", score=50.0),
        StudentScore(student_id="b", score=70.0),
        StudentScore(student_id="c", score=70.0),
        StudentScore(student_id="d", score=90.0),
    ]
    result = compute_percentiles(scores)

    # a(50): 0 below → 0/3*100 = 0.0
    assert result["a"] == 0.0
    # b(70) and c(70): 1 below → 1/3*100 ≈ 33.33
    assert result["b"] == result["c"]
    assert result["b"] == pytest.approx(33.33, abs=0.01)
    # d(90): 3 below → 3/3*100 = 100.0
    assert result["d"] == 100.0


# -- U-ANA-PCTL-07: Idempotency -------------------------------------------


def test_idempotency() -> None:
    """U-ANA-PCTL-07: Calling twice yields identical results."""
    scores = _make_scores([10.0, 20.0, 30.0, 40.0, 50.0])
    first = compute_percentiles(scores)
    second = compute_percentiles(scores)
    assert first == second


# -- U-ANA-PCTL-08: Order independence ------------------------------------


def test_input_order_independence() -> None:
    """U-ANA-PCTL-08: Shuffling input order doesn't change results."""
    ordered = _make_scores([10.0, 20.0, 30.0, 40.0, 50.0])
    shuffled = [ordered[3], ordered[0], ordered[4], ordered[1], ordered[2]]
    assert compute_percentiles(ordered) == compute_percentiles(shuffled)


# -- U-ANA-PCTL-09: percentile_for_score helper ---------------------------


def test_percentile_for_score_single() -> None:
    """U-ANA-PCTL-09: percentile_for_score with single-element list."""
    assert percentile_for_score(50.0, [50.0]) == 100.0


def test_percentile_for_score_population() -> None:
    """U-ANA-PCTL-09b: percentile_for_score against a population."""
    population = [10.0, 20.0, 30.0, 40.0, 50.0]
    # 30.0: 2 below, N=5 → 2/4*100 = 50.0
    assert percentile_for_score(30.0, population) == 50.0


def test_percentile_for_score_empty() -> None:
    """U-ANA-PCTL-09c: percentile_for_score with empty population."""
    assert percentile_for_score(50.0, []) == 0.0
