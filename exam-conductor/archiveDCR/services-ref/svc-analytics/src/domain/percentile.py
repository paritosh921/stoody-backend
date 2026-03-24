"""Percentile computation — pure domain logic.

Computes percentile rank for each student in a score set.
Percentile = (number of scores below this score) / (total - 1) * 100.
For a single student, percentile is 100.0.
Ties receive the same percentile.

This module is ZERO I/O -- no asyncio, no DB, no network imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StudentScore:
    """A single student's score for percentile computation."""

    student_id: str
    score: float


def compute_percentiles(
    scores: list[StudentScore],
) -> dict[str, float]:
    """Map each student_id to their percentile rank (0-100).

    Algorithm:
        percentile(s) = count(scores < s) / (N - 1) * 100
        When N=1, the single student gets percentile 100.0.
        Ties: students with the same score get the same percentile.

    Returns:
        dict mapping student_id -> percentile (float, 0-100).

    Idempotent: calling with the same input always produces the same
    output regardless of invocation count or ordering.
    """
    n = len(scores)
    if n == 0:
        return {}
    if n == 1:
        return {scores[0].student_id: 100.0}

    # Sort scores ascending to count "below" efficiently
    sorted_scores = sorted(scores, key=lambda s: s.score)

    # Build a map: score_value -> count of scores strictly below it
    # Walk through sorted list, tracking how many distinct groups below
    below_count: dict[float, int] = {}
    i = 0
    while i < n:
        val = sorted_scores[i].score
        # All entries at index < i have score <= val.
        # Count of scores strictly below val = index of first occurrence of val.
        below_count[val] = i
        # Skip all duplicates
        j = i + 1
        while j < n and sorted_scores[j].score == val:
            j += 1
        i = j

    divisor = n - 1
    result: dict[str, float] = {}
    for entry in scores:
        pct = below_count[entry.score] / divisor * 100.0
        # Round to 2 decimal places for clean output
        result[entry.student_id] = round(pct, 2)

    return result


def percentile_for_score(
    score: float,
    all_scores: list[float],
) -> float:
    """Compute percentile for a single score against a population.

    Useful for computing a single student's percentile without
    rebuilding the full map.
    """
    n = len(all_scores)
    if n == 0:
        return 0.0
    if n == 1:
        return 100.0
    below = sum(1 for s in all_scores if s < score)
    return round(below / (n - 1) * 100.0, 2)
