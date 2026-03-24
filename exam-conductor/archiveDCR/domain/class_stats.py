"""Class-level statistics — pure domain logic.

Computes aggregate statistics for an exam: mean, median, standard
deviation, pass rate, min, max, count. Also computes per-question
difficulty analysis.

This module is ZERO I/O -- no asyncio, no DB, no network imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassStats:
    """Aggregate exam statistics."""

    mean: float
    median: float
    std_dev: float
    pass_pct: float
    min_score: float
    max_score: float
    count: int


@dataclass(frozen=True, slots=True)
class QuestionDifficulty:
    """Per-question difficulty analysis."""

    question_id: str
    avg_score: float
    pct_attempted: float
    pct_correct: float


@dataclass(frozen=True, slots=True)
class QuestionResponse:
    """A single student's response to one question."""

    question_id: str
    score: float
    max_score: float
    attempted: bool


def compute_class_stats(
    scores: list[float],
    pass_threshold: float = 40.0,
) -> ClassStats:
    """Compute aggregate class statistics from a list of total scores.

    Args:
        scores: List of student total scores (as percentages 0-100,
            or raw scores — the pass_threshold must match the scale).
        pass_threshold: Minimum score to be considered passing.

    Returns:
        ClassStats with mean, median, std_dev, pass_pct, min, max, count.
        For an empty list, all values are 0.0 with count=0.
    """
    n = len(scores)
    if n == 0:
        return ClassStats(
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            pass_pct=0.0,
            min_score=0.0,
            max_score=0.0,
            count=0,
        )

    sorted_scores = sorted(scores)
    total = sum(sorted_scores)
    mean = total / n

    # Median
    if n % 2 == 1:
        median = sorted_scores[n // 2]
    else:
        mid = n // 2
        median = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0

    # Population standard deviation
    variance = sum((s - mean) ** 2 for s in sorted_scores) / n
    std_dev = math.sqrt(variance)

    # Pass percentage
    passing = sum(1 for s in sorted_scores if s >= pass_threshold)
    pass_pct = (passing / n) * 100.0

    return ClassStats(
        mean=round(mean, 2),
        median=round(median, 2),
        std_dev=round(std_dev, 2),
        pass_pct=round(pass_pct, 2),
        min_score=sorted_scores[0],
        max_score=sorted_scores[-1],
        count=n,
    )


def compute_question_difficulty(
    responses: list[QuestionResponse],
    total_students: int,
) -> list[QuestionDifficulty]:
    """Compute per-question difficulty from student responses.

    Args:
        responses: All question-level responses across students.
        total_students: Total number of students (for % attempted).

    Returns:
        List of QuestionDifficulty sorted by question_id.
    """
    if total_students == 0:
        return []

    # Group by question_id
    by_question: dict[str, list[QuestionResponse]] = {}
    for r in responses:
        by_question.setdefault(r.question_id, []).append(r)

    result: list[QuestionDifficulty] = []
    for qid in sorted(by_question.keys()):
        qr_list = by_question[qid]
        attempted_count = sum(1 for r in qr_list if r.attempted)
        correct_count = sum(
            1 for r in qr_list
            if r.attempted and r.max_score > 0 and r.score >= r.max_score
        )
        scores = [r.score for r in qr_list if r.attempted]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        pct_attempted = (attempted_count / total_students) * 100.0
        pct_correct = (
            (correct_count / attempted_count) * 100.0
            if attempted_count > 0
            else 0.0
        )

        result.append(
            QuestionDifficulty(
                question_id=qid,
                avg_score=round(avg_score, 2),
                pct_attempted=round(pct_attempted, 2),
                pct_correct=round(pct_correct, 2),
            )
        )

    return result
