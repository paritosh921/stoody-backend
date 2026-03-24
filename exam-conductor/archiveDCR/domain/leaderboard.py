"""Leaderboard generation — pure domain logic.

Generates ranked leaderboard entries from a set of student scores.
Supports scope filtering (section, grade, institute).
Ties are broken by student name (alphabetical, ascending).

This module is ZERO I/O -- no asyncio, no DB, no network imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LeaderboardScope(str, Enum):
    """Scope levels for leaderboard generation."""

    SECTION = "section"
    GRADE = "grade"
    INSTITUTE = "institute"


@dataclass(frozen=True, slots=True)
class ScoreEntry:
    """Input score record for leaderboard generation."""

    student_id: str
    student_name: str
    score: float
    percentile: float
    section: str = ""
    grade: str = ""
    institute: str = ""


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """A single row in the leaderboard output."""

    rank: int
    student_id: str
    student_name: str
    score: float
    percentile: float


def generate_leaderboard(
    scores: list[ScoreEntry],
    scope: LeaderboardScope = LeaderboardScope.INSTITUTE,
    scope_value: str = "",
) -> list[LeaderboardEntry]:
    """Generate a ranked leaderboard from score entries.

    Args:
        scores: List of student score entries.
        scope: The scope level to filter by.
        scope_value: The value to filter on (e.g. section name,
            grade name). Empty string means no filtering (all).

    Returns:
        Sorted list of LeaderboardEntry with dense ranking.
        Ties (same score) receive the same rank and are broken
        alphabetically by student_name.

    Idempotent: same inputs always produce the same output.
    """
    # Filter by scope if a scope_value is provided
    filtered = _filter_by_scope(scores, scope, scope_value)

    # Sort: primary = score descending, secondary = name ascending
    sorted_entries = sorted(
        filtered,
        key=lambda e: (-e.score, e.student_name),
    )

    return _assign_ranks(sorted_entries)


def _filter_by_scope(
    scores: list[ScoreEntry],
    scope: LeaderboardScope,
    scope_value: str,
) -> list[ScoreEntry]:
    """Filter entries by the given scope and value."""
    if not scope_value:
        return scores

    attr_map = {
        LeaderboardScope.SECTION: "section",
        LeaderboardScope.GRADE: "grade",
        LeaderboardScope.INSTITUTE: "institute",
    }
    attr = attr_map[scope]
    return [e for e in scores if getattr(e, attr) == scope_value]


def _assign_ranks(
    sorted_entries: list[ScoreEntry],
) -> list[LeaderboardEntry]:
    """Assign dense ranks to pre-sorted entries.

    Students with equal scores receive the same rank.
    The next distinct score gets rank = previous_rank + 1 (dense).
    """
    if not sorted_entries:
        return []

    result: list[LeaderboardEntry] = []
    current_rank = 1

    for i, entry in enumerate(sorted_entries):
        if i > 0 and entry.score < sorted_entries[i - 1].score:
            current_rank = i + 1

        result.append(
            LeaderboardEntry(
                rank=current_rank,
                student_id=entry.student_id,
                student_name=entry.student_name,
                score=entry.score,
                percentile=entry.percentile,
            )
        )

    return result
