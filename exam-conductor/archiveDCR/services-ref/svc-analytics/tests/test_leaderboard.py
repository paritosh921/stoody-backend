"""Unit tests for leaderboard generation — domain logic only, ZERO I/O.

Test IDs: U-ANA-LB-01 through U-ANA-LB-08
"""

from __future__ import annotations

from src.domain.leaderboard import (
    LeaderboardEntry,
    LeaderboardScope,
    ScoreEntry,
    generate_leaderboard,
)


# -- Helpers ---------------------------------------------------------------


def _entry(
    sid: str,
    name: str,
    score: float,
    percentile: float = 0.0,
    section: str = "",
    grade: str = "",
    institute: str = "",
) -> ScoreEntry:
    return ScoreEntry(
        student_id=sid,
        student_name=name,
        score=score,
        percentile=percentile,
        section=section,
        grade=grade,
        institute=institute,
    )


# -- U-ANA-LB-01: Empty input ---------------------------------------------


def test_empty_input() -> None:
    """U-ANA-LB-01: No scores → empty leaderboard."""
    result = generate_leaderboard([])
    assert result == []


# -- U-ANA-LB-02: Single student ------------------------------------------


def test_single_student() -> None:
    """U-ANA-LB-02: Single student gets rank 1."""
    scores = [_entry("s1", "Alice", 85.0, 100.0)]
    result = generate_leaderboard(scores)
    assert len(result) == 1
    assert result[0] == LeaderboardEntry(
        rank=1,
        student_id="s1",
        student_name="Alice",
        score=85.0,
        percentile=100.0,
    )


# -- U-ANA-LB-03: Sorted by score descending ------------------------------


def test_sorted_by_score_descending() -> None:
    """U-ANA-LB-03: Higher scores rank first."""
    scores = [
        _entry("s1", "Alice", 60.0, 0.0),
        _entry("s2", "Bob", 90.0, 100.0),
        _entry("s3", "Charlie", 75.0, 50.0),
    ]
    result = generate_leaderboard(scores)
    assert result[0].student_id == "s2"
    assert result[0].rank == 1
    assert result[1].student_id == "s3"
    assert result[1].rank == 2
    assert result[2].student_id == "s1"
    assert result[2].rank == 3


# -- U-ANA-LB-04: Ties get same rank, broken by name ----------------------


def test_ties_same_rank_alphabetical_tiebreak() -> None:
    """U-ANA-LB-04: Tied students share rank, sorted by name."""
    scores = [
        _entry("s1", "Charlie", 80.0, 50.0),
        _entry("s2", "Alice", 80.0, 50.0),
        _entry("s3", "Bob", 80.0, 50.0),
        _entry("s4", "Diana", 90.0, 100.0),
    ]
    result = generate_leaderboard(scores)

    # Diana first (highest score)
    assert result[0].student_id == "s4"
    assert result[0].rank == 1

    # Three tied students: alphabetical order Alice, Bob, Charlie
    assert result[1].student_name == "Alice"
    assert result[1].rank == 2
    assert result[2].student_name == "Bob"
    assert result[2].rank == 2
    assert result[3].student_name == "Charlie"
    assert result[3].rank == 2


# -- U-ANA-LB-05: Standard ranking (not dense after tie) ------------------


def test_standard_ranking_after_tie() -> None:
    """U-ANA-LB-05: After a tie, next rank skips appropriately.

    Scores: 90, 80, 80, 70 → ranks: 1, 2, 2, 4
    """
    scores = [
        _entry("s1", "Alice", 90.0),
        _entry("s2", "Bob", 80.0),
        _entry("s3", "Charlie", 80.0),
        _entry("s4", "Diana", 70.0),
    ]
    result = generate_leaderboard(scores)
    ranks = [r.rank for r in result]
    assert ranks == [1, 2, 2, 4]


# -- U-ANA-LB-06: Section scope filter ------------------------------------


def test_scope_filter_section() -> None:
    """U-ANA-LB-06: Only students in the matching section appear."""
    scores = [
        _entry("s1", "Alice", 90.0, section="A"),
        _entry("s2", "Bob", 80.0, section="B"),
        _entry("s3", "Charlie", 70.0, section="A"),
    ]
    result = generate_leaderboard(
        scores,
        scope=LeaderboardScope.SECTION,
        scope_value="A",
    )
    assert len(result) == 2
    assert all(
        r.student_id in ("s1", "s3") for r in result
    )


# -- U-ANA-LB-07: Grade scope filter --------------------------------------


def test_scope_filter_grade() -> None:
    """U-ANA-LB-07: Grade-level filtering works correctly."""
    scores = [
        _entry("s1", "Alice", 90.0, grade="10"),
        _entry("s2", "Bob", 85.0, grade="10"),
        _entry("s3", "Charlie", 95.0, grade="12"),
    ]
    result = generate_leaderboard(
        scores,
        scope=LeaderboardScope.GRADE,
        scope_value="10",
    )
    assert len(result) == 2
    assert result[0].student_id == "s1"
    assert result[1].student_id == "s2"


# -- U-ANA-LB-08: No scope_value means no filtering -----------------------


def test_no_scope_value_returns_all() -> None:
    """U-ANA-LB-08: Empty scope_value returns all students."""
    scores = [
        _entry("s1", "Alice", 90.0, section="A"),
        _entry("s2", "Bob", 80.0, section="B"),
    ]
    result = generate_leaderboard(
        scores,
        scope=LeaderboardScope.SECTION,
        scope_value="",
    )
    assert len(result) == 2
