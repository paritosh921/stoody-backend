"""U-SCR-01 .. U-SCR-10: Score FSM unit tests.

Tests every valid transition and rejects every invalid one.
Domain-only -- no DB, no network.
"""

import pytest

from src.domain.score_fsm import (
    ScoreState,
    ScoreTransitionError,
    TransitionResult,
    allowed_transitions,
    is_mutable,
    transition,
)


# ---- Valid transitions -------------------------------------------------------

class TestValidTransitions:
    """Happy-path transitions through the lifecycle."""

    def test_ai_draft_to_teacher_reviewed(self) -> None:
        r = transition(ScoreState.AI_DRAFT, ScoreState.TEACHER_REVIEWED)
        assert r == TransitionResult(ScoreState.AI_DRAFT, ScoreState.TEACHER_REVIEWED)

    def test_ai_draft_to_finalized(self) -> None:
        r = transition(ScoreState.AI_DRAFT, ScoreState.FINALIZED)
        assert r.new_state == ScoreState.FINALIZED

    def test_teacher_reviewed_to_finalized(self) -> None:
        r = transition(ScoreState.TEACHER_REVIEWED, ScoreState.FINALIZED)
        assert r.new_state == ScoreState.FINALIZED

    def test_teacher_reviewed_self_loop(self) -> None:
        """Multiple overrides keep state in teacher_reviewed."""
        r = transition(ScoreState.TEACHER_REVIEWED, ScoreState.TEACHER_REVIEWED)
        assert r.new_state == ScoreState.TEACHER_REVIEWED

    def test_finalized_to_published(self) -> None:
        r = transition(ScoreState.FINALIZED, ScoreState.PUBLISHED)
        assert r.new_state == ScoreState.PUBLISHED

    def test_published_to_objection_window(self) -> None:
        r = transition(ScoreState.PUBLISHED, ScoreState.OBJECTION_WINDOW)
        assert r.new_state == ScoreState.OBJECTION_WINDOW

    def test_objection_window_to_locked(self) -> None:
        r = transition(ScoreState.OBJECTION_WINDOW, ScoreState.LOCKED)
        assert r.new_state == ScoreState.LOCKED

    def test_objection_window_back_to_teacher_reviewed(self) -> None:
        """Re-scoring after objection returns to teacher_reviewed."""
        r = transition(ScoreState.OBJECTION_WINDOW, ScoreState.TEACHER_REVIEWED)
        assert r.new_state == ScoreState.TEACHER_REVIEWED


# ---- Invalid transitions -----------------------------------------------------

class TestInvalidTransitions:
    """Every backward / impossible jump must raise ScoreTransitionError."""

    @pytest.mark.parametrize(
        "current, target",
        [
            (ScoreState.AI_DRAFT, ScoreState.PUBLISHED),
            (ScoreState.AI_DRAFT, ScoreState.LOCKED),
            (ScoreState.AI_DRAFT, ScoreState.OBJECTION_WINDOW),
            (ScoreState.TEACHER_REVIEWED, ScoreState.AI_DRAFT),
            (ScoreState.TEACHER_REVIEWED, ScoreState.PUBLISHED),
            (ScoreState.FINALIZED, ScoreState.AI_DRAFT),
            (ScoreState.FINALIZED, ScoreState.TEACHER_REVIEWED),
            (ScoreState.FINALIZED, ScoreState.LOCKED),
            (ScoreState.PUBLISHED, ScoreState.AI_DRAFT),
            (ScoreState.PUBLISHED, ScoreState.FINALIZED),
            (ScoreState.PUBLISHED, ScoreState.LOCKED),
            (ScoreState.OBJECTION_WINDOW, ScoreState.AI_DRAFT),
            (ScoreState.OBJECTION_WINDOW, ScoreState.FINALIZED),
            (ScoreState.OBJECTION_WINDOW, ScoreState.PUBLISHED),
            (ScoreState.LOCKED, ScoreState.AI_DRAFT),
            (ScoreState.LOCKED, ScoreState.TEACHER_REVIEWED),
            (ScoreState.LOCKED, ScoreState.FINALIZED),
            (ScoreState.LOCKED, ScoreState.PUBLISHED),
            (ScoreState.LOCKED, ScoreState.OBJECTION_WINDOW),
            (ScoreState.LOCKED, ScoreState.LOCKED),
        ],
    )
    def test_invalid(self, current: ScoreState, target: ScoreState) -> None:
        with pytest.raises(ScoreTransitionError) as exc_info:
            transition(current, target)
        assert exc_info.value.current == current
        assert exc_info.value.requested == target


# ---- Helper functions --------------------------------------------------------

class TestAllowedTransitions:
    def test_locked_has_none(self) -> None:
        assert allowed_transitions(ScoreState.LOCKED) == frozenset()

    def test_ai_draft_has_two(self) -> None:
        assert allowed_transitions(ScoreState.AI_DRAFT) == frozenset(
            {ScoreState.TEACHER_REVIEWED, ScoreState.FINALIZED}
        )


class TestIsMutable:
    @pytest.mark.parametrize(
        "state, expected",
        [
            (ScoreState.AI_DRAFT, True),
            (ScoreState.TEACHER_REVIEWED, True),
            (ScoreState.FINALIZED, False),
            (ScoreState.PUBLISHED, False),
            (ScoreState.OBJECTION_WINDOW, True),
            (ScoreState.LOCKED, False),
        ],
    )
    def test_mutability(self, state: ScoreState, expected: bool) -> None:
        assert is_mutable(state) is expected
