"""Unit tests for exam lifecycle FSM — domain layer, no I/O.

Test IDs: U-ORCH-FSM-01 through U-ORCH-FSM-10.
"""

from __future__ import annotations

import pytest

from src.domain.exam_fsm import (
    ExamState,
    InvalidTransition,
    TransitionResult,
    get_allowed_transitions,
    transition,
)


# ---------------------------------------------------------------------------
# U-ORCH-FSM-01: Happy-path linear progression
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Full exam lifecycle from created to locked."""

    def test_created_to_armed(self) -> None:
        result = transition("created", "armed")
        assert result == TransitionResult(ExamState.CREATED, ExamState.ARMED)

    def test_armed_to_timer_running(self) -> None:
        result = transition("armed", "timer_running")
        assert result.to_state == ExamState.TIMER_RUNNING

    def test_timer_running_to_sync_pending(self) -> None:
        result = transition("timer_running", "sync_pending")
        assert result.to_state == ExamState.SYNC_PENDING

    def test_sync_pending_to_scoring(self) -> None:
        result = transition("sync_pending", "scoring")
        assert result.to_state == ExamState.SCORING

    def test_scoring_to_finalized(self) -> None:
        result = transition("scoring", "finalized")
        assert result.to_state == ExamState.FINALIZED

    def test_finalized_to_published(self) -> None:
        result = transition("finalized", "published")
        assert result.to_state == ExamState.PUBLISHED

    def test_published_to_locked(self) -> None:
        result = transition("published", "locked")
        assert result.to_state == ExamState.LOCKED

    def test_full_lifecycle(self) -> None:
        """Walk every state in order."""
        states = [
            "created", "armed", "timer_running", "sync_pending",
            "scoring", "finalized", "published", "locked",
        ]
        current = states[0]
        for target in states[1:]:
            result = transition(current, target)
            assert result.to_state.value == target
            current = target


# ---------------------------------------------------------------------------
# U-ORCH-FSM-02: Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_from_created(self) -> None:
        result = transition("created", "cancelled")
        assert result.to_state == ExamState.CANCELLED

    def test_cancel_from_armed(self) -> None:
        result = transition("armed", "cancelled")
        assert result.to_state == ExamState.CANCELLED

    def test_cancel_from_timer_running(self) -> None:
        result = transition("timer_running", "cancelled")
        assert result.to_state == ExamState.CANCELLED


# ---------------------------------------------------------------------------
# U-ORCH-FSM-03: Invalid transitions
# ---------------------------------------------------------------------------


class TestInvalidTransitions:
    def test_created_to_locked_rejected(self) -> None:
        with pytest.raises(InvalidTransition) as exc_info:
            transition("created", "locked")
        assert "created" in str(exc_info.value)
        assert "locked" in str(exc_info.value)

    def test_locked_to_anything(self) -> None:
        for target in ["created", "armed", "cancelled", "published"]:
            with pytest.raises(InvalidTransition):
                transition("locked", target)

    def test_cancelled_to_anything(self) -> None:
        for target in ["created", "armed", "timer_running"]:
            with pytest.raises(InvalidTransition):
                transition("cancelled", target)

    def test_skip_states(self) -> None:
        with pytest.raises(InvalidTransition):
            transition("created", "timer_running")

    def test_backward_transition(self) -> None:
        with pytest.raises(InvalidTransition):
            transition("armed", "created")

    def test_sync_pending_cannot_cancel(self) -> None:
        with pytest.raises(InvalidTransition):
            transition("sync_pending", "cancelled")

    def test_scoring_cannot_cancel(self) -> None:
        with pytest.raises(InvalidTransition):
            transition("scoring", "cancelled")


# ---------------------------------------------------------------------------
# U-ORCH-FSM-04: Unknown states
# ---------------------------------------------------------------------------


class TestUnknownStates:
    def test_unknown_source(self) -> None:
        with pytest.raises(ValueError, match="Unknown exam state"):
            transition("nonexistent", "armed")

    def test_unknown_target(self) -> None:
        with pytest.raises(ValueError, match="Unknown exam state"):
            transition("created", "nonexistent")


# ---------------------------------------------------------------------------
# U-ORCH-FSM-05: get_allowed_transitions
# ---------------------------------------------------------------------------


class TestAllowedTransitions:
    def test_from_created(self) -> None:
        allowed = get_allowed_transitions("created")
        assert set(allowed) == {"armed", "cancelled"}

    def test_from_locked(self) -> None:
        assert get_allowed_transitions("locked") == []

    def test_from_cancelled(self) -> None:
        assert get_allowed_transitions("cancelled") == []

    def test_unknown_returns_empty(self) -> None:
        assert get_allowed_transitions("bogus") == []
