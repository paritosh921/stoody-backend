"""Unit tests for hub_fsm.py — pure FSM logic, ZERO I/O.

Covers:
- All valid transitions in the happy path
- Every state reachable from 'created'
- Cancel from every non-terminal state
- Invalid transition raises InvalidTransition
- Unknown state raises InvalidState
- valid_events() returns correct events per state
- Terminal state detection
"""

from __future__ import annotations

import pytest

from src.hub_fsm import (
    ALL_STATES,
    ARMED,
    CANCELLED,
    CREATED,
    DONGLE_ACTIVATION,
    EVT_ACTIVATE_DONGLES,
    EVT_ARM,
    EVT_CANCEL,
    EVT_START_TIMER,
    EVT_START_UPLOAD,
    EVT_SYNC_ALL_COMPLETE,
    EVT_SYNC_PARTIAL,
    EVT_TIMER_EXPIRED,
    EVT_UPLOAD_DONE,
    PEN_SYNC,
    SYNC_COMPLETE,
    SYNC_PARTIAL,
    TERMINAL_STATES,
    TIMER_RUNNING,
    UPLOAD_COMPLETE,
    UPLOADING,
    InvalidState,
    InvalidTransition,
    TransitionResult,
    is_terminal,
    transition,
    valid_events,
)


# ===================================================================
# Happy path — full lifecycle
# ===================================================================

class TestHappyPath:
    """Walk the full exam lifecycle: created -> upload_complete."""

    def test_created_to_armed(self) -> None:
        r = transition(CREATED, EVT_ARM)
        assert r == TransitionResult(CREATED, EVT_ARM, ARMED)

    def test_armed_to_timer_running(self) -> None:
        r = transition(ARMED, EVT_START_TIMER)
        assert r.new_state == TIMER_RUNNING

    def test_timer_running_to_dongle_activation(self) -> None:
        r = transition(TIMER_RUNNING, EVT_TIMER_EXPIRED)
        assert r.new_state == DONGLE_ACTIVATION

    def test_dongle_activation_to_pen_sync(self) -> None:
        r = transition(DONGLE_ACTIVATION, EVT_ACTIVATE_DONGLES)
        assert r.new_state == PEN_SYNC

    def test_pen_sync_to_sync_complete(self) -> None:
        r = transition(PEN_SYNC, EVT_SYNC_ALL_COMPLETE)
        assert r.new_state == SYNC_COMPLETE

    def test_pen_sync_to_sync_partial(self) -> None:
        r = transition(PEN_SYNC, EVT_SYNC_PARTIAL)
        assert r.new_state == SYNC_PARTIAL

    def test_sync_complete_to_uploading(self) -> None:
        r = transition(SYNC_COMPLETE, EVT_START_UPLOAD)
        assert r.new_state == UPLOADING

    def test_sync_partial_to_uploading(self) -> None:
        r = transition(SYNC_PARTIAL, EVT_START_UPLOAD)
        assert r.new_state == UPLOADING

    def test_uploading_to_upload_complete(self) -> None:
        r = transition(UPLOADING, EVT_UPLOAD_DONE)
        assert r.new_state == UPLOAD_COMPLETE

    def test_full_lifecycle(self) -> None:
        """Walk the entire happy path sequentially."""
        events = [
            EVT_ARM, EVT_START_TIMER, EVT_TIMER_EXPIRED,
            EVT_ACTIVATE_DONGLES, EVT_SYNC_ALL_COMPLETE,
            EVT_START_UPLOAD, EVT_UPLOAD_DONE,
        ]
        state = CREATED
        for evt in events:
            r = transition(state, evt)
            state = r.new_state
        assert state == UPLOAD_COMPLETE


# ===================================================================
# Every state is reachable
# ===================================================================

class TestReachability:
    """Every state in ALL_STATES is reachable from 'created'."""

    def _walk_to(self, target: str) -> str:
        """Return the state reached by walking to *target*."""
        paths: dict[str, list[str]] = {
            CREATED: [],
            ARMED: [EVT_ARM],
            TIMER_RUNNING: [EVT_ARM, EVT_START_TIMER],
            DONGLE_ACTIVATION: [
                EVT_ARM, EVT_START_TIMER, EVT_TIMER_EXPIRED,
            ],
            PEN_SYNC: [
                EVT_ARM, EVT_START_TIMER, EVT_TIMER_EXPIRED,
                EVT_ACTIVATE_DONGLES,
            ],
            SYNC_COMPLETE: [
                EVT_ARM, EVT_START_TIMER, EVT_TIMER_EXPIRED,
                EVT_ACTIVATE_DONGLES, EVT_SYNC_ALL_COMPLETE,
            ],
            SYNC_PARTIAL: [
                EVT_ARM, EVT_START_TIMER, EVT_TIMER_EXPIRED,
                EVT_ACTIVATE_DONGLES, EVT_SYNC_PARTIAL,
            ],
            UPLOADING: [
                EVT_ARM, EVT_START_TIMER, EVT_TIMER_EXPIRED,
                EVT_ACTIVATE_DONGLES, EVT_SYNC_ALL_COMPLETE,
                EVT_START_UPLOAD,
            ],
            UPLOAD_COMPLETE: [
                EVT_ARM, EVT_START_TIMER, EVT_TIMER_EXPIRED,
                EVT_ACTIVATE_DONGLES, EVT_SYNC_ALL_COMPLETE,
                EVT_START_UPLOAD, EVT_UPLOAD_DONE,
            ],
            CANCELLED: [EVT_CANCEL],
        }
        state = CREATED
        for evt in paths[target]:
            r = transition(state, evt)
            state = r.new_state
        return state

    @pytest.mark.parametrize("target", sorted(ALL_STATES))
    def test_state_reachable(self, target: str) -> None:
        assert self._walk_to(target) == target


# ===================================================================
# Cancel from every non-terminal state
# ===================================================================

class TestCancel:
    """Cancel event is valid from every non-terminal state."""

    NON_TERMINAL = sorted(ALL_STATES - TERMINAL_STATES)

    @pytest.mark.parametrize("state", NON_TERMINAL)
    def test_cancel_from_state(self, state: str) -> None:
        r = transition(state, EVT_CANCEL)
        assert r.new_state == CANCELLED


# ===================================================================
# Invalid transitions
# ===================================================================

class TestInvalidTransitions:

    def test_cannot_skip_states(self) -> None:
        with pytest.raises(InvalidTransition):
            transition(CREATED, EVT_START_TIMER)

    def test_cannot_move_backward(self) -> None:
        with pytest.raises(InvalidTransition):
            transition(ARMED, EVT_ARM)

    def test_terminal_cannot_transition(self) -> None:
        with pytest.raises(InvalidTransition):
            transition(UPLOAD_COMPLETE, EVT_ARM)

    def test_cancelled_is_terminal(self) -> None:
        with pytest.raises(InvalidTransition):
            transition(CANCELLED, EVT_ARM)

    def test_unknown_event(self) -> None:
        with pytest.raises(InvalidTransition):
            transition(CREATED, "nonexistent_event")


# ===================================================================
# Unknown state
# ===================================================================

class TestInvalidState:

    def test_unknown_state_in_transition(self) -> None:
        with pytest.raises(InvalidState):
            transition("bogus", EVT_ARM)

    def test_unknown_state_in_valid_events(self) -> None:
        with pytest.raises(InvalidState):
            valid_events("bogus")


# ===================================================================
# valid_events() helper
# ===================================================================

class TestValidEvents:

    def test_created_events(self) -> None:
        evts = valid_events(CREATED)
        assert EVT_ARM in evts
        assert EVT_CANCEL in evts

    def test_terminal_no_events(self) -> None:
        assert valid_events(UPLOAD_COMPLETE) == []
        assert valid_events(CANCELLED) == []

    def test_pen_sync_has_two_outcomes(self) -> None:
        evts = valid_events(PEN_SYNC)
        assert EVT_SYNC_ALL_COMPLETE in evts
        assert EVT_SYNC_PARTIAL in evts
        assert EVT_CANCEL in evts


# ===================================================================
# Terminal detection
# ===================================================================

class TestTerminal:

    def test_upload_complete_is_terminal(self) -> None:
        assert is_terminal(UPLOAD_COMPLETE) is True

    def test_cancelled_is_terminal(self) -> None:
        assert is_terminal(CANCELLED) is True

    def test_created_is_not_terminal(self) -> None:
        assert is_terminal(CREATED) is False

    def test_uploading_is_not_terminal(self) -> None:
        assert is_terminal(UPLOADING) is False


# ===================================================================
# TransitionResult immutability
# ===================================================================

class TestTransitionResult:

    def test_frozen(self) -> None:
        r = transition(CREATED, EVT_ARM)
        with pytest.raises(AttributeError):
            r.new_state = "something"  # type: ignore[misc]

    def test_fields(self) -> None:
        r = transition(CREATED, EVT_ARM)
        assert r.old_state == CREATED
        assert r.event == EVT_ARM
        assert r.new_state == ARMED
