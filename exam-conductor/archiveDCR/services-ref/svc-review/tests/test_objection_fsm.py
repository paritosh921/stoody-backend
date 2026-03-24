"""Unit tests for domain/objection_fsm.py — ZERO I/O, pure logic.

Test IDs: U-REV-FSM-01 through U-REV-FSM-06
"""

import pytest

from src.domain.objection_fsm import (
    InvalidTransitionError,
    ObjectionEvent,
    ObjectionState,
    TransitionResult,
    is_terminal,
    transition,
    valid_events_for,
)


# -- U-REV-FSM-01: Happy path — full lifecycle --------------------------------


class TestHappyPath:
    """U-REV-FSM-01: filed -> assigned -> reviewing -> resolved."""

    def test_filed_to_assigned(self):
        """U-REV-FSM-01a: ASSIGN event moves filed -> assigned."""
        result = transition(ObjectionState.FILED, ObjectionEvent.ASSIGN)
        assert isinstance(result, TransitionResult)
        assert result.old_state == ObjectionState.FILED
        assert result.new_state == ObjectionState.ASSIGNED
        assert result.event == ObjectionEvent.ASSIGN

    def test_assigned_to_reviewing(self):
        """U-REV-FSM-01b: START_REVIEW event moves assigned -> reviewing."""
        result = transition(ObjectionState.ASSIGNED, ObjectionEvent.START_REVIEW)
        assert result.new_state == ObjectionState.REVIEWING

    def test_reviewing_to_resolved(self):
        """U-REV-FSM-01c: RESOLVE event moves reviewing -> resolved."""
        result = transition(ObjectionState.REVIEWING, ObjectionEvent.RESOLVE)
        assert result.new_state == ObjectionState.RESOLVED


# -- U-REV-FSM-02: Escalation path -------------------------------------------


class TestEscalation:
    """U-REV-FSM-02: reviewing -> escalated -> assigned (re-review)."""

    def test_reviewing_to_escalated(self):
        """U-REV-FSM-02a: ESCALATE event moves reviewing -> escalated."""
        result = transition(ObjectionState.REVIEWING, ObjectionEvent.ESCALATE)
        assert result.new_state == ObjectionState.ESCALATED

    def test_escalated_to_assigned(self):
        """U-REV-FSM-02b: ASSIGN event moves escalated -> assigned."""
        result = transition(ObjectionState.ESCALATED, ObjectionEvent.ASSIGN)
        assert result.new_state == ObjectionState.ASSIGNED


# -- U-REV-FSM-03: Invalid transitions raise ----------------------------------


class TestInvalidTransitions:
    """U-REV-FSM-03: Invalid transitions raise InvalidTransitionError."""

    def test_filed_cannot_resolve(self):
        """U-REV-FSM-03a: Cannot resolve directly from filed."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            transition(ObjectionState.FILED, ObjectionEvent.RESOLVE)
        assert exc_info.value.current == ObjectionState.FILED
        assert exc_info.value.event == ObjectionEvent.RESOLVE

    def test_filed_cannot_escalate(self):
        """U-REV-FSM-03b: Cannot escalate directly from filed."""
        with pytest.raises(InvalidTransitionError):
            transition(ObjectionState.FILED, ObjectionEvent.ESCALATE)

    def test_assigned_cannot_resolve(self):
        """U-REV-FSM-03c: Cannot resolve directly from assigned."""
        with pytest.raises(InvalidTransitionError):
            transition(ObjectionState.ASSIGNED, ObjectionEvent.RESOLVE)

    def test_assigned_cannot_escalate(self):
        """U-REV-FSM-03d: Cannot escalate directly from assigned."""
        with pytest.raises(InvalidTransitionError):
            transition(ObjectionState.ASSIGNED, ObjectionEvent.ESCALATE)

    def test_resolved_cannot_transition(self):
        """U-REV-FSM-03e: Resolved is terminal — no outgoing transitions."""
        with pytest.raises(InvalidTransitionError):
            transition(ObjectionState.RESOLVED, ObjectionEvent.ASSIGN)

    def test_resolved_cannot_escalate(self):
        """U-REV-FSM-03f: Resolved cannot be escalated."""
        with pytest.raises(InvalidTransitionError):
            transition(ObjectionState.RESOLVED, ObjectionEvent.ESCALATE)

    def test_escalated_cannot_resolve_directly(self):
        """U-REV-FSM-03g: Escalated cannot resolve — must re-assign first."""
        with pytest.raises(InvalidTransitionError):
            transition(ObjectionState.ESCALATED, ObjectionEvent.RESOLVE)


# -- U-REV-FSM-04: valid_events_for -------------------------------------------


class TestValidEvents:
    """U-REV-FSM-04: valid_events_for returns correct events per state."""

    def test_filed_allows_assign(self):
        """U-REV-FSM-04a: Filed state allows only ASSIGN."""
        events = valid_events_for(ObjectionState.FILED)
        assert events == [ObjectionEvent.ASSIGN]

    def test_assigned_allows_start_review(self):
        """U-REV-FSM-04b: Assigned state allows only START_REVIEW."""
        events = valid_events_for(ObjectionState.ASSIGNED)
        assert events == [ObjectionEvent.START_REVIEW]

    def test_reviewing_allows_resolve_and_escalate(self):
        """U-REV-FSM-04c: Reviewing allows RESOLVE and ESCALATE."""
        events = valid_events_for(ObjectionState.REVIEWING)
        assert set(events) == {ObjectionEvent.RESOLVE, ObjectionEvent.ESCALATE}

    def test_escalated_allows_assign(self):
        """U-REV-FSM-04d: Escalated allows only ASSIGN (re-assign)."""
        events = valid_events_for(ObjectionState.ESCALATED)
        assert events == [ObjectionEvent.ASSIGN]

    def test_resolved_has_no_events(self):
        """U-REV-FSM-04e: Resolved is terminal — no valid events."""
        events = valid_events_for(ObjectionState.RESOLVED)
        assert events == []


# -- U-REV-FSM-05: is_terminal ------------------------------------------------


class TestTerminal:
    """U-REV-FSM-05: Terminal state detection."""

    def test_resolved_is_terminal(self):
        """U-REV-FSM-05a: Resolved is terminal."""
        assert is_terminal(ObjectionState.RESOLVED) is True

    def test_filed_is_not_terminal(self):
        """U-REV-FSM-05b: Filed is not terminal."""
        assert is_terminal(ObjectionState.FILED) is False

    def test_escalated_is_not_terminal(self):
        """U-REV-FSM-05c: Escalated is not terminal (can re-assign)."""
        assert is_terminal(ObjectionState.ESCALATED) is False


# -- U-REV-FSM-06: Error message quality --------------------------------------


class TestErrorMessages:
    """U-REV-FSM-06: Error messages contain useful context."""

    def test_error_message_includes_state_and_event(self):
        """U-REV-FSM-06a: Error message includes current state and event."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            transition(ObjectionState.FILED, ObjectionEvent.RESOLVE)
        msg = str(exc_info.value)
        assert "filed" in msg
        assert "resolved" in msg

    def test_error_exposes_fields(self):
        """U-REV-FSM-06b: Exception exposes current and event attributes."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            transition(ObjectionState.RESOLVED, ObjectionEvent.ASSIGN)
        assert exc_info.value.current == ObjectionState.RESOLVED
        assert exc_info.value.event == ObjectionEvent.ASSIGN
