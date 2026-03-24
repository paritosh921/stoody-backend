"""Objection lifecycle FSM — ZERO I/O, pure domain logic.

States: filed -> assigned -> reviewing -> resolved
                              reviewing -> escalated

The ``transition`` function is the single authority for valid state
changes.  All I/O layers call this before persisting.

Test IDs: U-REV-FSM-01 through U-REV-FSM-06
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ObjectionState(StrEnum):
    """All valid objection states."""

    FILED = "filed"
    ASSIGNED = "assigned"
    REVIEWING = "reviewing"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ObjectionEvent(StrEnum):
    """Actions that trigger state transitions."""

    ASSIGN = "assigned"
    START_REVIEW = "reviewing"
    RESOLVE = "resolved"
    ESCALATE = "escalated"


# Exhaustive transition table.
# Key: (current_state, event) -> new_state
_TRANSITIONS: dict[tuple[ObjectionState, ObjectionEvent], ObjectionState] = {
    (ObjectionState.FILED, ObjectionEvent.ASSIGN): ObjectionState.ASSIGNED,
    (ObjectionState.ASSIGNED, ObjectionEvent.START_REVIEW): ObjectionState.REVIEWING,
    (ObjectionState.REVIEWING, ObjectionEvent.RESOLVE): ObjectionState.RESOLVED,
    (ObjectionState.REVIEWING, ObjectionEvent.ESCALATE): ObjectionState.ESCALATED,
    # Escalated objections can be re-assigned for fresh review.
    (ObjectionState.ESCALATED, ObjectionEvent.ASSIGN): ObjectionState.ASSIGNED,
}


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Successful transition outcome."""

    old_state: ObjectionState
    new_state: ObjectionState
    event: ObjectionEvent


class InvalidTransitionError(Exception):
    """Raised when a transition is not permitted by the FSM."""

    def __init__(
        self,
        current: ObjectionState,
        event: ObjectionEvent,
    ) -> None:
        self.current = current
        self.event = event
        super().__init__(
            f"Cannot apply event '{event}' in state '{current}'"
        )


def transition(
    current: ObjectionState,
    event: ObjectionEvent,
) -> TransitionResult:
    """Compute the next state for a given event.

    Parameters
    ----------
    current:
        The objection's current state.
    event:
        The action being applied.

    Returns
    -------
    TransitionResult
        Contains old state, new state, and the event.

    Raises
    ------
    InvalidTransitionError
        If the transition is not in the FSM table.
    """
    key = (current, event)
    new_state = _TRANSITIONS.get(key)
    if new_state is None:
        raise InvalidTransitionError(current, event)
    return TransitionResult(
        old_state=current,
        new_state=new_state,
        event=event,
    )


def valid_events_for(state: ObjectionState) -> list[ObjectionEvent]:
    """Return the list of events valid from *state*."""
    return [evt for (s, evt), _ in _TRANSITIONS.items() if s == state]


def is_terminal(state: ObjectionState) -> bool:
    """Return True if *state* has no outgoing transitions."""
    return not valid_events_for(state)
