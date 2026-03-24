"""Exam lifecycle finite state machine — ZERO I/O, pure logic.

States:
    created -> armed -> timer_running -> sync_pending -> scoring
    -> finalized -> published -> locked

Cancellation reachable from: created, armed, timer_running.

Every transition is validated; invalid transitions raise ``InvalidTransition``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExamState(StrEnum):
    """All possible exam lifecycle states."""

    CREATED = "created"
    ARMED = "armed"
    TIMER_RUNNING = "timer_running"
    SYNC_PENDING = "sync_pending"
    SCORING = "scoring"
    FINALIZED = "finalized"
    PUBLISHED = "published"
    LOCKED = "locked"
    CANCELLED = "cancelled"


class InvalidTransition(Exception):
    """Raised when a state transition is not allowed."""

    def __init__(self, from_state: str, to_state: str) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition: {from_state} -> {to_state}"
        )


# Adjacency map: source state -> set of valid target states
_TRANSITIONS: dict[ExamState, set[ExamState]] = {
    ExamState.CREATED: {ExamState.ARMED, ExamState.CANCELLED},
    ExamState.ARMED: {ExamState.TIMER_RUNNING, ExamState.CANCELLED},
    ExamState.TIMER_RUNNING: {ExamState.SYNC_PENDING, ExamState.CANCELLED},
    ExamState.SYNC_PENDING: {ExamState.SCORING},
    ExamState.SCORING: {ExamState.FINALIZED},
    ExamState.FINALIZED: {ExamState.PUBLISHED},
    ExamState.PUBLISHED: {ExamState.LOCKED},
    ExamState.LOCKED: set(),
    ExamState.CANCELLED: set(),
}

# All valid state strings for input parsing
VALID_STATES: frozenset[str] = frozenset(s.value for s in ExamState)


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Outcome of a successful state transition."""

    from_state: ExamState
    to_state: ExamState


def transition(current_state: str, target_state: str) -> TransitionResult:
    """Validate and return the transition result.

    Parameters
    ----------
    current_state:
        Current exam state (string value).
    target_state:
        Desired target state (string value).

    Returns
    -------
    TransitionResult on success.

    Raises
    ------
    InvalidTransition
        If the transition is not in the allowed adjacency map.
    ValueError
        If either state string is not a recognised exam state.
    """
    try:
        src = ExamState(current_state)
    except ValueError:
        raise ValueError(f"Unknown exam state: {current_state}") from None

    try:
        dst = ExamState(target_state)
    except ValueError:
        raise ValueError(f"Unknown exam state: {target_state}") from None

    allowed = _TRANSITIONS.get(src, set())
    if dst not in allowed:
        raise InvalidTransition(current_state, target_state)

    return TransitionResult(from_state=src, to_state=dst)


def get_allowed_transitions(current_state: str) -> list[str]:
    """Return the list of states reachable from *current_state*."""
    try:
        src = ExamState(current_state)
    except ValueError:
        return []
    return sorted(s.value for s in _TRANSITIONS.get(src, set()))
