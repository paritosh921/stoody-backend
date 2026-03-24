"""Hub lifecycle FSM -- ZERO I/O, pure state machine.

States match ``HUB_DEPLOYMENT_SPEC.md`` Section 3.1 ``exam_sessions.state``
CHECK constraint:

    created -> armed -> timer_running -> dongle_activation -> pen_sync
    -> sync_complete | sync_partial -> uploading -> upload_complete
    Any state -> cancelled

The ``transition()`` function is the single authority for valid transitions.
It returns the new state or raises ``InvalidTransition``.

This module must NEVER import asyncio, aiohttp, sqlite3, or any I/O library.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# States (match exam_sessions.state CHECK constraint)
# ---------------------------------------------------------------------------

CREATED = "created"
ARMED = "armed"
TIMER_RUNNING = "timer_running"
DONGLE_ACTIVATION = "dongle_activation"
PEN_SYNC = "pen_sync"
SYNC_COMPLETE = "sync_complete"
SYNC_PARTIAL = "sync_partial"
UPLOADING = "uploading"
UPLOAD_COMPLETE = "upload_complete"
CANCELLED = "cancelled"

ALL_STATES: frozenset[str] = frozenset(
    {
        CREATED,
        ARMED,
        TIMER_RUNNING,
        DONGLE_ACTIVATION,
        PEN_SYNC,
        SYNC_COMPLETE,
        SYNC_PARTIAL,
        UPLOADING,
        UPLOAD_COMPLETE,
        CANCELLED,
    }
)

TERMINAL_STATES: frozenset[str] = frozenset({UPLOAD_COMPLETE, CANCELLED})

# ---------------------------------------------------------------------------
# Events (trigger transitions)
# ---------------------------------------------------------------------------

EVT_ARM = "arm"
EVT_START_TIMER = "start_timer"
EVT_TIMER_EXPIRED = "timer_expired"
EVT_ACTIVATE_DONGLES = "activate_dongles"
EVT_SYNC_ALL_COMPLETE = "sync_all_complete"
EVT_SYNC_PARTIAL = "sync_partial"
EVT_START_UPLOAD = "start_upload"
EVT_UPLOAD_DONE = "upload_done"
EVT_CANCEL = "cancel"

# ---------------------------------------------------------------------------
# Transition table: (current_state, event) -> new_state
# ---------------------------------------------------------------------------

_TRANSITIONS: dict[tuple[str, str], str] = {
    (CREATED, EVT_ARM): ARMED,
    (ARMED, EVT_START_TIMER): TIMER_RUNNING,
    (TIMER_RUNNING, EVT_TIMER_EXPIRED): DONGLE_ACTIVATION,
    (DONGLE_ACTIVATION, EVT_ACTIVATE_DONGLES): PEN_SYNC,
    (PEN_SYNC, EVT_SYNC_ALL_COMPLETE): SYNC_COMPLETE,
    (PEN_SYNC, EVT_SYNC_PARTIAL): SYNC_PARTIAL,
    (SYNC_COMPLETE, EVT_START_UPLOAD): UPLOADING,
    (SYNC_PARTIAL, EVT_START_UPLOAD): UPLOADING,
    (UPLOADING, EVT_UPLOAD_DONE): UPLOAD_COMPLETE,
}

# Cancel is allowed from any non-terminal state.
for _state in ALL_STATES - TERMINAL_STATES:
    _TRANSITIONS[(_state, EVT_CANCEL)] = CANCELLED


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InvalidTransition(Exception):
    """Raised when a requested FSM transition is not allowed."""

    def __init__(self, current: str, event: str) -> None:
        self.current = current
        self.event = event
        super().__init__(
            f"Invalid transition: state={current!r}, event={event!r}"
        )


class InvalidState(Exception):
    """Raised when a state value is not recognized."""

    def __init__(self, state: str) -> None:
        self.state = state
        super().__init__(f"Unknown state: {state!r}")


# ---------------------------------------------------------------------------
# Transition function
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Immutable result of a successful state transition."""

    old_state: str
    event: str
    new_state: str


def transition(current: str, event: str) -> TransitionResult:
    """Compute the next state for *current* + *event*.

    Returns a :class:`TransitionResult` on success.
    Raises :class:`InvalidTransition` if the combination is not allowed.
    Raises :class:`InvalidState` if *current* is not a recognized state.
    """
    if current not in ALL_STATES:
        raise InvalidState(current)

    key = (current, event)
    new_state = _TRANSITIONS.get(key)
    if new_state is None:
        raise InvalidTransition(current, event)

    return TransitionResult(old_state=current, event=event, new_state=new_state)


def valid_events(current: str) -> list[str]:
    """Return sorted list of events valid from *current* state."""
    if current not in ALL_STATES:
        raise InvalidState(current)
    return sorted(evt for (st, evt) in _TRANSITIONS if st == current)


def is_terminal(state: str) -> bool:
    """Return True if *state* is a terminal (no further transitions)."""
    return state in TERMINAL_STATES
