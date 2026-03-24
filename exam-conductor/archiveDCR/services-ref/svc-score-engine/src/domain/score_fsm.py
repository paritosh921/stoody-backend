"""Score lifecycle finite state machine.

States:
    ai_draft           -- AI pipeline produced an initial score.
    teacher_reviewed   -- A teacher has reviewed / overridden at least one question.
    finalized          -- Evaluator review complete; scores locked for publication prep.
    published          -- Results visible to students/parents; objection window opens.
    objection_window   -- Objection window is open (may trigger re-scoring).
    locked             -- Final lock after objection window closes.

This module is ZERO I/O -- no asyncio, no DB, no network imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class ScoreState(str, Enum):
    AI_DRAFT = "ai_draft"
    TEACHER_REVIEWED = "teacher_reviewed"
    FINALIZED = "finalized"
    PUBLISHED = "published"
    OBJECTION_WINDOW = "objection_window"
    LOCKED = "locked"


class ScoreTransitionError(Exception):
    """Raised when a requested state transition is invalid."""

    def __init__(self, current: ScoreState, requested: ScoreState) -> None:
        self.current = current
        self.requested = requested
        super().__init__(
            f"Invalid transition: {current.value} -> {requested.value}"
        )


# ---- Transition table --------------------------------------------------------

_VALID_TRANSITIONS: dict[ScoreState, FrozenSet[ScoreState]] = {
    ScoreState.AI_DRAFT: frozenset(
        {ScoreState.TEACHER_REVIEWED, ScoreState.FINALIZED}
    ),
    ScoreState.TEACHER_REVIEWED: frozenset(
        {ScoreState.TEACHER_REVIEWED, ScoreState.FINALIZED}
    ),
    ScoreState.FINALIZED: frozenset({ScoreState.PUBLISHED}),
    ScoreState.PUBLISHED: frozenset({ScoreState.OBJECTION_WINDOW}),
    ScoreState.OBJECTION_WINDOW: frozenset(
        {ScoreState.LOCKED, ScoreState.TEACHER_REVIEWED}
    ),
    ScoreState.LOCKED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class TransitionResult:
    old_state: ScoreState
    new_state: ScoreState


def transition(current: ScoreState, target: ScoreState) -> TransitionResult:
    """Validate and return a state transition.

    Returns ``TransitionResult`` on success.
    Raises ``ScoreTransitionError`` if the transition is not allowed.
    """
    allowed = _VALID_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise ScoreTransitionError(current, target)
    return TransitionResult(old_state=current, new_state=target)


def allowed_transitions(current: ScoreState) -> FrozenSet[ScoreState]:
    """Return the set of states reachable from *current*."""
    return _VALID_TRANSITIONS.get(current, frozenset())


def is_mutable(state: ScoreState) -> bool:
    """Return True if individual question scores may still be changed."""
    return state in {
        ScoreState.AI_DRAFT,
        ScoreState.TEACHER_REVIEWED,
        ScoreState.OBJECTION_WINDOW,
    }
