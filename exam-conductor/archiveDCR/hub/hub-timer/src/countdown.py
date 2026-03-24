"""Core countdown timer engine.

Uses ``time.clock_gettime(time.CLOCK_MONOTONIC)`` so the countdown is
immune to NTP adjustments or ``settimeofday`` calls during an exam
(see FAILURE_MITIGATION_REGISTER F1).

This module is intentionally *pure logic with a clock dependency* -- it
owns no I/O, no asyncio, and no IPC.  The async loop in ``main.py``
drives it by calling helpers periodically and checking ``is_expired()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional


def _mono() -> float:
    """Return monotonic clock in fractional seconds."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


@dataclass
class TimerState:
    """Snapshot of a running (or finished) countdown."""

    exam_id: str
    total_sec: int
    remaining_sec: int
    started_at_mono: float   # CLOCK_MONOTONIC at arm-time
    started_at_epoch: int    # wall-clock epoch (for persistence only)
    expired: bool = False
    # Internal: the effective countdown ceiling relative to started_at_mono.
    # For fresh timers this equals total_sec; for resumed timers it equals
    # the remaining seconds that survived the reboot gap.
    _effective_total: int = 0


class CountdownTimer:
    """Drift-immune exam countdown.

    Lifecycle::

        timer.arm(exam_id, 3600)   # start 1-hour countdown
        ...
        timer.get_remaining()      # -> 3542
        ...
        timer.is_expired()         # -> True once remaining <= 0
        timer.cancel(exam_id)      # explicit stop
    """

    def __init__(self, clock_fn: Optional[Callable[[], float]] = None) -> None:
        # Allow injecting a fake clock for deterministic tests.
        self._clock = clock_fn or _mono
        self._state: Optional[TimerState] = None

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def arm(
        self,
        exam_id: str,
        duration_sec: int,
        *,
        resume_remaining: int | None = None,
        resume_epoch: int | None = None,
    ) -> TimerState:
        """Arm (or replace) the countdown.

        Parameters
        ----------
        exam_id:
            Exam session identifier.
        duration_sec:
            Original exam duration in seconds.
        resume_remaining:
            If recovering from a reboot, the *remaining* seconds that were
            persisted before the crash.
        resume_epoch:
            The ``last_updated`` epoch from the persistence row.  Used to
            subtract time elapsed while the process was down.
        """
        now_mono = self._clock()

        if resume_remaining is not None and resume_epoch is not None:
            elapsed_while_down = max(0, int(time.time()) - resume_epoch)
            effective = max(0, resume_remaining - elapsed_while_down)
        else:
            effective = duration_sec

        self._state = TimerState(
            exam_id=exam_id,
            total_sec=duration_sec,
            remaining_sec=effective,
            started_at_mono=now_mono,
            started_at_epoch=int(time.time()),
            expired=effective <= 0,
            _effective_total=effective,
        )
        return self._state

    def cancel(self, exam_id: str) -> bool:
        """Stop the countdown.  Returns True if there was an active timer."""
        if self._state is not None and self._state.exam_id == exam_id:
            self._state = None
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_remaining(self) -> int:
        """Remaining whole seconds, or 0 if expired / not armed."""
        if self._state is None:
            return 0
        self._recompute()
        return self._state.remaining_sec

    def is_expired(self) -> bool:
        """True when the countdown has reached zero."""
        if self._state is None:
            return False
        self._recompute()
        return self._state.expired

    def get_state(self) -> Optional[TimerState]:
        """Full snapshot (or None if not armed)."""
        if self._state is not None:
            self._recompute()
        return self._state

    @property
    def active(self) -> bool:
        return self._state is not None and not self._state.expired

    @property
    def exam_id(self) -> Optional[str]:
        return self._state.exam_id if self._state else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        """Derive remaining_sec from the monotonic clock."""
        assert self._state is not None
        elapsed = self._clock() - self._state.started_at_mono
        remaining = max(0, self._state._effective_total - int(elapsed))
        self._state.remaining_sec = remaining
        if remaining <= 0:
            self._state.expired = True
