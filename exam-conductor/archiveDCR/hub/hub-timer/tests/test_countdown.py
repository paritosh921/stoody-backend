"""Unit tests for the countdown timer engine.

Test IDs: U-TMR-01 through U-TMR-06.
Validation level: L3 (unit, no I/O).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.countdown import CountdownTimer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    """Deterministic clock that can be advanced manually."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# U-TMR-01: Arm timer and verify remaining decrements
# ---------------------------------------------------------------------------

def test_arm_and_remaining_decrements():
    """U-TMR-01: After arming, get_remaining returns correct countdown."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    timer.arm("exam-001", duration_sec=3600)
    assert timer.get_remaining() == 3600

    clock.advance(10)
    assert timer.get_remaining() == 3590

    clock.advance(100)
    assert timer.get_remaining() == 3490

    assert not timer.is_expired()
    assert timer.active


# ---------------------------------------------------------------------------
# U-TMR-02: Verify expiry fires at correct time
# ---------------------------------------------------------------------------

def test_expiry_at_correct_time():
    """U-TMR-02: Timer reports expired exactly when remaining hits 0."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    timer.arm("exam-002", duration_sec=60)

    clock.advance(59)
    assert not timer.is_expired()
    assert timer.get_remaining() == 1

    clock.advance(1)
    assert timer.is_expired()
    assert timer.get_remaining() == 0


def test_expiry_overshoot():
    """U-TMR-02b: Remaining never goes negative."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    timer.arm("exam-002b", duration_sec=30)
    clock.advance(100)

    assert timer.is_expired()
    assert timer.get_remaining() == 0


# ---------------------------------------------------------------------------
# U-TMR-03: CLOCK_MONOTONIC immunity to NTP adjustments
# ---------------------------------------------------------------------------

def test_monotonic_immune_to_wall_clock_changes():
    """U-TMR-03: Timer uses the injected monotonic clock, not time.time().

    We patch time.time() to jump backwards (simulating an NTP correction)
    and verify the countdown is unaffected.
    """
    clock = FakeClock(start=5000.0)
    timer = CountdownTimer(clock_fn=clock)

    timer.arm("exam-003", duration_sec=120)

    clock.advance(30)
    assert timer.get_remaining() == 90

    # Simulate NTP adjusting wall clock backwards by 60 seconds.
    # The monotonic clock is unaffected, so remaining should keep
    # counting down normally.
    original_time = time.time
    with patch("time.time", return_value=original_time() - 60):
        clock.advance(10)
        assert timer.get_remaining() == 80  # still monotonic-based


# ---------------------------------------------------------------------------
# U-TMR-04: Cancel mid-countdown
# ---------------------------------------------------------------------------

def test_cancel_mid_countdown():
    """U-TMR-04: Cancelling clears state and stops the timer."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    timer.arm("exam-004", duration_sec=600)
    clock.advance(100)
    assert timer.get_remaining() == 500

    result = timer.cancel("exam-004")
    assert result is True

    # After cancel, timer is idle.
    assert timer.get_remaining() == 0
    assert not timer.is_expired()
    assert not timer.active
    assert timer.get_state() is None


def test_cancel_wrong_exam_id():
    """U-TMR-04b: Cancelling with wrong exam_id is a no-op."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    timer.arm("exam-004", duration_sec=600)
    result = timer.cancel("wrong-id")
    assert result is False
    assert timer.active


def test_cancel_when_idle():
    """U-TMR-04c: Cancelling when no timer is armed is a no-op."""
    timer = CountdownTimer(clock_fn=FakeClock())
    result = timer.cancel("any-id")
    assert result is False


# ---------------------------------------------------------------------------
# U-TMR-05: Arm while already armed (replace)
# ---------------------------------------------------------------------------

def test_arm_replaces_existing_timer():
    """U-TMR-05: Arming a new timer replaces the previous one."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    timer.arm("exam-old", duration_sec=3600)
    clock.advance(100)
    assert timer.get_remaining() == 3500
    assert timer.exam_id == "exam-old"

    # Arm a new timer — should replace the old one.
    timer.arm("exam-new", duration_sec=1800)
    assert timer.exam_id == "exam-new"
    assert timer.get_remaining() == 1800
    assert timer.active


# ---------------------------------------------------------------------------
# U-TMR-06: Resume with elapsed time (boot recovery)
# ---------------------------------------------------------------------------

def test_arm_with_resume():
    """U-TMR-06: Resuming after reboot subtracts elapsed gap."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    now_epoch = int(time.time())
    # Simulate: timer was persisted with 500 s remaining, 30 s ago.
    with patch("time.time", return_value=now_epoch):
        timer.arm(
            "exam-resume",
            duration_sec=3600,
            resume_remaining=500,
            resume_epoch=now_epoch - 30,
        )

    # 500 - 30 = 470 effective remaining at arm time.
    assert timer.get_remaining() == 470

    clock.advance(10)
    assert timer.get_remaining() == 460


def test_arm_with_resume_already_expired():
    """U-TMR-06b: If elapsed gap exceeds remaining, timer is immediately expired."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    now_epoch = int(time.time())
    with patch("time.time", return_value=now_epoch):
        timer.arm(
            "exam-expired",
            duration_sec=3600,
            resume_remaining=20,
            resume_epoch=now_epoch - 60,  # 60 s elapsed > 20 s remaining
        )

    assert timer.is_expired()
    assert timer.get_remaining() == 0
