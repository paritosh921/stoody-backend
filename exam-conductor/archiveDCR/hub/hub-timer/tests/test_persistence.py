"""Unit tests for timer SQLite persistence and reboot recovery.

Test IDs: U-TMR-P01 through U-TMR-P03.
Validation level: L3 (unit — uses in-memory SQLite, no real I/O).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.countdown import CountdownTimer
from src.persistence import TimerPersistence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_hub.db"


@pytest.fixture()
def store(tmp_db: Path) -> TimerPersistence:
    p = TimerPersistence(db_path=tmp_db)
    p.open()
    yield p
    p.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# U-TMR-P01: Persist, close, reopen, verify resumed correctly
# ---------------------------------------------------------------------------

def test_persist_and_reload(tmp_db: Path):
    """U-TMR-P01: Written state survives close/reopen."""
    now_epoch = int(time.time())

    # Write
    store = TimerPersistence(db_path=tmp_db)
    store.open()
    store.persist_state(
        exam_id="exam-p01",
        start_epoch=now_epoch - 600,
        duration_sec=3600,
        remaining_sec=3000,
    )
    store.close()

    # Reopen and load
    store2 = TimerPersistence(db_path=tmp_db)
    store2.open()
    loaded = store2.load_state()
    store2.close()

    assert loaded is not None
    assert loaded.exam_id == "exam-p01"
    assert loaded.duration_sec == 3600
    assert loaded.remaining_sec == 3000
    assert loaded.start_epoch == now_epoch - 600
    # last_updated is set by persist_state to ~now
    assert abs(loaded.last_updated - now_epoch) < 5


def test_resume_countdown_after_simulated_reboot(tmp_db: Path):
    """U-TMR-P01b: Full round-trip -- arm, persist, 'reboot', resume."""
    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)

    timer.arm("exam-reboot", duration_sec=1800)
    clock.advance(300)
    remaining = timer.get_remaining()  # 1500
    assert remaining == 1500

    # Persist
    now_epoch = int(time.time())
    store = TimerPersistence(db_path=tmp_db)
    store.open()
    with patch("time.time", return_value=now_epoch):
        store.persist_state(
            exam_id="exam-reboot",
            start_epoch=now_epoch - 300,
            duration_sec=1800,
            remaining_sec=remaining,
        )
    store.close()

    # --- Simulate 15 seconds of downtime ---
    downtime = 15

    # Reload
    store2 = TimerPersistence(db_path=tmp_db)
    store2.open()
    saved = store2.load_state()
    store2.close()

    assert saved is not None

    clock2 = FakeClock(start=2000.0)
    timer2 = CountdownTimer(clock_fn=clock2)

    with patch("time.time", return_value=now_epoch + downtime):
        timer2.arm(
            saved.exam_id,
            saved.duration_sec,
            resume_remaining=saved.remaining_sec,
            resume_epoch=saved.last_updated,
        )

    # 1500 - 15 = 1485
    assert timer2.get_remaining() == 1485
    assert not timer2.is_expired()


# ---------------------------------------------------------------------------
# U-TMR-P02: Recovery with elapsed > remaining (already expired)
# ---------------------------------------------------------------------------

def test_recovery_already_expired(tmp_db: Path):
    """U-TMR-P02: If downtime exceeds remaining, timer is expired on load."""
    now_epoch = int(time.time())

    store = TimerPersistence(db_path=tmp_db)
    store.open()
    with patch("time.time", return_value=now_epoch - 120):
        store.persist_state(
            exam_id="exam-gone",
            start_epoch=now_epoch - 3700,
            duration_sec=3600,
            remaining_sec=60,  # only 60 s were left
        )
    store.close()

    # Reload 120 s later
    store2 = TimerPersistence(db_path=tmp_db)
    store2.open()
    saved = store2.load_state()
    store2.close()

    assert saved is not None

    clock = FakeClock()
    timer = CountdownTimer(clock_fn=clock)
    with patch("time.time", return_value=now_epoch):
        timer.arm(
            saved.exam_id,
            saved.duration_sec,
            resume_remaining=saved.remaining_sec,
            resume_epoch=saved.last_updated,
        )

    assert timer.is_expired()
    assert timer.get_remaining() == 0


# ---------------------------------------------------------------------------
# U-TMR-P03: Clean state after cancel
# ---------------------------------------------------------------------------

def test_clear_state_after_cancel(store: TimerPersistence):
    """U-TMR-P03: clear_state removes the row; load_state returns None."""
    now_epoch = int(time.time())

    store.persist_state(
        exam_id="exam-cancel",
        start_epoch=now_epoch,
        duration_sec=1800,
        remaining_sec=1800,
    )
    loaded = store.load_state()
    assert loaded is not None
    assert loaded.exam_id == "exam-cancel"

    store.clear_state("exam-cancel")
    assert store.load_state() is None


def test_clear_nonexistent_exam(store: TimerPersistence):
    """U-TMR-P03b: Clearing an exam that doesn't exist is a no-op."""
    store.clear_state("does-not-exist")
    assert store.load_state() is None


def test_persist_overwrites_previous(store: TimerPersistence):
    """Upsert semantics: second persist updates the row, not duplicates."""
    now_epoch = int(time.time())
    store.persist_state("e1", now_epoch, 3600, 3000)
    store.persist_state("e1", now_epoch, 3600, 2500)

    loaded = store.load_state()
    assert loaded is not None
    assert loaded.remaining_sec == 2500


def test_wal_mode_enabled(tmp_db: Path):
    """Verify that WAL journal mode is active after open()."""
    store = TimerPersistence(db_path=tmp_db)
    store.open()
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    store.close()
    assert mode.lower() == "wal"
