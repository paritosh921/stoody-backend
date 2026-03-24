"""Tests for per-pen sync state machine (ZERO I/O, domain logic).

Test IDs: U-SYN-01 .. U-SYN-12
Validation level: L3 (unit — pure domain, no I/O)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow imports from hub-pen-sync/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sync_state import (
    SyncEvent,
    SyncState,
    SyncStatus,
    record_chunk,
    record_store_confirm,
    set_buffer_info,
    transition,
)


def _make_state(**kwargs) -> SyncState:
    defaults = {"pen_mac": "AA:BB:CC:DD:EE:FF", "exam_id": "EXAM-01"}
    defaults.update(kwargs)
    return SyncState(**defaults)


# -----------------------------------------------------------------------
# U-SYN-01: initial state is PENDING
# -----------------------------------------------------------------------

def test_initial_state_pending() -> None:
    """U-SYN-01: freshly created SyncState has PENDING status."""
    state = _make_state()
    assert state.status == SyncStatus.PENDING
    assert state.retries_remaining == 3
    assert not state.is_terminal


# -----------------------------------------------------------------------
# U-SYN-02: PENDING -> CONNECTING on START
# -----------------------------------------------------------------------

def test_pending_to_connecting() -> None:
    """U-SYN-02: START event transitions PENDING to CONNECTING."""
    state = _make_state()
    transition(state, SyncEvent.START)
    assert state.status == SyncStatus.CONNECTING


# -----------------------------------------------------------------------
# U-SYN-03: CONNECTING -> SYNCING on CONNECTED
# -----------------------------------------------------------------------

def test_connecting_to_syncing() -> None:
    """U-SYN-03: CONNECTED event transitions to SYNCING."""
    state = _make_state(status=SyncStatus.CONNECTING)
    transition(state, SyncEvent.CONNECTED)
    assert state.status == SyncStatus.SYNCING


# -----------------------------------------------------------------------
# U-SYN-04: happy path through full lifecycle
# -----------------------------------------------------------------------

def test_happy_path_full_lifecycle() -> None:
    """U-SYN-04: PENDING -> CONNECTING -> SYNCING -> COMPLETE."""
    state = _make_state()
    transition(state, SyncEvent.START)
    assert state.status == SyncStatus.CONNECTING

    transition(state, SyncEvent.CONNECTED)
    assert state.status == SyncStatus.SYNCING

    transition(state, SyncEvent.CHUNK_RECEIVED)
    assert state.status == SyncStatus.SYNCING

    transition(state, SyncEvent.CHECKSUM_MATCH)
    assert state.status == SyncStatus.SYNCING

    transition(state, SyncEvent.STORE_CONFIRMED)
    assert state.status == SyncStatus.COMPLETE
    assert state.is_terminal


# -----------------------------------------------------------------------
# U-SYN-05: SYNCING -> FAILED on disconnect
# -----------------------------------------------------------------------

def test_syncing_to_failed_on_disconnect() -> None:
    """U-SYN-05: DISCONNECT during SYNCING leads to FAILED."""
    state = _make_state(status=SyncStatus.SYNCING)
    transition(state, SyncEvent.DISCONNECT)
    assert state.status == SyncStatus.FAILED
    assert state.error_detail == "BLE disconnect"
    assert state.is_terminal


# -----------------------------------------------------------------------
# U-SYN-06: SYNCING -> TIMEOUT on timeout
# -----------------------------------------------------------------------

def test_syncing_to_timeout() -> None:
    """U-SYN-06: TIMEOUT during SYNCING leads to TIMEOUT status."""
    state = _make_state(status=SyncStatus.SYNCING)
    transition(state, SyncEvent.TIMEOUT)
    assert state.status == SyncStatus.TIMEOUT
    assert state.error_detail == "Timeout"


# -----------------------------------------------------------------------
# U-SYN-07: retry from FAILED decrements counter
# -----------------------------------------------------------------------

def test_retry_from_failed() -> None:
    """U-SYN-07: RETRY from FAILED returns to CONNECTING, decrements retries."""
    state = _make_state(status=SyncStatus.FAILED, retries_remaining=2)
    transition(state, SyncEvent.RETRY)
    assert state.status == SyncStatus.CONNECTING
    assert state.retries_remaining == 1
    assert state.error_detail == ""


# -----------------------------------------------------------------------
# U-SYN-08: retry from TIMEOUT works the same
# -----------------------------------------------------------------------

def test_retry_from_timeout() -> None:
    """U-SYN-08: RETRY from TIMEOUT returns to CONNECTING."""
    state = _make_state(status=SyncStatus.TIMEOUT, retries_remaining=3)
    transition(state, SyncEvent.RETRY)
    assert state.status == SyncStatus.CONNECTING
    assert state.retries_remaining == 2


# -----------------------------------------------------------------------
# U-SYN-09: can_retry is False when retries exhausted
# -----------------------------------------------------------------------

def test_can_retry_exhausted() -> None:
    """U-SYN-09: can_retry is False when retries_remaining == 0."""
    state = _make_state(
        status=SyncStatus.FAILED, retries_remaining=0
    )
    assert not state.can_retry


# -----------------------------------------------------------------------
# U-SYN-10: invalid transition raises ValueError
# -----------------------------------------------------------------------

def test_invalid_transition_raises() -> None:
    """U-SYN-10: transitions not in the map raise ValueError."""
    state = _make_state(status=SyncStatus.COMPLETE)
    with pytest.raises(ValueError, match="Invalid transition"):
        transition(state, SyncEvent.START)


# -----------------------------------------------------------------------
# U-SYN-11: record_chunk updates counters
# -----------------------------------------------------------------------

def test_record_chunk_updates_counters() -> None:
    """U-SYN-11: record_chunk increments chunks_received and bytes."""
    state = _make_state(bytes_expected=1000)
    record_chunk(state, 256)
    assert state.chunks_received == 1
    assert state.bytes_received == 256

    record_chunk(state, 300)
    assert state.chunks_received == 2
    assert state.bytes_received == 556


# -----------------------------------------------------------------------
# U-SYN-12: progress_pct, checksum_verified, all_chunks_stored
# -----------------------------------------------------------------------

def test_derived_properties() -> None:
    """U-SYN-12: progress_pct, checksum_verified, all_chunks_stored."""
    state = _make_state()

    # progress when nothing expected
    assert state.progress_pct == 0.0

    set_buffer_info(state, total_bytes=1000, total_chunks=4, checksum_crc32="abcd1234")
    assert state.bytes_expected == 1000
    assert state.total_chunks == 4
    assert state.checksum_expected == "abcd1234"

    record_chunk(state, 500)
    assert state.progress_pct == 50.0

    # checksum not verified yet
    assert not state.checksum_verified
    state.checksum_actual = "abcd1234"
    assert state.checksum_verified

    # not all chunks stored yet
    assert not state.all_chunks_stored
    record_store_confirm(state, 0)
    record_store_confirm(state, 1)
    record_store_confirm(state, 2)
    record_store_confirm(state, 3)
    assert state.all_chunks_stored
    assert state.last_confirmed_chunk == 3


# -----------------------------------------------------------------------
# U-SYN-13: CHECKSUM_MISMATCH sets error detail
# -----------------------------------------------------------------------

def test_checksum_mismatch_error_detail() -> None:
    """U-SYN-13: CHECKSUM_MISMATCH populates error_detail."""
    state = _make_state(status=SyncStatus.SYNCING)
    state.checksum_expected = "aabbccdd"
    state.checksum_actual = "11223344"
    transition(state, SyncEvent.CHECKSUM_MISMATCH)
    assert state.status == SyncStatus.FAILED
    assert "aabbccdd" in state.error_detail
    assert "11223344" in state.error_detail


# -----------------------------------------------------------------------
# U-SYN-14: abort during SYNCING
# -----------------------------------------------------------------------

def test_abort_during_syncing() -> None:
    """U-SYN-14: ABORT during SYNCING transitions to FAILED."""
    state = _make_state(status=SyncStatus.SYNCING)
    transition(state, SyncEvent.ABORT)
    assert state.status == SyncStatus.FAILED
    assert state.error_detail == "Aborted by supervisor"
