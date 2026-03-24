"""Tests for upload ledger — chunk ack tracking, resume, completion.

Test IDs: U-UPL-08 .. U-UPL-16
Validation level: L3 (unit — in-memory SQLite, no network)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ledger import UploadLedger, ensure_ledger_table


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL;")
    ensure_ledger_table(c)
    return c


@pytest.fixture()
def ledger(conn: sqlite3.Connection) -> UploadLedger:
    return UploadLedger(conn)


# -----------------------------------------------------------------------
# U-UPL-08: init_pen creates a row
# -----------------------------------------------------------------------

def test_init_pen_creates_row(ledger: UploadLedger) -> None:
    """U-UPL-08: init_pen creates ledger row with correct defaults."""
    ledger.init_pen("E1", "AA:BB", total_chunks=5, upload_path="wifi")
    status = ledger.get_upload_status("E1")
    assert len(status["pens"]) == 1
    pen = status["pens"][0]
    assert pen["pen_mac"] == "AA:BB"
    assert pen["total_chunks"] == 5
    assert pen["acked_chunks"] == 0
    assert pen["complete"] is False


# -----------------------------------------------------------------------
# U-UPL-09: mark_chunk_acked records single chunk
# -----------------------------------------------------------------------

def test_mark_chunk_acked(ledger: UploadLedger) -> None:
    """U-UPL-09: marking a chunk as acked updates the JSON array."""
    ledger.init_pen("E1", "AA:BB", 3, "wifi")
    ledger.mark_chunk_acked("E1", "AA:BB", 0)

    pending = ledger.get_pending_chunks("E1", "AA:BB")
    assert 0 not in pending
    assert 1 in pending
    assert 2 in pending


# -----------------------------------------------------------------------
# U-UPL-10: duplicate ack is idempotent
# -----------------------------------------------------------------------

def test_duplicate_ack_idempotent(ledger: UploadLedger) -> None:
    """U-UPL-10: acking the same chunk twice does not corrupt the array."""
    ledger.init_pen("E1", "AA:BB", 3, "wifi")
    ledger.mark_chunk_acked("E1", "AA:BB", 1)
    ledger.mark_chunk_acked("E1", "AA:BB", 1)

    pending = ledger.get_pending_chunks("E1", "AA:BB")
    assert pending == [0, 2]


# -----------------------------------------------------------------------
# U-UPL-11: get_pending_chunks returns only missing chunks
# -----------------------------------------------------------------------

def test_pending_chunks_after_partial_upload(ledger: UploadLedger) -> None:
    """U-UPL-11: pending returns only unacked chunk indices."""
    ledger.init_pen("E1", "AA:BB", 5, "wifi")
    for i in [0, 2, 4]:
        ledger.mark_chunk_acked("E1", "AA:BB", i)

    pending = ledger.get_pending_chunks("E1", "AA:BB")
    assert pending == [1, 3]


# -----------------------------------------------------------------------
# U-UPL-12: is_pen_complete returns False until all acked
# -----------------------------------------------------------------------

def test_is_pen_complete_false_until_all_acked(ledger: UploadLedger) -> None:
    """U-UPL-12: pen is not complete with partial acks."""
    ledger.init_pen("E1", "AA:BB", 3, "wifi")
    ledger.mark_chunk_acked("E1", "AA:BB", 0)
    ledger.mark_chunk_acked("E1", "AA:BB", 1)
    assert not ledger.is_pen_complete("E1", "AA:BB")


# -----------------------------------------------------------------------
# U-UPL-13: is_pen_complete returns True when all acked
# -----------------------------------------------------------------------

def test_is_pen_complete_true_when_all_acked(ledger: UploadLedger) -> None:
    """U-UPL-13: pen is complete when every chunk index is acked."""
    ledger.init_pen("E1", "AA:BB", 3, "wifi")
    for i in range(3):
        ledger.mark_chunk_acked("E1", "AA:BB", i)
    assert ledger.is_pen_complete("E1", "AA:BB")


# -----------------------------------------------------------------------
# U-UPL-14: mark_upload_complete sets complete flag
# -----------------------------------------------------------------------

def test_mark_upload_complete(ledger: UploadLedger, conn: sqlite3.Connection) -> None:
    """U-UPL-14: mark_upload_complete sets complete=1 and completed_at."""
    ledger.init_pen("E1", "AA:BB", 2, "wifi")
    ledger.mark_chunk_acked("E1", "AA:BB", 0)
    ledger.mark_chunk_acked("E1", "AA:BB", 1)
    ledger.mark_upload_complete("E1", "AA:BB")

    row = conn.execute(
        "SELECT complete, completed_at FROM upload_ledger "
        "WHERE exam_id='E1' AND pen_mac='AA:BB'",
    ).fetchone()
    assert row[0] == 1
    assert row[1] is not None


# -----------------------------------------------------------------------
# U-UPL-15: get_upload_status returns per-pen summary
# -----------------------------------------------------------------------

def test_upload_status_multi_pen(ledger: UploadLedger) -> None:
    """U-UPL-15: status reports all pens for an exam."""
    ledger.init_pen("E1", "AA:BB", 3, "wifi")
    ledger.init_pen("E1", "CC:DD", 2, "mobile")
    ledger.mark_chunk_acked("E1", "AA:BB", 0)

    status = ledger.get_upload_status("E1")
    assert status["exam_id"] == "E1"
    assert len(status["pens"]) == 2

    by_mac = {p["pen_mac"]: p for p in status["pens"]}
    assert by_mac["AA:BB"]["acked_chunks"] == 1
    assert by_mac["CC:DD"]["acked_chunks"] == 0


# -----------------------------------------------------------------------
# U-UPL-16: resume — re-init preserves existing acked_chunks
# -----------------------------------------------------------------------

def test_reinit_preserves_acked_chunks(ledger: UploadLedger) -> None:
    """U-UPL-16: re-calling init_pen after a crash preserves acked state."""
    ledger.init_pen("E1", "AA:BB", 5, "wifi")
    ledger.mark_chunk_acked("E1", "AA:BB", 0)
    ledger.mark_chunk_acked("E1", "AA:BB", 2)

    # Simulate restart — re-init with same total_chunks
    ledger.init_pen("E1", "AA:BB", 5, "wifi")

    pending = ledger.get_pending_chunks("E1", "AA:BB")
    assert 0 not in pending
    assert 2 not in pending
    assert len(pending) == 3


# -----------------------------------------------------------------------
# U-UPL-16b: nonexistent pen returns empty pending
# -----------------------------------------------------------------------

def test_pending_nonexistent_pen(ledger: UploadLedger) -> None:
    """U-UPL-16b: get_pending_chunks for unknown pen returns []."""
    assert ledger.get_pending_chunks("E_NONE", "XX:YY") == []


def test_is_complete_nonexistent_pen(ledger: UploadLedger) -> None:
    """U-UPL-16c: is_pen_complete for unknown pen returns False."""
    assert not ledger.is_pen_complete("E_NONE", "XX:YY")
