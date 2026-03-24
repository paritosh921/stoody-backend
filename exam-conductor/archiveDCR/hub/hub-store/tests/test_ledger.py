"""Tests for SQLite ledger integration and USB metadata mirroring.

Test IDs: U-STR-09 .. U-STR-16
Validation level: L3 (unit — uses temp dirs and in-memory SQLite)
"""

from __future__ import annotations

import base64
import json
import sqlite3
import zlib
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import StoreConfig
from src.dual_writer import DualWriter
from src.ledger import ChunkLedger, open_ledger_db


def _encode(data: bytes) -> tuple[str, str]:
    b64 = base64.b64encode(data).decode("ascii")
    crc = format(zlib.crc32(data) & 0xFFFFFFFF, "08x")
    return b64, crc


@pytest.fixture()
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    return open_ledger_db(tmp_path / "hub.db")


@pytest.fixture()
def ledger(db_conn: sqlite3.Connection) -> ChunkLedger:
    return ChunkLedger(db_conn)


@pytest.fixture()
def store_with_ledger(tmp_path: Path) -> tuple[StoreConfig, ChunkLedger]:
    sd = tmp_path / "sd"
    usb = tmp_path / "usb"
    sd.mkdir()
    usb.mkdir()
    cfg = StoreConfig(sd_base=sd, usb_base=usb)
    conn = open_ledger_db(cfg.db_path)
    return cfg, ChunkLedger(conn)


# -----------------------------------------------------------------------
# U-STR-09: SQLite WAL mode is enabled
# -----------------------------------------------------------------------

def test_wal_mode_enabled(tmp_path: Path) -> None:
    """U-STR-09: open_ledger_db creates database in WAL journal mode."""
    conn = open_ledger_db(tmp_path / "hub.db")
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode == "wal"


# -----------------------------------------------------------------------
# U-STR-10: pen_sync_status and upload_ledger tables exist
# -----------------------------------------------------------------------

def test_tables_created(db_conn: sqlite3.Connection) -> None:
    """U-STR-10: required tables are created on open."""
    tables = {
        row[0]
        for row in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "pen_sync_status" in tables
    assert "upload_ledger" in tables


# -----------------------------------------------------------------------
# U-STR-11: chunk write creates/updates pen_sync_status row
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chunk_write_updates_sync_status(
    store_with_ledger: tuple[StoreConfig, ChunkLedger],
) -> None:
    """U-STR-11: after a chunk write, pen_sync_status has a 'syncing' row."""
    cfg, ledger = store_with_ledger
    writer = DualWriter(cfg, ledger=ledger)

    data = b"test-chunk"
    b64, crc = _encode(data)
    await writer.write_pen_chunk("EXAM-1", "AA:BB", 0, b64, crc)

    row = ledger.get_sync_status("EXAM-1", "AA:BB")
    assert row is not None
    assert row["status"] == "syncing"
    assert row["bytes_received"] == len(data)


# -----------------------------------------------------------------------
# U-STR-12: multiple chunks accumulate bytes_received
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bytes_accumulate(
    store_with_ledger: tuple[StoreConfig, ChunkLedger],
) -> None:
    """U-STR-12: bytes_received sums across multiple chunks."""
    cfg, ledger = store_with_ledger
    writer = DualWriter(cfg, ledger=ledger)

    d1 = b"chunk-zero"
    d2 = b"chunk-one-longer"
    b1, c1 = _encode(d1)
    b2, c2 = _encode(d2)

    await writer.write_pen_chunk("E1", "AA:BB", 0, b1, c1)
    await writer.write_pen_chunk("E1", "AA:BB", 1, b2, c2)

    row = ledger.get_sync_status("E1", "AA:BB")
    assert row is not None
    assert row["bytes_received"] == len(d1) + len(d2)


# -----------------------------------------------------------------------
# U-STR-13: metadata is written to BOTH SD and USB paths
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metadata_mirrored_to_usb(
    store_with_ledger: tuple[StoreConfig, ChunkLedger],
) -> None:
    """U-STR-13: strokes.meta.json exists on both SD and USB after write."""
    cfg, ledger = store_with_ledger
    writer = DualWriter(cfg, ledger=ledger)

    data = b"mirror-test"
    b64, crc = _encode(data)
    await writer.write_pen_chunk("E1", "AA:BB", 0, b64, crc)

    sd_meta = cfg.sd_data / "E1" / "AA-BB" / "strokes.meta.json"
    usb_meta = cfg.usb_data / "E1" / "AA-BB" / "strokes.meta.json"

    assert sd_meta.exists(), "Metadata missing on SD"
    assert usb_meta.exists(), "Metadata missing on USB"

    sd_content = json.loads(sd_meta.read_text())
    usb_content = json.loads(usb_meta.read_text())

    assert sd_content["bytes"] == len(data)
    assert usb_content["bytes"] == len(data)
    assert sd_content["checksum_crc32"] == usb_content["checksum_crc32"]


# -----------------------------------------------------------------------
# U-STR-14: metadata NOT mirrored to USB when USB unavailable
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metadata_not_mirrored_when_usb_missing(tmp_path: Path) -> None:
    """U-STR-14: when USB is missing, metadata only on SD."""
    sd = tmp_path / "sd"
    sd.mkdir()
    usb = tmp_path / "usb_gone"  # NOT created
    cfg = StoreConfig(sd_base=sd, usb_base=usb)

    writer = DualWriter(cfg)
    data = b"sd-only"
    b64, crc = _encode(data)
    await writer.write_pen_chunk("E1", "AA:BB", 0, b64, crc)

    assert (cfg.sd_data / "E1" / "AA-BB" / "strokes.meta.json").exists()
    assert not (cfg.usb_data / "E1" / "AA-BB" / "strokes.meta.json").exists()


# -----------------------------------------------------------------------
# U-STR-15: upload ledger initialisation
# -----------------------------------------------------------------------

def test_upload_ledger_init(ledger: ChunkLedger) -> None:
    """U-STR-15: init_upload_ledger creates a row with correct defaults."""
    ledger.init_upload_ledger("E1", "AA:BB", total_chunks=5)

    status = ledger.get_upload_status("E1", "AA:BB")
    assert status is not None
    assert status["total_chunks"] == 5
    assert status["acked_chunks"] == []
    assert status["complete"] == 0
    assert status["started_at"] is not None


# -----------------------------------------------------------------------
# U-STR-16: mark_sync_complete and mark_sync_failed
# -----------------------------------------------------------------------

def test_sync_lifecycle(ledger: ChunkLedger) -> None:
    """U-STR-16: record_chunk_received -> mark_sync_complete works."""
    ledger.record_chunk_received("E1", "AA:BB", 100)
    row = ledger.get_sync_status("E1", "AA:BB")
    assert row["status"] == "syncing"

    ledger.mark_sync_complete("E1", "AA:BB", "abcd1234")
    row = ledger.get_sync_status("E1", "AA:BB")
    assert row["status"] == "complete"
    assert row["checksum_actual"] == "abcd1234"
    assert row["sync_completed"] is not None


def test_sync_failure(ledger: ChunkLedger) -> None:
    """U-STR-16b: record_chunk_received -> mark_sync_failed stores error."""
    ledger.record_chunk_received("E1", "CC:DD", 50)
    ledger.mark_sync_failed("E1", "CC:DD", "timeout after 30s")

    row = ledger.get_sync_status("E1", "CC:DD")
    assert row["status"] == "failed"
    assert row["error_detail"] == "timeout after 30s"


def test_first_chunk_failure_creates_row(ledger: ChunkLedger) -> None:
    """U-STR-17: mark_sync_failed on first chunk (no prior row) must INSERT.

    Regression test: the original UPDATE-only implementation silently
    dropped failures when no row existed yet, leaving the ledger empty.
    """
    # No prior record_chunk_received — this is the very first interaction
    assert ledger.get_sync_status("E_NEW", "FF:FF") is None

    ledger.mark_sync_failed("E_NEW", "FF:FF", "CRC mismatch on first chunk")

    row = ledger.get_sync_status("E_NEW", "FF:FF")
    assert row is not None, "mark_sync_failed must create a row if none exists"
    assert row["status"] == "failed"
    assert row["error_detail"] == "CRC mismatch on first chunk"
    assert row["bytes_received"] == 0
    assert row["sync_started"] is not None
