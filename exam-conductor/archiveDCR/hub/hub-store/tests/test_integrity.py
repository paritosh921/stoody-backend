"""Tests for integrity verification: CRC-32, SD/USB comparison, SQLite check.

Test IDs: U-INT-01 .. U-INT-07
Validation level: L3 (unit — temp dirs and in-memory SQLite)
"""

from __future__ import annotations

import sqlite3
import zlib
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import StoreConfig
from src.integrity import (
    check_sqlite_integrity,
    compare_sd_usb,
    compute_crc32,
    verify_pen_crc,
)


@pytest.fixture()
def store_dirs(tmp_path: Path) -> StoreConfig:
    sd = tmp_path / "sd"
    usb = tmp_path / "usb"
    sd.mkdir()
    usb.mkdir()
    return StoreConfig(sd_base=sd, usb_base=usb)


def _write_pen_file(base: Path, exam_id: str, pen_mac: str, data: bytes) -> Path:
    pen_dir = base / "data" / exam_id / pen_mac.replace(":", "-")
    pen_dir.mkdir(parents=True, exist_ok=True)
    raw = pen_dir / "strokes_raw.bin"
    raw.write_bytes(data)
    return raw


# -----------------------------------------------------------------------
# U-INT-01: compute_crc32 returns correct hex string
# -----------------------------------------------------------------------

def test_compute_crc32(tmp_path: Path) -> None:
    """U-INT-01: CRC-32 matches stdlib zlib.crc32."""
    data = b"Hello, ExamPen!"
    p = tmp_path / "test.bin"
    p.write_bytes(data)

    expected = format(zlib.crc32(data) & 0xFFFFFFFF, "08x")
    assert compute_crc32(p) == expected


# -----------------------------------------------------------------------
# U-INT-02: verify_pen_crc passes when CRC matches
# -----------------------------------------------------------------------

def test_verify_pen_crc_pass(store_dirs: StoreConfig) -> None:
    """U-INT-02: verification succeeds when CRC is correct."""
    data = b"pen stroke bytes"
    _write_pen_file(store_dirs.sd_base, "E1", "AA:BB", data)
    crc = format(zlib.crc32(data) & 0xFFFFFFFF, "08x")

    result = verify_pen_crc(store_dirs, "E1", "AA:BB", crc)
    assert result.ok is True


# -----------------------------------------------------------------------
# U-INT-03: verify_pen_crc fails on mismatch
# -----------------------------------------------------------------------

def test_verify_pen_crc_fail(store_dirs: StoreConfig) -> None:
    """U-INT-03: verification fails when CRC does not match."""
    _write_pen_file(store_dirs.sd_base, "E1", "AA:BB", b"data")

    result = verify_pen_crc(store_dirs, "E1", "AA:BB", "00000000")
    assert result.ok is False
    assert "CRC mismatch" in result.detail


# -----------------------------------------------------------------------
# U-INT-04: verify_pen_crc fails when file missing
# -----------------------------------------------------------------------

def test_verify_pen_crc_missing(store_dirs: StoreConfig) -> None:
    """U-INT-04: missing file yields an error result."""
    result = verify_pen_crc(store_dirs, "E1", "NOPE", "deadbeef")
    assert result.ok is False
    assert "not found" in result.detail.lower()


# -----------------------------------------------------------------------
# U-INT-05: compare_sd_usb passes when copies are identical
# -----------------------------------------------------------------------

def test_compare_sd_usb_identical(store_dirs: StoreConfig) -> None:
    """U-INT-05: byte comparison succeeds for identical files."""
    data = b"identical"
    _write_pen_file(store_dirs.sd_base, "E1", "AA:BB", data)
    _write_pen_file(store_dirs.usb_base, "E1", "AA:BB", data)

    result = compare_sd_usb(store_dirs, "E1", "AA:BB")
    assert result.ok is True


# -----------------------------------------------------------------------
# U-INT-06: compare_sd_usb fails when content differs
# -----------------------------------------------------------------------

def test_compare_sd_usb_differs(store_dirs: StoreConfig) -> None:
    """U-INT-06: byte comparison fails if contents differ."""
    _write_pen_file(store_dirs.sd_base, "E1", "AA:BB", b"alpha")
    _write_pen_file(store_dirs.usb_base, "E1", "AA:BB", b"betax")

    result = compare_sd_usb(store_dirs, "E1", "AA:BB")
    assert result.ok is False


# -----------------------------------------------------------------------
# U-INT-07: SQLite integrity check on healthy DB
# -----------------------------------------------------------------------

def test_sqlite_integrity_ok(tmp_path: Path) -> None:
    """U-INT-07: PRAGMA integrity_check returns OK on a fresh DB."""
    db_path = tmp_path / "hub.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY);")
    conn.close()

    result = check_sqlite_integrity(db_path)
    assert result.ok is True
