"""Integrity verification for hub-store data.

Provides:
- CRC-32 checksum of a pen's entire raw stroke buffer.
- Byte-level comparison between SD and USB copies.
- SQLite ``PRAGMA integrity_check`` wrapper.

Referenced by FAILURE_MITIGATION_REGISTER.md S4 and STATE_OWNERSHIP_MAP.md
(hub-store is the sole writer and verifier of local stroke data).
"""

from __future__ import annotations

import logging
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path

from src.config import StoreConfig

logger = logging.getLogger(__name__)


def _safe_dir_name(name: str) -> str:
    """Sanitize a name for filesystem use (colons invalid on Windows)."""
    return name.replace(":", "-")

BUFFER_SIZE = 64 * 1024  # 64 KiB read buffer


@dataclass(slots=True)
class IntegrityResult:
    """Outcome of a single integrity check."""

    ok: bool
    detail: str


# -----------------------------------------------------------------------
# CRC-32 of pen buffer
# -----------------------------------------------------------------------

def compute_crc32(file_path: Path) -> str:
    """Return the CRC-32 hex string for *file_path* contents."""
    crc = 0
    with open(file_path, "rb") as f:
        while chunk := f.read(BUFFER_SIZE):
            crc = zlib.crc32(chunk, crc)
    return format(crc & 0xFFFFFFFF, "08x")


def verify_pen_crc(
    config: StoreConfig,
    exam_id: str,
    pen_mac: str,
    expected_crc: str,
) -> IntegrityResult:
    """Check that the SD copy of ``strokes_raw.bin`` matches *expected_crc*."""
    raw_path = config.sd_data / exam_id / _safe_dir_name(pen_mac) / "strokes_raw.bin"
    if not raw_path.exists():
        return IntegrityResult(ok=False, detail=f"File not found: {raw_path}")

    actual = compute_crc32(raw_path)
    if actual != expected_crc:
        return IntegrityResult(
            ok=False,
            detail=f"CRC mismatch: expected {expected_crc}, got {actual}",
        )
    return IntegrityResult(ok=True, detail="CRC OK")


# -----------------------------------------------------------------------
# SD / USB byte comparison
# -----------------------------------------------------------------------

def compare_sd_usb(
    config: StoreConfig,
    exam_id: str,
    pen_mac: str,
) -> IntegrityResult:
    """Byte-compare the SD and USB copies of ``strokes_raw.bin``."""
    sd_path = config.sd_data / exam_id / _safe_dir_name(pen_mac) / "strokes_raw.bin"
    usb_path = config.usb_data / exam_id / _safe_dir_name(pen_mac) / "strokes_raw.bin"

    if not sd_path.exists():
        return IntegrityResult(ok=False, detail=f"SD file missing: {sd_path}")
    if not usb_path.exists():
        return IntegrityResult(ok=False, detail=f"USB file missing: {usb_path}")

    sd_size = sd_path.stat().st_size
    usb_size = usb_path.stat().st_size
    if sd_size != usb_size:
        return IntegrityResult(
            ok=False,
            detail=f"Size mismatch: SD={sd_size} USB={usb_size}",
        )

    with open(sd_path, "rb") as sd_f, open(usb_path, "rb") as usb_f:
        offset = 0
        while True:
            sd_chunk = sd_f.read(BUFFER_SIZE)
            usb_chunk = usb_f.read(BUFFER_SIZE)
            if sd_chunk != usb_chunk:
                return IntegrityResult(
                    ok=False,
                    detail=f"Content mismatch starting at byte {offset}",
                )
            if not sd_chunk:
                break
            offset += len(sd_chunk)

    return IntegrityResult(ok=True, detail="SD and USB are byte-identical")


# -----------------------------------------------------------------------
# SQLite integrity check
# -----------------------------------------------------------------------

def check_sqlite_integrity(db_path: Path) -> IntegrityResult:
    """Run ``PRAGMA integrity_check`` on the hub SQLite database."""
    if not db_path.exists():
        return IntegrityResult(ok=False, detail=f"DB not found: {db_path}")

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.execute("PRAGMA integrity_check;")
            rows = cursor.fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return IntegrityResult(ok=False, detail=f"SQLite error: {exc}")

    if len(rows) == 1 and rows[0][0] == "ok":
        return IntegrityResult(ok=True, detail="SQLite integrity OK")

    issues = "; ".join(row[0] for row in rows[:10])
    return IntegrityResult(ok=False, detail=f"SQLite issues: {issues}")
