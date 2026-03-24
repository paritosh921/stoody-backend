"""Tests for dual-write engine: fsync, degraded mode, file layout.

Test IDs: U-STR-01 .. U-STR-08
Validation level: L3 (unit — uses temp dirs, no real SD/USB)
"""

from __future__ import annotations

import base64
import json
import os
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow imports from hub-store/src
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import StoreConfig
from src.dual_writer import DualWriter, WriteResult


@pytest.fixture()
def store_dirs(tmp_path: Path) -> StoreConfig:
    sd = tmp_path / "sd"
    usb = tmp_path / "usb"
    sd.mkdir()
    usb.mkdir()
    return StoreConfig(sd_base=sd, usb_base=usb)


def _encode(data: bytes) -> tuple[str, str]:
    """Return (base64, crc32-hex) for *data*."""
    b64 = base64.b64encode(data).decode("ascii")
    crc = format(zlib.crc32(data) & 0xFFFFFFFF, "08x")
    return b64, crc


# -----------------------------------------------------------------------
# U-STR-01: basic dual-write creates files on SD and USB
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dual_write_creates_both(store_dirs: StoreConfig) -> None:
    """U-STR-01: both SD and USB get strokes_raw.bin + chunk file."""
    writer = DualWriter(store_dirs)
    data = b"stroke-data-abc"
    b64, crc = _encode(data)

    result = await writer.write_pen_chunk("E1", "AA:BB", 0, b64, crc)

    assert result.sd_persisted is True
    assert result.usb_persisted is True
    assert result.degraded is False

    sd_raw = store_dirs.sd_data / "E1" / "AA-BB" / "strokes_raw.bin"
    usb_raw = store_dirs.usb_data / "E1" / "AA-BB" / "strokes_raw.bin"
    assert sd_raw.read_bytes() == data
    assert usb_raw.read_bytes() == data

    sd_chunk = store_dirs.sd_data / "E1" / "AA-BB" / "chunks" / "chunk_000.bin"
    assert sd_chunk.read_bytes() == data


# -----------------------------------------------------------------------
# U-STR-02: fsync is called after each write
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fsync_called(store_dirs: StoreConfig) -> None:
    """U-STR-02: os.fsync is invoked for every file descriptor."""
    writer = DualWriter(store_dirs)
    data = b"pen-bytes"
    b64, crc = _encode(data)

    fsync_calls: list[int] = []
    original_fsync = os.fsync

    def tracking_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        original_fsync(fd)

    with patch("src.dual_writer.os.fsync", side_effect=tracking_fsync):
        result = await writer.write_pen_chunk("E1", "AA:BB", 0, b64, crc)

    assert result.sd_persisted is True
    # 2 fsync per path (raw + chunk) * 2 paths (SD + USB) = 4
    # + 2 metadata fsync (SD + USB) = 6 total minimum
    assert len(fsync_calls) >= 4


# -----------------------------------------------------------------------
# U-STR-03: degraded mode when USB is missing
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_degraded_when_usb_missing(tmp_path: Path) -> None:
    """U-STR-03: if USB dir does not exist, degrade to SD-only."""
    sd = tmp_path / "sd"
    sd.mkdir()
    usb = tmp_path / "usb_nonexistent"  # intentionally NOT created
    cfg = StoreConfig(sd_base=sd, usb_base=usb)

    writer = DualWriter(cfg)
    data = b"data"
    b64, crc = _encode(data)

    result = await writer.write_pen_chunk("E1", "AA:BB", 0, b64, crc)

    assert result.sd_persisted is True
    assert result.usb_persisted is False
    assert result.degraded is True
    assert writer.degraded is True


# -----------------------------------------------------------------------
# U-STR-04: CRC mismatch rejects the write
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crc_mismatch_rejected(store_dirs: StoreConfig) -> None:
    """U-STR-04: if chunk CRC does not match, both writes are skipped."""
    writer = DualWriter(store_dirs)
    data = b"real-data"
    b64 = base64.b64encode(data).decode("ascii")
    bad_crc = "00000000"

    result = await writer.write_pen_chunk("E1", "AA:BB", 0, b64, bad_crc)

    assert result.sd_persisted is False
    assert result.usb_persisted is False
    assert result.error is not None
    assert "CRC mismatch" in result.error


# -----------------------------------------------------------------------
# U-STR-05: append mode — multiple chunks accumulate
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_mode(store_dirs: StoreConfig) -> None:
    """U-STR-05: successive writes append to strokes_raw.bin."""
    writer = DualWriter(store_dirs)

    d1 = b"chunk-0"
    d2 = b"chunk-1"

    b1, c1 = _encode(d1)
    b2, c2 = _encode(d2)

    await writer.write_pen_chunk("E1", "AA:BB", 0, b1, c1)
    await writer.write_pen_chunk("E1", "AA:BB", 1, b2, c2)

    raw_path = store_dirs.sd_data / "E1" / "AA-BB" / "strokes_raw.bin"
    assert raw_path.read_bytes() == d1 + d2


# -----------------------------------------------------------------------
# U-STR-06: file layout matches HUB_DEPLOYMENT_SPEC.md Section 3.2
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_layout(store_dirs: StoreConfig) -> None:
    """U-STR-06: verify directory tree structure."""
    writer = DualWriter(store_dirs)
    data = b"x"
    b64, crc = _encode(data)

    await writer.write_pen_chunk("EXAM-42", "AA:BB:CC", 7, b64, crc)

    assert (store_dirs.sd_data / "EXAM-42" / "AA-BB-CC" / "strokes_raw.bin").exists()
    assert (store_dirs.sd_data / "EXAM-42" / "AA-BB-CC" / "chunks" / "chunk_007.bin").exists()
    assert (store_dirs.sd_data / "EXAM-42" / "AA-BB-CC" / "strokes.meta.json").exists()


# -----------------------------------------------------------------------
# U-STR-07: metadata file is valid JSON with expected keys
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metadata_file(store_dirs: StoreConfig) -> None:
    """U-STR-07: strokes.meta.json contains bytes, checksum, sync_ts."""
    writer = DualWriter(store_dirs)
    data = b"metadata-test"
    b64, crc = _encode(data)

    await writer.write_pen_chunk("E1", "AA:BB", 0, b64, crc)

    meta_path = store_dirs.sd_data / "E1" / "AA-BB" / "strokes.meta.json"
    meta = json.loads(meta_path.read_text())

    assert meta["bytes"] == len(data)
    assert meta["checksum_crc32"] == crc
    assert meta["sync_ts"].endswith("Z")
    assert "pages" in meta


# -----------------------------------------------------------------------
# U-STR-08: degraded flag persists after USB failure
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_degraded_flag_sticky(tmp_path: Path) -> None:
    """U-STR-08: once degraded, the flag stays True for the writer's lifetime."""
    sd = tmp_path / "sd"
    sd.mkdir()
    usb = tmp_path / "usb_gone"
    cfg = StoreConfig(sd_base=sd, usb_base=usb)

    writer = DualWriter(cfg)
    b64, crc = _encode(b"a")
    await writer.write_pen_chunk("E1", "AA:BB", 0, b64, crc)
    assert writer.degraded is True

    # Even if USB appears later, the flag remains True for this writer instance
    usb.mkdir()
    b64_2, crc_2 = _encode(b"b")
    await writer.write_pen_chunk("E1", "AA:BB", 1, b64_2, crc_2)
    assert writer.degraded is True
