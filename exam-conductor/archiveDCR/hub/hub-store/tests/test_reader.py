"""Tests for read-only functions — MUST NOT mutate the filesystem.

STATE_OWNERSHIP_MAP.md Section 2.1 declares ``hub-store.get_pen_data()``
as a pure read.  These tests snapshot the filesystem before and after
every call to assert zero side effects.

Test IDs: U-RDR-01 .. U-RDR-07
Validation level: L3 (unit — temp dirs)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import StoreConfig
from src.reader import get_chunk, get_pen_data, list_pen_data


@pytest.fixture()
def store_dirs(tmp_path: Path) -> StoreConfig:
    sd = tmp_path / "sd"
    usb = tmp_path / "usb"
    sd.mkdir()
    usb.mkdir()
    return StoreConfig(sd_base=sd, usb_base=usb)


# -------------------------------------------------------------------
# Filesystem snapshot helper
# -------------------------------------------------------------------

def _snapshot(base: Path) -> dict[str, tuple[float, int, str]]:
    """Return {relpath: (mtime, size, md5)} for every file under *base*."""
    snap: dict[str, tuple[float, int, str]] = {}
    for root, _dirs, files in os.walk(base):
        for name in files:
            p = Path(root) / name
            stat = p.stat()
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            rel = str(p.relative_to(base))
            snap[rel] = (stat.st_mtime, stat.st_size, md5)
    return snap


def _seed_pen(cfg: StoreConfig, exam_id: str, pen_mac: str, data: bytes) -> None:
    """Create a minimal pen data tree on SD."""
    pen_dir = cfg.sd_data / exam_id / pen_mac.replace(":", "-")
    pen_dir.mkdir(parents=True, exist_ok=True)
    (pen_dir / "strokes_raw.bin").write_bytes(data)
    chunks = pen_dir / "chunks"
    chunks.mkdir(exist_ok=True)
    (chunks / "chunk_000.bin").write_bytes(data[:8] if len(data) >= 8 else data)


# -------------------------------------------------------------------
# U-RDR-01: get_pen_data returns correct bytes
# -------------------------------------------------------------------

def test_get_pen_data_returns_bytes(store_dirs: StoreConfig) -> None:
    """U-RDR-01: returns full strokes_raw.bin content."""
    payload = b"raw-stroke-bytes-0123456789"
    _seed_pen(store_dirs, "E1", "AA:BB", payload)

    result = get_pen_data(store_dirs, "E1", "AA:BB")
    assert result == payload


# -------------------------------------------------------------------
# U-RDR-02: get_pen_data does not mutate filesystem
# -------------------------------------------------------------------

def test_get_pen_data_no_side_effects(store_dirs: StoreConfig) -> None:
    """U-RDR-02: filesystem snapshot is identical before and after read."""
    _seed_pen(store_dirs, "E1", "AA:BB", b"data")
    before = _snapshot(store_dirs.sd_base)

    _ = get_pen_data(store_dirs, "E1", "AA:BB")

    after = _snapshot(store_dirs.sd_base)
    assert before == after, "get_pen_data mutated the filesystem!"


# -------------------------------------------------------------------
# U-RDR-03: get_pen_data returns empty bytes when missing
# -------------------------------------------------------------------

def test_get_pen_data_missing(store_dirs: StoreConfig) -> None:
    """U-RDR-03: missing file returns b'' (not an exception)."""
    result = get_pen_data(store_dirs, "E_NONE", "ZZ:ZZ")
    assert result == b""


# -------------------------------------------------------------------
# U-RDR-04: get_chunk returns correct chunk
# -------------------------------------------------------------------

def test_get_chunk_returns_bytes(store_dirs: StoreConfig) -> None:
    """U-RDR-04: returns the pre-chunked upload file."""
    _seed_pen(store_dirs, "E1", "AA:BB", b"some-payload")

    result = get_chunk(store_dirs, "E1", "AA:BB", 0)
    assert len(result) > 0


# -------------------------------------------------------------------
# U-RDR-05: get_chunk does not mutate filesystem
# -------------------------------------------------------------------

def test_get_chunk_no_side_effects(store_dirs: StoreConfig) -> None:
    """U-RDR-05: reading a chunk has zero side effects."""
    _seed_pen(store_dirs, "E1", "AA:BB", b"data-for-chunk")
    before = _snapshot(store_dirs.sd_base)

    _ = get_chunk(store_dirs, "E1", "AA:BB", 0)

    after = _snapshot(store_dirs.sd_base)
    assert before == after, "get_chunk mutated the filesystem!"


# -------------------------------------------------------------------
# U-RDR-06: list_pen_data returns pen MACs
# -------------------------------------------------------------------

def test_list_pen_data(store_dirs: StoreConfig) -> None:
    """U-RDR-06: enumerates pen MACs that have data."""
    _seed_pen(store_dirs, "E1", "AA:BB", b"d1")
    _seed_pen(store_dirs, "E1", "CC:DD", b"d2")

    result = list_pen_data(store_dirs, "E1")
    assert sorted(result) == ["AA-BB", "CC-DD"]


# -------------------------------------------------------------------
# U-RDR-07: list_pen_data does not mutate filesystem
# -------------------------------------------------------------------

def test_list_pen_data_no_side_effects(store_dirs: StoreConfig) -> None:
    """U-RDR-07: listing pens has zero side effects."""
    _seed_pen(store_dirs, "E1", "AA:BB", b"d")
    before = _snapshot(store_dirs.sd_base)

    _ = list_pen_data(store_dirs, "E1")

    after = _snapshot(store_dirs.sd_base)
    assert before == after, "list_pen_data mutated the filesystem!"
