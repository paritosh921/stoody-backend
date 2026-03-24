"""Tests for chunk assembly and CRC verification (ZERO I/O, domain logic).

Test IDs: U-CHK-01 .. U-CHK-11
Validation level: L3 (unit — pure domain, no I/O)
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import pytest

# Allow imports from hub-pen-sync/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunk_manager import (
    ChunkBuffer,
    ChunkHeader,
    build_chunk_frame,
    parse_chunk_header,
)
from src.config import CHUNK_FLAG_FIRST, CHUNK_FLAG_LAST, CHUNK_HEADER_SIZE


def _make_frame(
    index: int,
    total: int,
    payload: bytes,
    *,
    corrupt_crc: bool = False,
) -> bytes:
    """Build a valid chunk frame (or one with corrupted CRC)."""
    frame = build_chunk_frame(index, total, payload)
    if corrupt_crc:
        # Zero out the CRC field at offset 16 (4 bytes)
        frame = bytearray(frame)
        frame[16:20] = b"\x00\x00\x00\x00"
        frame = bytes(frame)
    return frame


# -----------------------------------------------------------------------
# U-CHK-01: parse valid chunk header
# -----------------------------------------------------------------------

def test_parse_valid_header() -> None:
    """U-CHK-01: parse a correctly formatted chunk header."""
    payload = b"hello-pen-data"
    frame = build_chunk_frame(0, 5, payload)

    header, err = parse_chunk_header(frame)
    assert err == ""
    assert header is not None
    assert header.version == 1
    assert header.chunk_index == 0
    assert header.total_chunks == 5
    assert header.payload_len == len(payload)
    assert header.payload_crc32 == (zlib.crc32(payload) & 0xFFFFFFFF)
    assert header.is_first is True
    assert header.is_last is False


# -----------------------------------------------------------------------
# U-CHK-02: parse rejects too-short data
# -----------------------------------------------------------------------

def test_parse_rejects_short_data() -> None:
    """U-CHK-02: data shorter than header size returns error."""
    _, err = parse_chunk_header(b"\x00" * 10)
    assert "too short" in err


# -----------------------------------------------------------------------
# U-CHK-03: single chunk add and retrieve
# -----------------------------------------------------------------------

def test_single_chunk_add() -> None:
    """U-CHK-03: add one chunk, verify it's stored and retrievable."""
    buf = ChunkBuffer()
    payload = b"test-data-123"
    frame = _make_frame(0, 1, payload)

    entry, err = buf.add_raw_chunk(frame)
    assert err == ""
    assert entry is not None
    assert entry.index == 0
    assert entry.payload == payload
    assert entry.crc_verified is True
    assert buf.received_count == 1
    assert buf.total_chunks == 1
    assert buf.is_complete is True


# -----------------------------------------------------------------------
# U-CHK-04: CRC mismatch rejects chunk
# -----------------------------------------------------------------------

def test_crc_mismatch_rejects() -> None:
    """U-CHK-04: corrupted CRC in frame is detected and rejected."""
    buf = ChunkBuffer()
    frame = _make_frame(0, 1, b"valid-data", corrupt_crc=True)

    entry, err = buf.add_raw_chunk(frame)
    assert entry is None
    assert "CRC mismatch" in err
    assert buf.received_count == 0


# -----------------------------------------------------------------------
# U-CHK-05: multi-chunk assembly in order
# -----------------------------------------------------------------------

def test_multi_chunk_in_order() -> None:
    """U-CHK-05: accumulate 3 chunks in order, verify completeness."""
    buf = ChunkBuffer()
    payloads = [b"chunk-0-data", b"chunk-1-data", b"chunk-2-data"]

    for i, p in enumerate(payloads):
        entry, err = buf.add_raw_chunk(_make_frame(i, 3, p))
        assert err == ""
        assert entry is not None

    assert buf.received_count == 3
    assert buf.total_chunks == 3
    assert buf.is_complete is True
    assert buf.missing_indices == []


# -----------------------------------------------------------------------
# U-CHK-06: out-of-order delivery accepted
# -----------------------------------------------------------------------

def test_out_of_order_delivery() -> None:
    """U-CHK-06: chunks arriving out of order are correctly assembled."""
    buf = ChunkBuffer()
    payloads = {0: b"first", 2: b"third", 1: b"second"}

    for idx, p in payloads.items():
        entry, err = buf.add_raw_chunk(_make_frame(idx, 3, p))
        assert err == ""

    assert buf.is_complete is True
    assert buf.get_chunk(0).payload == b"first"
    assert buf.get_chunk(1).payload == b"second"
    assert buf.get_chunk(2).payload == b"third"


# -----------------------------------------------------------------------
# U-CHK-07: missing indices tracked correctly
# -----------------------------------------------------------------------

def test_missing_indices() -> None:
    """U-CHK-07: missing_indices reports gaps in chunk sequence."""
    buf = ChunkBuffer(total_chunks=5)

    buf.add_raw_chunk(_make_frame(0, 5, b"a"))
    buf.add_raw_chunk(_make_frame(2, 5, b"c"))
    buf.add_raw_chunk(_make_frame(4, 5, b"e"))

    assert buf.missing_indices == [1, 3]
    assert buf.is_complete is False


# -----------------------------------------------------------------------
# U-CHK-08: whole-buffer checksum verification passes
# -----------------------------------------------------------------------

def test_whole_buffer_checksum_pass() -> None:
    """U-CHK-08: verify_whole_buffer succeeds when CRC matches."""
    payloads = [b"aaa", b"bbb", b"ccc"]

    # Compute expected CRC over concatenated payloads (rolling)
    crc = 0
    for p in payloads:
        crc = zlib.crc32(p, crc)
    expected_hex = format(crc & 0xFFFFFFFF, "08x")

    buf = ChunkBuffer(expected_buffer_crc32=expected_hex)

    for i, p in enumerate(payloads):
        buf.add_raw_chunk(_make_frame(i, 3, p))

    ok, actual_hex, err = buf.verify_whole_buffer()
    assert ok is True
    assert actual_hex == expected_hex
    assert err == ""


# -----------------------------------------------------------------------
# U-CHK-09: whole-buffer checksum mismatch detected
# -----------------------------------------------------------------------

def test_whole_buffer_checksum_mismatch() -> None:
    """U-CHK-09: verify_whole_buffer detects wrong checksum."""
    buf = ChunkBuffer(expected_buffer_crc32="deadbeef")

    buf.add_raw_chunk(_make_frame(0, 1, b"some-data"))

    ok, actual_hex, err = buf.verify_whole_buffer()
    assert ok is False
    assert "mismatch" in err
    assert actual_hex != ""


# -----------------------------------------------------------------------
# U-CHK-10: verify on incomplete buffer returns error
# -----------------------------------------------------------------------

def test_verify_incomplete_buffer() -> None:
    """U-CHK-10: verify_whole_buffer fails if chunks are missing."""
    buf = ChunkBuffer(total_chunks=3)
    buf.add_raw_chunk(_make_frame(0, 3, b"only-first"))

    ok, _, err = buf.verify_whole_buffer()
    assert ok is False
    assert "incomplete" in err.lower() or "missing" in err.lower()


# -----------------------------------------------------------------------
# U-CHK-11: reset clears accumulated chunks
# -----------------------------------------------------------------------

def test_reset_clears_chunks() -> None:
    """U-CHK-11: reset() clears all accumulated data for re-sync."""
    buf = ChunkBuffer()
    buf.add_raw_chunk(_make_frame(0, 2, b"data"))
    assert buf.received_count == 1

    buf.reset()
    assert buf.received_count == 0
    assert buf.is_complete is False


# -----------------------------------------------------------------------
# U-CHK-12: partial payload (truncated) is rejected
# -----------------------------------------------------------------------

def test_partial_payload_rejected() -> None:
    """U-CHK-12: frame with truncated payload is detected."""
    payload = b"full-payload-data"
    frame = build_chunk_frame(0, 1, payload)
    # Truncate the frame to cut into the payload
    truncated = frame[: CHUNK_HEADER_SIZE + 5]

    buf = ChunkBuffer()
    entry, err = buf.add_raw_chunk(truncated)
    assert entry is None
    assert "Incomplete" in err


# -----------------------------------------------------------------------
# U-CHK-13: FIRST and LAST flags set correctly by build_chunk_frame
# -----------------------------------------------------------------------

def test_first_last_flags() -> None:
    """U-CHK-13: build_chunk_frame sets correct flags."""
    # Single chunk: both FIRST and LAST
    frame = build_chunk_frame(0, 1, b"x")
    header, _ = parse_chunk_header(frame)
    assert header.is_first is True
    assert header.is_last is True

    # First of many: only FIRST
    frame = build_chunk_frame(0, 3, b"x")
    header, _ = parse_chunk_header(frame)
    assert header.is_first is True
    assert header.is_last is False

    # Middle: neither
    frame = build_chunk_frame(1, 3, b"x")
    header, _ = parse_chunk_header(frame)
    assert header.is_first is False
    assert header.is_last is False

    # Last: only LAST
    frame = build_chunk_frame(2, 3, b"x")
    header, _ = parse_chunk_header(frame)
    assert header.is_first is False
    assert header.is_last is True


# -----------------------------------------------------------------------
# U-CHK-14: received_bytes tracks total payload size
# -----------------------------------------------------------------------

def test_received_bytes_tracking() -> None:
    """U-CHK-14: received_bytes sums payload sizes across chunks."""
    buf = ChunkBuffer()
    p1 = b"short"
    p2 = b"a-longer-payload"

    buf.add_raw_chunk(_make_frame(0, 2, p1))
    buf.add_raw_chunk(_make_frame(1, 2, p2))

    assert buf.received_bytes == len(p1) + len(p2)
