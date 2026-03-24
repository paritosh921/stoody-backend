"""Chunk assembly and integrity verification — ZERO I/O, pure domain logic.

Implements chunk wire format from ble-gatt-spec.md Section 3 and
CRC-32 verification per chunk and whole-buffer.

The domain layer must NEVER import asyncio, bleak, or any I/O library
(CLAUDE.md Per-Service Layer Rules).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

from src.config import CHUNK_FLAG_FIRST, CHUNK_FLAG_LAST, CHUNK_HEADER_SIZE


@dataclass(slots=True, frozen=True)
class ChunkHeader:
    """Parsed header from ble-gatt-spec.md Section 3 chunk wire format."""

    version: int
    flags: int
    header_len: int
    chunk_index: int
    total_chunks: int
    payload_len: int
    payload_crc32: int

    @property
    def is_first(self) -> bool:
        return bool(self.flags & CHUNK_FLAG_FIRST)

    @property
    def is_last(self) -> bool:
        return bool(self.flags & CHUNK_FLAG_LAST)


@dataclass(slots=True)
class ChunkEntry:
    """A single verified chunk with its payload."""

    index: int
    payload: bytes
    crc_verified: bool


@dataclass(slots=True)
class ChunkBuffer:
    """Accumulates chunks, verifies per-chunk CRC and whole-buffer checksum.

    All operations are pure — no I/O, no side effects beyond internal state.
    """

    total_chunks: int = 0
    expected_total_bytes: int = 0
    expected_buffer_crc32: str = ""
    _chunks: dict[int, ChunkEntry] = field(default_factory=dict)

    @property
    def received_count(self) -> int:
        return len(self._chunks)

    @property
    def received_bytes(self) -> int:
        return sum(len(c.payload) for c in self._chunks.values())

    @property
    def is_complete(self) -> bool:
        return (
            self.total_chunks > 0
            and self.received_count >= self.total_chunks
        )

    @property
    def missing_indices(self) -> list[int]:
        if self.total_chunks <= 0:
            return []
        expected = set(range(self.total_chunks))
        return sorted(expected - set(self._chunks.keys()))

    def get_chunk(self, index: int) -> ChunkEntry | None:
        return self._chunks.get(index)

    # ---------------------------------------------------------------- parse

    def add_raw_chunk(self, raw: bytes) -> tuple[ChunkEntry | None, str]:
        """Parse a raw chunk frame, verify CRC, and accumulate.

        Returns ``(entry, error)`` where *error* is empty on success.
        """
        header, err = parse_chunk_header(raw)
        if err:
            return None, err

        payload = raw[CHUNK_HEADER_SIZE : CHUNK_HEADER_SIZE + header.payload_len]
        if len(payload) < header.payload_len:
            return None, (
                f"Incomplete payload: got {len(payload)}, "
                f"expected {header.payload_len}"
            )

        # Per-chunk CRC-32 verification (ble-gatt-spec.md Section 3)
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != header.payload_crc32:
            return None, (
                f"Chunk {header.chunk_index} CRC mismatch: "
                f"expected {header.payload_crc32:08x}, got {actual_crc:08x}"
            )

        if self.total_chunks == 0:
            self.total_chunks = header.total_chunks

        entry = ChunkEntry(
            index=header.chunk_index,
            payload=payload,
            crc_verified=True,
        )
        self._chunks[header.chunk_index] = entry
        return entry, ""

    # -------------------------------------------------------- whole-buffer

    def verify_whole_buffer(self) -> tuple[bool, str, str]:
        """Compute CRC-32 over all chunk payloads in index order.

        Returns ``(ok, actual_crc_hex, error)``.
        """
        if not self.is_complete:
            missing = self.missing_indices
            return False, "", f"Buffer incomplete: missing chunks {missing}"

        crc = 0
        for idx in range(self.total_chunks):
            entry = self._chunks.get(idx)
            if entry is None:
                return False, "", f"Missing chunk {idx} during verification"
            crc = zlib.crc32(entry.payload, crc)

        actual_hex = format(crc & 0xFFFFFFFF, "08x")

        if self.expected_buffer_crc32 and actual_hex != self.expected_buffer_crc32:
            return False, actual_hex, (
                f"Buffer checksum mismatch: "
                f"expected {self.expected_buffer_crc32}, got {actual_hex}"
            )

        return True, actual_hex, ""

    # --------------------------------------------------------- reset

    def reset(self) -> None:
        """Clear all accumulated chunks for a re-sync attempt."""
        self._chunks.clear()


# ===================================================================
# Parsing helpers
# ===================================================================

def parse_chunk_header(raw: bytes) -> tuple[ChunkHeader | None, str]:
    """Parse the 20-byte chunk header from ble-gatt-spec.md Section 3.

    Returns ``(header, error)`` where *error* is empty on success.
    """
    if len(raw) < CHUNK_HEADER_SIZE:
        return None, (
            f"Raw data too short for header: {len(raw)} < {CHUNK_HEADER_SIZE}"
        )

    # Wire layout (ble-gatt-spec.md Section 3):
    #   offset 0:  version     u8
    #   offset 1:  flags       u8
    #   offset 2:  header_len  u16 LE
    #   offset 4:  chunk_index u32 LE
    #   offset 8:  total_chunks u32 LE
    #   offset 12: payload_len u32 LE
    #   offset 16: payload_crc32 u32 LE
    (
        version,
        flags,
        header_len,
        chunk_index,
        total_chunks,
        payload_len,
        payload_crc32,
    ) = struct.unpack_from("<BBHIIII", raw, 0)

    return ChunkHeader(
        version=version,
        flags=flags,
        header_len=header_len,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        payload_len=payload_len,
        payload_crc32=payload_crc32,
    ), ""


def build_chunk_frame(
    chunk_index: int,
    total_chunks: int,
    payload: bytes,
    *,
    flags: int = 0,
    version: int = 1,
) -> bytes:
    """Build a complete chunk frame (header + payload) for testing.

    Automatically sets FIRST/LAST flags and computes payload CRC-32.
    """
    if chunk_index == 0:
        flags |= CHUNK_FLAG_FIRST
    if chunk_index == total_chunks - 1:
        flags |= CHUNK_FLAG_LAST

    payload_crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(
        "<BBHIIII",
        version,
        flags,
        CHUNK_HEADER_SIZE,
        chunk_index,
        total_chunks,
        len(payload),
        payload_crc,
    )
    return header + payload
