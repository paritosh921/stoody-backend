"""GATT characteristic read — adapter layer for BLE pen communication.

Reads pen stroke buffer via BLE GATT using bleak. Follows the P05
offline sync flow (P05_pen_SDK.md) and ExamPen GATT spec
(ble-gatt-spec.md Section 1).

Protocol:
  1. Read Buffer Status characteristic (total_bytes, bytes_remaining, crc32)
  2. Write Sync Control = 0x01 (start)
  3. Subscribe to Stroke Buffer notifications
  4. Receive chunk frames (ble-gatt-spec.md Section 3)
  5. Each chunk is parsed and returned to caller for storage
  6. On disconnect: checkpoint last confirmed chunk for resume

IMPORTANT: This module does NOT clear the pen buffer. Buffer clear
(0x03) is ONLY sent by sync_orchestrator after hub-store confirms
dual-write for ALL chunks + checksum match.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import (
    BUFFER_STATUS_UUID,
    CHUNK_HEADER_SIZE,
    PEN_METADATA_UUID,
    STROKE_BUFFER_UUID,
    SYNC_CMD_ABORT,
    SYNC_CMD_CLEAR_BUFFER,
    SYNC_CMD_START,
    SYNC_CONTROL_UUID,
    PenSyncConfig,
)

logger = logging.getLogger(__name__)


class BleClient(Protocol):
    """Protocol for BLE client abstraction (allows mocking bleak)."""

    async def read_gatt_char(self, uuid: str) -> bytes: ...
    async def write_gatt_char(
        self, uuid: str, data: bytes, response: bool = True
    ) -> None: ...
    async def start_notify(
        self, uuid: str, callback: Any
    ) -> None: ...
    async def stop_notify(self, uuid: str) -> None: ...

    @property
    def is_connected(self) -> bool: ...


@dataclass(slots=True, frozen=True)
class BufferStatus:
    """Parsed Buffer Status characteristic (12 bytes, little-endian)."""

    total_bytes: int
    bytes_remaining: int
    checksum_crc32: str  # hex string


@dataclass(slots=True, frozen=True)
class PenMetadata:
    """Parsed Pen Metadata characteristic (8 bytes, little-endian)."""

    fw_version: int
    battery_pct: int
    pen_serial: int
    page_count: int


async def read_buffer_status(client: BleClient) -> BufferStatus:
    """Read the Buffer Status GATT characteristic.

    Returns parsed status with total bytes and CRC-32 checksum.
    Raises ``RuntimeError`` if the characteristic is unreadable.
    """
    raw = await client.read_gatt_char(BUFFER_STATUS_UUID)
    if len(raw) < 12:
        raise RuntimeError(
            f"Buffer status too short: {len(raw)} bytes (need 12)"
        )
    total_bytes, bytes_remaining, crc32_val = struct.unpack_from("<III", raw)
    return BufferStatus(
        total_bytes=total_bytes,
        bytes_remaining=bytes_remaining,
        checksum_crc32=format(crc32_val & 0xFFFFFFFF, "08x"),
    )


async def read_pen_metadata(client: BleClient) -> PenMetadata:
    """Read the Pen Metadata GATT characteristic."""
    raw = await client.read_gatt_char(PEN_METADATA_UUID)
    if len(raw) < 8:
        raise RuntimeError(
            f"Pen metadata too short: {len(raw)} bytes (need 8)"
        )
    # Layout: fw_version u16, battery_pct u8, pen_serial u32, page_count u8
    fw, battery, serial, pages = struct.unpack_from("<HBIB", raw, 0)
    return PenMetadata(
        fw_version=fw,
        battery_pct=battery,
        pen_serial=serial,
        page_count=pages,
    )


async def start_sync(client: BleClient) -> None:
    """Write Sync Control = 0x01 to begin data transfer."""
    await client.write_gatt_char(
        SYNC_CONTROL_UUID, bytes([SYNC_CMD_START]), response=True
    )
    logger.info("Sent sync start command (0x01)")


async def abort_sync(client: BleClient) -> None:
    """Write Sync Control = 0x02 to abort the current transfer."""
    await client.write_gatt_char(
        SYNC_CONTROL_UUID, bytes([SYNC_CMD_ABORT]), response=True
    )
    logger.info("Sent sync abort command (0x02)")


async def clear_pen_buffer(client: BleClient) -> None:
    """Write Sync Control = 0x03 to clear the pen's stroke buffer.

    CRITICAL DATA SAFETY: This MUST only be called after hub-store
    confirms dual-write for ALL chunks AND checksum matches.
    Pen data is irreplaceable once cleared.
    """
    await client.write_gatt_char(
        SYNC_CONTROL_UUID, bytes([SYNC_CMD_CLEAR_BUFFER]), response=True
    )
    logger.info("Sent buffer clear command (0x03) — pen data erased")


async def receive_chunks(
    client: BleClient,
    config: PenSyncConfig,
    on_chunk: Any,  # Callable[[bytes], Awaitable[None]]
) -> None:
    """Subscribe to Stroke Buffer notifications and forward raw frames.

    Calls ``on_chunk(raw_frame)`` for each notification received.
    Blocks until notifications stop or timeout is reached.
    """
    chunk_event: asyncio.Event = asyncio.Event()
    transfer_done: asyncio.Event = asyncio.Event()

    async def _notification_handler(
        _characteristic: Any, data: bytearray
    ) -> None:
        chunk_event.set()
        try:
            await on_chunk(bytes(data))
        except Exception:
            logger.exception("Error in on_chunk callback")
        # Check if this is the last chunk (flag 0x02)
        if len(data) >= 2 and (data[1] & 0x02):
            transfer_done.set()

    await client.start_notify(STROKE_BUFFER_UUID, _notification_handler)

    try:
        # Wait for transfer to complete or timeout
        timeout = config.retry_timeout_sec * config.max_retries
        await asyncio.wait_for(transfer_done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Chunk receive timed out after %.1fs", timeout)
        raise
    finally:
        try:
            await client.stop_notify(STROKE_BUFFER_UUID)
        except Exception:
            logger.debug("stop_notify failed (pen may have disconnected)")
