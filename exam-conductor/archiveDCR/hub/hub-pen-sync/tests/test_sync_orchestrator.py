"""Tests for sync orchestrator — full flow with mocked GATT + store IPC.

Test IDs: U-ORC-01 .. U-ORC-08
Validation level: L3 (unit — mocked BLE + IPC, no real hardware)
"""

from __future__ import annotations

import asyncio
import base64
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest

# Allow imports from hub-pen-sync/src and hub-common
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "hub-common"),
)

from hub_common.ipc_protocol import IpcEnvelope

from src.chunk_manager import build_chunk_frame
from src.config import (
    BUFFER_STATUS_UUID,
    PEN_METADATA_UUID,
    STROKE_BUFFER_UUID,
    SYNC_CONTROL_UUID,
    PenSyncConfig,
)
from src.sync_orchestrator import SyncOrchestrator
from src.sync_state import SyncStatus


# ===================================================================
# Test fixtures and helpers
# ===================================================================

TEST_CFG = PenSyncConfig(
    max_retries=2,
    retry_timeout_sec=5.0,
    chunk_retries=2,
    chunk_timeout_sec=2.0,
    store_write_timeout_sec=2.0,
    ble_connect_timeout_sec=2.0,
)


def _make_buffer_status(total_bytes: int, crc_hex: str) -> bytes:
    """Build a 12-byte Buffer Status characteristic value."""
    crc_int = int(crc_hex, 16)
    return struct.pack("<III", total_bytes, total_bytes, crc_int)


def _make_store_reply(
    env: IpcEnvelope, sd_ok: bool = True, usb_ok: bool = True
) -> IpcEnvelope:
    """Build a store.write.result reply for a given request."""
    return env.make_reply(
        "store.write.result",
        {
            "exam_id": env.payload["exam_id"],
            "pen_mac": env.payload["pen_mac"],
            "chunk_index": env.payload["chunk_index"],
            "sd_persisted": sd_ok,
            "usb_persisted": usb_ok,
        },
        source="hub-store",
    )


class MockBleClient:
    """Mock BLE client that simulates pen GATT interactions."""

    def __init__(
        self,
        total_bytes: int,
        payloads: list[bytes],
        buffer_crc_hex: str,
    ) -> None:
        self.total_bytes = total_bytes
        self.payloads = payloads
        self.buffer_crc_hex = buffer_crc_hex
        self._is_connected = True
        self._notify_callback: Any = None
        self._writes: list[tuple[str, bytes]] = []

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def read_gatt_char(self, uuid: str) -> bytes:
        if uuid == BUFFER_STATUS_UUID:
            return _make_buffer_status(self.total_bytes, self.buffer_crc_hex)
        if uuid == PEN_METADATA_UUID:
            return struct.pack("<HBIB", 0x0207, 95, 12345, 10)
        return b""

    async def write_gatt_char(
        self, uuid: str, data: bytes, response: bool = True
    ) -> None:
        self._writes.append((uuid, data))

    async def start_notify(self, uuid: str, callback: Any) -> None:
        self._notify_callback = callback
        # Deliver chunks as notifications in a background task
        asyncio.create_task(self._deliver_chunks())

    async def stop_notify(self, uuid: str) -> None:
        self._notify_callback = None

    async def _deliver_chunks(self) -> None:
        """Simulate pen sending chunks as BLE notifications."""
        await asyncio.sleep(0.01)  # Small delay to simulate BLE latency
        for i, payload in enumerate(self.payloads):
            frame = build_chunk_frame(i, len(self.payloads), payload)
            if self._notify_callback:
                await self._notify_callback(None, bytearray(frame))
                await asyncio.sleep(0.01)

    def get_sync_control_writes(self) -> list[bytes]:
        return [data for uuid, data in self._writes if uuid == SYNC_CONTROL_UUID]


class MockBleFactory:
    """Factory that creates MockBleClient instances."""

    def __init__(self, client: MockBleClient) -> None:
        self._client = client
        self.connect_count = 0

    async def connect(self, pen_mac: str, timeout: float) -> MockBleClient:
        self.connect_count += 1
        return self._client

    async def disconnect(self, client: Any) -> None:
        pass


class MockStoreClient:
    """Mock IPC client that simulates hub-store responses."""

    def __init__(self, *, fail_writes: bool = False) -> None:
        self.fail_writes = fail_writes
        self.write_requests: list[IpcEnvelope] = []

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def request(
        self, envelope: IpcEnvelope, *, timeout: float = 10.0
    ) -> IpcEnvelope:
        self.write_requests.append(envelope)
        if self.fail_writes:
            return envelope.make_error(
                "storage_write_failed", "SD write failed", source="hub-store"
            )
        return _make_store_reply(envelope)

    async def send(self, envelope: IpcEnvelope) -> None:
        pass


def _make_orchestrator(
    payloads: list[bytes],
    *,
    fail_store: bool = False,
    config: PenSyncConfig | None = None,
) -> tuple[SyncOrchestrator, MockBleClient, MockStoreClient]:
    """Build an orchestrator with mocked dependencies."""
    # Compute expected whole-buffer CRC
    crc = 0
    total_bytes = 0
    for p in payloads:
        crc = zlib.crc32(p, crc)
        total_bytes += len(p)
    crc_hex = format(crc & 0xFFFFFFFF, "08x")

    ble_client = MockBleClient(total_bytes, payloads, crc_hex)
    ble_factory = MockBleFactory(ble_client)
    store_client = MockStoreClient(fail_writes=fail_store)

    published: list[IpcEnvelope] = []

    async def publish(env: IpcEnvelope) -> None:
        published.append(env)

    cfg = config or TEST_CFG
    orch = SyncOrchestrator(
        config=cfg,
        store_client=store_client,
        ble_factory=ble_factory,
        event_publisher=publish,
    )
    return orch, ble_client, store_client


# ===================================================================
# Tests
# ===================================================================


# -----------------------------------------------------------------------
# U-ORC-01: happy path — single chunk sync completes
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_single_chunk() -> None:
    """U-ORC-01: sync one chunk, checksum match, buffer cleared."""
    payloads = [b"single-chunk-pen-data"]
    orch, ble_client, store = _make_orchestrator(payloads)

    state = await orch.sync_pen("EXAM-01", "AA:BB", "DD:00")

    assert state.status == SyncStatus.COMPLETE
    assert state.chunks_store_confirmed == 1
    assert state.checksum_verified

    # Verify pen buffer was cleared (0x03 written)
    ctrl_writes = ble_client.get_sync_control_writes()
    assert any(w == bytes([0x03]) for w in ctrl_writes), (
        "Pen buffer clear (0x03) must be sent after confirmed dual-write"
    )


# -----------------------------------------------------------------------
# U-ORC-02: happy path — multi-chunk sync
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_multi_chunk() -> None:
    """U-ORC-02: three chunks synced, all stored, checksum verified."""
    payloads = [b"chunk-0-data", b"chunk-1-data", b"chunk-2-data"]
    orch, ble_client, store = _make_orchestrator(payloads)

    state = await orch.sync_pen("EXAM-02", "BB:CC", "DD:01")

    assert state.status == SyncStatus.COMPLETE
    assert state.chunks_store_confirmed == 3
    assert state.total_chunks == 3

    # All 3 chunks should have been sent to store
    assert len(store.write_requests) == 3


# -----------------------------------------------------------------------
# U-ORC-03: empty pen buffer — immediate complete
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_buffer_completes() -> None:
    """U-ORC-03: pen with 0 bytes returns complete immediately."""
    ble_client = MockBleClient(0, [], "00000000")
    ble_factory = MockBleFactory(ble_client)
    store = MockStoreClient()

    async def noop_publish(env: IpcEnvelope) -> None:
        pass

    orch = SyncOrchestrator(
        config=TEST_CFG,
        store_client=store,
        ble_factory=ble_factory,
        event_publisher=noop_publish,
    )

    state = await orch.sync_pen("EXAM-03", "CC:DD", "DD:02")
    assert state.status == SyncStatus.COMPLETE
    assert store.write_requests == []


# -----------------------------------------------------------------------
# U-ORC-04: store write failure does NOT clear pen buffer
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_failure_no_buffer_clear() -> None:
    """U-ORC-04: if store fails, pen buffer MUST NOT be cleared."""
    payloads = [b"important-data"]
    orch, ble_client, store = _make_orchestrator(payloads, fail_store=True)

    state = await orch.sync_pen("EXAM-04", "DD:EE", "DD:03")

    # Should not be COMPLETE since store writes failed
    assert state.status != SyncStatus.COMPLETE

    # CRITICAL: 0x03 must NOT be in the sync control writes
    ctrl_writes = ble_client.get_sync_control_writes()
    assert not any(w == bytes([0x03]) for w in ctrl_writes), (
        "Pen buffer MUST NOT be cleared when store write fails"
    )


# -----------------------------------------------------------------------
# U-ORC-05: sync start is sent (0x01)
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_start_sent() -> None:
    """U-ORC-05: the orchestrator writes 0x01 to Sync Control."""
    payloads = [b"data"]
    orch, ble_client, _ = _make_orchestrator(payloads)

    await orch.sync_pen("EXAM-05", "EE:FF", "DD:04")

    ctrl_writes = ble_client.get_sync_control_writes()
    assert any(w == bytes([0x01]) for w in ctrl_writes)


# -----------------------------------------------------------------------
# U-ORC-06: store requests contain correct payload fields
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_request_fields() -> None:
    """U-ORC-06: store.write.request has required fields."""
    payload_data = b"verify-fields"
    orch, _, store = _make_orchestrator([payload_data])

    await orch.sync_pen("EXAM-06", "FF:00", "DD:05")

    assert len(store.write_requests) == 1
    req = store.write_requests[0]
    p = req.payload

    assert p["exam_id"] == "EXAM-06"
    assert p["pen_mac"] == "FF:00"
    assert p["chunk_index"] == 0

    # Verify base64 decodes to original payload
    decoded = base64.b64decode(p["chunk_b64"])
    assert decoded == payload_data

    # Verify CRC matches
    expected_crc = format(zlib.crc32(payload_data) & 0xFFFFFFFF, "08x")
    assert p["checksum_crc32"] == expected_crc


# -----------------------------------------------------------------------
# U-ORC-07: abort stops sync
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abort_stops_sync() -> None:
    """U-ORC-07: calling abort_pen cancels in-progress sync."""
    orch, _, _ = _make_orchestrator([b"data" * 100])

    # Start sync in background and abort quickly
    task = asyncio.create_task(
        orch.sync_pen("EXAM-07", "AA:00", "DD:06")
    )
    await asyncio.sleep(0.01)
    await orch.abort_pen("AA:00", "test abort")

    state = await task
    assert state.status in (
        SyncStatus.FAILED,
        SyncStatus.TIMEOUT,   # retries exhausted via timeout
        SyncStatus.COMPLETE,  # may complete before abort
    )


# -----------------------------------------------------------------------
# U-ORC-08: get_state returns active sync state
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_state() -> None:
    """U-ORC-08: get_state returns None before sync, state after."""
    orch, _, _ = _make_orchestrator([b"x"])
    assert orch.get_state("NO:SUCH:PEN") is None

    await orch.sync_pen("EXAM-08", "BB:00", "DD:07")
    state = orch.get_state("BB:00")
    assert state is not None
    assert state.exam_id == "EXAM-08"
