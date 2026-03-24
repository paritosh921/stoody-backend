"""Tests for UplinkHandlers — progress/completion events are broadcast.

Test IDs: U-UPL-24 .. U-UPL-27
Validation level: L3 (unit — mocked broadcast, mocked store IPC)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hub_common.ipc_protocol import IpcEnvelope
from hub_common.message_types import (
    UPLINK_UPLOAD_COMPLETE_EVENT,
    UPLINK_UPLOAD_PROGRESS_EVENT,
)

from src.config import UplinkConfig
from src.ipc_handlers import EVENT_TARGETS, UplinkHandlers
from src.ledger import UploadLedger, ensure_ledger_table
from src.upload_manager import PenUploadSpec, UploadManager


@pytest.fixture()
def config() -> UplinkConfig:
    return UplinkConfig(
        backend_url="https://backend.test",
        ingest_endpoint="/api/v1/strokes/ingest",
        health_endpoint="/health",
        upload_timeout_sec=5,
        retry_interval_sec=0,
        chunk_batch_size=4,
    )


@pytest.fixture()
def ledger() -> UploadLedger:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    ensure_ledger_table(conn)
    return UploadLedger(conn)


def _make_store_client(chunks: dict[int, dict]) -> AsyncMock:
    """Build a mock IpcClient that returns chunk data."""
    client = AsyncMock()

    async def mock_request(env: IpcEnvelope, **kw: Any) -> IpcEnvelope:
        idx = env.payload["chunk_index"]
        if idx in chunks:
            return IpcEnvelope(
                msg_type="store.read.result",
                source="hub-store",
                target="hub-uplink",
                payload=chunks[idx],
            )
        return IpcEnvelope(
            msg_type="store.read.error",
            source="hub-store",
            target="hub-uplink",
            payload={"code": "storage_read_failed", "message": "not found"},
        )

    client.request = mock_request
    return client


def _mock_http_session() -> MagicMock:
    """Return a mock aiohttp.ClientSession that always returns 202."""
    resp = AsyncMock()
    resp.status = 202
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.post = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


# -----------------------------------------------------------------------
# U-UPL-24: progress events are broadcast for each chunk
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_progress_events_broadcast(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-24: emit_progress broadcasts to all EVENT_TARGETS per chunk."""
    broadcast_calls: list[IpcEnvelope] = []

    async def fake_broadcast(env: IpcEnvelope) -> None:
        broadcast_calls.append(env)

    chunks = {
        i: {"chunk_b64": f"d{i}", "checksum_crc32": f"c{i}"}
        for i in range(3)
    }
    store = _make_store_client(chunks)

    handlers = UplinkHandlers(
        config, ledger, broadcast_fn=fake_broadcast,
    )
    mgr = UploadManager(
        config, ledger, store,
        progress_callback=handlers.emit_progress,
    )
    handlers.set_upload_manager(mgr)

    spec = PenUploadSpec("E1", "AA:BB", 3, "wifi")

    with patch("src.upload_manager.aiohttp.ClientSession",
               return_value=_mock_http_session()):
        await mgr.upload_pen(spec)

    # 3 chunks x 3 targets = 9 progress broadcasts.
    progress_envs = [
        e for e in broadcast_calls
        if e.msg_type == UPLINK_UPLOAD_PROGRESS_EVENT
    ]
    assert len(progress_envs) == 3 * len(EVENT_TARGETS)

    # Verify envelope fields.
    first = progress_envs[0]
    assert first.payload["exam_id"] == "E1"
    assert first.payload["pen_mac"] == "AA:BB"
    assert first.payload["acked_count"] == 1
    assert first.payload["total_chunks"] == 3


# -----------------------------------------------------------------------
# U-UPL-25: progress events contain correct acked count
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_progress_acked_count_increments(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-25: acked_count increments with each chunk upload."""
    broadcast_calls: list[IpcEnvelope] = []

    async def fake_broadcast(env: IpcEnvelope) -> None:
        broadcast_calls.append(env)

    chunks = {
        i: {"chunk_b64": f"d{i}", "checksum_crc32": f"c{i}"}
        for i in range(3)
    }
    store = _make_store_client(chunks)

    handlers = UplinkHandlers(
        config, ledger, broadcast_fn=fake_broadcast,
    )
    mgr = UploadManager(
        config, ledger, store,
        progress_callback=handlers.emit_progress,
    )
    handlers.set_upload_manager(mgr)

    with patch("src.upload_manager.aiohttp.ClientSession",
               return_value=_mock_http_session()):
        await mgr.upload_pen(PenUploadSpec("E1", "AA:BB", 3, "wifi"))

    # Pick one target (first of each triple).
    progress = [
        e for e in broadcast_calls
        if e.msg_type == UPLINK_UPLOAD_PROGRESS_EVENT
        and e.target == EVENT_TARGETS[0]
    ]
    acked_counts = [e.payload["acked_count"] for e in progress]
    assert acked_counts == [1, 2, 3]


# -----------------------------------------------------------------------
# U-UPL-26: completion events broadcast after pen upload
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_completion_event_broadcast(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-26: _broadcast_pen_complete sends to all EVENT_TARGETS."""
    broadcast_calls: list[IpcEnvelope] = []

    async def fake_broadcast(env: IpcEnvelope) -> None:
        broadcast_calls.append(env)

    handlers = UplinkHandlers(
        config, ledger, broadcast_fn=fake_broadcast,
    )

    await handlers._broadcast_pen_complete("E1", "AA:BB")

    assert len(broadcast_calls) == len(EVENT_TARGETS)
    for env in broadcast_calls:
        assert env.msg_type == UPLINK_UPLOAD_COMPLETE_EVENT
        assert env.payload["exam_id"] == "E1"
        assert env.payload["pen_mac"] == "AA:BB"
        assert env.payload["complete"] is True

    targets_sent = {e.target for e in broadcast_calls}
    assert targets_sent == set(EVENT_TARGETS)


# -----------------------------------------------------------------------
# U-UPL-27: no progress events when no callback wired
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_progress_without_callback(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-27: UploadManager without callback does not raise."""
    chunks = {0: {"chunk_b64": "d0", "checksum_crc32": "c0"}}
    store = _make_store_client(chunks)

    mgr = UploadManager(config, ledger, store)  # no callback

    with patch("src.upload_manager.aiohttp.ClientSession",
               return_value=_mock_http_session()):
        result = await mgr.upload_pen(
            PenUploadSpec("E1", "AA:BB", 1, "wifi"),
        )

    assert result is True
    assert ledger.is_pen_complete("E1", "AA:BB")
