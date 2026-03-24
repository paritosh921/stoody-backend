"""Tests for upload manager — upload flow with mocked HTTP + store IPC.

Test IDs: U-UPL-17 .. U-UPL-23
Validation level: L3 (unit — mocked aiohttp, mocked IPC client)
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hub_common.ipc_protocol import IpcEnvelope

from src.config import UplinkConfig
from src.ledger import UploadLedger, ensure_ledger_table
from src.upload_manager import PenUploadSpec, UploadManager


@pytest.fixture()
def config() -> UplinkConfig:
    return UplinkConfig(
        backend_url="https://backend.test",
        ingest_endpoint="/api/v1/strokes/ingest",
        health_endpoint="/health",
        upload_timeout_sec=5,
        retry_interval_sec=0,  # no delay in tests
        chunk_batch_size=4,
    )


@pytest.fixture()
def ledger() -> UploadLedger:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    ensure_ledger_table(conn)
    return UploadLedger(conn)


def _make_store_client(chunks: dict[int, dict]) -> AsyncMock:
    """Build a mock IpcClient that returns chunk data for store.read.request."""
    client = AsyncMock()

    async def mock_request(env: IpcEnvelope, **kwargs: Any) -> IpcEnvelope:
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


def _spec(total: int = 3) -> PenUploadSpec:
    return PenUploadSpec(
        exam_id="E1", pen_mac="AA:BB", total_chunks=total, upload_path="wifi",
    )


# -----------------------------------------------------------------------
# U-UPL-17: successful upload of all chunks
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_all_chunks_success(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-17: all chunks uploaded and acked -> pen marked complete."""
    chunks = {
        i: {"chunk_b64": f"data{i}", "checksum_crc32": f"crc{i}"}
        for i in range(3)
    }
    store = _make_store_client(chunks)

    mock_resp = AsyncMock()
    mock_resp.status = 202
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.upload_manager.aiohttp.ClientSession", return_value=mock_session):
        mgr = UploadManager(config, ledger, store)
        result = await mgr.upload_pen(_spec(3))

    assert result is True
    assert ledger.is_pen_complete("E1", "AA:BB")
    assert ledger.get_pending_chunks("E1", "AA:BB") == []


# -----------------------------------------------------------------------
# U-UPL-18: ledger updated ONLY after backend ACK
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ledger_updated_only_after_ack(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-18: chunk not acked in ledger before backend responds 202."""
    chunks = {0: {"chunk_b64": "d0", "checksum_crc32": "c0"}}
    store = _make_store_client(chunks)

    call_count = 0

    mock_resp = AsyncMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    async def _check_ledger_then_202(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        # Before returning 202, verify ledger has NOT been updated
        pending = ledger.get_pending_chunks("E1", "AA:BB")
        assert 0 in pending, "Ledger must not be updated before ACK"
        mock_resp.status = 202
        return mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=_check_ledger_then_202)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.upload_manager.aiohttp.ClientSession", return_value=mock_session):
        mgr = UploadManager(config, ledger, store)
        spec = PenUploadSpec("E1", "AA:BB", 1, "wifi")
        await mgr.upload_pen(spec)

    assert call_count == 1
    assert ledger.is_pen_complete("E1", "AA:BB")


# -----------------------------------------------------------------------
# U-UPL-19: resume sends only pending chunks
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_skips_acked_chunks(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-19: on resume, already-acked chunks are not re-uploaded."""
    # Pre-ack chunks 0 and 1
    ledger.init_pen("E1", "AA:BB", 3, "wifi")
    ledger.mark_chunk_acked("E1", "AA:BB", 0)
    ledger.mark_chunk_acked("E1", "AA:BB", 1)

    chunks = {2: {"chunk_b64": "d2", "checksum_crc32": "c2"}}
    store = _make_store_client(chunks)

    uploaded_indices: list[int] = []

    mock_resp = AsyncMock()
    mock_resp.status = 202
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    def _capture_post(*args: Any, **kwargs: Any) -> Any:
        body = kwargs.get("json", {})
        uploaded_indices.append(body.get("chunk_index", -1))
        return mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=_capture_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.upload_manager.aiohttp.ClientSession", return_value=mock_session):
        mgr = UploadManager(config, ledger, store)
        await mgr.upload_pen(_spec(3))

    assert uploaded_indices == [2], "Only chunk 2 should have been uploaded"
    assert ledger.is_pen_complete("E1", "AA:BB")


# -----------------------------------------------------------------------
# U-UPL-20: retry on transient HTTP failure
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_on_http_failure(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-20: upload retries on 500 and succeeds on next attempt."""
    chunks = {0: {"chunk_b64": "d0", "checksum_crc32": "c0"}}
    store = _make_store_client(chunks)

    attempt = 0

    mock_resp = AsyncMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    def _fail_then_succeed(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempt
        attempt += 1
        mock_resp.status = 500 if attempt == 1 else 202
        return mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=_fail_then_succeed)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.upload_manager.aiohttp.ClientSession", return_value=mock_session):
        mgr = UploadManager(config, ledger, store)
        spec = PenUploadSpec("E1", "AA:BB", 1, "wifi")
        await mgr.upload_pen(spec)

    assert attempt == 2
    assert ledger.is_pen_complete("E1", "AA:BB")


# -----------------------------------------------------------------------
# U-UPL-21: idempotency key format matches spec
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idempotency_key_format(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-21: idempotency_key is '{exam_id}:{pen_mac}:{chunk_index}'."""
    chunks = {0: {"chunk_b64": "d", "checksum_crc32": "c"}}
    store = _make_store_client(chunks)

    captured_body: dict = {}

    mock_resp = AsyncMock()
    mock_resp.status = 202
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    def _capture(*args: Any, **kwargs: Any) -> Any:
        captured_body.update(kwargs.get("json", {}))
        return mock_resp

    mock_session = AsyncMock()
    mock_session.post = MagicMock(side_effect=_capture)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.upload_manager.aiohttp.ClientSession", return_value=mock_session):
        mgr = UploadManager(config, ledger, store)
        spec = PenUploadSpec("E1", "AA:BB", 1, "wifi")
        await mgr.upload_pen(spec)

    assert captured_body["idempotency_key"] == "E1:AA:BB:0"


# -----------------------------------------------------------------------
# U-UPL-22: already-complete pen is a no-op
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_already_complete_pen_noop(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-22: uploading a pen with all chunks acked does nothing."""
    ledger.init_pen("E1", "AA:BB", 2, "wifi")
    ledger.mark_chunk_acked("E1", "AA:BB", 0)
    ledger.mark_chunk_acked("E1", "AA:BB", 1)

    store = _make_store_client({})
    mgr = UploadManager(config, ledger, store)
    result = await mgr.upload_pen(_spec(2))

    assert result is True
    assert ledger.is_pen_complete("E1", "AA:BB")


# -----------------------------------------------------------------------
# U-UPL-23: progress callback invoked per chunk
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_progress_callback_invoked(
    config: UplinkConfig, ledger: UploadLedger,
) -> None:
    """U-UPL-23: progress callback fires for each uploaded chunk."""
    chunks = {i: {"chunk_b64": f"d{i}", "checksum_crc32": f"c{i}"} for i in range(2)}
    store = _make_store_client(chunks)

    progress_calls: list[tuple] = []

    async def on_progress(*args: Any) -> None:
        progress_calls.append(args)

    mock_resp = AsyncMock()
    mock_resp.status = 202
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.upload_manager.aiohttp.ClientSession", return_value=mock_session):
        mgr = UploadManager(config, ledger, store, progress_callback=on_progress)
        await mgr.upload_pen(_spec(2))

    assert len(progress_calls) == 2
