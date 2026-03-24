"""Tests for exampen_common.nats_client — publish/subscribe with mocked NATS.

Test IDs: U-COMMON-NATS-01 through U-COMMON-NATS-05
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exampen_common.nats_client import NatsClient, create_nats_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_js() -> MagicMock:
    js = AsyncMock()
    js.publish = AsyncMock()
    js.subscribe = AsyncMock()
    return js


def _mock_conn(js: MagicMock | None = None) -> MagicMock:
    conn = AsyncMock()
    conn.is_closed = False
    conn.drain = AsyncMock()
    conn.jetstream = MagicMock(return_value=js or _mock_js())
    return conn


# ---------------------------------------------------------------------------
# U-COMMON-NATS-01: NatsClient.publish serializes JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_serializes_json():
    js = _mock_js()
    client = NatsClient(conn=_mock_conn(js), js=js)

    payload = {"exam_id": "e1", "score": 95}
    await client.publish("score.updated", payload)

    js.publish.assert_awaited_once()
    call_args = js.publish.call_args
    assert call_args[0][0] == "score.updated"
    data = json.loads(call_args[0][1].decode("utf-8"))
    assert data["exam_id"] == "e1"
    assert data["score"] == 95


# ---------------------------------------------------------------------------
# U-COMMON-NATS-02: NatsClient.publish passes headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_passes_headers():
    js = _mock_js()
    client = NatsClient(conn=_mock_conn(js), js=js)

    await client.publish("test.subj", {"a": 1}, headers={"X-Trace": "abc"})

    call_kwargs = js.publish.call_args
    assert call_kwargs[1]["headers"] == {"X-Trace": "abc"}


# ---------------------------------------------------------------------------
# U-COMMON-NATS-03: NatsClient.subscribe registers callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_registers():
    js = _mock_js()
    client = NatsClient(conn=_mock_conn(js), js=js)

    handler = AsyncMock()
    await client.subscribe("stroke.raw", handler, durable="proc", queue="workers")

    js.subscribe.assert_awaited_once()
    call_args = js.subscribe.call_args
    assert call_args[0][0] == "stroke.raw"
    assert call_args[1]["queue"] == "workers"


# ---------------------------------------------------------------------------
# U-COMMON-NATS-04: NatsClient.close drains connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_drains():
    conn = _mock_conn()
    client = NatsClient(conn=conn, js=_mock_js())

    await client.close()
    conn.drain.assert_awaited_once()


# ---------------------------------------------------------------------------
# U-COMMON-NATS-05: create_nats_client factory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_nats_client_factory():
    mock_conn = _mock_conn()

    with patch("exampen_common.nats_client.nats.connect", return_value=mock_conn):
        client = await create_nats_client(url="nats://test:4222")

    assert client.connection is mock_conn
    assert client.jetstream is not None
