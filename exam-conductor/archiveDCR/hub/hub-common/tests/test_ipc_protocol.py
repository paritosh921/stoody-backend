"""Tests for IPC envelope serialization, round-trip, and timeout handling.

Test IDs: U-IPC-01 .. U-IPC-08
Validation level: L3 (unit, no I/O)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import os

import pytest

from hub_common.ipc_protocol import IpcEnvelope, IpcClient, IpcServer


# -------------------------------------------------------------------
# U-IPC-01: Envelope round-trip serialization
# -------------------------------------------------------------------

class TestEnvelopeSerialization:
    def test_round_trip(self) -> None:
        """U-IPC-01: serialize -> deserialize preserves all fields."""
        env = IpcEnvelope(
            msg_type="store.write.request",
            source="hub-pen-sync",
            target="hub-store",
            payload={"exam_id": "E1", "pen_mac": "AA:BB:CC:DD:EE:FF"},
            expects_reply=True,
        )
        line = env.to_json_line()
        assert line.endswith(b"\n")

        restored = IpcEnvelope.from_json_line(line)
        assert restored.msg_id == env.msg_id
        assert restored.msg_type == env.msg_type
        assert restored.source == env.source
        assert restored.target == env.target
        assert restored.payload == env.payload
        assert restored.expects_reply is True
        assert restored.sent_at == env.sent_at

    def test_json_line_is_valid_json(self) -> None:
        """U-IPC-02: the JSON-lines output is parseable JSON."""
        env = IpcEnvelope(
            msg_type="timer.arm.request",
            source="hub-supervisor",
            target="hub-timer",
        )
        data = json.loads(env.to_json_line().decode("utf-8"))
        assert data["msg_type"] == "timer.arm.request"

    def test_correlation_id_none_by_default(self) -> None:
        """U-IPC-03: correlation_id defaults to None."""
        env = IpcEnvelope(
            msg_type="ble.scan.start.request",
            source="hub-supervisor",
            target="hub-ble-mgr",
        )
        assert env.correlation_id is None

    def test_sent_at_is_iso8601_utc(self) -> None:
        """U-IPC-04: sent_at ends with Z (UTC)."""
        env = IpcEnvelope(
            msg_type="pen.sync.request",
            source="hub-supervisor",
            target="hub-pen-sync",
        )
        assert env.sent_at.endswith("Z")


# -------------------------------------------------------------------
# U-IPC-05: Reply and error envelope factories
# -------------------------------------------------------------------

class TestEnvelopeFactories:
    def test_make_reply_sets_correlation_id(self) -> None:
        """U-IPC-05: reply.correlation_id == request.msg_id."""
        req = IpcEnvelope(
            msg_type="store.write.request",
            source="hub-pen-sync",
            target="hub-store",
            expects_reply=True,
        )
        reply = req.make_reply(
            "store.write.result",
            {"sd_persisted": True, "usb_persisted": True},
        )
        assert reply.correlation_id == req.msg_id
        assert reply.source == "hub-store"
        assert reply.target == "hub-pen-sync"
        assert reply.expects_reply is False

    def test_make_error_derives_type(self) -> None:
        """U-IPC-06: error type is namespace.error."""
        req = IpcEnvelope(
            msg_type="store.write.request",
            source="hub-pen-sync",
            target="hub-store",
            expects_reply=True,
        )
        err = req.make_error("storage_write_failed", "SD full")
        assert err.msg_type == "store.write.error"
        assert err.payload["code"] == "storage_write_failed"


# -------------------------------------------------------------------
# U-IPC-07 / U-IPC-08: Client-server round-trip + timeout
# -------------------------------------------------------------------

@pytest.fixture()
def unix_socket_path(tmp_path):
    import asyncio
    if hasattr(asyncio, "start_unix_server"):
        return str(tmp_path / "test.sock")
    # Windows fallback: use TCP loopback with a free port
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_client_server_round_trip(unix_socket_path: str) -> None:
    """U-IPC-07: client sends request, server dispatches and replies."""
    async def echo_handler(env: IpcEnvelope) -> IpcEnvelope:
        return env.make_reply(
            "store.write.result",
            {"echo": True, **env.payload},
        )

    server = IpcServer(unix_socket_path, module_id="hub-store")
    server.register("store.write.request", echo_handler)
    await server.start()

    client = IpcClient(unix_socket_path, source_id="hub-pen-sync")
    await client.connect()

    req = IpcEnvelope(
        msg_type="store.write.request",
        source="hub-pen-sync",
        target="hub-store",
        payload={"exam_id": "E1"},
        expects_reply=True,
    )
    reply = await client.request(req, timeout=5.0)

    assert reply.correlation_id == req.msg_id
    assert reply.payload["echo"] is True
    assert reply.payload["exam_id"] == "E1"

    await client.close()
    await server.stop()


@pytest.mark.asyncio
async def test_client_request_timeout(unix_socket_path: str) -> None:
    """U-IPC-08: client raises TimeoutError when server never replies."""
    async def silent_handler(env: IpcEnvelope) -> None:
        # Returns None -> no reply written
        return None

    server = IpcServer(unix_socket_path, module_id="hub-store")
    server.register("store.write.request", silent_handler)
    await server.start()

    client = IpcClient(unix_socket_path, source_id="hub-pen-sync")
    await client.connect()

    req = IpcEnvelope(
        msg_type="store.write.request",
        source="hub-pen-sync",
        target="hub-store",
        payload={},
        expects_reply=True,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        await client.request(req, timeout=0.3)

    await client.close()
    await server.stop()
