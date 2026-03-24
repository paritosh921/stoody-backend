"""Unit tests for IPC handlers, server, client, and cmd_id mapping.

Test IDs: U-INVIG-IPC-01 through U-INVIG-IPC-08.
Validation level: L3 (unit) / L4 (integration -- real TCP sockets).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.config import MODULE_ID
from src.ipc_client import IpcClient
from src.ipc_handlers import (
    COMMAND_NAME_TO_ID,
    MSG_FSM_SNAPSHOT_REQUEST,
    MSG_FSM_SNAPSHOT_RESULT,
    MSG_INVIG_AUTH_STATE_EVENT,
    MSG_INVIG_COMMAND_EVENT,
    MSG_SUPERVISOR_HEALTH_REQUEST,
    Envelope,
    InvigIpcHandlers,
)
from src.ipc_server import InvigIpcServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Collector:
    """Collects envelopes passed to a broadcast callback."""

    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    async def __call__(self, env: Envelope) -> None:
        self.envelopes.append(env)


# ---------------------------------------------------------------------------
# U-INVIG-IPC-01: IPC server starts and receives a health request
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ipc_server_starts_and_receives_health():
    """U-INVIG-IPC-01: Server starts, client connects, health request works."""
    server = InvigIpcServer(
        "/tmp/test-invig.sock", module_id=MODULE_ID, use_tcp=True,
    )
    ipc = InvigIpcHandlers(broadcast_fn=server.broadcast)
    server.register(MSG_SUPERVISOR_HEALTH_REQUEST, ipc.handle_health)

    await server.start()
    assert server.tcp_port is not None

    client = IpcClient(f"localhost:{server.tcp_port}", use_tcp=True)
    assert await client.connect()

    env = Envelope(
        msg_type=MSG_SUPERVISOR_HEALTH_REQUEST,
        source="hub-supervisor",
        target=MODULE_ID,
        expects_reply=True,
    )
    reply = await client.request(env, timeout_sec=2.0)

    assert reply is not None
    assert reply.payload["module"] == MODULE_ID
    assert reply.payload["healthy"] is True

    await client.close()
    await server.stop()


# ---------------------------------------------------------------------------
# U-INVIG-IPC-02: Broadcast reaches connected clients
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_reaches_client():
    """U-INVIG-IPC-02: broadcast() delivers envelope to connected client."""
    server = InvigIpcServer(
        "/tmp/test-invig2.sock", module_id=MODULE_ID, use_tcp=True,
    )
    await server.start()
    assert server.tcp_port is not None

    reader, writer = await asyncio.open_connection(
        "127.0.0.1", server.tcp_port,
    )
    # Give the server time to register the connection.
    await asyncio.sleep(0.05)

    env = Envelope(
        msg_type=MSG_INVIG_AUTH_STATE_EVENT,
        source=MODULE_ID,
        target="*",
        payload={"invig_id": "AA:BB", "connected": True, "authenticated": True},
    )
    await server.broadcast(env)
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    data = json.loads(line)

    assert data["msg_type"] == MSG_INVIG_AUTH_STATE_EVENT
    assert data["payload"]["invig_id"] == "AA:BB"

    writer.close()
    await server.stop()


# ---------------------------------------------------------------------------
# U-INVIG-IPC-03: publish_command uses numeric cmd_id, not name string
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_command_numeric_cmd_id():
    """U-INVIG-IPC-03: cmd_id in the IPC payload is a numeric int, not a string."""
    collector = _Collector()
    ipc = InvigIpcHandlers(broadcast_fn=collector)

    await ipc.publish_command(
        cmd_name="exam_start",
        cmd_id=0x01,
        request_id="req-100",
        payload={"exam_id": "e1", "duration_sec": 3600},
    )

    assert len(collector.envelopes) == 1
    env = collector.envelopes[0]
    assert env.msg_type == MSG_INVIG_COMMAND_EVENT
    assert isinstance(env.payload["cmd_id"], int)
    assert env.payload["cmd_id"] == 0x01
    assert env.payload["cmd_name"] == "exam_start"
    assert env.payload["request_id"] == "req-100"


# ---------------------------------------------------------------------------
# U-INVIG-IPC-04: COMMAND_NAME_TO_ID covers all GATT spec commands
# ---------------------------------------------------------------------------

def test_command_name_to_id_mapping():
    """U-INVIG-IPC-04: Every command in the GATT spec has a numeric mapping."""
    expected = {
        "start_exam": 0x01,
        "exam_start": 0x01,
        "stop_exam": 0x02,
        "exam_stop": 0x02,
        "start_registration_scan": 0x03,
        "manual_register": 0x04,
        "start_upload": 0x05,
        "trigger_upload": 0x05,
        "request_snapshot": 0x06,
    }
    assert COMMAND_NAME_TO_ID == expected


# ---------------------------------------------------------------------------
# U-INVIG-IPC-05: publish_command resolves name -> numeric even if
#   cmd_id param is wrong (name is authoritative)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_command_resolves_from_name():
    """U-INVIG-IPC-05: Name lookup takes priority over passed cmd_id."""
    collector = _Collector()
    ipc = InvigIpcHandlers(broadcast_fn=collector)

    # Pass a deliberately wrong numeric cmd_id; the name should win.
    await ipc.publish_command(
        cmd_name="manual_register",
        cmd_id=0xFF,
        request_id="req-200",
        payload={"exam_id": "e2", "pen_mac": "AA:BB", "student_id": "S1"},
    )

    env = collector.envelopes[0]
    assert env.payload["cmd_id"] == 0x04  # resolved from name, not 0xFF


# ---------------------------------------------------------------------------
# U-INVIG-IPC-06: Status feed requests supervisor snapshot via IPC client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_feed_requests_snapshot():
    """U-INVIG-IPC-06: A mock supervisor returns a snapshot, status feed updates."""
    # Stand up a mock supervisor server that replies with a snapshot.
    sv_server = InvigIpcServer(
        "/tmp/test-sv.sock", module_id="hub-supervisor", use_tcp=True,
    )

    async def _snapshot_handler(env: Envelope) -> Envelope | None:
        return Envelope(
            msg_type=MSG_FSM_SNAPSHOT_RESULT,
            source="hub-supervisor",
            target=env.source,
            correlation_id=env.msg_id,
            payload={
                "exam_id": "exam-snap",
                "state": "timer_running",
                "timer": {"remaining_sec": 1200},
                "wifi": {"connected": True, "band": "5GHz", "signal_dbm": -40},
                "storage": {"sd_ok": True, "usb_ok": True, "degraded": False},
                "sync": {
                    "complete": 10, "in_progress": 2,
                    "failed": 0, "pending": 28,
                },
            },
        )

    sv_server.register(MSG_FSM_SNAPSHOT_REQUEST, _snapshot_handler)
    await sv_server.start()

    client = IpcClient(f"localhost:{sv_server.tcp_port}", use_tcp=True)

    from src.main import _request_supervisor_snapshot
    from src.status_feed import StatusFeedCollector

    feed = StatusFeedCollector()
    await _request_supervisor_snapshot(client, feed)

    d = feed.to_dict()
    assert d["exam_id"] == "exam-snap"
    assert d["state"] == "timer_running"
    assert d["timer_remaining_sec"] == 1200
    assert d["sync"]["in_progress"] == 2

    await client.close()
    await sv_server.stop()


# ---------------------------------------------------------------------------
# U-INVIG-IPC-07: Graceful degradation when supervisor unreachable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_graceful_degradation():
    """U-INVIG-IPC-07: If supervisor is down, status feed keeps default state."""
    from src.main import _request_supervisor_snapshot
    from src.status_feed import StatusFeedCollector

    # Client pointing to a port where nothing is listening.
    client = IpcClient("localhost:1", use_tcp=True)
    feed = StatusFeedCollector()

    # Should not raise -- just keep default idle state.
    await _request_supervisor_snapshot(client, feed)

    d = feed.to_dict()
    assert d["state"] == "idle"
    assert d["exam_id"] is None

    await client.close()


# ---------------------------------------------------------------------------
# U-INVIG-IPC-08: publish_auth_state broadcasts correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_auth_state():
    """U-INVIG-IPC-08: Auth state event carries correct payload shape."""
    collector = _Collector()
    ipc = InvigIpcHandlers(broadcast_fn=collector)

    await ipc.publish_auth_state(
        invig_id="AA:BB:CC:DD:EE:01",
        connected=True,
        authenticated=False,
    )

    assert len(collector.envelopes) == 1
    env = collector.envelopes[0]
    assert env.msg_type == MSG_INVIG_AUTH_STATE_EVENT
    assert env.payload["invig_id"] == "AA:BB:CC:DD:EE:01"
    assert env.payload["connected"] is True
    assert env.payload["authenticated"] is False
