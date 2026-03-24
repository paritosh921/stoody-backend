"""Integration tests for hub-timer IPC server.

Test IDs: I-TMR-IPC-01 through I-TMR-IPC-04.
Validation level: L4 (integration -- real async sockets, no mocks).

Each test spins up a ``TimerIpcServer`` on a TCP loopback port (works on
Linux and Windows), sends messages via a raw asyncio stream, and asserts
on the responses / broadcasts received.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from src.config import MODULE_ID
from src.countdown import CountdownTimer
from src.ipc_handlers import (
    MSG_SUPERVISOR_HEALTH_REQUEST,
    MSG_TIMER_ARM_REQUEST,
    MSG_TIMER_CANCEL_REQUEST,
    MSG_TIMER_SNAPSHOT_REQUEST,
    MSG_TIMER_TICK,
    TimerIpcHandlers,
)
from src.ipc_server import Envelope, TimerIpcServer
from src.persistence import TimerPersistence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _make_request(
    msg_type: str,
    payload: dict | None = None,
    *,
    expects_reply: bool = False,
) -> bytes:
    """Build a JSON-line request that the server can parse."""
    env = {
        "msg_id": str(uuid.uuid4()),
        "msg_type": msg_type,
        "source": "test-client",
        "target": MODULE_ID,
        "sent_at": "2026-03-18T00:00:00Z",
        "correlation_id": None,
        "expects_reply": expects_reply,
        "payload": payload or {},
    }
    return json.dumps(env, separators=(",", ":")).encode() + b"\n"


async def _read_line(reader: asyncio.StreamReader, timeout: float = 2.0) -> dict:
    """Read one JSON-line from the stream with a timeout."""
    raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
    assert raw, "Connection closed before a line was received"
    return json.loads(raw.decode())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def timer(clock: FakeClock) -> CountdownTimer:
    return CountdownTimer(clock_fn=clock)


@pytest.fixture()
async def server_stack(tmp_path: Path, timer: CountdownTimer):
    """Yield (server, handlers, persist) with the server started."""
    persist = TimerPersistence(db_path=tmp_path / "test.db")
    persist.open()

    server = TimerIpcServer(
        str(tmp_path / "timer.sock"),
        module_id=MODULE_ID,
        use_tcp=True,
    )

    expired_fired: set[str] = set()

    async def on_arm(exam_id: str, duration_sec: int) -> None:
        timer.arm(exam_id, duration_sec)
        state = timer.get_state()
        assert state is not None
        persist.persist_state(
            exam_id, state.started_at_epoch, duration_sec, state.remaining_sec,
        )
        expired_fired.discard(exam_id)

    async def on_cancel(exam_id: str) -> None:
        if timer.cancel(exam_id):
            persist.clear_state(exam_id)
            expired_fired.discard(exam_id)

    handlers = TimerIpcHandlers(
        timer, server.broadcast, on_arm=on_arm, on_cancel=on_cancel,
    )

    server.register(MSG_TIMER_ARM_REQUEST, handlers.handle_arm)
    server.register(MSG_TIMER_CANCEL_REQUEST, handlers.handle_cancel)
    server.register(MSG_TIMER_SNAPSHOT_REQUEST, handlers.handle_snapshot)
    server.register(MSG_SUPERVISOR_HEALTH_REQUEST, handlers.handle_health)

    await server.start()
    yield server, handlers, persist
    await server.stop()
    persist.close()


@pytest.fixture()
async def client(server_stack):
    """Open a TCP connection to the test server and yield (reader, writer)."""
    server, _, _ = server_stack
    port = server.tcp_port
    assert port is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    yield reader, writer
    writer.close()
    await writer.wait_closed()


# ---------------------------------------------------------------------------
# I-TMR-IPC-01: Arm timer via IPC
# ---------------------------------------------------------------------------

async def test_arm_via_ipc(server_stack, client, timer: CountdownTimer):
    """I-TMR-IPC-01: Sending timer.arm.request over socket arms the timer."""
    reader, writer = client

    req = _make_request(
        MSG_TIMER_ARM_REQUEST,
        {"exam_id": "exam-ipc-01", "duration_sec": 600},
    )
    writer.write(req)
    await writer.drain()

    # Give the server a moment to process.
    await asyncio.sleep(0.05)

    assert timer.active
    assert timer.exam_id == "exam-ipc-01"
    assert timer.get_remaining() == 600


# ---------------------------------------------------------------------------
# I-TMR-IPC-02: Cancel timer via IPC
# ---------------------------------------------------------------------------

async def test_cancel_via_ipc(server_stack, client, timer: CountdownTimer):
    """I-TMR-IPC-02: Sending timer.cancel.request cancels the active timer."""
    reader, writer = client

    # Arm first.
    writer.write(_make_request(
        MSG_TIMER_ARM_REQUEST,
        {"exam_id": "exam-ipc-02", "duration_sec": 300},
    ))
    await writer.drain()
    await asyncio.sleep(0.05)
    assert timer.active

    # Cancel.
    writer.write(_make_request(
        MSG_TIMER_CANCEL_REQUEST,
        {"exam_id": "exam-ipc-02"},
    ))
    await writer.drain()
    await asyncio.sleep(0.05)

    assert not timer.active
    assert timer.get_state() is None


# ---------------------------------------------------------------------------
# I-TMR-IPC-03: Snapshot request returns reply
# ---------------------------------------------------------------------------

async def test_snapshot_request_reply(
    server_stack, client, timer: CountdownTimer,
):
    """I-TMR-IPC-03: snapshot request returns a correlated reply with state."""
    reader, writer = client

    # Arm.
    writer.write(_make_request(
        MSG_TIMER_ARM_REQUEST,
        {"exam_id": "exam-ipc-03", "duration_sec": 1800},
    ))
    await writer.drain()
    await asyncio.sleep(0.05)

    # Snapshot.
    msg_id = str(uuid.uuid4())
    snap_env = {
        "msg_id": msg_id,
        "msg_type": MSG_TIMER_SNAPSHOT_REQUEST,
        "source": "test-client",
        "target": MODULE_ID,
        "sent_at": "2026-03-18T00:00:00Z",
        "correlation_id": None,
        "expects_reply": True,
        "payload": {},
    }
    writer.write(json.dumps(snap_env, separators=(",", ":")).encode() + b"\n")
    await writer.drain()

    reply = await _read_line(reader)

    assert reply["msg_type"] == "timer.snapshot.result"
    assert reply["correlation_id"] == msg_id
    assert reply["payload"]["exam_id"] == "exam-ipc-03"
    assert reply["payload"]["state"] == "running"
    assert reply["payload"]["remaining_sec"] == 1800


# ---------------------------------------------------------------------------
# I-TMR-IPC-04: Tick broadcasts reach connected client
# ---------------------------------------------------------------------------

async def test_tick_broadcast_received(
    server_stack, client, timer: CountdownTimer, clock: FakeClock,
):
    """I-TMR-IPC-04: publish_tick sends timer.tick to all connected clients."""
    server, handlers, _ = server_stack
    reader, writer = client

    # Arm.
    writer.write(_make_request(
        MSG_TIMER_ARM_REQUEST,
        {"exam_id": "exam-ipc-04", "duration_sec": 120},
    ))
    await writer.drain()
    await asyncio.sleep(0.05)

    # Trigger a tick broadcast from the handler.
    clock.advance(5)
    await handlers.publish_tick()

    tick = await _read_line(reader)

    assert tick["msg_type"] == MSG_TIMER_TICK
    assert tick["payload"]["exam_id"] == "exam-ipc-04"
    assert tick["payload"]["remaining_sec"] == 115
    assert tick["payload"]["total_sec"] == 120
