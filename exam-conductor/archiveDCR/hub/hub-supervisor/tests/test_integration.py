"""L4 integration tests — end-to-end hub FSM flow over real IPC (TCP).

Uses TCP loopback (Windows compatible) between supervisor and mock child
module servers.  HTTP backend + BLE hardware are mocked; IPC transport
is real JSON-lines over TCP.  Test IDs: I-SUP-INT-01 through I-SUP-INT-10.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from hub_common.ipc_protocol import IpcClient, IpcEnvelope
from hub_common.message_types import (
    BLE_SCAN_START_REQUEST, FSM_TRANSITION_REQUEST,
    PEN_SYNC_REQUEST, TIMER_ARM_REQUEST, UPLINK_UPLOAD_REQUEST,
)
from src.config import MODULE_ID
from src.first_boot import is_first_boot, run_provisioning
from src.hub_fsm import (
    ARMED, CREATED, DONGLE_ACTIVATION, PEN_SYNC,
    SYNC_COMPLETE, TIMER_RUNNING, UPLOAD_COMPLETE, UPLOADING,
)
from src.interaction_log import InteractionLog, LogEntry
from src.ipc_handlers import SupervisorIpcHandlers
from src.orchestrator import Orchestrator

# -- helpers ----------------------------------------------------------

_DDL = [
    "CREATE TABLE IF NOT EXISTS exam_sessions (exam_id TEXT PRIMARY KEY,"
    " invig_id TEXT NOT NULL, duration_min INTEGER NOT NULL,"
    " state TEXT NOT NULL DEFAULT 'created', created_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS hub_config (hub_id TEXT PRIMARY KEY,"
    " backend_url TEXT NOT NULL, uplink_mode TEXT NOT NULL DEFAULT 'wifi',"
    " region TEXT NOT NULL DEFAULT 'US', provisioned_at TEXT NOT NULL,"
    " last_backend_sync TEXT)",
    "CREATE TABLE IF NOT EXISTS invig_codes (code TEXT PRIMARY KEY,"
    " valid_from TEXT NOT NULL, valid_until TEXT NOT NULL, fetched_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS pen_inventory (pen_mac TEXT PRIMARY KEY,"
    " pen_serial TEXT, fw_version TEXT, registered_at TEXT NOT NULL,"
    " last_seen TEXT, battery_pct INTEGER)",
]


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL;")
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()
    return conn


def _insert_exam(conn: sqlite3.Connection, eid: str, state: str) -> None:
    conn.execute(
        "INSERT INTO exam_sessions VALUES (?,?,?,?,?)",
        (eid, "invig-1", 60, state, "2026-03-19T10:00:00Z"),
    )
    conn.commit()


class MockChildServer:
    """TCP server that captures received IPC envelopes."""
    def __init__(self) -> None:
        self.received: list[IpcEnvelope] = []
        self._server: asyncio.AbstractServer | None = None
        self.port: int = 0

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            while line := await r.readline():
                self.received.append(IpcEnvelope.from_json_line(line))
        except asyncio.CancelledError:
            pass
        finally:
            w.close()


# -- fixtures ---------------------------------------------------------

@pytest.fixture()
def db() -> sqlite3.Connection:
    return _make_db()

@pytest.fixture()
def ilog(db: sqlite3.Connection) -> InteractionLog:
    log = InteractionLog()
    log.open(conn=db)
    return log

@pytest.fixture()
async def child_servers():
    servers: dict[str, MockChildServer] = {}
    for name in ("hub-timer", "hub-ble-mgr", "hub-pen-sync", "hub-uplink"):
        s = MockChildServer()
        await s.start()
        servers[name] = s
    yield servers
    for s in servers.values():
        await s.stop()


def _ipc_fn(servers: dict[str, MockChildServer]) -> Any:
    """Build IPC send function that routes to real mock TCP servers."""
    async def send(target: str, env: IpcEnvelope) -> IpcEnvelope | None:
        srv = servers.get(target)
        if srv is None:
            return None
        client = IpcClient(f"127.0.0.1:{srv.port}", source_id=MODULE_ID)
        await client.connect()
        try:
            await client.send(env)
        finally:
            await client.close()
        return None
    return send


def _req(from_st: str, to_st: str, eid: str = "exam-1", **extra: Any) -> IpcEnvelope:
    """Shortcut to build an FSM transition request envelope."""
    return IpcEnvelope(
        msg_type=FSM_TRANSITION_REQUEST, source="test",
        target=MODULE_ID, expects_reply=True,
        payload={"exam_id": eid, "from_state": from_st, "to_state": to_st, **extra},
    )

# -- I-SUP-INT-01: First-boot detection ------------------------------

class TestFirstBoot:
    def test_missing_conf(self, tmp_path: Path) -> None:
        """I-SUP-INT-01: Missing hub.conf flags first-boot."""
        assert is_first_boot(str(tmp_path / "nope.conf")) is True

    def test_existing_conf(self, tmp_path: Path) -> None:
        (tmp_path / "hub.conf").write_text("[hub]\nhub_id=X\n")
        assert is_first_boot(str(tmp_path / "hub.conf")) is False

# -- I-SUP-INT-02: Provisioning mock ---------------------------------

class TestProvisioning:
    async def test_provision_stores_config(self, db: sqlite3.Connection, tmp_path: Path) -> None:
        """I-SUP-INT-02: Provision populates DB and writes hub.conf."""
        async def mock_post(url: str, payload: dict) -> dict:
            assert "/api/v1/hubs/provision" in url
            return {"hub_id": "EPH-42", "institute_id": "I-1",
                    "invig_codes": [{"code": "A", "valid_from": "x", "valid_until": "y"}],
                    "pen_inventory": [{"pen_mac": "AA:BB", "pen_serial": "P1", "fw_version": "1"}]}

        conf = str(tmp_path / "hub.conf")
        r = await run_provisioning("CODE", "https://ex.com", "wifi", db,
                                   "2026-03-19T00:00:00Z", http_post_fn=mock_post, config_path=conf)
        assert r.hub_id == "EPH-42"
        assert Path(conf).exists()
        assert db.execute("SELECT hub_id FROM hub_config").fetchone()[0] == "EPH-42"

# -- I-SUP-INT-03: Invigilator auth logged ----------------------------

class TestInvigAuth:
    async def test_auth_event_logged(self, ilog: InteractionLog) -> None:
        """I-SUP-INT-03: Invig BLE auth success is recorded."""
        ilog.append(LogEntry(source="hub-invig-ble", event_type="invig_auth_ok",
                             invig_id="INV-1", detail={"code_used": "ABC"}))
        assert any(e["event_type"] == "invig_auth_ok" for e in ilog.recent(10))

# -- I-SUP-INT-04: Exam arm (created -> armed) -----------------------

class TestExamArm:
    async def test_arm(self, db, ilog, child_servers) -> None:
        """I-SUP-INT-04: created->armed persists and validates."""
        _insert_exam(db, "exam-1", CREATED)
        h = SupervisorIpcHandlers(db, Orchestrator(ilog, ipc_send_fn=_ipc_fn(child_servers)), ilog)
        reply = await h.handle_transition(_req(CREATED, ARMED))
        assert reply.payload["state"] == ARMED
        assert db.execute("SELECT state FROM exam_sessions WHERE exam_id='exam-1'").fetchone()[0] == ARMED

# -- I-SUP-INT-05: Timer start (armed -> timer_running) --------------

class TestTimerStart:
    async def test_timer_ipc(self, db, ilog, child_servers) -> None:
        """I-SUP-INT-05: armed->timer_running sends timer.arm IPC."""
        _insert_exam(db, "exam-1", ARMED)
        h = SupervisorIpcHandlers(db, Orchestrator(ilog, ipc_send_fn=_ipc_fn(child_servers)), ilog)
        reply = await h.handle_transition(_req(ARMED, TIMER_RUNNING, duration_sec=3600))
        assert reply.payload["state"] == TIMER_RUNNING
        await asyncio.sleep(0.05)
        assert any(m.msg_type == TIMER_ARM_REQUEST for m in child_servers["hub-timer"].received)

# -- I-SUP-INT-06: Timer expire -> dongle activation ------------------

class TestTimerExpire:
    async def test_ble_scan_ipc(self, db, ilog, child_servers) -> None:
        """I-SUP-INT-06: timer_running->dongle_activation sends ble.scan.start."""
        _insert_exam(db, "exam-1", TIMER_RUNNING)
        h = SupervisorIpcHandlers(db, Orchestrator(ilog, ipc_send_fn=_ipc_fn(child_servers)), ilog)
        reply = await h.handle_transition(_req(TIMER_RUNNING, DONGLE_ACTIVATION))
        assert reply.payload["state"] == DONGLE_ACTIVATION
        await asyncio.sleep(0.05)
        assert any(m.msg_type == BLE_SCAN_START_REQUEST for m in child_servers["hub-ble-mgr"].received)

# -- I-SUP-INT-07: Pen sync per-pen IPC ------------------------------

class TestPenSync:
    async def test_per_pen_ipc(self, db, ilog, child_servers) -> None:
        """I-SUP-INT-07: pen_sync sends pen.sync.request per pen."""
        _insert_exam(db, "exam-1", DONGLE_ACTIVATION)
        pens = [{"pen_mac": "AA:01", "dongle_mac": "D:01"},
                {"pen_mac": "AA:02", "dongle_mac": "D:02"}]
        h = SupervisorIpcHandlers(db, Orchestrator(ilog, ipc_send_fn=_ipc_fn(child_servers)), ilog)
        await h.handle_transition(_req(DONGLE_ACTIVATION, PEN_SYNC, pens=pens))
        await asyncio.sleep(0.05)
        msgs = [m for m in child_servers["hub-pen-sync"].received if m.msg_type == PEN_SYNC_REQUEST]
        assert len(msgs) == 2

# -- I-SUP-INT-08: Sync complete -> upload ----------------------------

class TestSyncToUpload:
    async def test_upload_ipc(self, db, ilog, child_servers) -> None:
        """I-SUP-INT-08: sync_complete->uploading sends uplink.upload.request."""
        _insert_exam(db, "exam-1", SYNC_COMPLETE)
        h = SupervisorIpcHandlers(db, Orchestrator(ilog, ipc_send_fn=_ipc_fn(child_servers)), ilog)
        await h.handle_transition(_req(SYNC_COMPLETE, UPLOADING, upload_path="wifi"))
        await asyncio.sleep(0.05)
        assert any(m.msg_type == UPLINK_UPLOAD_REQUEST for m in child_servers["hub-uplink"].received)

# -- I-SUP-INT-09: Upload complete (terminal) ------------------------

class TestUploadComplete:
    async def test_terminal(self, db, ilog, child_servers) -> None:
        """I-SUP-INT-09: uploading->upload_complete reaches terminal."""
        _insert_exam(db, "exam-1", UPLOADING)
        h = SupervisorIpcHandlers(db, Orchestrator(ilog, ipc_send_fn=_ipc_fn(child_servers)), ilog)
        reply = await h.handle_transition(_req(UPLOADING, UPLOAD_COMPLETE))
        assert reply.payload["state"] == UPLOAD_COMPLETE
        assert db.execute("SELECT state FROM exam_sessions WHERE exam_id='exam-1'").fetchone()[0] == UPLOAD_COMPLETE

# -- I-SUP-INT-10: Full flow interaction log --------------------------

class TestFullFlowLog:
    async def test_all_transitions_logged(self, db, ilog, child_servers) -> None:
        """I-SUP-INT-10: Every FSM transition is recorded in interaction_log."""
        _insert_exam(db, "exam-1", CREATED)
        h = SupervisorIpcHandlers(db, Orchestrator(ilog, ipc_send_fn=_ipc_fn(child_servers)), ilog)
        flow = [(CREATED, ARMED), (ARMED, TIMER_RUNNING),
                (TIMER_RUNNING, DONGLE_ACTIVATION), (DONGLE_ACTIVATION, PEN_SYNC),
                (PEN_SYNC, SYNC_COMPLETE), (SYNC_COMPLETE, UPLOADING),
                (UPLOADING, UPLOAD_COMPLETE)]
        for fr, to in flow:
            await h.handle_transition(_req(fr, to))
        await asyncio.sleep(0.05)
        entries = ilog.recent(100)
        assert sum(1 for e in entries if e["event_type"] == "fsm_transition") == len(flow)
