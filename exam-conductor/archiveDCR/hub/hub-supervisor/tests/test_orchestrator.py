"""Unit tests for orchestrator.py — FSM side effects trigger correct IPC.

All IPC communication is mocked via an injected ``ipc_send_fn``.
"""

from __future__ import annotations

import sqlite3

import pytest

from hub_common.ipc_protocol import IpcEnvelope
from hub_common.message_types import (
    BLE_SCAN_START_REQUEST,
    BLE_SCAN_STOP_REQUEST,
    PEN_SYNC_REQUEST,
    TIMER_ARM_REQUEST,
    TIMER_CANCEL_REQUEST,
    UPLINK_UPLOAD_REQUEST,
)

from src.hub_fsm import (
    ARMED,
    CANCELLED,
    CREATED,
    DONGLE_ACTIVATION,
    PEN_SYNC,
    SYNC_COMPLETE,
    TIMER_RUNNING,
    UPLOAD_COMPLETE,
    UPLOADING,
)
from src.interaction_log import InteractionLog
from src.orchestrator import Orchestrator


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture()
def ilog() -> InteractionLog:
    """In-memory interaction log."""
    log = InteractionLog()
    conn = sqlite3.connect(":memory:")
    log.open(conn=conn)
    return log


@pytest.fixture()
def sent_messages() -> list[tuple[str, IpcEnvelope]]:
    """Accumulator for IPC messages sent by the orchestrator."""
    return []


@pytest.fixture()
def orchestrator(
    ilog: InteractionLog, sent_messages: list[tuple[str, IpcEnvelope]]
) -> Orchestrator:
    async def mock_send(
        target: str, env: IpcEnvelope
    ) -> IpcEnvelope | None:
        sent_messages.append((target, env))
        return None

    return Orchestrator(ilog, ipc_send_fn=mock_send)


# ===================================================================
# Side-effect tests per state
# ===================================================================

class TestOnArmed:
    async def test_logs_armed_event(
        self, orchestrator: Orchestrator, ilog: InteractionLog,
        sent_messages: list[tuple[str, IpcEnvelope]],
    ) -> None:
        await orchestrator.on_transition("exam-1", ARMED)
        # No IPC messages sent for armed (just validation)
        assert len(sent_messages) == 0
        # But an interaction log entry is created
        entries = ilog.recent(10)
        assert any(e["event_type"] == "exam_armed" for e in entries)


class TestOnTimerRunning:
    async def test_sends_timer_arm(
        self, orchestrator: Orchestrator,
        sent_messages: list[tuple[str, IpcEnvelope]],
    ) -> None:
        await orchestrator.on_transition(
            "exam-1", TIMER_RUNNING,
            context={"duration_sec": 3600, "armed_by": "invigilator-1"},
        )
        assert len(sent_messages) == 1
        target, env = sent_messages[0]
        assert target == "hub-timer"
        assert env.msg_type == TIMER_ARM_REQUEST
        assert env.payload["exam_id"] == "exam-1"
        assert env.payload["duration_sec"] == 3600

    async def test_logs_timer_start(
        self, orchestrator: Orchestrator, ilog: InteractionLog,
    ) -> None:
        await orchestrator.on_transition(
            "exam-1", TIMER_RUNNING, context={"duration_sec": 1800},
        )
        entries = ilog.recent(10)
        assert any(e["event_type"] == "exam_timer_start" for e in entries)


class TestOnDongleActivation:
    async def test_sends_ble_scan_start(
        self, orchestrator: Orchestrator,
        sent_messages: list[tuple[str, IpcEnvelope]],
    ) -> None:
        await orchestrator.on_transition("exam-1", DONGLE_ACTIVATION)
        assert len(sent_messages) == 1
        target, env = sent_messages[0]
        assert target == "hub-ble-mgr"
        assert env.msg_type == BLE_SCAN_START_REQUEST
        assert env.payload["mode"] == "sync"


class TestOnPenSync:
    async def test_sends_sync_per_pen(
        self, orchestrator: Orchestrator,
        sent_messages: list[tuple[str, IpcEnvelope]],
    ) -> None:
        pens = [
            {"pen_mac": "AA:BB:CC:DD:EE:01", "dongle_mac": "11:22:33:44:55:01"},
            {"pen_mac": "AA:BB:CC:DD:EE:02", "dongle_mac": "11:22:33:44:55:02"},
        ]
        await orchestrator.on_transition(
            "exam-1", PEN_SYNC, context={"pens": pens},
        )
        assert len(sent_messages) == 2
        for target, env in sent_messages:
            assert target == "hub-pen-sync"
            assert env.msg_type == PEN_SYNC_REQUEST

    async def test_no_pens_sends_nothing(
        self, orchestrator: Orchestrator,
        sent_messages: list[tuple[str, IpcEnvelope]],
    ) -> None:
        await orchestrator.on_transition(
            "exam-1", PEN_SYNC, context={"pens": []},
        )
        assert len(sent_messages) == 0


class TestOnUploading:
    async def test_sends_upload_request(
        self, orchestrator: Orchestrator,
        sent_messages: list[tuple[str, IpcEnvelope]],
    ) -> None:
        await orchestrator.on_transition(
            "exam-1", UPLOADING, context={"upload_path": "wifi"},
        )
        assert len(sent_messages) == 1
        target, env = sent_messages[0]
        assert target == "hub-uplink"
        assert env.msg_type == UPLINK_UPLOAD_REQUEST
        assert env.payload["path"] == "wifi"

    async def test_default_upload_path_is_auto(
        self, orchestrator: Orchestrator,
        sent_messages: list[tuple[str, IpcEnvelope]],
    ) -> None:
        await orchestrator.on_transition("exam-1", UPLOADING)
        _, env = sent_messages[0]
        assert env.payload["path"] == "auto"


class TestOnCancelled:
    async def test_sends_timer_cancel_and_scan_stop(
        self, orchestrator: Orchestrator,
        sent_messages: list[tuple[str, IpcEnvelope]],
    ) -> None:
        await orchestrator.on_transition(
            "exam-1", CANCELLED, context={"reason": "invigilator_abort"},
        )
        assert len(sent_messages) == 2
        targets = {t for t, _ in sent_messages}
        assert "hub-timer" in targets
        assert "hub-ble-mgr" in targets
        types = {e.msg_type for _, e in sent_messages}
        assert TIMER_CANCEL_REQUEST in types
        assert BLE_SCAN_STOP_REQUEST in types


# ===================================================================
# No side effects for states without handlers
# ===================================================================

class TestNoSideEffects:
    @pytest.mark.parametrize("state", [CREATED, SYNC_COMPLETE, UPLOAD_COMPLETE])
    async def test_no_ipc_for_state(
        self, orchestrator: Orchestrator,
        sent_messages: list[tuple[str, IpcEnvelope]],
        state: str,
    ) -> None:
        await orchestrator.on_transition("exam-1", state)
        # Some states (like CREATED) have no handler, so 0 messages.
        # SYNC_COMPLETE and UPLOAD_COMPLETE also have no side effects.
        # (ARMED logs but doesn't send IPC — already tested above.)
        if state != ARMED:
            assert len(sent_messages) == 0


# ===================================================================
# Error resilience
# ===================================================================

class TestErrorResilience:
    async def test_ipc_failure_does_not_crash(
        self, ilog: InteractionLog,
    ) -> None:
        """Orchestrator catches IPC errors and logs them."""
        async def failing_send(
            target: str, env: IpcEnvelope
        ) -> IpcEnvelope | None:
            raise OSError("connection refused")

        orch = Orchestrator(ilog, ipc_send_fn=failing_send)
        # Should not raise
        await orch.on_transition(
            "exam-1", TIMER_RUNNING, context={"duration_sec": 60},
        )
        entries = ilog.recent(10)
        assert any(e["event_type"] == "side_effect_error" for e in entries)
