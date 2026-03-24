"""IPC message handlers for hub-ble-mgr.

Handles incoming requests from hub-supervisor and hub-pen-sync, and
emits BLE events (scan results, dongle health, connection status).

Message types follow ``new-docs/hub/ipc-protocol.md`` Section 3.3.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from src.config import MODULE_ID
from src.connection_manager import ConnectionManager
from src.dongle_manager import DongleManager, DongleStatus
from src.pen_discovery import PenDiscovery, PenInfo

# TODO: Import from hub-common when available on PYTHONPATH.
# Local fallback: define message type constants inline.
try:
    from hub_common.message_types import (
        BLE_CONNECT_REQUEST,
        BLE_CONNECT_RESULT,
        BLE_DONGLE_HEALTH_EVENT,
        BLE_SCAN_RESULT_EVENT,
        BLE_SCAN_START_REQUEST,
        BLE_SCAN_STOP_REQUEST,
    )
except ImportError:  # pragma: no cover
    BLE_SCAN_START_REQUEST = "ble.scan.start.request"
    BLE_SCAN_STOP_REQUEST = "ble.scan.stop.request"
    BLE_SCAN_RESULT_EVENT = "ble.scan.result.event"
    BLE_DONGLE_HEALTH_EVENT = "ble.dongle.health.event"
    BLE_CONNECT_REQUEST = "ble.connect.request"
    BLE_CONNECT_RESULT = "ble.connect.result"

# Additional message types used locally.
MSG_SUPERVISOR_HEALTH_REQUEST = "supervisor.health.request"
MSG_SUPERVISOR_HEALTH_RESULT = "supervisor.health.result"
MSG_PEN_CONNECTED_EVENT = "ble.pen.connected"
MSG_PEN_DISCONNECTED_EVENT = "ble.pen.disconnected"

logger = logging.getLogger(__name__)

# Type alias for the async broadcast callback.
BroadcastFn = Callable[..., Awaitable[None]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------

def _make_envelope(
    msg_type: str,
    source: str,
    target: str,
    payload: dict[str, Any],
    correlation_id: str = "",
) -> dict[str, Any]:
    """Build a dict envelope suitable for broadcast or reply.

    Works with both the local BleIpcServer.Envelope and hub-common's
    IpcEnvelope -- callers that need the hub-common type can wrap this.
    """
    return {
        "msg_type": msg_type,
        "source": source,
        "target": target,
        "payload": payload,
        "correlation_id": correlation_id,
    }


def _reply(request: Any, reply_type: str, payload: dict) -> Any:
    """Build a reply envelope correlated to *request*."""
    # If the request has a make_reply helper (hub-common IpcEnvelope), use it.
    if hasattr(request, "make_reply"):
        return request.make_reply(reply_type, payload, source=MODULE_ID)

    # Local Envelope from ipc_server.py.
    try:
        from src.ipc_server import Envelope
        return Envelope(
            msg_type=reply_type,
            source=MODULE_ID,
            target=getattr(request, "source", ""),
            payload=payload,
            correlation_id=getattr(request, "msg_id", ""),
        )
    except ImportError:
        pass

    return _make_envelope(
        reply_type, MODULE_ID,
        getattr(request, "source", ""),
        payload,
        correlation_id=getattr(request, "msg_id", ""),
    )


def _event(msg_type: str, target: str, payload: dict) -> Any:
    """Build a fire-and-forget event envelope."""
    try:
        from src.ipc_server import Envelope
        return Envelope(
            msg_type=msg_type,
            source=MODULE_ID,
            target=target,
            payload=payload,
        )
    except ImportError:
        pass

    return _make_envelope(msg_type, MODULE_ID, target, payload)


# ---------------------------------------------------------------------------
# BleIpcHandlers
# ---------------------------------------------------------------------------

class BleIpcHandlers:
    """Handler dispatch for hub-ble-mgr IPC messages.

    ``broadcast_fn`` is required -- it is called for fire-and-forget
    events (scan results, health, connect/disconnect) that must reach
    all connected clients.  Passing None is a programming error.
    """

    def __init__(
        self,
        dongle_mgr: DongleManager,
        discovery: PenDiscovery,
        conn_mgr: ConnectionManager,
        broadcast_fn: BroadcastFn,
    ) -> None:
        self._dongle_mgr = dongle_mgr
        self._discovery = discovery
        self._conn_mgr = conn_mgr
        self._broadcast = broadcast_fn

    # -- Inbound handlers ---------------------------------------------------

    async def handle_scan_start(self, env: Any) -> Any:
        """Handle ``ble.scan.start.request``."""
        payload = env.payload
        timeout_sec = payload.get("timeout_sec", 60)
        logger.info(
            "Scan start requested (exam=%s, mode=%s, timeout=%ds)",
            payload.get("exam_id"), payload.get("mode"), timeout_sec,
        )
        await self._discovery.start_scan(timeout_sec=timeout_sec)
        return None  # fire-and-forget ACK via scan result events

    async def handle_scan_stop(self, env: Any) -> Any:
        """Handle ``ble.scan.stop.request``."""
        payload = env.payload
        logger.info(
            "Scan stop requested (exam=%s, reason=%s)",
            payload.get("exam_id"), payload.get("reason"),
        )
        await self._discovery.stop_scan()
        return None

    async def handle_connect(self, env: Any) -> Any:
        """Handle ``ble.connect.request``."""
        payload = env.payload
        pen_mac: str = payload["pen_mac"]
        dongle_mac: str = payload.get("dongle_mac", "")

        record = await self._conn_mgr.connect_pen(
            pen_mac, dongle_mac or None,
        )

        if record is None:
            return _reply(env, BLE_CONNECT_RESULT, {
                "exam_id": payload.get("exam_id", ""),
                "pen_mac": pen_mac,
                "dongle_mac": dongle_mac,
                "connection_id": "",
                "error": "connection_failed",
            })

        return _reply(env, BLE_CONNECT_RESULT, {
            "exam_id": payload.get("exam_id", ""),
            "pen_mac": pen_mac,
            "dongle_mac": record.dongle_mac,
            "connection_id": record.connection_id,
        })

    async def handle_health(self, env: Any) -> Any:
        """Handle ``supervisor.health.request``."""
        summary = self._dongle_mgr.summary()
        return _reply(env, MSG_SUPERVISOR_HEALTH_RESULT, {
            "module": MODULE_ID,
            "healthy": True,
            "scanning": self._discovery.scanning,
            **summary,
        })

    # -- Outbound event emitters (called by domain callbacks) ---------------

    async def emit_scan_result(self, pen: PenInfo) -> None:
        """Emit ``ble.scan.result.event`` for a discovered pen."""
        env = _event(BLE_SCAN_RESULT_EVENT, "hub-supervisor", {
            "exam_id": "",
            "pen_mac": pen.mac,
            "dongle_mac": pen.dongle_mac,
            "rssi": pen.rssi,
            "battery_pct": pen.battery_pct,
        })
        await self._broadcast(env)

    async def emit_dongle_health(
        self, dongle_mac: str, status: DongleStatus, detail: str,
    ) -> None:
        """Emit ``ble.dongle.health.event``."""
        env = _event(BLE_DONGLE_HEALTH_EVENT, "hub-supervisor", {
            "dongle_mac": dongle_mac,
            "status": status.value,
            "detail": detail,
        })
        await self._broadcast(env)

    async def emit_pen_connected(self, record: Any) -> None:
        """Emit ``ble.pen.connected``."""
        env = _event(MSG_PEN_CONNECTED_EVENT, "hub-supervisor", {
            "pen_mac": record.pen_mac,
            "dongle_mac": record.dongle_mac,
            "connection_id": record.connection_id,
        })
        await self._broadcast(env)

    async def emit_pen_disconnected(
        self, pen_mac: str, dongle_mac: str,
    ) -> None:
        """Emit ``ble.pen.disconnected``."""
        env = _event(MSG_PEN_DISCONNECTED_EVENT, "hub-supervisor", {
            "pen_mac": pen_mac,
            "dongle_mac": dongle_mac,
        })
        await self._broadcast(env)
