"""Bridge between TUI and hub-supervisor / module IPC sockets.

Polls each hub module via ``IpcClient``, caches last-known state so
screens always display something even when IPC is unreachable.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from hub_common.config import HubConfig, load_hub_config
from hub_common.ipc_protocol import IpcClient, IpcEnvelope

logger = logging.getLogger(__name__)
POLL_INTERVAL_SEC = 1.0

_WIN_TCP_PORTS: dict[str, int] = {
    "hub-supervisor": 19100, "hub-ble-mgr": 19101,
    "hub-timer": 19102, "hub-uplink": 19103, "hub-store": 19104,
}


def _resolve_socket(cfg: HubConfig, module_id: str) -> str:
    if sys.platform == "win32" or not hasattr(asyncio, "open_unix_connection"):
        return f"localhost:{_WIN_TCP_PORTS.get(module_id, 19100)}"
    return cfg.socket_path(module_id)


@dataclass
class SupervisorSnapshot:
    connected: bool = False
    exam_id: str = ""
    state: str = "DISCONNECTED"
    timer_remaining_sec: int = 0
    timer_state: str = ""
    dongles: list[dict[str, Any]] = field(default_factory=list)
    storage: dict[str, Any] = field(default_factory=dict)
    upload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DongleStatus:
    connected: bool = False
    dongles: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WifiStatus:
    connected: bool = False
    ssid: str = ""
    band: str = ""
    channel: str = ""
    signal: str = ""
    ip: str = ""
    gateway: str = ""
    dns: str = ""
    backend_reachable: bool = False
    latency_ms: int = 0


@dataclass
class StoreHealth:
    connected: bool = False
    sd_ok: bool = False
    usb_ok: bool = False
    degraded: bool = False
    sd_free: str = ""
    usb_free: str = ""


@dataclass
class SyncAggregated:
    total: int = 0
    complete: int = 0
    in_progress: int = 0
    failed: int = 0
    pending: int = 0


class HubIpcBridge:
    """Async bridge polling hub module IPC sockets at 1 Hz."""

    def __init__(self, cfg: HubConfig | None = None) -> None:
        self._cfg = cfg or load_hub_config()
        self._clients: dict[str, IpcClient] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self.supervisor = SupervisorSnapshot()
        self.dongles = DongleStatus()
        self.wifi = WifiStatus()
        self.store = StoreHealth()
        self.sync = SyncAggregated()

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Connect to module sockets and begin 1 Hz polling."""
        for module_id in ("hub-supervisor", "hub-ble-mgr", "hub-timer",
                          "hub-uplink", "hub-store"):
            path = _resolve_socket(self._cfg, module_id)
            client = IpcClient(path, source_id="hub-tui")
            self._clients[module_id] = client
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Cancel polling and close all IPC connections."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

    # -- polling -------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Periodically request snapshots from each module."""
        while True:
            await asyncio.gather(
                self._poll_supervisor(),
                self._poll_dongles(),
                self._poll_wifi(),
                self._poll_store(),
                return_exceptions=True,
            )
            await asyncio.sleep(POLL_INTERVAL_SEC)

    # -- individual poll methods ---------------------------------------------

    async def _ensure_connected(self, module_id: str) -> IpcClient:
        client = self._clients[module_id]
        if client._writer is None:
            await client.connect()
        return client

    async def _poll_supervisor(self) -> None:
        try:
            client = await self._ensure_connected("hub-supervisor")
            env = IpcEnvelope(
                msg_type="fsm.snapshot.request",
                source="hub-tui",
                target="hub-supervisor",
                payload={},
            )
            reply = await client.request(env, timeout=3.0)
            p = reply.payload
            self.supervisor = SupervisorSnapshot(
                connected=True,
                exam_id=p.get("exam_id", ""),
                state=p.get("state", "UNKNOWN"),
                timer_remaining_sec=p.get("timer", {}).get("remaining_sec", 0),
                timer_state=p.get("timer", {}).get("state", ""),
                dongles=p.get("dongles", []),
                storage=p.get("storage", {}),
                upload=p.get("upload", {}),
            )
            # Derive sync aggregated from supervisor snapshot dongles/upload.
            self._derive_sync(p)
        except Exception:
            logger.debug("Supervisor IPC unreachable", exc_info=True)
            self.supervisor.connected = False

    async def _poll_dongles(self) -> None:
        try:
            client = await self._ensure_connected("hub-ble-mgr")
            env = IpcEnvelope(
                msg_type="ble.status.request",
                source="hub-tui",
                target="hub-ble-mgr",
                payload={},
            )
            reply = await client.request(env, timeout=3.0)
            self.dongles = DongleStatus(
                connected=True,
                dongles=reply.payload.get("dongles", []),
            )
        except Exception:
            logger.debug("BLE-mgr IPC unreachable", exc_info=True)
            self.dongles.connected = False

    async def _poll_wifi(self) -> None:
        try:
            client = await self._ensure_connected("hub-uplink")
            env = IpcEnvelope(
                msg_type="uplink.status.request",
                source="hub-tui",
                target="hub-uplink",
                payload={},
            )
            reply = await client.request(env, timeout=3.0)
            p = reply.payload
            self.wifi = WifiStatus(
                connected=True,
                ssid=p.get("ssid", ""),
                band=p.get("band", ""),
                channel=str(p.get("channel", "")),
                signal=p.get("signal", ""),
                ip=p.get("ip", ""),
                gateway=p.get("gateway", ""),
                dns=p.get("dns", ""),
                backend_reachable=p.get("backend_reachable", False),
                latency_ms=p.get("latency_ms", 0),
            )
        except Exception:
            logger.debug("Uplink IPC unreachable", exc_info=True)
            self.wifi.connected = False

    async def _poll_store(self) -> None:
        try:
            client = await self._ensure_connected("hub-store")
            env = IpcEnvelope(
                msg_type="store.health.request",
                source="hub-tui",
                target="hub-store",
                payload={},
            )
            reply = await client.request(env, timeout=3.0)
            p = reply.payload
            self.store = StoreHealth(
                connected=True,
                sd_ok=p.get("sd_ok", False),
                usb_ok=p.get("usb_ok", False),
                degraded=p.get("degraded", False),
                sd_free=p.get("sd_free", ""),
                usb_free=p.get("usb_free", ""),
            )
        except Exception:
            logger.debug("Store IPC unreachable", exc_info=True)
            self.store.connected = False

    # -- helpers -------------------------------------------------------------

    def _derive_sync(self, payload: dict[str, Any]) -> None:
        """Build aggregated sync counts from supervisor snapshot."""
        upload = payload.get("upload", {})
        self.sync = SyncAggregated(
            total=upload.get("total", 0),
            complete=upload.get("complete", 0),
            in_progress=upload.get("in_progress", 0),
            failed=upload.get("failed", 0),
            pending=upload.get("pending", 0),
        )

    # -- one-shot commands ---------------------------------------------------

    async def _ipc_request(
        self, module: str, msg_type: str, payload: dict | None = None,
    ) -> dict[str, Any]:
        """Generic IPC request helper. Returns payload or ``{error: ...}``."""
        try:
            client = await self._ensure_connected(module)
            env = IpcEnvelope(
                msg_type=msg_type, source="hub-tui",
                target=module, payload=payload or {},
            )
            reply = await client.request(env, timeout=5.0)
            return reply.payload
        except Exception as exc:
            return {"error": str(exc)}

    async def request_dongle_reset(self, dongle_mac: str) -> dict[str, Any]:
        return await self._ipc_request(
            "hub-ble-mgr", "ble.dongle.reset.request",
            {"dongle_mac": dongle_mac},
        )

    async def request_snapshot(self) -> dict[str, Any]:
        return await self._ipc_request("hub-supervisor", "fsm.snapshot.request")

    async def request_dongle_status(self) -> list[dict[str, Any]]:
        result = await self._ipc_request("hub-ble-mgr", "ble.status.request")
        return result.get("dongles", self.dongles.dongles)

    async def request_wifi_status(self) -> dict[str, Any]:
        return await self._ipc_request("hub-uplink", "uplink.status.request")
