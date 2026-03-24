"""Tests for HubIpcBridge with mocked IPC servers on TCP loopback."""
from __future__ import annotations

import pytest

from hub_common.config import HubConfig
from hub_common.ipc_protocol import IpcClient, IpcEnvelope, IpcServer
from src.ipc_bridge import HubIpcBridge

_SUPERVISOR_PORT = 19200
_BLE_MGR_PORT = 19201
_TIMER_PORT = 19202
_UPLINK_PORT = 19203
_STORE_PORT = 19204


def _test_config() -> HubConfig:
    return HubConfig(
        hub_id="TEST-001", backend_url="https://example.com",
        uplink_mode="wifi", region="US",
        sd_data_path="/tmp/exampen-test/sd",
        usb_data_path="/tmp/exampen-test/usb",
        socket_dir="/tmp/exampen-test/run",
    )


def _make_bridge() -> HubIpcBridge:
    import src.ipc_bridge as mod
    mod._WIN_TCP_PORTS.update({
        "hub-supervisor": _SUPERVISOR_PORT, "hub-ble-mgr": _BLE_MGR_PORT,
        "hub-timer": _TIMER_PORT, "hub-uplink": _UPLINK_PORT,
        "hub-store": _STORE_PORT,
    })
    return HubIpcBridge(cfg=_test_config())


def _make_client(port: int) -> IpcClient:
    return IpcClient(f"localhost:{port}", source_id="hub-tui")


async def _start_mock_server(
    port: int, module_id: str, handlers: dict[str, dict],
) -> IpcServer:
    server = IpcServer(f"localhost:{port}", module_id=module_id)
    for msg_type, payload in handlers.items():
        async def _handler(
            env: IpcEnvelope, _payload=payload, _module=module_id,
        ) -> IpcEnvelope:
            return env.make_reply(
                env.msg_type.rsplit(".", 1)[0] + ".result",
                _payload, source=_module,
            )
        server.register(msg_type, _handler)
    await server.start()
    return server

@pytest.mark.asyncio
async def test_supervisor_snapshot() -> None:
    """Bridge caches supervisor state after poll."""
    server = await _start_mock_server(
        _SUPERVISOR_PORT,
        "hub-supervisor",
        {
            "fsm.snapshot.request": {
                "exam_id": "EX-001",
                "state": "EXAM_ARMED",
                "timer": {"remaining_sec": 1200, "state": "running"},
                "dongles": [],
                "storage": {},
                "upload": {
                    "total": 40,
                    "complete": 30,
                    "in_progress": 5,
                    "failed": 1,
                    "pending": 4,
                },
            },
        },
    )
    bridge = _make_bridge()
    try:
        client = _make_client(_SUPERVISOR_PORT)
        bridge._clients["hub-supervisor"] = client
        await client.connect()
        await bridge._poll_supervisor()

        assert bridge.supervisor.connected is True
        assert bridge.supervisor.state == "EXAM_ARMED"
        assert bridge.supervisor.exam_id == "EX-001"
        assert bridge.supervisor.timer_remaining_sec == 1200
        assert bridge.sync.total == 40
        assert bridge.sync.complete == 30
        assert bridge.sync.failed == 1
    finally:
        await bridge.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_dongle_status() -> None:
    """Bridge caches dongle list from ble-mgr."""
    server = await _start_mock_server(
        _BLE_MGR_PORT,
        "hub-ble-mgr",
        {
            "ble.status.request": {
                "dongles": [
                    {
                        "dongle_id": "D1",
                        "dongle_mac": "AA:BB:CC:DD:EE:01",
                        "hci_path": "hci0",
                        "pens": "8/8",
                        "status": "healthy",
                    },
                    {
                        "dongle_id": "D2",
                        "dongle_mac": "AA:BB:CC:DD:EE:02",
                        "hci_path": "hci1",
                        "pens": "7/8",
                        "status": "degraded",
                    },
                ],
            },
        },
    )
    bridge = _make_bridge()
    try:
        client = _make_client(_BLE_MGR_PORT)
        bridge._clients["hub-ble-mgr"] = client
        await client.connect()
        await bridge._poll_dongles()

        assert bridge.dongles.connected is True
        assert len(bridge.dongles.dongles) == 2
        assert bridge.dongles.dongles[0]["dongle_id"] == "D1"
        assert bridge.dongles.dongles[1]["status"] == "degraded"
    finally:
        await bridge.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_wifi_status() -> None:
    """Bridge caches wifi info from hub-uplink."""
    server = await _start_mock_server(
        _UPLINK_PORT,
        "hub-uplink",
        {
            "uplink.status.request": {
                "ssid": "SchoolNet-5G",
                "band": "5 GHz",
                "channel": 36,
                "signal": "-42",
                "ip": "192.168.1.105",
                "gateway": "192.168.1.1",
                "dns": "8.8.8.8",
                "backend_reachable": True,
                "latency_ms": 34,
            },
        },
    )
    bridge = _make_bridge()
    try:
        client = _make_client(_UPLINK_PORT)
        bridge._clients["hub-uplink"] = client
        await client.connect()
        await bridge._poll_wifi()

        assert bridge.wifi.connected is True
        assert bridge.wifi.ssid == "SchoolNet-5G"
        assert bridge.wifi.channel == "36"
        assert bridge.wifi.backend_reachable is True
        assert bridge.wifi.latency_ms == 34
    finally:
        await bridge.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_store_health() -> None:
    """Bridge caches store health from hub-store."""
    server = await _start_mock_server(
        _STORE_PORT,
        "hub-store",
        {
            "store.health.request": {
                "sd_ok": True,
                "usb_ok": True,
                "degraded": False,
                "sd_free": "12.3 GB",
                "usb_free": "8.1 GB",
            },
        },
    )
    bridge = _make_bridge()
    try:
        client = _make_client(_STORE_PORT)
        bridge._clients["hub-store"] = client
        await client.connect()
        await bridge._poll_store()

        assert bridge.store.connected is True
        assert bridge.store.sd_ok is True
        assert bridge.store.usb_ok is True
        assert bridge.store.sd_free == "12.3 GB"
        assert bridge.store.usb_free == "8.1 GB"
        assert bridge.store.degraded is False
    finally:
        await bridge.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_graceful_degradation_when_unreachable() -> None:
    """When IPC socket is unreachable, bridge shows disconnected state."""
    bridge = _make_bridge()
    # No servers started — all polls should fail gracefully.
    bridge._clients["hub-supervisor"] = _make_client(19299)
    bridge._clients["hub-ble-mgr"] = _make_client(19298)
    bridge._clients["hub-uplink"] = _make_client(19297)
    bridge._clients["hub-store"] = _make_client(19296)

    await bridge._poll_supervisor()
    await bridge._poll_dongles()
    await bridge._poll_wifi()
    await bridge._poll_store()

    assert bridge.supervisor.connected is False
    assert bridge.dongles.connected is False
    assert bridge.wifi.connected is False
    assert bridge.store.connected is False

    await bridge.stop()


@pytest.mark.asyncio
async def test_request_dongle_reset() -> None:
    """Bridge sends dongle reset command and returns result."""
    async def _reset_handler(env: IpcEnvelope) -> IpcEnvelope:
        mac = env.payload.get("dongle_mac", "")
        return env.make_reply(
            "ble.dongle.reset.result",
            {"dongle_mac": mac, "reset": True},
            source="hub-ble-mgr",
        )

    server = IpcServer(f"localhost:{_BLE_MGR_PORT}", module_id="hub-ble-mgr")
    server.register("ble.dongle.reset.request", _reset_handler)
    await server.start()

    bridge = _make_bridge()
    try:
        client = _make_client(_BLE_MGR_PORT)
        bridge._clients["hub-ble-mgr"] = client
        await client.connect()

        result = await bridge.request_dongle_reset("AA:BB:CC:DD:EE:01")
        assert result.get("reset") is True
        assert result.get("dongle_mac") == "AA:BB:CC:DD:EE:01"
    finally:
        await bridge.stop()
        await server.stop()
