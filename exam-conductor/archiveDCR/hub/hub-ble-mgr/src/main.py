"""hub-ble-mgr module entry point.

Responsibilities:
  1. Enumerate BLE dongles on startup via the real BlueZ adapter.
  2. Start IPC server and register message handlers.
  3. Wire broadcast callback so all outbound events reach IPC clients.
  4. Start the health monitor background task.
  5. Run the async event loop.
  6. Clean shutdown on cancellation.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from src.ble_adapter import (
    BleakConnectorAdapter,
    BleakScannerAdapter,
    BlueZAdapter,
    HciHealthProbe,
)
from src.config import HEALTH_CHECK_INTERVAL_SEC, IPC_SOCKET_PATH, MODULE_ID
from src.connection_manager import ConnectionManager
from src.dongle_manager import DongleManager
from src.health_monitor import HealthMonitor
from src.ipc_handlers import (
    MSG_SUPERVISOR_HEALTH_REQUEST,
    BleIpcHandlers,
)
from src.ipc_server import BleIpcServer
from src.pen_discovery import PenDiscovery

# Message type constants -- prefer hub-common, fallback inline.
try:
    from hub_common.message_types import (
        BLE_CONNECT_REQUEST,
        BLE_SCAN_START_REQUEST,
        BLE_SCAN_STOP_REQUEST,
    )
except ImportError:  # pragma: no cover
    BLE_SCAN_START_REQUEST = "ble.scan.start.request"
    BLE_SCAN_STOP_REQUEST = "ble.scan.stop.request"
    BLE_CONNECT_REQUEST = "ble.connect.request"

logger = logging.getLogger(MODULE_ID)


async def run(
    *,
    ipc_server: BleIpcServer | None = None,
    adapter: BlueZAdapter | None = None,
) -> None:
    """Main async entry point -- called by ``__main__`` or tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # --- BLE adapter layer ------------------------------------------------
    ble_adapter = adapter or BlueZAdapter()
    scanner_adapter = BleakScannerAdapter()
    connector_adapter = BleakConnectorAdapter()
    health_probe = HciHealthProbe()

    # --- Domain components ------------------------------------------------
    dongle_mgr = DongleManager()

    discovery = PenDiscovery(dongle_mgr, scanner=scanner_adapter)
    conn_mgr = ConnectionManager(dongle_mgr, connector=connector_adapter)
    health_mon = HealthMonitor(
        dongle_mgr,
        probe=health_probe,
        interval_sec=HEALTH_CHECK_INTERVAL_SEC,
    )

    # --- IPC server -------------------------------------------------------
    server = ipc_server or BleIpcServer(
        IPC_SOCKET_PATH, module_id=MODULE_ID,
    )

    # --- Wire handlers with broadcast callback ----------------------------
    handlers = BleIpcHandlers(
        dongle_mgr, discovery, conn_mgr,
        broadcast_fn=server.broadcast,
    )

    # Wire domain callbacks -> IPC event emission.
    discovery._on_pen_discovered = handlers.emit_scan_result
    conn_mgr._on_connected = handlers.emit_pen_connected
    conn_mgr._on_disconnected = handlers.emit_pen_disconnected
    health_mon._on_health_change = handlers.emit_dongle_health

    # --- Register IPC message handlers ------------------------------------
    server.register(BLE_SCAN_START_REQUEST, handlers.handle_scan_start)
    server.register(BLE_SCAN_STOP_REQUEST, handlers.handle_scan_stop)
    server.register(BLE_CONNECT_REQUEST, handlers.handle_connect)
    server.register(MSG_SUPERVISOR_HEALTH_REQUEST, handlers.handle_health)

    # --- Start IPC server -------------------------------------------------
    await server.start()
    logger.info("IPC server started on %s", server.address)

    # --- Enumerate dongles on startup -------------------------------------
    dongles = await dongle_mgr.refresh(ble_adapter)
    logger.info(
        "Startup enumeration complete: %d dongle(s) found", len(dongles),
    )

    # --- Start health monitor background task -----------------------------
    health_mon.start()
    logger.info("Health monitor started (interval=%.1fs)", HEALTH_CHECK_INTERVAL_SEC)

    logger.info("hub-ble-mgr ready on %s", server.address)

    # --- Event loop: block until cancelled --------------------------------
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass

    # --- Shutdown ---------------------------------------------------------
    health_mon.stop()
    await conn_mgr.disconnect_all()
    await server.stop()
    logger.info("hub-ble-mgr stopped")


def main() -> None:
    """CLI entry point."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("hub-ble-mgr shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
