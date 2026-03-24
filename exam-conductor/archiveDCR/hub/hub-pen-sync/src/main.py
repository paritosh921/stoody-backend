"""hub-pen-sync module entry point.

Registers IPC handlers for ``pen.sync.request``,
``pen.sync.abort.request``, and ``supervisor.health.request``,
then runs the async event loop.

This module reads pen stroke buffers via BLE GATT and passes chunks
to hub-store for dual-write. It NEVER writes to local storage directly
(STATE_OWNERSHIP_MAP.md: hub-store owns the write path).
"""

from __future__ import annotations

import asyncio
import logging
import sys

from hub_common.config import load_hub_config
from hub_common.ipc_protocol import IpcClient, IpcEnvelope, IpcServer
from hub_common.message_types import PEN_SYNC_ABORT_REQUEST, PEN_SYNC_REQUEST

from src.config import PenSyncConfig
from src.ipc_handlers import HEALTH_REQUEST, PenSyncHandlers
from src.sync_orchestrator import SyncOrchestrator

MODULE_ID = "hub-pen-sync"
logger = logging.getLogger(MODULE_ID)


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    hub_cfg = load_hub_config()
    sync_cfg = PenSyncConfig()

    # IPC client to hub-store for dual-write requests
    store_client = IpcClient(
        hub_cfg.socket_path("hub-store"), source_id=MODULE_ID
    )
    await store_client.connect()

    # BLE client factory — uses bleak on real hardware
    from src._ble_factory import BleakClientFactory

    ble_factory = BleakClientFactory()

    # Event publisher — fire-and-forget IPC sends
    async def publish_event(env: IpcEnvelope) -> None:
        # For events, we connect-send-close per target
        # (lightweight for infrequent progress events)
        try:
            target_path = hub_cfg.socket_path(env.target)
            client = IpcClient(target_path, source_id=MODULE_ID)
            await client.connect()
            await client.send(env)
            await client.close()
        except Exception:
            logger.debug("Event publish failed to %s", env.target)

    orchestrator = SyncOrchestrator(
        config=sync_cfg,
        store_client=store_client,
        ble_factory=ble_factory,
        event_publisher=publish_event,
    )

    handlers = PenSyncHandlers(orchestrator)

    socket_path = hub_cfg.socket_path(MODULE_ID)
    server = IpcServer(socket_path, module_id=MODULE_ID)
    server.register(PEN_SYNC_REQUEST, handlers.handle_sync_request)
    server.register(PEN_SYNC_ABORT_REQUEST, handlers.handle_abort_request)
    server.register(HEALTH_REQUEST, handlers.handle_health_request)

    logger.info("hub-pen-sync starting on %s", socket_path)
    await server.serve_forever()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("hub-pen-sync shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
