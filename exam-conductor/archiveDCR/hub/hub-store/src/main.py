"""hub-store module entry point.

Registers IPC handlers for ``store.write.request`` and
``store.read.request``, then runs the async event loop.

On startup the SQLite ledger (WAL mode) is opened and the
``pen_sync_status`` / ``upload_ledger`` tables are created if missing.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from hub_common.config import load_hub_config
from hub_common.ipc_protocol import IpcServer
from hub_common.message_types import STORE_READ_REQUEST, STORE_WRITE_REQUEST

from src.config import StoreConfig
from src.dual_writer import DualWriter
from src.ipc_handlers import StoreHandlers
from src.ledger import ChunkLedger, open_ledger_db

MODULE_ID = "hub-store"
logger = logging.getLogger(MODULE_ID)


def _build_config() -> StoreConfig:
    hub_cfg = load_hub_config()
    from pathlib import Path

    return StoreConfig(
        sd_base=Path(hub_cfg.sd_data_path),
        usb_base=Path(hub_cfg.usb_data_path),
    )


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = _build_config()

    # -- SQLite ledger (WAL) ------------------------------------------
    db_conn = open_ledger_db(cfg.db_path)
    ledger = ChunkLedger(db_conn)
    logger.info("SQLite ledger initialised at %s", cfg.db_path)

    writer = DualWriter(cfg, ledger=ledger)
    handlers = StoreHandlers(writer, cfg)

    hub_cfg = load_hub_config()
    socket_path = hub_cfg.socket_path(MODULE_ID)

    server = IpcServer(socket_path, module_id=MODULE_ID)
    server.register(STORE_WRITE_REQUEST, handlers.handle_write_request)
    server.register(STORE_READ_REQUEST, handlers.handle_read_request)

    logger.info("hub-store starting on %s", socket_path)
    await server.serve_forever()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("hub-store shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
