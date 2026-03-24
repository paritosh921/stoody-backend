"""hub-uplink module entry point.

Registers IPC handlers for ``uplink.upload.request``,
``uplink.status.request``, and ``supervisor.health.request``,
then runs the async event loop.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from hub_common.config import load_hub_config
from hub_common.ipc_protocol import IpcClient, IpcServer
from hub_common.message_types import (
    SUPERVISOR_HEALTH_REQUEST,
    UPLINK_STATUS_REQUEST,
    UPLINK_UPLOAD_REQUEST,
)

from src.config import load_uplink_config
from src.ipc_handlers import UplinkHandlers
from src.ledger import UploadLedger, ensure_ledger_table
from src.upload_manager import UploadManager

MODULE_ID = "hub-uplink"
logger = logging.getLogger(MODULE_ID)

# Retry parameters for connecting to hub-store.
_STORE_CONNECT_MAX_RETRIES = 5
_STORE_CONNECT_BACKOFF_SEC = 2.0


async def _connect_store_client(client: IpcClient) -> None:
    """Connect *client* to hub-store with exponential backoff.

    Retries up to ``_STORE_CONNECT_MAX_RETRIES`` times.  If all
    attempts fail the last exception propagates so the caller can
    decide whether to abort or continue without store connectivity.
    """
    for attempt in range(1, _STORE_CONNECT_MAX_RETRIES + 1):
        try:
            await client.connect()
            logger.info("Connected to hub-store IPC socket")
            return
        except (OSError, ConnectionRefusedError) as exc:
            delay = _STORE_CONNECT_BACKOFF_SEC * attempt
            logger.warning(
                "hub-store connect attempt %d/%d failed (%s) — "
                "retrying in %.1fs",
                attempt, _STORE_CONNECT_MAX_RETRIES, exc, delay,
            )
            if attempt == _STORE_CONNECT_MAX_RETRIES:
                raise
            await asyncio.sleep(delay)


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    hub_cfg = load_hub_config()
    uplink_cfg = load_uplink_config(hub_cfg.backend_url)

    # -- SQLite ledger (WAL) -----------------------------------------------
    import sqlite3

    db_path = Path(hub_cfg.sd_data_path) / "hub.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    ensure_ledger_table(conn)
    ledger = UploadLedger(conn)
    logger.info("Upload ledger ready at %s", db_path)

    # -- IPC client to hub-store -------------------------------------------
    store_socket = hub_cfg.socket_path("hub-store")
    store_client = IpcClient(store_socket, source_id=MODULE_ID)
    await _connect_store_client(store_client)

    # -- IPC server --------------------------------------------------------
    socket_path = hub_cfg.socket_path(MODULE_ID)
    server = IpcServer(socket_path, module_id=MODULE_ID)

    # -- Upload manager (with progress callback) ---------------------------
    handlers = UplinkHandlers(uplink_cfg, ledger, broadcast_fn=server.broadcast)
    mgr = UploadManager(
        uplink_cfg, ledger, store_client,
        progress_callback=handlers.emit_progress,
    )
    handlers.set_upload_manager(mgr)

    # -- Register IPC handlers ---------------------------------------------
    server.register(UPLINK_UPLOAD_REQUEST, handlers.handle_upload_request)
    server.register(UPLINK_STATUS_REQUEST, handlers.handle_status_request)
    server.register(SUPERVISOR_HEALTH_REQUEST, handlers.handle_health_request)

    logger.info("hub-uplink starting on %s", socket_path)
    try:
        await server.serve_forever()
    finally:
        await store_client.close()
        logger.info("hub-store IPC client closed")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("hub-uplink shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
