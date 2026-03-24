"""hub-supervisor entry point.

Responsibilities:
  1. Load config, open SQLite (WAL mode).
  2. Detect first-boot (missing ``/etc/exampen/hub.conf``).
  3. Start IPC server on the supervisor socket.
  4. Spawn child modules via ProcessManager.
  5. Start FSM orchestration loop.
  6. Clean shutdown on cancellation.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Any

from hub_common.ipc_protocol import IpcServer
from hub_common.message_types import FSM_SNAPSHOT_REQUEST, FSM_TRANSITION_REQUEST

from src.config import MODULE_ID, SQLITE_DB_PATH, SUPERVISOR_SOCKET
from src.first_boot import is_first_boot
from src.interaction_log import InteractionLog, LogEntry
from src.ipc_handlers import SUPERVISOR_SHUTDOWN, SupervisorIpcHandlers
from src.orchestrator import Orchestrator
from src.process_manager import ModuleInfo, ProcessManager

logger = logging.getLogger(MODULE_ID)


def _open_db(db_path: str | None = None) -> sqlite3.Connection:
    """Open SQLite with WAL mode and create tables if needed."""
    path = db_path or str(SQLITE_DB_PATH)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


async def run(
    *,
    db_path: str | None = None,
    socket_path: str | None = None,
    skip_spawn: bool = False,
    ipc_send_fn: Any | None = None,
    config_path: str | None = None,
) -> None:
    """Main supervisor loop.

    Parameters are overridable for testing:

    - *db_path*: SQLite path (default: production path from config).
    - *socket_path*: Supervisor IPC socket (``host:port`` on Windows).
    - *skip_spawn*: Skip spawning child processes.
    - *ipc_send_fn*: Injected IPC send function for the orchestrator
      (e.g. route messages to mock child TCP servers in tests).
    - *config_path*: Override ``hub.conf`` path for first-boot detection.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    db_conn = _open_db(db_path)
    ilog = InteractionLog()
    ilog.open(conn=db_conn)
    orchestrator = Orchestrator(ilog, ipc_send_fn=ipc_send_fn)

    first_boot = is_first_boot(config_path)
    ilog.append(LogEntry(
        source=MODULE_ID,
        event_type="hub_boot",
        detail={"first_boot": first_boot},
    ))

    if first_boot:
        logger.info("First boot detected — awaiting TUI provisioning")

    # -- IPC server --------------------------------------------------------
    sock = socket_path or SUPERVISOR_SOCKET
    server = IpcServer(sock, module_id=MODULE_ID)

    # Build process manager (empty list for now; children populated below)
    from src.config import CHILD_COMMANDS, CHILD_SOCKETS, OPTIONAL_MODULES
    modules = [
        ModuleInfo(
            name=name,
            socket_path=CHILD_SOCKETS[name],
            command=CHILD_COMMANDS[name],
            optional=name in OPTIONAL_MODULES,
        )
        for name in CHILD_COMMANDS
    ]
    pm = ProcessManager(modules)

    shutdown_event = asyncio.Event()

    async def do_shutdown() -> None:
        shutdown_event.set()

    handlers = SupervisorIpcHandlers(
        db_conn,
        orchestrator,
        ilog,
        shutdown_fn=do_shutdown,
        get_module_health=pm.get_status_summary,
    )
    server.register(FSM_TRANSITION_REQUEST, handlers.handle_transition)
    server.register(FSM_SNAPSHOT_REQUEST, handlers.handle_snapshot)
    server.register(SUPERVISOR_SHUTDOWN, handlers.handle_shutdown)

    await server.start()
    logger.info("Supervisor IPC listening on %s", sock)

    # -- Spawn children ----------------------------------------------------
    if not skip_spawn:
        await pm.spawn_all()
        pm.start_watchdog()

    # -- Wait for shutdown -------------------------------------------------
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Supervisor shutting down")
        await pm.stop_all()
        await server.stop()
        ilog.close()
        logger.info("Supervisor stopped")


def main() -> None:
    """CLI entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
