"""hub-invig-ble module entry point.

Responsibilities:
  1. Initialise auth handler with SQLite code store.
  2. Start IPC server for incoming messages from other hub modules.
  3. Connect IPC client to hub-supervisor for snapshot requests.
  4. Start BLE peripheral (GATT server) for invigilator mobile app.
  5. Run 1 Hz status feed loop (requests supervisor snapshot each tick).
  6. Clean shutdown on cancellation.
"""

from __future__ import annotations

import asyncio
import logging

from src.auth_handler import AuthHandler, AuthResult, CodeStore
from src.command_handler import CommandHandler, CommandResult
from src.config import (
    IPC_SOCKET_PATH,
    MODULE_ID,
    STATUS_FEED_INTERVAL_SEC,
    SUPERVISOR_SOCKET_PATH,
)
from src.ipc_client import IpcClient
from src.ipc_handlers import (
    MSG_FSM_SNAPSHOT_REQUEST,
    MSG_INVIG_AUTH_STATE_EVENT,
    MSG_INVIG_COMMAND_EVENT,
    MSG_SUPERVISOR_HEALTH_REQUEST,
    Envelope,
    InvigIpcHandlers,
)
from src.ipc_server import InvigIpcServer
from src.peripheral import BlePeripheralBackend, InvigilatorPeripheral
from src.status_feed import StatusFeedCollector

log = logging.getLogger(MODULE_ID)


async def _request_supervisor_snapshot(
    client: IpcClient,
    status_feed: StatusFeedCollector,
) -> None:
    """Ask supervisor for an FSM snapshot and update the status feed.

    Graceful degradation: if supervisor is unreachable the status feed
    keeps the last known snapshot (or the default idle snapshot).
    """
    env = Envelope(
        msg_type=MSG_FSM_SNAPSHOT_REQUEST,
        source=MODULE_ID,
        target="hub-supervisor",
        expects_reply=True,
        payload={},
    )
    reply = await client.request(env, timeout_sec=1.0)
    if reply is not None and reply.payload:
        status_feed.update_from_ipc(reply.payload)


async def run(
    backend: BlePeripheralBackend | None = None,
    code_store: CodeStore | None = None,
    ipc_server: InvigIpcServer | None = None,
    supervisor_client: IpcClient | None = None,
) -> None:
    """Main loop -- called by ``__main__`` or by tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # --- Components --------------------------------------------------------
    store = code_store or CodeStore()
    if code_store is None:
        store.open()

    auth = AuthHandler(store)
    cmd = CommandHandler(auth)
    status_feed = StatusFeedCollector()

    # --- IPC server --------------------------------------------------------
    server = ipc_server or InvigIpcServer(
        IPC_SOCKET_PATH, module_id=MODULE_ID,
    )

    # --- IPC client to supervisor ------------------------------------------
    sv_client = supervisor_client or IpcClient(SUPERVISOR_SOCKET_PATH)

    ipc = InvigIpcHandlers(
        broadcast_fn=server.broadcast,
        supervisor_send_fn=sv_client.send,
    )

    # --- Register inbound IPC handlers ------------------------------------
    server.register(MSG_SUPERVISOR_HEALTH_REQUEST, ipc.handle_health)
    server.register(MSG_INVIG_AUTH_STATE_EVENT, ipc.handle_auth_state_event)
    server.register(MSG_INVIG_COMMAND_EVENT, ipc.handle_command_event)

    await server.start()
    log.info("IPC server started on %s", server.address)

    # --- Peripheral callbacks ---------------------------------------------
    class _Callbacks:
        async def on_auth_result(self, result: AuthResult) -> None:
            await ipc.publish_auth_state(
                invig_id=result.ble_addr,
                connected=True,
                authenticated=result.success,
            )

        async def on_command(self, result: CommandResult) -> None:
            if result.accepted and result.cmd_name:
                await ipc.publish_command(
                    cmd_name=result.cmd_name,
                    cmd_id=result.cmd_id,
                    request_id=result.request_id,
                    payload=result.payload,
                )

    if backend is None:
        log.warning("No BLE backend -- running IPC-only (no BLE peripheral)")
        # Still run the IPC server so other modules can reach us.
        try:
            while True:
                await asyncio.sleep(STATUS_FEED_INTERVAL_SEC)
                await _request_supervisor_snapshot(sv_client, status_feed)
        except asyncio.CancelledError:
            pass
        finally:
            await server.stop()
            await sv_client.close()
            store.close()
            log.info("hub-invig-ble stopped (IPC-only mode)")
        return

    peripheral = InvigilatorPeripheral(
        backend=backend,
        auth_handler=auth,
        command_handler=cmd,
        status_feed=status_feed,
        callbacks=_Callbacks(),
    )

    await peripheral.start()
    log.info("hub-invig-ble started on %s", server.address)

    # --- 1 Hz status feed loop --------------------------------------------
    try:
        while True:
            await asyncio.sleep(STATUS_FEED_INTERVAL_SEC)
            await _request_supervisor_snapshot(sv_client, status_feed)
            await peripheral.push_status()
    except asyncio.CancelledError:
        log.info("Shutting down hub-invig-ble")
    finally:
        await peripheral.stop()
        await server.stop()
        await sv_client.close()
        store.close()
        log.info("hub-invig-ble stopped")


def main() -> None:
    """CLI entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
