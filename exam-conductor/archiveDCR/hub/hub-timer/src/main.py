"""hub-timer module entry point.

Responsibilities:
  1. Open SQLite, check for a persisted timer (reboot recovery).
  2. Start an IPC server (Unix socket on Linux, TCP on Windows).
  3. Register IPC handlers for timer.arm, timer.cancel, snapshot, health.
  4. Run the async event loop: tick broadcasts (1 s), persist (10 s), expiry check.
  5. Clean shutdown on cancellation.
"""

from __future__ import annotations

import asyncio
import logging

from src.config import (
    IPC_SOCKET_PATH,
    MODULE_ID,
    PERSIST_INTERVAL_SEC,
    TICK_BROADCAST_INTERVAL_SEC,
)
from src.countdown import CountdownTimer
from src.ipc_handlers import (
    MSG_SUPERVISOR_HEALTH_REQUEST,
    MSG_TIMER_ARM_REQUEST,
    MSG_TIMER_CANCEL_REQUEST,
    MSG_TIMER_SNAPSHOT_REQUEST,
    TimerIpcHandlers,
)
from src.ipc_server import TimerIpcServer
from src.persistence import TimerPersistence

log = logging.getLogger(MODULE_ID)


async def run(
    persistence: TimerPersistence | None = None,
    timer: CountdownTimer | None = None,
    ipc_server: TimerIpcServer | None = None,
) -> None:
    """Main loop -- called by ``__main__`` or by tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # --- Initialise components --------------------------------------------
    persist = persistence or TimerPersistence()
    persist.open()

    tmr = timer or CountdownTimer()
    server = ipc_server or TimerIpcServer(
        IPC_SOCKET_PATH, module_id=MODULE_ID,
    )
    expired_fired: set[str] = set()

    # --- Callbacks for IPC handlers ---------------------------------------

    async def on_arm(exam_id: str, duration_sec: int) -> None:
        tmr.arm(exam_id, duration_sec)
        state = tmr.get_state()
        assert state is not None
        persist.persist_state(
            exam_id, state.started_at_epoch, duration_sec, state.remaining_sec,
        )
        expired_fired.discard(exam_id)
        log.info("Timer armed: exam=%s duration=%ds", exam_id, duration_sec)

    async def on_cancel(exam_id: str) -> None:
        if tmr.cancel(exam_id):
            persist.clear_state(exam_id)
            expired_fired.discard(exam_id)
            log.info("Timer cancelled: exam=%s", exam_id)

    handlers = TimerIpcHandlers(
        tmr, server.broadcast, on_arm=on_arm, on_cancel=on_cancel,
    )

    # --- Register IPC message handlers ------------------------------------
    server.register(MSG_TIMER_ARM_REQUEST, handlers.handle_arm)
    server.register(MSG_TIMER_CANCEL_REQUEST, handlers.handle_cancel)
    server.register(MSG_TIMER_SNAPSHOT_REQUEST, handlers.handle_snapshot)
    server.register(MSG_SUPERVISOR_HEALTH_REQUEST, handlers.handle_health)

    # --- Start IPC server -------------------------------------------------
    await server.start()
    log.info("IPC server started on %s", server.address)

    # --- Boot recovery ----------------------------------------------------
    saved = persist.load_state()
    if saved is not None:
        log.info(
            "Recovering timer: exam=%s remaining=%ds last_updated=%d",
            saved.exam_id,
            saved.remaining_sec,
            saved.last_updated,
        )
        tmr.arm(
            saved.exam_id,
            saved.duration_sec,
            resume_remaining=saved.remaining_sec,
            resume_epoch=saved.last_updated,
        )
        if tmr.is_expired():
            await handlers.publish_expired(saved.exam_id)
            persist.clear_state(saved.exam_id)
            expired_fired.add(saved.exam_id)
            log.info("Recovered timer already expired: exam=%s", saved.exam_id)

    # --- Event loop -------------------------------------------------------
    persist_counter = 0.0
    try:
        while True:
            await asyncio.sleep(TICK_BROADCAST_INTERVAL_SEC)

            # Tick broadcast
            if tmr.active:
                await handlers.publish_tick()

            # Expiry check
            if (
                tmr.is_expired()
                and tmr.exam_id
                and tmr.exam_id not in expired_fired
            ):
                exam_id = tmr.exam_id
                await handlers.publish_expired(exam_id)
                persist.clear_state(exam_id)
                expired_fired.add(exam_id)
                log.info("Timer expired: exam=%s", exam_id)

            # Periodic persistence
            persist_counter += TICK_BROADCAST_INTERVAL_SEC
            if persist_counter >= PERSIST_INTERVAL_SEC and tmr.active:
                persist_counter = 0.0
                state = tmr.get_state()
                if state is not None:
                    persist.persist_state(
                        state.exam_id,
                        state.started_at_epoch,
                        state.total_sec,
                        state.remaining_sec,
                    )
    except asyncio.CancelledError:
        log.info("Shutting down hub-timer")
    finally:
        await server.stop()
        persist.close()
        log.info("hub-timer stopped")


def main() -> None:
    """CLI entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
