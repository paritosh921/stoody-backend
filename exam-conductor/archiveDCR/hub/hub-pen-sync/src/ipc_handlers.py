"""IPC message handlers for pen.sync.* messages.

Maps incoming IPC requests to the sync orchestrator, then returns
well-formed reply envelopes.

Message catalog (ipc-protocol.md Section 3.4):
  pen.sync.request      -> start sync for a specific pen
  pen.sync.abort.request -> abort in-progress sync
  supervisor.health.request -> report sync module status
"""

from __future__ import annotations

import logging

from hub_common.ipc_protocol import IpcEnvelope

from src.sync_orchestrator import SyncOrchestrator

logger = logging.getLogger(__name__)

MODULE_ID = "hub-pen-sync"

# Health request message type (not in hub-common yet — defined locally)
HEALTH_REQUEST = "supervisor.health.request"


class PenSyncHandlers:
    """IPC handler registry for hub-pen-sync."""

    def __init__(self, orchestrator: SyncOrchestrator) -> None:
        self._orch = orchestrator

    # --------------------------------------------------------------- sync

    async def handle_sync_request(self, env: IpcEnvelope) -> IpcEnvelope:
        """Handle ``pen.sync.request``: start sync for one pen.

        Payload: {exam_id, pen_mac, dongle_mac}
        This is a long-running operation — the reply confirms the request
        was accepted. Progress and completion come as events.
        """
        p = env.payload
        exam_id: str = p.get("exam_id", "")
        pen_mac: str = p.get("pen_mac", "")
        dongle_mac: str = p.get("dongle_mac", "")

        if not exam_id or not pen_mac:
            return env.make_error(
                "validation_failed",
                "Missing exam_id or pen_mac",
                source=MODULE_ID,
            )

        # Check if already syncing this pen
        existing = self._orch.get_state(pen_mac)
        if existing and not existing.is_terminal:
            return env.make_error(
                "busy",
                f"Sync already in progress for {pen_mac}",
                source=MODULE_ID,
            )

        # Launch sync as background task — don't block the IPC handler
        import asyncio

        asyncio.create_task(
            self._run_sync(exam_id, pen_mac, dongle_mac)
        )

        return env.make_reply(
            "pen.sync.accepted",
            {
                "exam_id": exam_id,
                "pen_mac": pen_mac,
                "status": "accepted",
            },
            source=MODULE_ID,
        )

    async def _run_sync(
        self, exam_id: str, pen_mac: str, dongle_mac: str
    ) -> None:
        """Background task to run the full sync flow."""
        try:
            await self._orch.sync_pen(exam_id, pen_mac, dongle_mac)
        except Exception:
            logger.exception("Sync failed for pen %s", pen_mac)

    # --------------------------------------------------------------- abort

    async def handle_abort_request(self, env: IpcEnvelope) -> IpcEnvelope:
        """Handle ``pen.sync.abort.request``: cancel in-progress sync."""
        p = env.payload
        pen_mac: str = p.get("pen_mac", "")
        reason: str = p.get("reason", "supervisor request")

        if not pen_mac:
            return env.make_error(
                "validation_failed",
                "Missing pen_mac",
                source=MODULE_ID,
            )

        await self._orch.abort_pen(pen_mac, reason)

        return env.make_reply(
            "pen.sync.abort.result",
            {"pen_mac": pen_mac, "aborted": True},
            source=MODULE_ID,
        )

    # --------------------------------------------------------------- health

    async def handle_health_request(self, env: IpcEnvelope) -> IpcEnvelope:
        """Handle ``supervisor.health.request``: report sync status."""
        active = {}
        for mac, state in self._orch._active_syncs.items():
            active[mac] = {
                "exam_id": state.exam_id,
                "status": state.status.name.lower(),
                "progress_pct": round(state.progress_pct, 1),
                "chunks_received": state.chunks_received,
                "total_chunks": state.total_chunks,
                "retries_remaining": state.retries_remaining,
            }

        return env.make_reply(
            "supervisor.health.result",
            {
                "module": MODULE_ID,
                "active_syncs": active,
                "sync_count": len(active),
            },
            source=MODULE_ID,
        )
