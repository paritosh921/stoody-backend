"""IPC message handlers for uplink.* messages.

Maps incoming requests to upload manager, connectivity probes, and
ledger queries.  Publishes progress/complete/error events to
hub-supervisor, hub-tui, and hub-invig-ble via the server broadcast.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from hub_common.ipc_protocol import IpcEnvelope
from hub_common.message_types import (
    SUPERVISOR_HEALTH_RESULT,
    UPLINK_UPLOAD_COMPLETE_EVENT,
    UPLINK_UPLOAD_PROGRESS_EVENT,
    UPLINK_STATUS_RESULT,
)

from src.config import UplinkConfig
from src.connectivity import check_backend_reachable, check_wifi_status
from src.ledger import UploadLedger
from src.path_selector import select_path
from src.upload_manager import PenUploadSpec, UploadManager

logger = logging.getLogger(__name__)

MODULE_ID = "hub-uplink"

# Targets that receive progress and completion events.
EVENT_TARGETS = ("hub-supervisor", "hub-tui", "hub-invig-ble")

# Type alias for the IPC server broadcast function.
BroadcastFn = Callable[[IpcEnvelope], Awaitable[None]]


class UplinkHandlers:
    """Stateful handler registry for hub-uplink IPC messages."""

    def __init__(
        self,
        config: UplinkConfig,
        ledger: UploadLedger,
        *,
        broadcast_fn: BroadcastFn,
    ) -> None:
        self._cfg = config
        self._ledger = ledger
        self._broadcast = broadcast_fn
        self._mgr: UploadManager | None = None
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}

    def set_upload_manager(self, mgr: UploadManager) -> None:
        """Inject the upload manager after construction (breaks cycle)."""
        self._mgr = mgr

    # -- progress callback for UploadManager --------------------------------

    async def emit_progress(
        self,
        exam_id: str,
        pen_mac: str,
        chunk_index: int,
        acked_count: int,
        total_chunks: int,
        upload_path: str,
    ) -> None:
        """Broadcast an ``uplink.upload.progress.event`` to all targets."""
        payload = {
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "chunk_index": chunk_index,
            "acked_count": acked_count,
            "total_chunks": total_chunks,
            "upload_path": upload_path,
        }
        for target in EVENT_TARGETS:
            env = _make_event(UPLINK_UPLOAD_PROGRESS_EVENT, target, payload)
            await self._broadcast(env)

    # -- uplink.upload.request ----------------------------------------------

    async def handle_upload_request(self, env: IpcEnvelope) -> IpcEnvelope:
        """Start upload for an exam -- runs in a background task.

        Expects payload: ``{exam_id, path: "wifi"|"mobile"|"auto"}``.
        """
        assert self._mgr is not None, "upload manager not wired"
        p = env.payload
        exam_id: str = p["exam_id"]
        requested_path: str = p.get("path", "auto")

        # If an upload for this exam is already running, ack immediately.
        if exam_id in self._active_tasks and not self._active_tasks[exam_id].done():
            return env.make_reply(
                UPLINK_UPLOAD_PROGRESS_EVENT,
                {"exam_id": exam_id, "message": "upload already in progress"},
                source=MODULE_ID,
            )

        task = asyncio.create_task(
            self._run_exam_upload(exam_id, requested_path),
        )
        self._active_tasks[exam_id] = task

        return env.make_reply(
            UPLINK_UPLOAD_PROGRESS_EVENT,
            {"exam_id": exam_id, "message": "upload started"},
            source=MODULE_ID,
        )

    # -- uplink.status.request ----------------------------------------------

    async def handle_status_request(self, env: IpcEnvelope) -> IpcEnvelope:
        """Report upload progress for a given exam."""
        exam_id: str = env.payload["exam_id"]
        status = self._ledger.get_upload_status(exam_id)
        return env.make_reply(
            UPLINK_STATUS_RESULT,
            status,
            source=MODULE_ID,
        )

    # -- supervisor.health.request ------------------------------------------

    async def handle_health_request(self, env: IpcEnvelope) -> IpcEnvelope:
        """Report module health: connectivity + upload status."""
        wifi = await check_wifi_status()
        backend_ok = await check_backend_reachable(
            self._cfg.backend_url, self._cfg.health_endpoint,
        )
        return env.make_reply(
            SUPERVISOR_HEALTH_RESULT,
            {
                "module": MODULE_ID,
                "healthy": wifi.connected or backend_ok,
                "detail": {
                    "wifi_connected": wifi.connected,
                    "wifi_ssid": wifi.ssid,
                    "wifi_signal_dbm": wifi.signal_dbm,
                    "wifi_band": wifi.band,
                    "backend_reachable": backend_ok,
                    "active_uploads": len(
                        [t for t in self._active_tasks.values() if not t.done()],
                    ),
                },
            },
            source=MODULE_ID,
        )

    # -- background upload orchestration ------------------------------------

    async def _run_exam_upload(
        self, exam_id: str, requested_path: str,
    ) -> None:
        """Upload all synced pens for *exam_id* (runs as a Task)."""
        assert self._mgr is not None
        try:
            wifi = await check_wifi_status()
            backend_ok = await check_backend_reachable(
                self._cfg.backend_url, self._cfg.health_endpoint,
            )
            decision = select_path(
                wifi_available=wifi.connected,
                backend_reachable=backend_ok,
                mobile_connected=(requested_path == "mobile"),
            )
            upload_path = (
                requested_path
                if requested_path in ("wifi", "mobile")
                else decision.path.value
            )

            # Fetch pen list from ledger (initialised by pen-sync complete).
            status = self._ledger.get_upload_status(exam_id)
            for pen in status.get("pens", []):
                if pen["complete"]:
                    continue
                spec = PenUploadSpec(
                    exam_id=exam_id,
                    pen_mac=pen["pen_mac"],
                    total_chunks=pen["total_chunks"],
                    upload_path=upload_path,
                )
                await self._mgr.upload_pen(spec)

                # Broadcast completion event for this pen.
                await self._broadcast_pen_complete(exam_id, pen["pen_mac"])

            logger.info("Exam %s upload complete", exam_id)

        except Exception:
            logger.exception("Exam %s upload failed", exam_id)
        finally:
            self._active_tasks.pop(exam_id, None)

    async def _broadcast_pen_complete(
        self, exam_id: str, pen_mac: str,
    ) -> None:
        """Send ``uplink.upload.complete.event`` to all targets."""
        payload = {"exam_id": exam_id, "pen_mac": pen_mac, "complete": True}
        for target in EVENT_TARGETS:
            env = _make_event(UPLINK_UPLOAD_COMPLETE_EVENT, target, payload)
            await self._broadcast(env)


def _make_event(
    msg_type: str, target: str, payload: dict[str, Any],
) -> IpcEnvelope:
    """Build a fire-and-forget event envelope."""
    return IpcEnvelope(
        msg_type=msg_type,
        source=MODULE_ID,
        target=target,
        expects_reply=False,
        payload=payload,
    )
