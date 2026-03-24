"""FSM side-effect executor.

On each FSM transition the orchestrator executes the appropriate IPC
commands to child modules.  The FSM state is already persisted to SQLite
BEFORE this module runs (crash-safe by design).

Side effects are best-effort: if a child module is unreachable the
error is logged and the interaction_log records the failure, but the
FSM state is NOT rolled back (per STATE_OWNERSHIP_MAP.md Section 3.1).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from hub_common.ipc_protocol import IpcClient, IpcEnvelope
from hub_common.message_types import (
    BLE_SCAN_START_REQUEST,
    BLE_SCAN_STOP_REQUEST,
    PEN_SYNC_REQUEST,
    TIMER_ARM_REQUEST,
    TIMER_CANCEL_REQUEST,
    UPLINK_UPLOAD_REQUEST,
)

from src.config import CHILD_SOCKETS, MODULE_ID
from src.hub_fsm import (
    ARMED,
    CANCELLED,
    DONGLE_ACTIVATION,
    PEN_SYNC,
    TIMER_RUNNING,
    UPLOADING,
)
from src.interaction_log import InteractionLog, LogEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Execute FSM transition side effects by sending IPC to children.

    Parameters
    ----------
    interaction_log:
        Forensic audit logger.
    ipc_send_fn:
        Optional override for sending IPC messages (for testing).
        Signature: ``async (target_module, envelope) -> reply_or_none``.
    """

    def __init__(
        self,
        interaction_log: InteractionLog,
        *,
        ipc_send_fn: Any | None = None,
    ) -> None:
        self._log = interaction_log
        self._ipc_send = ipc_send_fn or self._default_ipc_send

    # -- public API ---------------------------------------------------------

    async def on_transition(
        self,
        exam_id: str,
        new_state: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Execute side effects for a transition into *new_state*.

        *context* carries extra data such as ``duration_sec``,
        ``pen_macs``, ``invig_id``, etc.
        """
        ctx = context or {}
        handler = self._SIDE_EFFECTS.get(new_state)
        if handler is not None:
            try:
                await handler(self, exam_id, ctx)
            except Exception:
                logger.exception(
                    "Side-effect error for state %s exam %s",
                    new_state, exam_id,
                )
                self._log.append(LogEntry(
                    source=MODULE_ID,
                    event_type="side_effect_error",
                    exam_id=exam_id,
                    severity="error",
                    detail={"state": new_state},
                ))

    async def collect_status(self) -> dict[str, Any]:
        """Aggregate health/status from all child modules (placeholder)."""
        return {"modules": "placeholder"}

    # -- side-effect handlers -----------------------------------------------

    async def _on_armed(
        self, exam_id: str, ctx: dict[str, Any]
    ) -> None:
        """Validate prerequisites (WiFi, dongles, invigilator)."""
        self._log.append(LogEntry(
            source=MODULE_ID,
            event_type="exam_armed",
            exam_id=exam_id,
            detail={"prerequisites": "validated"},
        ))
        logger.info("Exam %s armed — prerequisites validated", exam_id)

    async def _on_timer_running(
        self, exam_id: str, ctx: dict[str, Any]
    ) -> None:
        """Send ``timer.arm`` IPC to hub-timer."""
        duration_sec = ctx.get("duration_sec", 0)
        armed_by = ctx.get("armed_by", "supervisor")
        env = IpcEnvelope(
            msg_type=TIMER_ARM_REQUEST,
            source=MODULE_ID,
            target="hub-timer",
            expects_reply=False,
            payload={
                "exam_id": exam_id,
                "duration_sec": duration_sec,
                "armed_by": armed_by,
            },
        )
        await self._ipc_send("hub-timer", env)
        self._log.append(LogEntry(
            source=MODULE_ID,
            event_type="exam_timer_start",
            exam_id=exam_id,
            detail={"duration_sec": duration_sec},
        ))

    async def _on_dongle_activation(
        self, exam_id: str, ctx: dict[str, Any]
    ) -> None:
        """Send ``ble.scan.start`` IPC to hub-ble-mgr."""
        env = IpcEnvelope(
            msg_type=BLE_SCAN_START_REQUEST,
            source=MODULE_ID,
            target="hub-ble-mgr",
            expects_reply=False,
            payload={
                "exam_id": exam_id,
                "mode": "sync",
                "timeout_sec": ctx.get("scan_timeout_sec", 300),
            },
        )
        await self._ipc_send("hub-ble-mgr", env)
        self._log.append(LogEntry(
            source=MODULE_ID,
            event_type="dongle_activated",
            exam_id=exam_id,
        ))

    async def _on_pen_sync(
        self, exam_id: str, ctx: dict[str, Any]
    ) -> None:
        """Send ``pen.sync.request`` for each discovered pen."""
        pen_macs: list[dict[str, str]] = ctx.get("pens", [])
        for pen in pen_macs:
            env = IpcEnvelope(
                msg_type=PEN_SYNC_REQUEST,
                source=MODULE_ID,
                target="hub-pen-sync",
                expects_reply=False,
                payload={
                    "exam_id": exam_id,
                    "pen_mac": pen.get("pen_mac", ""),
                    "dongle_mac": pen.get("dongle_mac", ""),
                },
            )
            await self._ipc_send("hub-pen-sync", env)
        self._log.append(LogEntry(
            source=MODULE_ID,
            event_type="pen_sync_started",
            exam_id=exam_id,
            detail={"pen_count": len(pen_macs)},
        ))

    async def _on_uploading(
        self, exam_id: str, ctx: dict[str, Any]
    ) -> None:
        """Send ``uplink.upload.request`` to hub-uplink."""
        env = IpcEnvelope(
            msg_type=UPLINK_UPLOAD_REQUEST,
            source=MODULE_ID,
            target="hub-uplink",
            expects_reply=False,
            payload={
                "exam_id": exam_id,
                "path": ctx.get("upload_path", "auto"),
            },
        )
        await self._ipc_send("hub-uplink", env)
        self._log.append(LogEntry(
            source=MODULE_ID,
            event_type="upload_start",
            exam_id=exam_id,
        ))

    async def _on_cancelled(
        self, exam_id: str, ctx: dict[str, Any]
    ) -> None:
        """Cancel timer and stop scans."""
        cancel_timer = IpcEnvelope(
            msg_type=TIMER_CANCEL_REQUEST,
            source=MODULE_ID,
            target="hub-timer",
            expects_reply=False,
            payload={"exam_id": exam_id, "reason": "exam_cancelled"},
        )
        await self._ipc_send("hub-timer", cancel_timer)

        stop_scan = IpcEnvelope(
            msg_type=BLE_SCAN_STOP_REQUEST,
            source=MODULE_ID,
            target="hub-ble-mgr",
            expects_reply=False,
            payload={"exam_id": exam_id, "reason": "exam_cancelled"},
        )
        await self._ipc_send("hub-ble-mgr", stop_scan)

        self._log.append(LogEntry(
            source=MODULE_ID,
            event_type="exam_cancelled",
            exam_id=exam_id,
            detail={"reason": ctx.get("reason", "")},
        ))

    # -- state -> handler map ------------------------------------------------

    _SIDE_EFFECTS: dict[str, Any] = {
        ARMED: _on_armed,
        TIMER_RUNNING: _on_timer_running,
        DONGLE_ACTIVATION: _on_dongle_activation,
        PEN_SYNC: _on_pen_sync,
        UPLOADING: _on_uploading,
        CANCELLED: _on_cancelled,
    }

    # -- default IPC sender --------------------------------------------------

    async def _default_ipc_send(
        self, target_module: str, envelope: IpcEnvelope
    ) -> IpcEnvelope | None:
        """Send an IPC message to a child module's socket."""
        socket_path = CHILD_SOCKETS.get(target_module)
        if socket_path is None:
            logger.error("No socket path for module %s", target_module)
            return None
        client = IpcClient(socket_path, source_id=MODULE_ID)
        try:
            await client.connect()
            if envelope.expects_reply:
                return await client.request(envelope)
            else:
                await client.send(envelope)
                return None
        except (OSError, TimeoutError) as exc:
            logger.warning(
                "IPC send to %s failed: %s", target_module, exc
            )
            return None
        finally:
            await client.close()
