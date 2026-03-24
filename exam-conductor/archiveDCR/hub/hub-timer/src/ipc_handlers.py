"""IPC message handlers for hub-timer.

Handles incoming requests from hub-supervisor and other modules, and
publishes timer.tick / timer.expired events via broadcast.

Message types follow the catalog in ``new-docs/hub/ipc-protocol.md``
section 3.2 (Timer).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from src.config import MODULE_ID
from src.countdown import CountdownTimer
from src.ipc_server import Envelope

# -- Inbound message types -------------------------------------------------
MSG_TIMER_ARM_REQUEST = "timer.arm.request"
MSG_TIMER_CANCEL_REQUEST = "timer.cancel.request"
MSG_TIMER_SNAPSHOT_REQUEST = "timer.snapshot.request"
MSG_SUPERVISOR_HEALTH_REQUEST = "supervisor.health.request"

# -- Outbound message types ------------------------------------------------
MSG_TIMER_TICK = "timer.tick"
MSG_TIMER_EXPIRED_EVENT = "timer.expired.event"
MSG_TIMER_SNAPSHOT_RESULT = "timer.snapshot.result"
MSG_SUPERVISOR_HEALTH_RESULT = "supervisor.health.result"

log = logging.getLogger(__name__)

# Type alias for the async broadcast callback.
BroadcastFn = Callable[[Envelope], Coroutine[Any, Any, None]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reply(
    request: Envelope,
    reply_type: str,
    payload: dict[str, Any],
) -> Envelope:
    """Build a reply envelope correlated to *request*."""
    return Envelope(
        msg_type=reply_type,
        source=MODULE_ID,
        target=request.source,
        correlation_id=request.msg_id,
        payload=payload,
    )


def _event(
    msg_type: str,
    target: str,
    payload: dict[str, Any],
) -> Envelope:
    """Build a fire-and-forget event envelope."""
    return Envelope(
        msg_type=msg_type,
        source=MODULE_ID,
        target=target,
        payload=payload,
    )


class TimerIpcHandlers:
    """Handler dispatch for hub-timer IPC messages.

    ``broadcast_fn`` is called for fire-and-forget events (tick, expired)
    that must reach all connected clients.  Request/reply handlers return
    an ``Envelope`` that the server writes back on the originating
    connection only.
    """

    def __init__(
        self,
        timer: CountdownTimer,
        broadcast_fn: BroadcastFn,
        on_arm: Optional[Callable[..., Coroutine]] = None,
        on_cancel: Optional[Callable[..., Coroutine]] = None,
    ) -> None:
        self._timer = timer
        self._broadcast = broadcast_fn
        self._on_arm = on_arm
        self._on_cancel = on_cancel

    # ------------------------------------------------------------------
    # Server handler adapters  (Envelope -> Envelope | None)
    # ------------------------------------------------------------------

    async def handle_arm(self, env: Envelope) -> Envelope | None:
        """Handle ``timer.arm.request``."""
        payload = env.payload
        exam_id: str = payload["exam_id"]
        duration_sec: int = payload["duration_sec"]
        log.info("Arming timer for exam %s (%d s)", exam_id, duration_sec)
        if self._on_arm is not None:
            await self._on_arm(exam_id, duration_sec)
        return None

    async def handle_cancel(self, env: Envelope) -> Envelope | None:
        """Handle ``timer.cancel.request``."""
        payload = env.payload
        exam_id: str = payload["exam_id"]
        reason: str = payload.get("reason", "")
        log.info("Cancelling timer for exam %s (reason: %s)", exam_id, reason)
        if self._on_cancel is not None:
            await self._on_cancel(exam_id)
        return None

    async def handle_snapshot(self, env: Envelope) -> Envelope | None:
        """Handle ``timer.snapshot.request`` -- returns reply."""
        state = self._timer.get_state()
        if state is None:
            payload = {"exam_id": None, "state": "idle", "remaining_sec": 0}
        else:
            payload = {
                "exam_id": state.exam_id,
                "state": "expired" if state.expired else "running",
                "remaining_sec": state.remaining_sec,
                "started_at": datetime.fromtimestamp(
                    state.started_at_epoch, tz=timezone.utc,
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "expires_at": datetime.fromtimestamp(
                    state.started_at_epoch + state._effective_total,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        return _reply(env, MSG_TIMER_SNAPSHOT_RESULT, payload)

    async def handle_health(self, env: Envelope) -> Envelope | None:
        """Handle ``supervisor.health.request`` -- returns reply."""
        payload = {
            "module": MODULE_ID,
            "healthy": True,
            "timer_active": self._timer.active,
            "exam_id": self._timer.exam_id,
            "remaining_sec": self._timer.get_remaining(),
        }
        return _reply(env, MSG_SUPERVISOR_HEALTH_RESULT, payload)

    # ------------------------------------------------------------------
    # Outbound broadcasts (called by the main loop)
    # ------------------------------------------------------------------

    async def publish_tick(self) -> None:
        """Broadcast ``timer.tick`` to all connected clients."""
        state = self._timer.get_state()
        if state is None:
            return
        env = _event(
            MSG_TIMER_TICK,
            target="*",
            payload={
                "exam_id": state.exam_id,
                "remaining_sec": state.remaining_sec,
                "total_sec": state.total_sec,
            },
        )
        await self._broadcast(env)

    async def publish_expired(self, exam_id: str) -> None:
        """Broadcast ``timer.expired.event`` to all connected clients."""
        env = _event(
            MSG_TIMER_EXPIRED_EVENT,
            target="*",
            payload={
                "exam_id": exam_id,
                "expired_at": _utc_now_iso(),
            },
        )
        await self._broadcast(env)
