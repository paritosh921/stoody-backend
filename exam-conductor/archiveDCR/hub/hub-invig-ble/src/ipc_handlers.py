"""IPC message handlers for hub-invig-ble.

Publishes events to ``hub-supervisor`` and ``hub-tui`` when authentication
state changes or when the invigilator issues a command.  Also requests
supervisor snapshots for the status feed.

Message types follow ``new-docs/hub/ipc-protocol.md`` Section 3.7.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from src.config import MODULE_ID

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Envelope (mirrors hub_common.ipc_protocol.IpcEnvelope / hub-timer Envelope)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Envelope:
    """Minimal IPC envelope compatible with hub-common."""

    msg_type: str
    source: str
    target: str
    payload: dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    expects_reply: bool = False
    sent_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )

    def to_line(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode() + b"\n"

    @classmethod
    def from_line(cls, raw: bytes | str) -> "Envelope":
        text = raw.decode() if isinstance(raw, bytes) else raw
        d: dict[str, Any] = json.loads(text.strip())
        return cls(
            msg_id=d["msg_id"],
            msg_type=d["msg_type"],
            source=d["source"],
            target=d["target"],
            sent_at=d["sent_at"],
            correlation_id=d.get("correlation_id"),
            expects_reply=d.get("expects_reply", False),
            payload=d.get("payload", {}),
        )

    def make_error(self, code: str, message: str, *, source: str) -> "Envelope":
        return Envelope(
            msg_type=self.msg_type.rsplit(".", 1)[0] + ".error",
            source=source,
            target=self.source,
            correlation_id=self.msg_id,
            payload={"code": code, "message": message},
        )


# ---------------------------------------------------------------------------
# Message type constants (ipc-protocol.md Section 3.7 + 3.1)
# ---------------------------------------------------------------------------

MSG_INVIG_AUTH_STATE_EVENT = "invig.auth.state.event"
MSG_INVIG_COMMAND_EVENT = "invig.command.event"
MSG_FSM_SNAPSHOT_REQUEST = "fsm.snapshot.request"
MSG_FSM_SNAPSHOT_RESULT = "fsm.snapshot.result"
MSG_SUPERVISOR_HEALTH_REQUEST = "supervisor.health.request"
MSG_SUPERVISOR_HEALTH_RESULT = "supervisor.health.result"

# ---------------------------------------------------------------------------
# Command name -> numeric ID mapping (ble-gatt-spec.md Section 4)
# ---------------------------------------------------------------------------

COMMAND_NAME_TO_ID: dict[str, int] = {
    "start_exam": 0x01,
    "exam_start": 0x01,
    "stop_exam": 0x02,
    "exam_stop": 0x02,
    "start_registration_scan": 0x03,
    "manual_register": 0x04,
    "start_upload": 0x05,
    "trigger_upload": 0x05,
    "request_snapshot": 0x06,
}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BroadcastFn = Callable[[Envelope], Awaitable[None]]
SendFn = Callable[[Envelope], Awaitable[None]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


# ---------------------------------------------------------------------------
# IPC handler class
# ---------------------------------------------------------------------------

class InvigIpcHandlers:
    """Outbound IPC event publisher for the invigilator BLE module.

    ``broadcast_fn`` sends to all connected IPC clients (supervisor, TUI).
    ``supervisor_send_fn`` sends a request to the supervisor and returns
    the correlated reply (for snapshot requests).
    """

    def __init__(
        self,
        broadcast_fn: BroadcastFn,
        supervisor_send_fn: SendFn | None = None,
    ) -> None:
        self._broadcast = broadcast_fn
        self._supervisor_send = supervisor_send_fn

    # -- Inbound handler adapters ------------------------------------------

    async def handle_health(self, env: Envelope) -> Envelope | None:
        """Handle ``supervisor.health.request`` -- returns reply."""
        payload = {
            "module": MODULE_ID,
            "healthy": True,
        }
        return Envelope(
            msg_type=MSG_SUPERVISOR_HEALTH_RESULT,
            source=MODULE_ID,
            target=env.source,
            correlation_id=env.msg_id,
            payload=payload,
        )

    async def handle_auth_state_event(self, env: Envelope) -> Envelope | None:
        """Handle inbound ``invig.auth.state.event`` (echo / ack)."""
        log.debug("Received auth state event: %s", env.payload)
        return None

    async def handle_command_event(self, env: Envelope) -> Envelope | None:
        """Handle inbound ``invig.command.event`` (echo / ack)."""
        log.debug("Received command event: %s", env.payload)
        return None

    # -- Auth events --------------------------------------------------------

    async def publish_auth_state(
        self,
        invig_id: str,
        connected: bool,
        authenticated: bool,
    ) -> None:
        """Emit ``invig.auth.state.event`` to supervisor and TUI."""
        env = _event(
            MSG_INVIG_AUTH_STATE_EVENT,
            target="hub-supervisor",
            payload={
                "invig_id": invig_id,
                "connected": connected,
                "authenticated": authenticated,
            },
        )
        await self._broadcast(env)
        log.info(
            "Auth state event: invig=%s connected=%s auth=%s",
            invig_id, connected, authenticated,
        )

    # -- Command events -----------------------------------------------------

    async def publish_command(
        self,
        cmd_name: str,
        cmd_id: int,
        request_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit ``invig.command.event`` to supervisor.

        ``cmd_id`` in the IPC payload is the **numeric** command ID from
        ``ble-gatt-spec.md`` Section 4 (e.g. ``0x01`` for ``start_exam``),
        not the human-readable command name.
        """
        numeric_id = COMMAND_NAME_TO_ID.get(cmd_name, cmd_id)
        env = _event(
            MSG_INVIG_COMMAND_EVENT,
            target="hub-supervisor",
            payload={
                "cmd_name": cmd_name,
                "cmd_id": numeric_id,
                "request_id": request_id,
                "payload": payload,
            },
        )
        await self._broadcast(env)
        log.info(
            "Command event: cmd=%s cmd_id=0x%02x request_id=%s",
            cmd_name, numeric_id, request_id,
        )

    # -- Snapshot request ---------------------------------------------------

    async def request_snapshot(self, exam_id: str) -> Envelope | None:
        """Request an FSM snapshot from the supervisor.

        Returns the reply :class:`Envelope`, or ``None`` if no supervisor
        send function is configured (e.g. in unit tests).
        """
        if self._supervisor_send is None:
            return None
        env = Envelope(
            msg_type=MSG_FSM_SNAPSHOT_REQUEST,
            source=MODULE_ID,
            target="hub-supervisor",
            expects_reply=True,
            payload={"exam_id": exam_id},
        )
        await self._supervisor_send(env)
        return None  # reply handled asynchronously by caller
