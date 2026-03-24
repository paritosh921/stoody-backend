"""IPC message handlers for hub-supervisor.

Handles:
- ``fsm.transition.request`` -- validate and execute FSM transition
- ``fsm.snapshot.request``   -- return current FSM state + module health
- ``supervisor.shutdown``    -- graceful shutdown sequence

All handlers receive an :class:`IpcEnvelope` and return an optional reply.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Callable, Coroutine

from hub_common.ipc_protocol import IpcEnvelope
from hub_common.message_types import (
    FSM_SNAPSHOT_REQUEST,
    FSM_SNAPSHOT_RESULT,
    FSM_TRANSITION_ERROR,
    FSM_TRANSITION_REQUEST,
    FSM_TRANSITION_RESULT,
)

from src.config import MODULE_ID
from src.hub_fsm import InvalidState, InvalidTransition, transition
from src.interaction_log import InteractionLog, LogEntry
from src.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# Supervisor-specific message types (not in the common catalog yet)
SUPERVISOR_SHUTDOWN = "supervisor.shutdown"

# Type alias for the shutdown callback.
ShutdownFn = Callable[[], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# FSM state persistence helpers
# ---------------------------------------------------------------------------

_PERSIST_STATE_SQL = (
    "UPDATE exam_sessions SET state = ? WHERE exam_id = ?"
)


def persist_fsm_state(
    conn: sqlite3.Connection, exam_id: str, new_state: str
) -> None:
    """Persist FSM state to SQLite BEFORE side effects (crash-safe).

    Per STATE_OWNERSHIP_MAP.md Section 3.1:
      Hub FSM transitions persisted to SQLite ``exam_sessions.state``
      BEFORE side effects execute.
    """
    conn.execute(_PERSIST_STATE_SQL, (new_state, exam_id))
    conn.commit()


def load_fsm_state(
    conn: sqlite3.Connection, exam_id: str
) -> str | None:
    """Load current FSM state for an exam from SQLite."""
    row = conn.execute(
        "SELECT state FROM exam_sessions WHERE exam_id = ?", (exam_id,)
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Handler class
# ---------------------------------------------------------------------------

class SupervisorIpcHandlers:
    """IPC dispatch for hub-supervisor messages."""

    def __init__(
        self,
        db_conn: sqlite3.Connection,
        orchestrator: Orchestrator,
        interaction_log: InteractionLog,
        *,
        shutdown_fn: ShutdownFn | None = None,
        get_module_health: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self._db = db_conn
        self._orchestrator = orchestrator
        self._ilog = interaction_log
        self._shutdown_fn = shutdown_fn
        self._get_module_health = get_module_health

    # -- fsm.transition.request ---------------------------------------------

    async def handle_transition(
        self, env: IpcEnvelope
    ) -> IpcEnvelope | None:
        """Validate + execute an FSM transition.

        Payload: ``{exam_id, from_state, to_state, reason, actor}``
        """
        p = env.payload
        exam_id: str = p["exam_id"]
        from_state: str = p["from_state"]
        to_state: str = p["to_state"]
        reason: str = p.get("reason", "")
        actor: str = p.get("actor", env.source)

        # Map to_state to the correct event for the FSM.
        event = self._state_to_event(to_state)
        if event is None:
            return env.make_error(
                "invalid_state_transition",
                f"Cannot derive event for target state {to_state!r}",
                source=MODULE_ID,
            )

        # Validate current DB state matches expected from_state.
        db_state = load_fsm_state(self._db, exam_id)
        if db_state is None:
            return env.make_error(
                "unknown_exam",
                f"Exam {exam_id} not found in local DB",
                source=MODULE_ID,
            )
        if db_state != from_state:
            return env.make_error(
                "invalid_state_transition",
                f"Stale state: DB has {db_state!r}, request has {from_state!r}",
                source=MODULE_ID,
            )

        try:
            result = transition(from_state, event)
        except (InvalidTransition, InvalidState) as exc:
            return env.make_error(
                "invalid_state_transition",
                str(exc),
                source=MODULE_ID,
            )

        # --- Persist BEFORE side effects (crash-safe) ---
        persist_fsm_state(self._db, exam_id, result.new_state)

        # --- Log transition ---
        self._ilog.append(LogEntry(
            source=MODULE_ID,
            event_type="fsm_transition",
            exam_id=exam_id,
            detail={
                "from": result.old_state,
                "to": result.new_state,
                "event": event,
                "actor": actor,
                "reason": reason,
            },
        ))

        # --- Execute side effects ---
        context = dict(p)
        await self._orchestrator.on_transition(
            exam_id, result.new_state, context=context,
        )

        return env.make_reply(
            FSM_TRANSITION_RESULT,
            {"exam_id": exam_id, "state": result.new_state, "persisted": True},
            source=MODULE_ID,
        )

    # -- fsm.snapshot.request -----------------------------------------------

    async def handle_snapshot(
        self, env: IpcEnvelope
    ) -> IpcEnvelope | None:
        """Return current FSM state + module health."""
        exam_id: str = env.payload.get("exam_id", "")
        state = load_fsm_state(self._db, exam_id) if exam_id else None
        module_health = (
            self._get_module_health() if self._get_module_health else {}
        )
        return env.make_reply(
            FSM_SNAPSHOT_RESULT,
            {
                "exam_id": exam_id,
                "state": state or "unknown",
                "modules": module_health,
                "timer": {},
                "dongles": {},
                "storage": {},
                "upload": {},
            },
            source=MODULE_ID,
        )

    # -- supervisor.shutdown ------------------------------------------------

    async def handle_shutdown(
        self, env: IpcEnvelope
    ) -> IpcEnvelope | None:
        """Initiate graceful shutdown sequence."""
        logger.info("Shutdown requested by %s", env.source)
        self._ilog.append(LogEntry(
            source=MODULE_ID,
            event_type="hub_shutdown",
            detail={"requested_by": env.source},
        ))
        if self._shutdown_fn is not None:
            await self._shutdown_fn()
        return env.make_reply(
            "supervisor.shutdown.result",
            {"status": "shutting_down"},
            source=MODULE_ID,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _state_to_event(target_state: str) -> str | None:
        """Map a target state name to the FSM event that reaches it."""
        from src.hub_fsm import (
            EVT_ACTIVATE_DONGLES,
            EVT_ARM,
            EVT_CANCEL,
            EVT_START_TIMER,
            EVT_START_UPLOAD,
            EVT_SYNC_ALL_COMPLETE,
            EVT_SYNC_PARTIAL,
            EVT_TIMER_EXPIRED,
            EVT_UPLOAD_DONE,
        )
        mapping: dict[str, str] = {
            "armed": EVT_ARM,
            "timer_running": EVT_START_TIMER,
            "dongle_activation": EVT_TIMER_EXPIRED,
            "pen_sync": EVT_ACTIVATE_DONGLES,
            "sync_complete": EVT_SYNC_ALL_COMPLETE,
            "sync_partial": EVT_SYNC_PARTIAL,
            "uploading": EVT_START_UPLOAD,
            "upload_complete": EVT_UPLOAD_DONE,
            "cancelled": EVT_CANCEL,
        }
        return mapping.get(target_state)
