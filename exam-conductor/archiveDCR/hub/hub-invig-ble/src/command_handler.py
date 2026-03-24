"""Invigilator command relay handler.

Parses commands received over the invigilator GATT Command characteristic,
validates auth state, and produces structured command payloads for IPC
dispatch to ``hub-supervisor``.

Domain logic only -- no I/O, no asyncio, no BLE imports.

Command IDs and payload formats follow ``ble-gatt-spec.md`` Section 4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.auth_handler import AuthHandler


# ---------------------------------------------------------------------------
# Command IDs (ble-gatt-spec.md Section 4)
# ---------------------------------------------------------------------------

CMD_START_EXAM: int = 0x01
CMD_STOP_EXAM: int = 0x02
CMD_START_REGISTRATION_SCAN: int = 0x03
CMD_MANUAL_REGISTER: int = 0x04
CMD_START_UPLOAD: int = 0x05
CMD_REQUEST_SNAPSHOT: int = 0x06

KNOWN_CMD_IDS: set[int] = {
    CMD_START_EXAM,
    CMD_STOP_EXAM,
    CMD_START_REGISTRATION_SCAN,
    CMD_MANUAL_REGISTER,
    CMD_START_UPLOAD,
    CMD_REQUEST_SNAPSHOT,
}

CMD_NAMES: dict[int, str] = {
    CMD_START_EXAM: "exam_start",
    CMD_STOP_EXAM: "exam_stop",
    CMD_START_REGISTRATION_SCAN: "start_registration_scan",
    CMD_MANUAL_REGISTER: "manual_register",
    CMD_START_UPLOAD: "trigger_upload",
    CMD_REQUEST_SNAPSHOT: "request_snapshot",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CommandResult:
    """Outcome of a command parse + validation."""

    accepted: bool
    error_code: str | None  # None on success
    cmd_name: str | None  # Friendly name from CMD_NAMES
    cmd_id: int
    request_id: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Command parser / validator
# ---------------------------------------------------------------------------

class CommandHandler:
    """Parse and validate invigilator commands.

    Parameters
    ----------
    auth_handler:
        Used to check whether the requesting BLE address is authenticated.
    """

    def __init__(self, auth_handler: AuthHandler) -> None:
        self._auth = auth_handler

    def handle(
        self,
        raw: bytes,
        ble_addr: str,
    ) -> CommandResult:
        """Parse *raw* command bytes from the Command characteristic.

        Wire format (ble-gatt-spec.md Section 4):
          offset 0:   cmd_id (1 byte, u8)
          offset 1:   request_id (16 bytes, UTF-8/ASCII)
          offset 17:  payload (N bytes, UTF-8 JSON)

        Returns a :class:`CommandResult` with ``accepted=True`` on success.
        """
        # 1. Auth check.
        if not self._auth.is_authenticated(ble_addr):
            return CommandResult(
                accepted=False,
                error_code="auth_required",
                cmd_name=None,
                cmd_id=0,
                request_id="",
                payload={},
            )

        # 2. Minimum length: 1 (cmd_id) + 16 (request_id) = 17 bytes.
        if len(raw) < 17:
            return CommandResult(
                accepted=False,
                error_code="invalid_payload",
                cmd_name=None,
                cmd_id=0,
                request_id="",
                payload={},
            )

        cmd_id = raw[0]
        request_id = raw[1:17].decode("ascii", errors="replace").rstrip("\x00")

        # 3. Unknown command check.
        if cmd_id not in KNOWN_CMD_IDS:
            return CommandResult(
                accepted=False,
                error_code="unsupported_command",
                cmd_name=None,
                cmd_id=cmd_id,
                request_id=request_id,
                payload={},
            )

        # 4. Parse JSON payload (may be empty for some commands).
        payload: dict[str, Any] = {}
        if len(raw) > 17:
            try:
                payload = json.loads(raw[17:].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return CommandResult(
                    accepted=False,
                    error_code="invalid_payload",
                    cmd_name=CMD_NAMES.get(cmd_id),
                    cmd_id=cmd_id,
                    request_id=request_id,
                    payload={},
                )

        cmd_name = CMD_NAMES[cmd_id]

        # 5. Per-command validation.
        validation_error = self._validate(cmd_id, payload)
        if validation_error is not None:
            return CommandResult(
                accepted=False,
                error_code=validation_error,
                cmd_name=cmd_name,
                cmd_id=cmd_id,
                request_id=request_id,
                payload=payload,
            )

        return CommandResult(
            accepted=True,
            error_code=None,
            cmd_name=cmd_name,
            cmd_id=cmd_id,
            request_id=request_id,
            payload=payload,
        )

    # -- per-command validation ---------------------------------------------

    @staticmethod
    def _validate(cmd_id: int, payload: dict[str, Any]) -> str | None:
        """Return an error code string if *payload* is invalid, else None."""
        if cmd_id == CMD_START_EXAM:
            if "exam_id" not in payload or "duration_sec" not in payload:
                return "invalid_payload"
        elif cmd_id == CMD_STOP_EXAM:
            if "exam_id" not in payload:
                return "invalid_payload"
        elif cmd_id == CMD_START_REGISTRATION_SCAN:
            if "exam_id" not in payload:
                return "invalid_payload"
        elif cmd_id == CMD_MANUAL_REGISTER:
            for key in ("exam_id", "pen_mac", "student_id"):
                if key not in payload:
                    return "invalid_payload"
        elif cmd_id == CMD_START_UPLOAD:
            if "exam_id" not in payload:
                return "invalid_payload"
        return None


# ---------------------------------------------------------------------------
# Provisional pen binding (local-only, per HUB_DEPLOYMENT_SPEC Section 4.3)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProvisionalBinding:
    """A local pen-student binding pending server confirmation."""

    exam_id: str
    pen_mac: str
    student_id: str
    status: str = "provisional"


def create_provisional_binding(
    payload: dict[str, Any],
) -> ProvisionalBinding:
    """Create a :class:`ProvisionalBinding` from a ``manual_register`` payload.

    The hub never creates authoritative bindings -- ``svc-exam-orch`` is the
    single writable owner.  This provisional record is for display and local
    workflow continuity only.
    """
    return ProvisionalBinding(
        exam_id=payload["exam_id"],
        pen_mac=payload["pen_mac"],
        student_id=payload["student_id"],
    )
