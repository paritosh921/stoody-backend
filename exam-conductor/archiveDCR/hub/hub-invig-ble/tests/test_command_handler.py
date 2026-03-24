"""Unit tests for invigilator command handler.

Test IDs: U-INVIG-CMD-01 through U-INVIG-CMD-06.
Validation level: L3 (unit, no I/O).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from src.auth_handler import AuthHandler, CodeStore
from src.command_handler import (
    CMD_MANUAL_REGISTER,
    CMD_REQUEST_SNAPSHOT,
    CMD_START_EXAM,
    CMD_START_UPLOAD,
    CMD_STOP_EXAM,
    CommandHandler,
    ProvisionalBinding,
    create_provisional_binding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BLE_ADDR = "AA:BB:CC:DD:EE:02"
VALID_CODE = "CODE12345678"


def _make_store() -> tuple[CodeStore, sqlite3.Connection]:
    """In-memory code store with a valid code."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE invig_codes (
            code TEXT PRIMARY KEY, valid_from TEXT, valid_until TEXT, fetched_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO invig_codes VALUES (?, ?, ?, ?)",
        (
            VALID_CODE,
            "2026-03-19T00:00:00",
            "2026-03-20T00:00:00",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    store = CodeStore()
    store.open(conn=conn)
    return store, conn


def _fixed_clock() -> datetime:
    return datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc)


def _build_raw(cmd_id: int, request_id: str, payload: dict | None = None) -> bytes:
    """Build raw command bytes matching the GATT wire format."""
    rid_bytes = request_id.encode("ascii")[:16].ljust(16, b"\x00")
    header = bytes([cmd_id]) + rid_bytes
    if payload is not None:
        header += json.dumps(payload).encode("utf-8")
    return header


@pytest.fixture()
def auth_handler() -> AuthHandler:
    store, _ = _make_store()
    handler = AuthHandler(store, clock_fn=_fixed_clock)
    # Pre-authenticate the test address.
    handler.authenticate(VALID_CODE, BLE_ADDR)
    return handler


@pytest.fixture()
def cmd_handler(auth_handler: AuthHandler) -> CommandHandler:
    return CommandHandler(auth_handler)


# ---------------------------------------------------------------------------
# U-INVIG-CMD-01: Command rejected without auth
# ---------------------------------------------------------------------------

def test_command_rejected_without_auth():
    """U-INVIG-CMD-01: Commands from unauthenticated addresses are rejected."""
    store, _ = _make_store()
    auth = AuthHandler(store, clock_fn=_fixed_clock)
    # Do NOT authenticate.
    cmd = CommandHandler(auth)

    raw = _build_raw(CMD_START_EXAM, "req-001", {"exam_id": "e1", "duration_sec": 60})
    result = cmd.handle(raw, BLE_ADDR)

    assert result.accepted is False
    assert result.error_code == "auth_required"


# ---------------------------------------------------------------------------
# U-INVIG-CMD-02: Start exam parsed correctly
# ---------------------------------------------------------------------------

def test_start_exam_parsed(cmd_handler: CommandHandler):
    """U-INVIG-CMD-02: start_exam command is parsed and accepted."""
    raw = _build_raw(
        CMD_START_EXAM, "req-002", {"exam_id": "exam-A", "duration_sec": 3600},
    )
    result = cmd_handler.handle(raw, BLE_ADDR)

    assert result.accepted is True
    assert result.cmd_name == "exam_start"
    assert result.cmd_id == CMD_START_EXAM
    assert result.request_id == "req-002"
    assert result.payload["exam_id"] == "exam-A"
    assert result.payload["duration_sec"] == 3600


# ---------------------------------------------------------------------------
# U-INVIG-CMD-03: Stop exam parsed correctly
# ---------------------------------------------------------------------------

def test_stop_exam_parsed(cmd_handler: CommandHandler):
    """U-INVIG-CMD-03: stop_exam command is parsed and accepted."""
    raw = _build_raw(
        CMD_STOP_EXAM, "req-003", {"exam_id": "exam-A", "reason": "time_up"},
    )
    result = cmd_handler.handle(raw, BLE_ADDR)

    assert result.accepted is True
    assert result.cmd_name == "exam_stop"
    assert result.payload["reason"] == "time_up"


# ---------------------------------------------------------------------------
# U-INVIG-CMD-04: Manual register creates provisional binding
# ---------------------------------------------------------------------------

def test_manual_register_accepted(cmd_handler: CommandHandler):
    """U-INVIG-CMD-04: manual_register command is accepted with valid payload."""
    payload = {
        "exam_id": "exam-B",
        "pen_mac": "11:22:33:44:55:66",
        "student_id": "S-001",
    }
    raw = _build_raw(CMD_MANUAL_REGISTER, "req-004", payload)
    result = cmd_handler.handle(raw, BLE_ADDR)

    assert result.accepted is True
    assert result.cmd_name == "manual_register"
    assert result.payload["pen_mac"] == "11:22:33:44:55:66"


def test_provisional_binding_creation():
    """U-INVIG-CMD-04b: ProvisionalBinding is created from payload."""
    payload = {
        "exam_id": "exam-B",
        "pen_mac": "11:22:33:44:55:66",
        "student_id": "S-001",
    }
    binding = create_provisional_binding(payload)

    assert isinstance(binding, ProvisionalBinding)
    assert binding.exam_id == "exam-B"
    assert binding.pen_mac == "11:22:33:44:55:66"
    assert binding.student_id == "S-001"
    assert binding.status == "provisional"


def test_manual_register_missing_field(cmd_handler: CommandHandler):
    """U-INVIG-CMD-04c: manual_register rejected if required field missing."""
    payload = {"exam_id": "exam-B", "pen_mac": "11:22:33:44:55:66"}
    # Missing student_id.
    raw = _build_raw(CMD_MANUAL_REGISTER, "req-004c", payload)
    result = cmd_handler.handle(raw, BLE_ADDR)

    assert result.accepted is False
    assert result.error_code == "invalid_payload"


# ---------------------------------------------------------------------------
# U-INVIG-CMD-05: Unknown command rejected
# ---------------------------------------------------------------------------

def test_unknown_command_rejected(cmd_handler: CommandHandler):
    """U-INVIG-CMD-05: Unknown cmd_id returns unsupported_command."""
    raw = _build_raw(0xFF, "req-005", {"foo": "bar"})
    result = cmd_handler.handle(raw, BLE_ADDR)

    assert result.accepted is False
    assert result.error_code == "unsupported_command"


# ---------------------------------------------------------------------------
# U-INVIG-CMD-06: Malformed payload rejected
# ---------------------------------------------------------------------------

def test_malformed_json_rejected(cmd_handler: CommandHandler):
    """U-INVIG-CMD-06: Invalid JSON payload returns invalid_payload."""
    rid = "req-006".encode("ascii")[:16].ljust(16, b"\x00")
    raw = bytes([CMD_START_EXAM]) + rid + b"not-valid-json{{"
    result = cmd_handler.handle(raw, BLE_ADDR)

    assert result.accepted is False
    assert result.error_code == "invalid_payload"


def test_too_short_payload_rejected(cmd_handler: CommandHandler):
    """U-INVIG-CMD-06b: Payload shorter than 17 bytes is rejected."""
    result = cmd_handler.handle(b"\x01short", BLE_ADDR)

    assert result.accepted is False
    assert result.error_code == "invalid_payload"


def test_start_exam_missing_required_fields(cmd_handler: CommandHandler):
    """U-INVIG-CMD-06c: start_exam without duration_sec is rejected."""
    raw = _build_raw(CMD_START_EXAM, "req-006c", {"exam_id": "e1"})
    result = cmd_handler.handle(raw, BLE_ADDR)

    assert result.accepted is False
    assert result.error_code == "invalid_payload"
