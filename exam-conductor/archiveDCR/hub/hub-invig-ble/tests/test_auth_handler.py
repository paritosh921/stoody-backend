"""Unit tests for invigilator authentication handler.

Test IDs: U-INVIG-AUTH-01 through U-INVIG-AUTH-06.
Validation level: L3 (unit, no I/O -- uses in-memory SQLite).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.auth_handler import AuthHandler, AuthResult, CodeStore
from src.config import AUTH_LOCKOUT_DURATION_SEC, AUTH_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with the ``invig_codes`` table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE invig_codes (
            code        TEXT PRIMARY KEY,
            valid_from  TEXT NOT NULL,
            valid_until TEXT NOT NULL,
            fetched_at  TEXT NOT NULL
        )
    """)
    return conn


def _insert_code(
    conn: sqlite3.Connection,
    code: str,
    valid_from: datetime,
    valid_until: datetime,
) -> None:
    conn.execute(
        "INSERT INTO invig_codes (code, valid_from, valid_until, fetched_at) "
        "VALUES (?, ?, ?, ?)",
        (
            code,
            valid_from.isoformat(),
            valid_until.isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


class FakeClock:
    """Deterministic clock returning a fixed datetime, advanceable."""

    def __init__(self, now: datetime | None = None) -> None:
        self._now = now or datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


BLE_ADDR = "AA:BB:CC:DD:EE:01"
VALID_CODE = "CODE12345678"


@pytest.fixture()
def db() -> sqlite3.Connection:
    conn = _make_db()
    # Insert a code valid for 24 hours around the fake clock's default time.
    _insert_code(
        conn,
        VALID_CODE,
        valid_from=datetime(2026, 3, 19, 0, 0, 0, tzinfo=timezone.utc),
        valid_until=datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc),
    )
    return conn


@pytest.fixture()
def store(db: sqlite3.Connection) -> CodeStore:
    s = CodeStore()
    s.open(conn=db)
    return s


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def handler(store: CodeStore, clock: FakeClock) -> AuthHandler:
    return AuthHandler(store, clock_fn=clock)


# ---------------------------------------------------------------------------
# U-INVIG-AUTH-01: Valid code accepted
# ---------------------------------------------------------------------------

def test_valid_code_accepted(handler: AuthHandler):
    """U-INVIG-AUTH-01: A valid, non-expired code is accepted."""
    result = handler.authenticate(VALID_CODE, BLE_ADDR)

    assert result.success is True
    assert result.reason == "ok"
    assert result.ble_addr == BLE_ADDR
    assert result.attempts_remaining == AUTH_MAX_ATTEMPTS
    assert handler.is_authenticated(BLE_ADDR)


# ---------------------------------------------------------------------------
# U-INVIG-AUTH-02: Invalid code rejected
# ---------------------------------------------------------------------------

def test_invalid_code_rejected(handler: AuthHandler):
    """U-INVIG-AUTH-02: A code not in the store is rejected."""
    result = handler.authenticate("BADCODE12345", BLE_ADDR)

    assert result.success is False
    assert result.reason == "invalid_code"
    assert result.attempts_remaining == AUTH_MAX_ATTEMPTS - 1
    assert not handler.is_authenticated(BLE_ADDR)


# ---------------------------------------------------------------------------
# U-INVIG-AUTH-03: Expired code rejected
# ---------------------------------------------------------------------------

def test_expired_code_rejected(
    store: CodeStore, db: sqlite3.Connection,
):
    """U-INVIG-AUTH-03: A code past its valid_until is rejected."""
    # Insert a code that expired yesterday.
    _insert_code(
        db,
        "EXPIREDCODE1",
        valid_from=datetime(2026, 3, 17, 0, 0, 0, tzinfo=timezone.utc),
        valid_until=datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc),
    )
    clock = FakeClock()  # now = 2026-03-19T12:00
    handler = AuthHandler(store, clock_fn=clock)

    result = handler.authenticate("EXPIREDCODE1", BLE_ADDR)

    assert result.success is False
    assert result.reason == "expired_code"


def test_not_yet_valid_code_rejected(
    store: CodeStore, db: sqlite3.Connection,
):
    """U-INVIG-AUTH-03b: A code whose valid_from is in the future is rejected."""
    _insert_code(
        db,
        "FUTURECODE12",
        valid_from=datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc),
        valid_until=datetime(2026, 3, 21, 0, 0, 0, tzinfo=timezone.utc),
    )
    clock = FakeClock()  # now = 2026-03-19T12:00 (before valid_from)
    handler = AuthHandler(store, clock_fn=clock)

    result = handler.authenticate("FUTURECODE12", BLE_ADDR)

    assert result.success is False
    assert result.reason == "expired_code"


# ---------------------------------------------------------------------------
# U-INVIG-AUTH-04: Lockout after 5 consecutive failures
# ---------------------------------------------------------------------------

def test_lockout_after_max_failures(handler: AuthHandler):
    """U-INVIG-AUTH-04: After 5 failed attempts the address is locked out."""
    for i in range(AUTH_MAX_ATTEMPTS):
        result = handler.authenticate("WRONGCODE123", BLE_ADDR)
        assert result.success is False

    # The 5th failure should report 0 remaining.
    assert result.attempts_remaining == 0

    # 6th attempt should be locked out.
    result = handler.authenticate(VALID_CODE, BLE_ADDR)
    assert result.success is False
    assert result.reason == "locked_out"
    assert result.attempts_remaining == 0


def test_attempts_remaining_decrements(handler: AuthHandler):
    """U-INVIG-AUTH-04b: attempts_remaining decrements correctly."""
    for i in range(1, AUTH_MAX_ATTEMPTS + 1):
        result = handler.authenticate("WRONGCODE123", BLE_ADDR)
        expected_remaining = max(0, AUTH_MAX_ATTEMPTS - i)
        assert result.attempts_remaining == expected_remaining


# ---------------------------------------------------------------------------
# U-INVIG-AUTH-05: Lockout expiry allows retry
# ---------------------------------------------------------------------------

def test_lockout_expiry(handler: AuthHandler, clock: FakeClock):
    """U-INVIG-AUTH-05: After lockout duration expires, auth attempts succeed."""
    # Trigger lockout.
    for _ in range(AUTH_MAX_ATTEMPTS):
        handler.authenticate("WRONGCODE123", BLE_ADDR)

    result = handler.authenticate(VALID_CODE, BLE_ADDR)
    assert result.reason == "locked_out"

    # Advance clock past lockout.
    clock.advance(AUTH_LOCKOUT_DURATION_SEC + 1)

    # Should now be able to authenticate again.
    result = handler.authenticate(VALID_CODE, BLE_ADDR)
    assert result.success is True
    assert result.reason == "ok"


# ---------------------------------------------------------------------------
# U-INVIG-AUTH-06: Successful auth clears failure count
# ---------------------------------------------------------------------------

def test_success_clears_failure_count(handler: AuthHandler):
    """U-INVIG-AUTH-06: A successful auth resets the failure counter."""
    # Rack up some failures (but less than lockout threshold).
    for _ in range(AUTH_MAX_ATTEMPTS - 1):
        handler.authenticate("WRONGCODE123", BLE_ADDR)

    # Succeed.
    result = handler.authenticate(VALID_CODE, BLE_ADDR)
    assert result.success is True

    # Failure counter should be reset -- next failure gets full attempts.
    handler.disconnect(BLE_ADDR)
    result = handler.authenticate("WRONGCODE123", BLE_ADDR)
    assert result.attempts_remaining == AUTH_MAX_ATTEMPTS - 1


def test_disconnect_clears_auth(handler: AuthHandler):
    """U-INVIG-AUTH-06b: Disconnecting clears authenticated state."""
    handler.authenticate(VALID_CODE, BLE_ADDR)
    assert handler.is_authenticated(BLE_ADDR)

    handler.disconnect(BLE_ADDR)
    assert not handler.is_authenticated(BLE_ADDR)
