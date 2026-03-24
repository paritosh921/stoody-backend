"""Invigilator authentication handler.

Validates rotating 24-hour auth codes against the cached ``invig_codes``
SQLite table.  Tracks failed attempts per BLE address and enforces lockout
after ``AUTH_MAX_ATTEMPTS`` consecutive failures within a sliding window.

Domain logic only -- no I/O, no asyncio, no BLE imports.  The caller
(``peripheral.py``) feeds in the code and BLE address; this module returns
a pure result.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from src.config import AUTH_LOCKOUT_DURATION_SEC, AUTH_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AuthResult:
    """Outcome of an authentication attempt."""

    success: bool
    reason: str  # "ok", "invalid_code", "expired_code", "locked_out"
    ble_addr: str
    attempts_remaining: int = AUTH_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Per-address failure tracker
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _FailureRecord:
    count: int = 0
    locked_until: float = 0.0  # epoch seconds


# ---------------------------------------------------------------------------
# Code store protocol
# ---------------------------------------------------------------------------

class CodeStore:
    """Read-only accessor for the ``invig_codes`` table.

    Accepts either a real SQLite connection or a test stub.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._conn: sqlite3.Connection | None = None
        self._db_path = db_path

    def open(self, conn: sqlite3.Connection | None = None) -> None:
        if conn is not None:
            self._conn = conn
        elif self._db_path:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def lookup(self, code: str) -> dict | None:
        """Return the ``invig_codes`` row for *code*, or ``None``."""
        if self._conn is None:
            return None
        cur = self._conn.execute(
            "SELECT code, valid_from, valid_until FROM invig_codes WHERE code = ?",
            (code,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "code": row["code"],
            "valid_from": row["valid_from"],
            "valid_until": row["valid_until"],
        }


# ---------------------------------------------------------------------------
# Auth handler (pure domain logic)
# ---------------------------------------------------------------------------

class AuthHandler:
    """Validate invigilator auth codes and enforce lockout policy.

    Parameters
    ----------
    code_store:
        Accessor for the ``invig_codes`` SQLite table.
    clock_fn:
        Callable returning the current time as a UTC ``datetime``.
        Defaults to ``datetime.now(timezone.utc)``; inject a fake for tests.
    """

    def __init__(
        self,
        code_store: CodeStore,
        clock_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = code_store
        self._clock = clock_fn or (lambda: datetime.now(timezone.utc))
        self._failures: dict[str, _FailureRecord] = {}
        self._authenticated: dict[str, str] = {}  # ble_addr -> invig code

    # -- public API ---------------------------------------------------------

    def authenticate(self, code: str, ble_addr: str) -> AuthResult:
        """Validate *code* from device at *ble_addr*.

        Returns an :class:`AuthResult` describing the outcome.
        """
        now = self._clock()
        now_epoch = now.timestamp()

        # 1. Check lockout.
        rec = self._failures.get(ble_addr)
        if rec is not None and rec.locked_until > now_epoch:
            return AuthResult(
                success=False,
                reason="locked_out",
                ble_addr=ble_addr,
                attempts_remaining=0,
            )

        # 2. Lookup code in store.
        row = self._store.lookup(code)
        if row is None:
            return self._fail(ble_addr, "invalid_code", now_epoch)

        # 3. Check temporal validity.
        valid_from = datetime.fromisoformat(row["valid_from"]).replace(
            tzinfo=timezone.utc,
        )
        valid_until = datetime.fromisoformat(row["valid_until"]).replace(
            tzinfo=timezone.utc,
        )
        if now < valid_from or now >= valid_until:
            return self._fail(ble_addr, "expired_code", now_epoch)

        # 4. Success -- clear failure record and record auth.
        self._failures.pop(ble_addr, None)
        self._authenticated[ble_addr] = code
        return AuthResult(
            success=True,
            reason="ok",
            ble_addr=ble_addr,
            attempts_remaining=AUTH_MAX_ATTEMPTS,
        )

    def is_authenticated(self, ble_addr: str) -> bool:
        """Return whether *ble_addr* has an active authenticated session."""
        return ble_addr in self._authenticated

    def disconnect(self, ble_addr: str) -> None:
        """Clear auth state for a disconnected device."""
        self._authenticated.pop(ble_addr, None)

    # -- internals ----------------------------------------------------------

    def _fail(
        self, ble_addr: str, reason: str, now_epoch: float,
    ) -> AuthResult:
        rec = self._failures.get(ble_addr)
        if rec is None:
            rec = _FailureRecord()
            self._failures[ble_addr] = rec

        # Reset count if previous lockout has expired.
        if rec.locked_until > 0 and rec.locked_until <= now_epoch:
            rec.count = 0
            rec.locked_until = 0.0

        rec.count += 1
        remaining = max(0, AUTH_MAX_ATTEMPTS - rec.count)

        if rec.count >= AUTH_MAX_ATTEMPTS:
            rec.locked_until = now_epoch + AUTH_LOCKOUT_DURATION_SEC

        return AuthResult(
            success=False,
            reason=reason,
            ble_addr=ble_addr,
            attempts_remaining=remaining,
        )
