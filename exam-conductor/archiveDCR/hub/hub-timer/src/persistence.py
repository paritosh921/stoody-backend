"""SQLite persistence for reboot recovery.

Schema follows HUB_DEPLOYMENT_SPEC section 3.1 ``active_timer`` table.
WAL mode is enabled for crash resilience and concurrent read access by
other hub modules that share the same ``hub.db``.

Recovery logic (per FAILURE_MITIGATION_REGISTER F4):
    remaining = saved_remaining - (now_epoch - last_updated)
    If negative -> timer already expired while the process was down.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import SQLITE_DB_PATH


@dataclass(frozen=True)
class PersistedTimer:
    """Row from the ``active_timer`` table."""

    exam_id: str
    start_epoch: int
    duration_sec: int
    remaining_sec: int
    last_updated: int


class TimerPersistence:
    """Read/write the ``active_timer`` table in the hub SQLite database."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = str(db_path or SQLITE_DB_PATH)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the database and ensure the table exists."""
        self._conn = sqlite3.connect(self._db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=3000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS active_timer (
                exam_id       TEXT PRIMARY KEY,
                start_epoch   INTEGER NOT NULL,
                duration_sec  INTEGER NOT NULL,
                remaining_sec INTEGER NOT NULL,
                last_updated  INTEGER NOT NULL
            )
            """
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def persist_state(
        self,
        exam_id: str,
        start_epoch: int,
        duration_sec: int,
        remaining_sec: int,
    ) -> None:
        """Upsert the current timer state.  Called every PERSIST_INTERVAL_SEC."""
        assert self._conn is not None, "Database not opened"
        now_epoch = int(time.time())
        self._conn.execute(
            """
            INSERT INTO active_timer
                (exam_id, start_epoch, duration_sec, remaining_sec, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(exam_id) DO UPDATE SET
                remaining_sec = excluded.remaining_sec,
                last_updated  = excluded.last_updated
            """,
            (exam_id, start_epoch, duration_sec, remaining_sec, now_epoch),
        )

    def clear_state(self, exam_id: str) -> None:
        """Remove the timer row on cancel or expiry."""
        assert self._conn is not None, "Database not opened"
        self._conn.execute(
            "DELETE FROM active_timer WHERE exam_id = ?",
            (exam_id,),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_state(self) -> Optional[PersistedTimer]:
        """Load the persisted timer (if any) for boot recovery.

        Returns None if no active timer row exists.
        """
        assert self._conn is not None, "Database not opened"
        row = self._conn.execute(
            """
            SELECT exam_id, start_epoch, duration_sec, remaining_sec, last_updated
            FROM active_timer
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return PersistedTimer(
            exam_id=row[0],
            start_epoch=row[1],
            duration_sec=row[2],
            remaining_sec=row[3],
            last_updated=row[4],
        )
