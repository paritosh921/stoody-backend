"""Forensic audit logger — append-only interaction_log table.

Schema matches ``HUB_DEPLOYMENT_SPEC.md`` Section 3.4.
Every state transition, command, and event is logged.  This table is
NEVER updated or deleted — it is an immutable audit trail.

Each entry: timestamp, source, event_type, exam_id, pen_mac, invig_id,
detail (JSON), severity.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS interaction_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    exam_id     TEXT,
    pen_mac     TEXT,
    invig_id    TEXT,
    detail      TEXT,
    severity    TEXT    NOT NULL DEFAULT 'info'
                CHECK (severity IN ('debug','info','warn','error','critical'))
);
"""


# ---------------------------------------------------------------------------
# Dataclass for a single log entry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LogEntry:
    """Represents one interaction_log row."""

    source: str
    event_type: str
    severity: str = "info"
    exam_id: str | None = None
    pen_mac: str | None = None
    invig_id: str | None = None
    detail: dict[str, Any] | None = None
    timestamp: str | None = None  # set automatically if None

    def _resolve_timestamp(self) -> str:
        if self.timestamp is not None:
            return self.timestamp
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# InteractionLog writer
# ---------------------------------------------------------------------------

class InteractionLog:
    """Append-only writer for the ``interaction_log`` SQLite table.

    The table is created if it does not exist when :meth:`open` is called.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle -----------------------------------------------------------

    def open(self, conn: sqlite3.Connection | None = None) -> None:
        """Open (or reuse) a SQLite connection and ensure the table exists."""
        if conn is not None:
            self._conn = conn
        elif self._db_path is not None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL;")
        else:
            raise ValueError("Either db_path or conn must be provided")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- write ---------------------------------------------------------------

    def append(self, entry: LogEntry) -> int:
        """Insert a log entry.  Returns the ``log_id``."""
        if self._conn is None:
            raise RuntimeError("InteractionLog is not open")
        ts = entry._resolve_timestamp()
        detail_json = json.dumps(entry.detail) if entry.detail else None
        cur = self._conn.execute(
            "INSERT INTO interaction_log "
            "(timestamp, source, event_type, exam_id, pen_mac, invig_id, "
            " detail, severity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts,
                entry.source,
                entry.event_type,
                entry.exam_id,
                entry.pen_mac,
                entry.invig_id,
                detail_json,
                entry.severity,
            ),
        )
        self._conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    # -- read (for tests / diagnostics) --------------------------------------

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent *limit* log rows as dicts."""
        if self._conn is None:
            raise RuntimeError("InteractionLog is not open")
        cur = self._conn.execute(
            "SELECT log_id, timestamp, source, event_type, exam_id, "
            "pen_mac, invig_id, detail, severity "
            "FROM interaction_log ORDER BY log_id DESC LIMIT ?",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count(self) -> int:
        """Return total number of log entries."""
        if self._conn is None:
            raise RuntimeError("InteractionLog is not open")
        cur = self._conn.execute("SELECT COUNT(*) FROM interaction_log")
        return cur.fetchone()[0]
