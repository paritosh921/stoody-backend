"""SQLite ledger for pen sync status and upload tracking.

Wraps the ``pen_sync_status`` and ``upload_ledger`` tables defined in
HUB_DEPLOYMENT_SPEC.md Section 3.1.  The database is opened in WAL mode
for crash-safe concurrent reads during uploads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ schema

_PEN_SYNC_STATUS_DDL = """\
CREATE TABLE IF NOT EXISTS pen_sync_status (
    exam_id         TEXT NOT NULL,
    pen_mac         TEXT NOT NULL,
    dongle_mac      TEXT,
    sync_started    TEXT,
    sync_completed  TEXT,
    bytes_expected  INTEGER,
    bytes_received  INTEGER,
    checksum_expected TEXT,
    checksum_actual TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','connecting','syncing',
                           'complete','failed','timeout')),
    error_detail    TEXT,
    PRIMARY KEY (exam_id, pen_mac)
);
"""

_UPLOAD_LEDGER_DDL = """\
CREATE TABLE IF NOT EXISTS upload_ledger (
    exam_id         TEXT NOT NULL,
    pen_mac         TEXT NOT NULL,
    total_chunks    INTEGER NOT NULL,
    acked_chunks    TEXT NOT NULL DEFAULT '[]',
    upload_path     TEXT CHECK (upload_path IN ('wifi','mobile')),
    complete        INTEGER NOT NULL DEFAULT 0,
    started_at      TEXT,
    completed_at    TEXT,
    PRIMARY KEY (exam_id, pen_mac)
);
"""


def open_ledger_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the hub SQLite database in WAL mode.

    Returns a :class:`sqlite3.Connection` with tables guaranteed to exist.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_PEN_SYNC_STATUS_DDL)
    conn.execute(_UPLOAD_LEDGER_DDL)
    logger.info("Ledger DB ready at %s (WAL mode)", db_path)
    return conn


# ------------------------------------------------------------------ class

class ChunkLedger:
    """Thin wrapper over SQLite for pen-sync and upload tracking."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- pen_sync_status ------------------------------------------------

    def record_chunk_received(
        self,
        exam_id: str,
        pen_mac: str,
        new_bytes: int,
    ) -> None:
        """Upsert a ``pen_sync_status`` row after a successful chunk write.

        Increments ``bytes_received`` and sets ``status = 'syncing'``.
        """
        now = _iso_now()
        self._conn.execute(
            """
            INSERT INTO pen_sync_status
                (exam_id, pen_mac, bytes_received, status, sync_started)
            VALUES (?, ?, ?, 'syncing', ?)
            ON CONFLICT (exam_id, pen_mac) DO UPDATE SET
                bytes_received = COALESCE(pen_sync_status.bytes_received, 0) + ?,
                status         = 'syncing'
            """,
            (exam_id, pen_mac, new_bytes, now, new_bytes),
        )

    def mark_sync_complete(
        self,
        exam_id: str,
        pen_mac: str,
        checksum_actual: str,
    ) -> None:
        """Mark a pen sync as ``complete`` with verified checksum."""
        now = _iso_now()
        self._conn.execute(
            """
            UPDATE pen_sync_status
               SET status           = 'complete',
                   sync_completed   = ?,
                   checksum_actual  = ?
             WHERE exam_id = ? AND pen_mac = ?
            """,
            (now, checksum_actual, exam_id, pen_mac),
        )

    def mark_sync_failed(
        self,
        exam_id: str,
        pen_mac: str,
        error_detail: str,
    ) -> None:
        """Mark a pen sync as ``failed`` with an error description.

        Uses INSERT ... ON CONFLICT so the failure is recorded even when
        no prior chunk was successfully written (i.e., the very first
        chunk for this pen failed).
        """
        now = _iso_now()
        self._conn.execute(
            """
            INSERT INTO pen_sync_status
                (exam_id, pen_mac, bytes_received, status, error_detail, sync_started)
            VALUES (?, ?, 0, 'failed', ?, ?)
            ON CONFLICT (exam_id, pen_mac) DO UPDATE SET
                status       = 'failed',
                error_detail = ?
            """,
            (exam_id, pen_mac, error_detail, now, error_detail),
        )

    def get_sync_status(
        self, exam_id: str, pen_mac: str
    ) -> dict | None:
        """Return the current ``pen_sync_status`` row as a dict, or *None*."""
        cur = self._conn.execute(
            "SELECT * FROM pen_sync_status WHERE exam_id = ? AND pen_mac = ?",
            (exam_id, pen_mac),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    # -- upload_ledger --------------------------------------------------

    def init_upload_ledger(
        self,
        exam_id: str,
        pen_mac: str,
        total_chunks: int,
    ) -> None:
        """Create or reset an ``upload_ledger`` row before upload begins."""
        now = _iso_now()
        self._conn.execute(
            """
            INSERT INTO upload_ledger
                (exam_id, pen_mac, total_chunks, acked_chunks, complete, started_at)
            VALUES (?, ?, ?, '[]', 0, ?)
            ON CONFLICT (exam_id, pen_mac) DO UPDATE SET
                total_chunks = ?,
                acked_chunks = '[]',
                complete     = 0,
                started_at   = ?
            """,
            (exam_id, pen_mac, total_chunks, now, total_chunks, now),
        )

    def get_upload_status(
        self, exam_id: str, pen_mac: str
    ) -> dict | None:
        """Return the current ``upload_ledger`` row as a dict, or *None*."""
        cur = self._conn.execute(
            "SELECT * FROM upload_ledger WHERE exam_id = ? AND pen_mac = ?",
            (exam_id, pen_mac),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        result = dict(zip(cols, row))
        result["acked_chunks"] = json.loads(result["acked_chunks"])
        return result


# ---------------------------------------------------------------- helpers

def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
