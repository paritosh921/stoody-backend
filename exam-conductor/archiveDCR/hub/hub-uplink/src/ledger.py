"""SQLite upload ledger wrapper for hub-uplink.

Operates on the ``upload_ledger`` table defined in HUB_DEPLOYMENT_SPEC.md
Section 3.1.  The table itself is created by ``hub-store`` (via
``hub_common.migrations``); this module only reads and updates rows.

KEY INVARIANT (STATE_OWNERSHIP_MAP.md Section 3.1):
  Ledger is updated ONLY after backend ACK — never before.
  If the process crashes between HTTP POST and ledger update, the
  chunk is re-sent on next attempt (backend deduplicates via
  idempotency key).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

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


def ensure_ledger_table(conn: sqlite3.Connection) -> None:
    """Create the ``upload_ledger`` table if it does not exist."""
    conn.execute(_UPLOAD_LEDGER_DDL)


class UploadLedger:
    """Thin wrapper over the ``upload_ledger`` SQLite table.

    All mutations happen AFTER backend ACK per the transactional
    boundary defined in STATE_OWNERSHIP_MAP.md Section 3.1.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- initialisation -----------------------------------------------------

    def init_pen(
        self,
        exam_id: str,
        pen_mac: str,
        total_chunks: int,
        upload_path: str,
    ) -> None:
        """Create or reset a ledger row before upload begins."""
        now = _iso_now()
        self._conn.execute(
            """
            INSERT INTO upload_ledger
                (exam_id, pen_mac, total_chunks, acked_chunks,
                 upload_path, complete, started_at)
            VALUES (?, ?, ?, '[]', ?, 0, ?)
            ON CONFLICT (exam_id, pen_mac) DO UPDATE SET
                total_chunks = ?,
                acked_chunks = upload_ledger.acked_chunks,
                upload_path  = ?,
                complete     = 0,
                started_at   = COALESCE(upload_ledger.started_at, ?)
            """,
            (exam_id, pen_mac, total_chunks, upload_path, now,
             total_chunks, upload_path, now),
        )

    # -- per-chunk ACK ------------------------------------------------------

    def mark_chunk_acked(
        self, exam_id: str, pen_mac: str, chunk_index: int,
    ) -> None:
        """Append *chunk_index* to the ``acked_chunks`` JSON array.

        Called ONLY after the backend returns a 202 ACK for this chunk.
        Duplicate indices are silently ignored.
        """
        row = self._conn.execute(
            "SELECT acked_chunks FROM upload_ledger "
            "WHERE exam_id = ? AND pen_mac = ?",
            (exam_id, pen_mac),
        ).fetchone()
        if row is None:
            logger.warning(
                "mark_chunk_acked: no ledger row for %s/%s", exam_id, pen_mac,
            )
            return

        acked: list[int] = json.loads(row[0])
        if chunk_index not in acked:
            acked.append(chunk_index)
            acked.sort()
        self._conn.execute(
            "UPDATE upload_ledger SET acked_chunks = ? "
            "WHERE exam_id = ? AND pen_mac = ?",
            (json.dumps(acked), exam_id, pen_mac),
        )

    # -- query --------------------------------------------------------------

    def get_pending_chunks(
        self, exam_id: str, pen_mac: str,
    ) -> list[int]:
        """Return chunk indices not yet ACKd for this pen."""
        row = self._conn.execute(
            "SELECT total_chunks, acked_chunks FROM upload_ledger "
            "WHERE exam_id = ? AND pen_mac = ?",
            (exam_id, pen_mac),
        ).fetchone()
        if row is None:
            return []
        total: int = row[0]
        acked: list[int] = json.loads(row[1])
        return [i for i in range(total) if i not in acked]

    def is_pen_complete(self, exam_id: str, pen_mac: str) -> bool:
        """True when all chunks have been ACKd."""
        row = self._conn.execute(
            "SELECT total_chunks, acked_chunks, complete FROM upload_ledger "
            "WHERE exam_id = ? AND pen_mac = ?",
            (exam_id, pen_mac),
        ).fetchone()
        if row is None:
            return False
        if row[2] == 1:
            return True
        acked: list[int] = json.loads(row[1])
        return len(acked) >= row[0]

    # -- completion ---------------------------------------------------------

    def mark_upload_complete(self, exam_id: str, pen_mac: str) -> None:
        """Set ``complete = 1`` once ALL chunks are ACKd."""
        now = _iso_now()
        self._conn.execute(
            "UPDATE upload_ledger SET complete = 1, completed_at = ? "
            "WHERE exam_id = ? AND pen_mac = ?",
            (now, exam_id, pen_mac),
        )

    # -- exam-level summary -------------------------------------------------

    def get_upload_status(self, exam_id: str) -> dict:
        """Per-pen upload summary for the entire exam.

        Returns ``{exam_id, pens: [{pen_mac, acked, total, complete}, ...]}``.
        """
        rows = self._conn.execute(
            "SELECT pen_mac, total_chunks, acked_chunks, complete "
            "FROM upload_ledger WHERE exam_id = ?",
            (exam_id,),
        ).fetchall()
        pens = []
        for pen_mac, total, acked_json, complete in rows:
            acked: list[int] = json.loads(acked_json)
            pens.append({
                "pen_mac": pen_mac,
                "acked_chunks": len(acked),
                "total_chunks": total,
                "complete": bool(complete),
            })
        return {"exam_id": exam_id, "pens": pens}


# ---------------------------------------------------------------- helpers

def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
