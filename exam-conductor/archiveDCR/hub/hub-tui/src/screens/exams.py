"""Exam history screen — list past sessions with per-pen detail view.

Reads exam history from the local SQLite database
(``/var/lib/exampen/hub.db``) rather than IPC, since historical data
is persisted in the DB and does not change at real-time frequency.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Header, Static

from src.widgets.footer import HubFooter
from src.widgets.status_table import StatusTable

logger = logging.getLogger(__name__)

_EXAM_COLS = ["Exam ID", "Date", "Duration", "Pens Synced", "Upload", "Invigilator"]
_PEN_DETAIL_COLS = ["Pen MAC", "Student", "Sync", "Data Size", "Upload"]

# DB path — overridable via env var for testing.
_DEFAULT_DB_PATH = "/var/lib/exampen/hub.db"


def _get_db_path() -> str:
    return os.environ.get("EXAMPEN_DB_PATH", _DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# SQLite queries (synchronous, run in worker thread via run_worker)
# ---------------------------------------------------------------------------

def _query_exam_history(db_path: str) -> list[list[str]]:
    """Fetch exam sessions ordered by most recent first."""
    rows: list[list[str]] = []
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.execute(
            """
            SELECT
                es.exam_id,
                DATE(es.created_at) AS exam_date,
                es.duration_min || ' min' AS duration,
                (
                    SELECT COUNT(*) FROM pen_sync_status ps
                    WHERE ps.exam_id = es.exam_id AND ps.status = 'complete'
                ) || '/' || (
                    SELECT COUNT(*) FROM pen_sync_status ps
                    WHERE ps.exam_id = es.exam_id
                ) AS pens_synced,
                CASE
                    WHEN NOT EXISTS (
                        SELECT 1 FROM upload_ledger ul
                        WHERE ul.exam_id = es.exam_id AND ul.complete = 0
                    ) AND EXISTS (
                        SELECT 1 FROM upload_ledger ul
                        WHERE ul.exam_id = es.exam_id
                    ) THEN 'complete'
                    WHEN EXISTS (
                        SELECT 1 FROM upload_ledger ul
                        WHERE ul.exam_id = es.exam_id AND ul.complete = 1
                    ) THEN 'partial'
                    ELSE 'pending'
                END AS upload_status,
                es.invig_id
            FROM exam_sessions es
            ORDER BY es.created_at DESC
            LIMIT 50
            """
        )
        for row in cursor.fetchall():
            rows.append([str(c) if c is not None else "" for c in row])
        conn.close()
    except Exception:
        logger.warning("Failed to query exam history", exc_info=True)
    return rows


def _query_pen_detail(db_path: str, exam_id: str) -> list[list[str]]:
    """Fetch per-pen breakdown for a specific exam."""
    rows: list[list[str]] = []
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.execute(
            """
            SELECT
                ps.pen_mac,
                COALESCE(pb.student_name, pb.student_id, 'Unknown'),
                ps.status,
                CASE
                    WHEN ps.bytes_received IS NOT NULL
                    THEN ROUND(ps.bytes_received / 1048576.0, 1) || ' MB'
                    ELSE '--'
                END AS data_size,
                CASE
                    WHEN ul.complete = 1 THEN 'uploaded'
                    WHEN ul.complete = 0 THEN 'pending'
                    ELSE 'none'
                END AS upload_status
            FROM pen_sync_status ps
            LEFT JOIN pen_bindings pb
                ON pb.exam_id = ps.exam_id AND pb.pen_mac = ps.pen_mac
            LEFT JOIN upload_ledger ul
                ON ul.exam_id = ps.exam_id AND ul.pen_mac = ps.pen_mac
            WHERE ps.exam_id = ?
            ORDER BY ps.pen_mac
            """,
            (exam_id,),
        )
        for row in cursor.fetchall():
            rows.append([str(c) if c is not None else "" for c in row])
        conn.close()
    except Exception:
        logger.warning("Failed to query pen detail for %s", exam_id, exc_info=True)
    return rows


class ExamsScreen(Screen):
    """Exam session history with SQLite-backed data and detail view."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("d", "show_detail", "Detail", show=True),
        Binding("r", "refresh_data", "Refresh", show=True),
    ]

    DEFAULT_CSS = """
    ExamsScreen #exams-container {
        padding: 1 2;
    }
    ExamsScreen .screen-title {
        text-style: bold;
        margin-bottom: 1;
    }
    ExamsScreen .section-title {
        text-style: bold;
        margin-top: 1;
    }
    ExamsScreen .info-line {
        margin: 0 1;
        color: $text-muted;
    }
    """

    _showing_detail: bool = False
    _exam_rows: list[list[str]] = []

    def __init__(self, db_path: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._db_path = db_path or _get_db_path()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="exams-container"):
            yield Static("[5] Exam History", classes="screen-title")

            yield StatusTable(
                columns=_EXAM_COLS,
                rows=[],
                status_column=4,
                id="exam-table",
            )

            yield Static(
                "Press [d] for per-pen detail of first exam  |  [r] to refresh",
                classes="info-line",
            )

            # Detail section (shown on [d])
            yield Static("", id="detail-title", classes="section-title")
            yield StatusTable(
                columns=_PEN_DETAIL_COLS,
                rows=[],
                status_column=2,
                id="pen-detail-table",
            )
        yield HubFooter()

    def on_mount(self) -> None:
        """Load exam history on screen mount."""
        self._load_exams()

    def _load_exams(self) -> None:
        """Fetch exam history from SQLite (sync, fast on small dataset)."""
        self._exam_rows = _query_exam_history(self._db_path)
        table = self.query_one("#exam-table", StatusTable)
        if self._exam_rows:
            table.set_data(self._exam_rows, status_column=4)
        else:
            table.set_data([])
            info = self.query_one(".info-line", Static)
            info.update(
                "No exam history found.  "
                f"DB path: {self._db_path}"
            )

    def action_refresh_data(self) -> None:
        """Reload exam data from SQLite."""
        self._load_exams()
        # Also collapse detail if open.
        if self._showing_detail:
            self.query_one("#detail-title", Static).update("")
            self.query_one("#pen-detail-table", StatusTable).set_data([])
            self._showing_detail = False

    def action_show_detail(self) -> None:
        """Toggle per-pen detail for the first exam entry."""
        title = self.query_one("#detail-title", Static)
        table = self.query_one("#pen-detail-table", StatusTable)

        if self._showing_detail:
            title.update("")
            table.set_data([])
            self._showing_detail = False
            return

        if not self._exam_rows:
            title.update("[dim]No exams to show detail for[/dim]")
            return

        exam_id = self._exam_rows[0][0]
        pen_rows = _query_pen_detail(self._db_path, exam_id)
        title.update(f"Per-Pen Breakdown: {exam_id}")
        table.set_data(pen_rows, columns=_PEN_DETAIL_COLS, status_column=2)
        self._showing_detail = True

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
