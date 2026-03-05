"""Diagnostics screen - inspect desktop diagnostics reports."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static


class DiagnosticsScreen(Screen):
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("enter", "inspect_selected", "Inspect"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="diagnostics-container"):
            yield Static(" Desktop Diagnostics ", classes="screen-title")
            yield DataTable(id="diag-table")
            yield Static(" Select a row and press Enter to inspect archive details.", id="diag-status-bar")
            yield Static("", id="diag-details")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#diag-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Uploaded",
            "Ticket",
            "User",
            "Tenant",
            "DB",
            "Size(MB)",
            "App",
            "Pen",
        )
        self._rows: list[dict] = []
        self._detail_cache: dict[str, str] = {}
        self._loading_key: str | None = None
        self.load_data()

    def _set_status(self, message: str) -> None:
        self.query_one("#diag-status-bar", Static).update(f" {message}")

    def _set_details(self, text: str) -> None:
        # Render raw diagnostic text as plain text (no Rich markup parsing).
        self.query_one("#diag-details", Static).update(Text(text))

    def action_refresh(self) -> None:
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        rows = db.list_diagnostics_reports(limit=400)
        self.app.call_from_thread(self._populate, rows)

    def _populate(self, rows: list[dict]) -> None:
        table = self.query_one("#diag-table", DataTable)
        table.clear()
        self._rows = rows
        for row in rows:
            uploaded_at = row.get("uploaded_at")
            if isinstance(uploaded_at, datetime):
                uploaded = uploaded_at.strftime("%Y-%m-%d %H:%M")
            else:
                uploaded = str(uploaded_at or "")
            size_mb = (float(row.get("size_bytes", 0) or 0.0)) / (1024.0 * 1024.0)
            table.add_row(
                uploaded,
                row.get("ticket_id", ""),
                str(row.get("username") or row.get("user_id") or ""),
                str(row.get("tenant_name") or row.get("tenant_id") or ""),
                str(row.get("db_name") or ""),
                f"{size_mb:.2f}",
                str(row.get("app_version") or ""),
                str(row.get("pen_mac") or ""),
            )
        if rows:
            self._set_status(f"Loaded {len(rows)} diagnostics reports.")
            self.action_inspect_selected()
        else:
            self._set_status("No diagnostics reports found.")
            self._set_details(" No diagnostics reports available.")

    def _selected_row(self) -> dict | None:
        table = self.query_one("#diag-table", DataTable)
        if table.cursor_row is None or table.cursor_row < 0:
            return None
        if table.cursor_row >= len(self._rows):
            return None
        return self._rows[table.cursor_row]

    def action_inspect_selected(self) -> None:
        row = self._selected_row()
        if not row:
            self._set_status("No report selected.")
            return
        key = f"{row.get('db_name')}::{row.get('_id')}"
        if key in self._detail_cache:
            self._set_details(self._detail_cache[key])
            self._set_status(f"Showing cached details for {row.get('ticket_id', '')}.")
            return
        if self._loading_key == key:
            return
        self._loading_key = key
        self._set_details(" Loading report details...")
        self._set_status(f"Inspecting {row.get('ticket_id', '')} ...")
        self._load_details(row)

    @work(thread=True)
    def _load_details(self, row: dict) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        key = f"{row.get('db_name')}::{row.get('_id')}"
        try:
            detail = db.get_diagnostics_report_details(
                report_id=row["_id"],
                db_name=row["db_name"],
            )
            text = detail.get("pretty_text", "No detail.")
            status = f"Loaded details for {row.get('ticket_id', '')}."
        except Exception as exc:
            text = f"Failed to load report details:\n{exc}"
            status = f"Failed to inspect {row.get('ticket_id', '')}."
        self.app.call_from_thread(self._apply_details, key, text, status)

    def _apply_details(self, key: str, text: str, status: str) -> None:
        self._loading_key = None
        self._detail_cache[key] = text
        self._set_details(text)
        self._set_status(status)
