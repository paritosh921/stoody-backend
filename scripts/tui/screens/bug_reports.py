"""Desktop bug reports screen - inspect Help-tab messages from agent users."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Static


class BugReportsScreen(Screen):
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("enter", "inspect_selected", "Inspect"),
        ("c", "copy_details", "Copy Full Msg"),
        ("j", "details_down", "Details Down"),
        ("k", "details_up", "Details Up"),
        ("pagedown", "details_page_down", "Details PgDn"),
        ("pageup", "details_page_up", "Details PgUp"),
        ("home", "details_home", "Details Top"),
        ("end", "details_end", "Details End"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="bug-reports-container"):
            yield Static(" Desktop User Messages ", classes="screen-title")
            yield DataTable(id="bug-table")
            yield Static(
                " Select row + Enter to inspect. Press c to copy full message text.",
                id="bug-status-bar",
            )
            with Horizontal(id="bug-actions"):
                yield Button("Copy Full Message", id="bug-copy", variant="primary")
            with VerticalScroll(id="bug-details-pane"):
                yield Static("", id="bug-details")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#bug-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Created",
            "Ticket",
            "User",
            "Tenant",
            "DB",
            "App",
            "Pen",
            "Title",
        )
        self._rows: list[dict] = []
        self._detail_cache: dict[str, str] = {}
        self._loading_key: str | None = None
        self._current_detail_text: str = ""
        self.load_data()

    def _set_status(self, message: str) -> None:
        self.query_one("#bug-status-bar", Static).update(f" {message}")

    def _set_details(self, text: str) -> None:
        self._current_detail_text = text
        self.query_one("#bug-details", Static).update(Text(text))
        try:
            self.query_one("#bug-details-pane", VerticalScroll).scroll_home(animate=False)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "bug-copy":
            self.action_copy_details()

    def action_copy_details(self) -> None:
        text = (self._current_detail_text or "").strip()
        if not text:
            self._set_status("No message details loaded to copy.")
            return
        copy_fn = getattr(self.app, "copy_to_clipboard", None)
        if not callable(copy_fn):
            self._set_status("Clipboard copy not supported in this terminal.")
            return
        try:
            copy_fn(text)
            self._set_status(f"Copied full message text ({len(text)} chars) to clipboard.")
        except Exception as exc:
            self._set_status(f"Copy failed: {exc}")

    def _scroll_details(self, amount: int) -> None:
        try:
            pane = self.query_one("#bug-details-pane", VerticalScroll)
            pane.scroll_relative(y=amount, animate=False)
        except Exception:
            pass

    def action_details_down(self) -> None:
        self._scroll_details(3)

    def action_details_up(self) -> None:
        self._scroll_details(-3)

    def action_details_page_down(self) -> None:
        self._scroll_details(18)

    def action_details_page_up(self) -> None:
        self._scroll_details(-18)

    def action_details_home(self) -> None:
        try:
            self.query_one("#bug-details-pane", VerticalScroll).scroll_home(animate=False)
        except Exception:
            pass

    def action_details_end(self) -> None:
        try:
            self.query_one("#bug-details-pane", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def action_refresh(self) -> None:
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        rows = db.list_desktop_bug_reports(limit=500)
        self.app.call_from_thread(self._populate, rows)

    def _populate(self, rows: list[dict]) -> None:
        table = self.query_one("#bug-table", DataTable)
        table.clear()
        self._rows = rows
        for row in rows:
            created_at = row.get("created_at")
            if isinstance(created_at, datetime):
                created = created_at.strftime("%Y-%m-%d %H:%M")
            else:
                created = str(created_at or "")
            title = str(row.get("title") or "").strip()
            table.add_row(
                created,
                row.get("ticket_id", ""),
                str(row.get("username") or row.get("user_id") or ""),
                str(row.get("tenant_name") or row.get("tenant_id") or ""),
                str(row.get("db_name") or ""),
                str(row.get("app_version") or ""),
                str(row.get("pen_mac") or ""),
                title[:48] + ("..." if len(title) > 48 else ""),
            )
        if rows:
            self._set_status(f"Loaded {len(rows)} desktop user messages.")
            self.action_inspect_selected()
        else:
            self._set_status("No desktop user messages found.")
            self._set_details(" No desktop user messages available.")

    def _selected_row(self) -> dict | None:
        table = self.query_one("#bug-table", DataTable)
        if table.cursor_row is None or table.cursor_row < 0:
            return None
        if table.cursor_row >= len(self._rows):
            return None
        return self._rows[table.cursor_row]

    def action_inspect_selected(self) -> None:
        row = self._selected_row()
        if not row:
            self._set_status("No message selected.")
            return
        key = f"{row.get('db_name')}::{row.get('_id')}"
        if key in self._detail_cache:
            self._set_details(self._detail_cache[key])
            self._set_status(f"Showing cached details for {row.get('ticket_id', '')}.")
            return
        if self._loading_key == key:
            return
        self._loading_key = key
        self._set_details(" Loading message details...")
        self._set_status(f"Inspecting {row.get('ticket_id', '')} ...")
        self._load_details(row)

    @work(thread=True)
    def _load_details(self, row: dict) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        key = f"{row.get('db_name')}::{row.get('_id')}"
        try:
            detail = db.get_desktop_bug_report_details(
                report_id=row["_id"],
                db_name=row["db_name"],
            )
            text = detail.get("pretty_text", "No detail.")
            status = f"Loaded details for {row.get('ticket_id', '')}."
        except Exception as exc:
            text = f"Failed to load message details:\n{exc}"
            status = f"Failed to inspect {row.get('ticket_id', '')}."
        self.app.call_from_thread(self._apply_details, key, text, status)

    def _apply_details(self, key: str, text: str, status: str) -> None:
        self._loading_key = None
        self._detail_cache[key] = text
        self._set_details(text)
        self._set_status(status)

