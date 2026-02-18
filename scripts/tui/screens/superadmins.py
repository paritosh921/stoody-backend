"""Super-Admin management screen with lifecycle actions."""

from textual.app import ComposeResult
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
)
from textual.containers import Horizontal, Vertical
from textual import work


STATUS_DISPLAY = {
    "active": "Active",
    "suspended": "Suspended",
    "deactivated": "Deactivated",
}


class ConfirmActionScreen(ModalScreen[dict | None]):
    """Modal dialog for confirming lifecycle actions with optional reason."""

    DEFAULT_CSS = """
    ConfirmActionScreen {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    .confirm-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    .confirm-body {
        height: auto;
        margin: 0 0 1 0;
    }

    #reason-input {
        margin: 0 0 1 0;
    }

    .confirm-buttons {
        height: auto;
        align: center middle;
    }

    .confirm-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        action: str,
        sa_name: str,
        sa_email: str,
        needs_reason: bool = False,
    ) -> None:
        super().__init__()
        self.action = action
        self.sa_name = sa_name
        self.sa_email = sa_email
        self.needs_reason = needs_reason

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(f" {self.action.title()} Super-Admin ", classes="confirm-title")
            yield Static(
                f"Are you sure you want to [bold]{self.action}[/bold] "
                f"[cyan]{self.sa_name}[/cyan] ({self.sa_email})?",
                classes="confirm-body",
            )
            if self.needs_reason:
                yield Label("Reason (optional):")
                yield Input(id="reason-input", placeholder="Enter reason...")
            with Horizontal(classes="confirm-buttons"):
                yield Button("Confirm", variant="error", id="btn-confirm")
                yield Button("Cancel", variant="primary", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm":
            reason = ""
            if self.needs_reason:
                reason = self.query_one("#reason-input", Input).value
            self.dismiss({"action": self.action, "reason": reason})
        else:
            self.dismiss(None)


class SuperAdminScreen(Screen):
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("s", "suspend", "Suspend"),
        ("a", "activate", "Activate"),
        ("x", "deactivate", "Deactivate"),
        ("delete", "delete_sa", "Delete"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" Super-Admins ", classes="screen-title")
        yield DataTable(id="sa-table")
        yield Static("", id="sa-status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#sa-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Name", "Email", "Auth Code", "Status", "Tenants", "Last Login")
        self._sa_rows: list = []
        self.load_data()

    def _set_status(self, msg: str) -> None:
        bar = self.query_one("#sa-status-bar", Static)
        bar.update(f" {msg}")

    def action_refresh(self) -> None:
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        db = self.app.db  # type: ignore
        rows = db.list_superadmins()
        self.app.call_from_thread(self._populate, rows)

    def _populate(self, rows: list) -> None:
        table = self.query_one("#sa-table", DataTable)
        table.clear()
        self._sa_rows = rows
        for sa in rows:
            last_login = sa.get("last_login", "")
            if hasattr(last_login, "strftime"):
                last_login = last_login.strftime("%Y-%m-%d %H:%M")
            sa_status = sa.get("status", "active")
            table.add_row(
                sa.get("name", ""),
                sa.get("email", ""),
                sa.get("authorization_code", ""),
                STATUS_DISPLAY.get(sa_status, sa_status),
                str(sa.get("tenant_count", 0)),
                str(last_login or "Never"),
            )

    def _get_selected_sa(self) -> dict | None:
        table = self.query_one("#sa-table", DataTable)
        if table.cursor_row is None or table.cursor_row < 0:
            self._set_status("No row selected.")
            return None
        if table.cursor_row >= len(self._sa_rows):
            self._set_status("No row selected.")
            return None
        return self._sa_rows[table.cursor_row]

    # ---- Lifecycle actions ----

    def action_suspend(self) -> None:
        sa = self._get_selected_sa()
        if not sa:
            return
        if sa.get("status") == "suspended":
            self._set_status("Already suspended.")
            return
        self.app.push_screen(
            ConfirmActionScreen("suspend", sa.get("name", ""), sa["email"], needs_reason=True),
            callback=self._on_confirm,
        )

    def action_activate(self) -> None:
        sa = self._get_selected_sa()
        if not sa:
            return
        if sa.get("status") == "active":
            self._set_status("Already active.")
            return
        self.app.push_screen(
            ConfirmActionScreen("activate", sa.get("name", ""), sa["email"]),
            callback=self._on_confirm,
        )

    def action_deactivate(self) -> None:
        sa = self._get_selected_sa()
        if not sa:
            return
        if sa.get("status") == "deactivated":
            self._set_status("Already deactivated.")
            return
        self.app.push_screen(
            ConfirmActionScreen("deactivate", sa.get("name", ""), sa["email"], needs_reason=True),
            callback=self._on_confirm,
        )

    def action_delete_sa(self) -> None:
        sa = self._get_selected_sa()
        if not sa:
            return
        self.app.push_screen(
            ConfirmActionScreen("delete", sa.get("name", ""), sa["email"], needs_reason=False),
            callback=self._on_confirm,
        )

    def _on_confirm(self, result: dict | None) -> None:
        if result is None:
            self._set_status("Cancelled.")
            return
        sa = self._get_selected_sa()
        if not sa:
            return
        action = result["action"]
        reason = result.get("reason", "")
        self._execute_action(sa["_id"], action, reason)

    @work(thread=True)
    def _execute_action(self, sa_id: str, action: str, reason: str) -> None:
        db = self.app.db  # type: ignore
        try:
            if action == "suspend":
                res = db.suspend_superadmin(sa_id, reason)
                msg = f"Suspended. {res['tenants_affected']} tenant(s) platform-suspended."
            elif action == "activate":
                res = db.activate_superadmin(sa_id)
                msg = f"Activated. {res['tenants_affected']} tenant(s) platform access restored."
            elif action == "deactivate":
                res = db.deactivate_superadmin(sa_id, reason)
                msg = f"Deactivated. {res['tenants_affected']} tenant(s) platform-suspended."
            elif action == "delete":
                res = db.delete_superadmin(sa_id)
                msg = f"Deleted. {res['tenants_orphaned']} tenant(s) orphaned."
            else:
                msg = f"Unknown action: {action}"
        except Exception as exc:
            msg = f"Error: {exc}"

        self.app.call_from_thread(self._set_status, msg)
        # Reload data
        db2 = self.app.db  # type: ignore
        rows = db2.list_superadmins()
        self.app.call_from_thread(self._populate, rows)
