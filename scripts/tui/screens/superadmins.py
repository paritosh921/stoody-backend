"""Super-Admin management screen."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static
from textual.worker import work


class SuperAdminScreen(Screen):
    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" Super-Admins ", classes="screen-title")
        yield DataTable(id="sa-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#sa-table", DataTable)
        table.add_columns("Name", "Email", "Auth Code", "Active", "Tenants", "Last Login")
        self.load_data()

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
        for sa in rows:
            last_login = sa.get("last_login", "")
            if hasattr(last_login, "strftime"):
                last_login = last_login.strftime("%Y-%m-%d %H:%M")
            table.add_row(
                sa.get("name", ""),
                sa.get("email", ""),
                sa.get("authorization_code", ""),
                "Yes" if sa.get("is_active") else "No",
                str(sa.get("tenant_count", 0)),
                str(last_login or "Never"),
            )
