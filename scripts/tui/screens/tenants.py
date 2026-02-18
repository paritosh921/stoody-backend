"""Tenant cost overview screen - Excel-like cost DataTable."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Select, Static
from textual.worker import work

from ..widgets.cost_table import CostTable


class TenantOverviewScreen(Screen):
    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._superadmins: list = []
        self._selected_sa_id: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="tenants-container"):
            yield Static(" Tenant Cost Overview ", classes="screen-title")
            yield Static("Select Super-Admin:", classes="field-label")
            yield Select([], id="sa-select")
            yield CostTable(id="cost-table")
            yield Static("", id="totals-line")
        yield Footer()

    def on_mount(self) -> None:
        self.load_superadmins()

    def action_refresh(self) -> None:
        if self._selected_sa_id:
            self.load_costs(self._selected_sa_id)
        else:
            self.load_superadmins()

    @work(thread=True)
    def load_superadmins(self) -> None:
        db = self.app.db  # type: ignore
        rows = db.list_superadmins()
        self.app.call_from_thread(self._populate_select, rows)

    def _populate_select(self, rows: list) -> None:
        self._superadmins = rows
        options = [(f"{sa['name']} ({sa['email']})", sa["_id"]) for sa in rows]
        try:
            select = self.query_one("#sa-select", Select)
            select.set_options(options)
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "sa-select" or event.value is Select.BLANK:
            return
        self._selected_sa_id = str(event.value)
        self.load_costs(self._selected_sa_id)

    @work(thread=True)
    def load_costs(self, sa_id: str) -> None:
        db = self.app.db  # type: ignore
        result = db.compute_costs_for_superadmin(sa_id)
        self.app.call_from_thread(self._populate_table, result)

    def _populate_table(self, result: dict) -> None:
        cs = result["pricing"].get("currency_symbol", "$")
        table = self.query_one("#cost-table", CostTable)
        table.load_costs(result["tenant_costs"], cs)

        totals = (
            f"  Tenants: {cs}{result['total_tenants_cost']:.2f}"
            f"  |  Base Fee: {cs}{result['superadmin_base_fee']:.2f}"
            f"  |  GRAND TOTAL: {cs}{result['total_platform_cost']:.2f}"
        )
        try:
            self.query_one("#totals-line", Static).update(totals)
        except Exception:
            pass
