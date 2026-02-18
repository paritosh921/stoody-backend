"""Pricing configuration screen."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)
from textual import work


CURRENCIES = [("USD ($)", "USD"), ("EUR (€)", "EUR"), ("INR (₹)", "INR")]
BILLING_CYCLES = [("Monthly", "monthly"), ("Annual", "annual")]
TIER_NAMES = ["core", "advanced", "max"]
ROLES = ["student", "tutor", "admin"]
PERIODS = ["monthly", "annual"]


class PricingScreen(Screen):
    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._superadmins: list = []
        self._selected_sa_id: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" Pricing Configuration ", classes="screen-title")
        with Horizontal(id="pricing-layout"):
            with Vertical(id="pricing-left"):
                yield Static("Select Super-Admin:", classes="field-label")
                yield DataTable(id="sa-picker")
            with VerticalScroll(id="pricing-right"):
                yield Static("Tiered Pricing Rates", classes="field-label")
                yield Static("Currency:")
                yield Select(CURRENCIES, id="currency-select", value="USD")

                # --- Core Tier ---
                yield Static("--- Core Tier ---", classes="field-label")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Student Monthly:")
                        yield Input(placeholder="0.50", id="core-student-monthly", type="number")
                    with Vertical():
                        yield Label("Student Annual:")
                        yield Input(placeholder="5.00", id="core-student-annual", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Tutor Monthly:")
                        yield Input(placeholder="2.00", id="core-tutor-monthly", type="number")
                    with Vertical():
                        yield Label("Tutor Annual:")
                        yield Input(placeholder="20.00", id="core-tutor-annual", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Admin Monthly:")
                        yield Input(placeholder="10.00", id="core-admin-monthly", type="number")
                    with Vertical():
                        yield Label("Admin Annual:")
                        yield Input(placeholder="100.00", id="core-admin-annual", type="number")

                # --- Advanced Tier ---
                yield Static("--- Advanced Tier ---", classes="field-label")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Student Monthly:")
                        yield Input(placeholder="1.00", id="advanced-student-monthly", type="number")
                    with Vertical():
                        yield Label("Student Annual:")
                        yield Input(placeholder="10.00", id="advanced-student-annual", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Tutor Monthly:")
                        yield Input(placeholder="4.00", id="advanced-tutor-monthly", type="number")
                    with Vertical():
                        yield Label("Tutor Annual:")
                        yield Input(placeholder="40.00", id="advanced-tutor-annual", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Admin Monthly:")
                        yield Input(placeholder="15.00", id="advanced-admin-monthly", type="number")
                    with Vertical():
                        yield Label("Admin Annual:")
                        yield Input(placeholder="150.00", id="advanced-admin-annual", type="number")

                # --- Max Tier ---
                yield Static("--- Max Tier ---", classes="field-label")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Student Monthly:")
                        yield Input(placeholder="2.00", id="max-student-monthly", type="number")
                    with Vertical():
                        yield Label("Student Annual:")
                        yield Input(placeholder="20.00", id="max-student-annual", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Tutor Monthly:")
                        yield Input(placeholder="8.00", id="max-tutor-monthly", type="number")
                    with Vertical():
                        yield Label("Tutor Annual:")
                        yield Input(placeholder="80.00", id="max-tutor-annual", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Admin Monthly:")
                        yield Input(placeholder="25.00", id="max-admin-monthly", type="number")
                    with Vertical():
                        yield Label("Admin Annual:")
                        yield Input(placeholder="250.00", id="max-admin-annual", type="number")

                # --- Super-Admin Fee ---
                yield Static("--- Super-Admin Fee ---", classes="field-label")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Monthly:")
                        yield Input(placeholder="100.00", id="sa-fee-monthly", type="number")
                    with Vertical():
                        yield Label("Annual:")
                        yield Input(placeholder="1000.00", id="sa-fee-annual", type="number")

                # --- Billing ---
                yield Static("--- Billing ---", classes="field-label")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Cycle:")
                        yield Select(BILLING_CYCLES, id="billing-cycle-select", value="monthly")
                    with Vertical():
                        yield Label("Billing Day (1-28):")
                        yield Input(placeholder="1", id="billing-day", type="integer")

                yield Button("Save Pricing", id="save-pricing", variant="primary")
                yield Static("", id="pricing-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#sa-picker", DataTable)
        table.add_columns("Name", "Email")
        table.cursor_type = "row"
        self.load_superadmins()

    def action_refresh(self) -> None:
        self.load_superadmins()

    @work(thread=True)
    def load_superadmins(self) -> None:
        db = self.app.db  # type: ignore
        rows = db.list_superadmins()
        self.app.call_from_thread(self._populate_sa, rows)

    def _populate_sa(self, rows: list) -> None:
        self._superadmins = rows
        table = self.query_one("#sa-picker", DataTable)
        table.clear()
        for sa in rows:
            table.add_row(sa.get("name", ""), sa.get("email", ""), key=sa["_id"])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "sa-picker":
            return
        row_key = str(event.row_key.value)
        self._selected_sa_id = row_key
        self.load_pricing(row_key)

    @work(thread=True)
    def load_pricing(self, sa_id: str) -> None:
        db = self.app.db  # type: ignore
        pricing = db.get_pricing(sa_id)
        self.app.call_from_thread(self._fill_form, pricing)

    def _fill_form(self, pricing: dict) -> None:
        try:
            self.query_one("#currency-select", Select).value = pricing.get("currency", "USD")
        except Exception:
            pass

        tiers = pricing.get("tiers", {})
        for tier_name in TIER_NAMES:
            tier_data = tiers.get(tier_name, {})
            for role in ROLES:
                for period in PERIODS:
                    field_id = f"{tier_name}-{role}-{period}"
                    val = tier_data.get(f"{role}_{period}", 0)
                    try:
                        self.query_one(f"#{field_id}", Input).value = str(val)
                    except Exception:
                        pass

        sa_fee = pricing.get("superadmin_fee", {})
        try:
            self.query_one("#sa-fee-monthly", Input).value = str(sa_fee.get("monthly", 100.0))
        except Exception:
            pass
        try:
            self.query_one("#sa-fee-annual", Input).value = str(sa_fee.get("annual", 1000.0))
        except Exception:
            pass

        try:
            self.query_one("#billing-cycle-select", Select).value = pricing.get("billing_cycle", "monthly")
        except Exception:
            pass
        try:
            self.query_one("#billing-day", Input).value = str(pricing.get("billing_day", 1))
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-pricing":
            self.save_pricing()

    @work(thread=True)
    def save_pricing(self) -> None:
        if not self._selected_sa_id:
            self.app.call_from_thread(self._set_status, "Select a super-admin first.")
            return

        def _get_val(field_id: str) -> float:
            try:
                return float(self.query_one(f"#{field_id}", Input).value)
            except (ValueError, Exception):
                return 0.0

        currency = "USD"
        try:
            currency = str(self.query_one("#currency-select", Select).value)
        except Exception:
            pass

        tiers = {}
        for tier_name in TIER_NAMES:
            tier_data = {}
            for role in ROLES:
                for period in PERIODS:
                    field_id = f"{tier_name}-{role}-{period}"
                    tier_data[f"{role}_{period}"] = _get_val(field_id)
            tiers[tier_name] = tier_data

        billing_cycle = "monthly"
        try:
            billing_cycle = str(self.query_one("#billing-cycle-select", Select).value)
        except Exception:
            pass

        billing_day = 1
        try:
            billing_day = int(float(self.query_one("#billing-day", Input).value))
            billing_day = max(1, min(28, billing_day))
        except (ValueError, Exception):
            pass

        fields = {
            "currency": currency,
            "tiers": tiers,
            "superadmin_fee": {
                "monthly": _get_val("sa-fee-monthly"),
                "annual": _get_val("sa-fee-annual"),
            },
            "billing_cycle": billing_cycle,
            "billing_day": billing_day,
        }

        db = self.app.db  # type: ignore
        db.upsert_pricing(self._selected_sa_id, fields)
        self.app.call_from_thread(self._set_status, "Pricing saved successfully!")

    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#pricing-status", Static).update(msg)
        except Exception:
            pass
