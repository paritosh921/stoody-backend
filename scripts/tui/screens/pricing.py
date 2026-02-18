"""Pricing configuration screen."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
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


CURRENCIES = [("USD ($)", "USD"), ("EUR (\u20ac)", "EUR"), ("INR (\u20b9)", "INR")]


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
            with Vertical(id="pricing-right"):
                yield Static("Pricing Details", classes="field-label")
                yield Static("Currency:")
                yield Select(CURRENCIES, id="currency-select", value="USD")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Core Tier:")
                        yield Input(placeholder="50.0", id="tier-core", type="number")
                    with Vertical():
                        yield Label("Advanced Tier:")
                        yield Input(placeholder="120.0", id="tier-advanced", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Max Tier:")
                        yield Input(placeholder="250.0", id="tier-max", type="number")
                    with Vertical():
                        yield Label("Custom Tier:")
                        yield Input(placeholder="200.0", id="tier-custom", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Per Student:")
                        yield Input(placeholder="0.50", id="per-student", type="number")
                    with Vertical():
                        yield Label("Per Tutor:")
                        yield Input(placeholder="2.00", id="per-tutor", type="number")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Per Admin:")
                        yield Input(placeholder="10.00", id="per-admin", type="number")
                    with Vertical():
                        yield Label("Base Fee:")
                        yield Input(placeholder="100.00", id="base-fee", type="number")
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
        tr = pricing.get("tier_rates", {})
        field_map = {
            "tier-core": str(tr.get("core", 50.0)),
            "tier-advanced": str(tr.get("advanced", 120.0)),
            "tier-max": str(tr.get("max", 250.0)),
            "tier-custom": str(tr.get("custom", 200.0)),
            "per-student": str(pricing.get("flat_per_student", 0.5)),
            "per-tutor": str(pricing.get("flat_per_tutor", 2.0)),
            "per-admin": str(pricing.get("flat_per_admin", 10.0)),
            "base-fee": str(pricing.get("superadmin_base_fee", 100.0)),
        }
        for field_id, val in field_map.items():
            try:
                self.query_one(f"#{field_id}", Input).value = val
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

        fields = {
            "currency": currency,
            "tier_rates": {
                "core": _get_val("tier-core"),
                "advanced": _get_val("tier-advanced"),
                "max": _get_val("tier-max"),
                "custom": _get_val("tier-custom"),
            },
            "flat_per_student": _get_val("per-student"),
            "flat_per_tutor": _get_val("per-tutor"),
            "flat_per_admin": _get_val("per-admin"),
            "superadmin_base_fee": _get_val("base-fee"),
        }

        db = self.app.db  # type: ignore
        db.upsert_pricing(self._selected_sa_id, fields)
        self.app.call_from_thread(self._set_status, "Pricing saved successfully!")

    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#pricing-status", Static).update(msg)
        except Exception:
            pass
