"""Billing & payment tracking screen."""

from datetime import datetime

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


PAYMENT_METHODS = [
    ("Bank Transfer", "bank_transfer"),
    ("Credit Card", "credit_card"),
    ("UPI", "upi"),
    ("Cash", "cash"),
    ("Other", "other"),
]


class BillingScreen(Screen):
    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._superadmins: list = []
        self._selected_sa_id: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" Billing & Payments ", classes="screen-title")
        with Horizontal(id="billing-layout"):
            with Vertical(id="billing-left"):
                yield Static("Select Super-Admin:", classes="field-label")
                yield Select([], id="billing-sa-select")
                yield Static("", id="billing-summary-panel")
            with VerticalScroll(id="billing-right"):
                yield Static("Payment History", classes="field-label")
                yield DataTable(id="payments-table")
                yield Static("--- Record Payment ---", classes="field-label")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Amount:")
                        yield Input(placeholder="0.00", id="pay-amount", type="number")
                    with Vertical():
                        yield Label("Method:")
                        yield Select(PAYMENT_METHODS, id="pay-method", value="bank_transfer")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Reference:")
                        yield Input(placeholder="TXN-001", id="pay-reference")
                    with Vertical():
                        yield Label("Date (YYYY-MM-DD):")
                        yield Input(placeholder="", id="pay-date")
                with Horizontal(classes="form-row"):
                    with Vertical():
                        yield Label("Notes:")
                        yield Input(placeholder="", id="pay-notes")
                yield Button("Record Payment", id="record-payment", variant="primary")
                yield Static("", id="billing-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#payments-table", DataTable)
        table.add_columns("Date", "Amount", "Method", "Reference", "Notes")
        self.load_superadmins()

    def action_refresh(self) -> None:
        if self._selected_sa_id:
            self.load_billing(self._selected_sa_id)
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
            select = self.query_one("#billing-sa-select", Select)
            select.set_options(options)
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "billing-sa-select" or event.value is Select.BLANK:
            return
        self._selected_sa_id = str(event.value)
        self.load_billing(self._selected_sa_id)

    @work(thread=True)
    def load_billing(self, sa_id: str) -> None:
        db = self.app.db  # type: ignore
        summary = db.get_billing_summary(sa_id)
        payments = db.list_payments(sa_id, limit=100)
        self.app.call_from_thread(self._populate_billing, summary, payments)

    def _populate_billing(self, summary: dict, payments: list) -> None:
        cs = summary.get("currency_symbol", "$")
        cycle = summary.get("billing_cycle", "monthly")
        text = (
            f"  Billing Cycle: {cycle.upper()}\n"
            f"  Billing Day: {summary.get('billing_day', 1)}\n"
            f"  Period: {summary.get('period_start', '')} to {summary.get('period_end', '')}\n"
            f"  \n"
            f"  Period Cost:  {cs}{summary.get('current_period_cost', 0):.2f}\n"
            f"  Paid:         {cs}{summary.get('paid_this_period', 0):.2f}\n"
            f"  Balance Due:  {cs}{summary.get('balance_due', 0):.2f}\n"
            f"  \n"
            f"  Next Due:     {summary.get('next_due_date', '')}\n"
            f"  All-Time Paid:{cs}{summary.get('total_paid_all_time', 0):.2f}"
        )
        try:
            self.query_one("#billing-summary-panel", Static).update(text)
        except Exception:
            pass

        table = self.query_one("#payments-table", DataTable)
        table.clear()
        for p in payments:
            pay_date = p.get("payment_date", "")
            if isinstance(pay_date, str) and len(pay_date) > 10:
                pay_date = pay_date[:10]
            table.add_row(
                str(pay_date),
                f"{cs}{p.get('amount', 0):.2f}",
                p.get("payment_method", ""),
                p.get("reference", ""),
                p.get("notes", "")[:30],
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "record-payment":
            self.do_record_payment()

    @work(thread=True)
    def do_record_payment(self) -> None:
        if not self._selected_sa_id:
            self.app.call_from_thread(self._set_status, "Select a super-admin first.")
            return

        try:
            amount = float(self.query_one("#pay-amount", Input).value)
        except (ValueError, Exception):
            self.app.call_from_thread(self._set_status, "Invalid amount.")
            return

        if amount <= 0:
            self.app.call_from_thread(self._set_status, "Amount must be greater than 0.")
            return

        method = "bank_transfer"
        try:
            method = str(self.query_one("#pay-method", Select).value)
        except Exception:
            pass

        reference = ""
        try:
            reference = self.query_one("#pay-reference", Input).value
        except Exception:
            pass

        notes = ""
        try:
            notes = self.query_one("#pay-notes", Input).value
        except Exception:
            pass

        pay_date = None
        try:
            date_str = self.query_one("#pay-date", Input).value.strip()
            if date_str:
                pay_date = datetime.fromisoformat(date_str)
        except Exception:
            pass

        db = self.app.db  # type: ignore
        db.record_payment(
            self._selected_sa_id,
            amount=amount,
            payment_method=method,
            reference=reference,
            notes=notes,
            payment_date=pay_date,
        )

        # Clear form
        self.app.call_from_thread(self._clear_form)
        self.app.call_from_thread(self._set_status, f"Payment of {amount:.2f} recorded!")

        # Reload
        self.load_billing(self._selected_sa_id)

    def _clear_form(self) -> None:
        for field_id in ("pay-amount", "pay-reference", "pay-date", "pay-notes"):
            try:
                self.query_one(f"#{field_id}", Input).value = ""
            except Exception:
                pass

    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#billing-status", Static).update(msg)
        except Exception:
            pass
