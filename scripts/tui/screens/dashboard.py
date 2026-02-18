"""Dashboard screen - platform overview stats."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from textual.worker import Worker, work


class StatCard(Static):
    """A simple stat display card."""

    def __init__(self, label: str, value: str = "...", **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="stat-label")
        yield Static(self._value, classes="stat-value", id=f"val-{self.id}")

    def update_value(self, value: str) -> None:
        self._value = value
        try:
            self.query_one(f"#val-{self.id}").update(value)
        except Exception:
            pass


class DashboardScreen(Screen):
    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="dashboard-container"):
            yield Static(" Platform Dashboard ", classes="screen-title")
            with Horizontal(classes="stat-row"):
                yield StatCard("Super-Admins", id="sa-count")
                yield StatCard("Total Tenants", id="tenant-count")
                yield StatCard("Active Tenants", id="active-count")
                yield StatCard("Total Students", id="student-count")
                yield StatCard("Total Tutors", id="tutor-count")
        yield Footer()

    def on_mount(self) -> None:
        self.load_data()

    def action_refresh(self) -> None:
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        db = self.app.db  # type: ignore
        stats = db.get_aggregate_stats()
        self.app.call_from_thread(self._apply_stats, stats)

    def _apply_stats(self, stats: dict) -> None:
        mapping = {
            "sa-count": str(stats["superadmin_count"]),
            "tenant-count": str(stats["tenant_count"]),
            "active-count": str(stats["active_tenants"]),
            "student-count": str(stats["total_students"]),
            "tutor-count": str(stats["total_tutors"]),
        }
        for widget_id, value in mapping.items():
            try:
                card = self.query_one(f"#{widget_id}", StatCard)
                card.update_value(value)
            except Exception:
                pass
