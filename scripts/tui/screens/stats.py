"""Stats screen - per-super-admin summary and per-tier breakdown."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static
from textual import work


class StatsScreen(Screen):
    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="stats-container"):
            yield Static(" Aggregated Stats ", classes="screen-title")
            yield Static("Per Super-Admin Summary", classes="field-label")
            yield DataTable(id="sa-summary-table")
            yield Static("Per Tier Breakdown", classes="field-label")
            yield DataTable(id="tier-table")
        yield Footer()

    def on_mount(self) -> None:
        sa_table = self.query_one("#sa-summary-table", DataTable)
        sa_table.add_columns("Name", "Email", "Tenants", "Students", "Tutors", "Admins", "Total Cost")

        tier_table = self.query_one("#tier-table", DataTable)
        tier_table.add_columns("Tier", "Tenant Count", "Avg Cost")

        self.load_data()

    def action_refresh(self) -> None:
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        db = self.app.db  # type: ignore
        superadmins = db.list_superadmins()

        sa_rows = []
        tier_stats: dict = {}

        for sa in superadmins:
            costs = db.compute_costs_for_superadmin(sa["_id"])
            total_students = sum(tc["student_count"] for tc in costs["tenant_costs"])
            total_tutors = sum(tc["tutor_count"] for tc in costs["tenant_costs"])
            total_admins = sum(tc["admin_count"] for tc in costs["tenant_costs"])

            cs = costs["pricing"].get("currency_symbol", "$")
            sa_rows.append({
                "name": sa.get("name", ""),
                "email": sa.get("email", ""),
                "tenants": len(costs["tenant_costs"]),
                "students": total_students,
                "tutors": total_tutors,
                "admins": total_admins,
                "total_cost": f"{cs}{costs['total_platform_cost']:.2f}",
            })

            for tc in costs["tenant_costs"]:
                tier = tc["tier"]
                if tier not in tier_stats:
                    tier_stats[tier] = {"count": 0, "total_cost": 0.0}
                tier_stats[tier]["count"] += 1
                tier_stats[tier]["total_cost"] += tc["total_cost"]

        tier_rows = []
        for tier, data in sorted(tier_stats.items()):
            avg = data["total_cost"] / data["count"] if data["count"] else 0
            tier_rows.append({
                "tier": tier,
                "count": data["count"],
                "avg_cost": f"${avg:.2f}",
            })

        self.app.call_from_thread(self._populate, sa_rows, tier_rows)

    def _populate(self, sa_rows: list, tier_rows: list) -> None:
        sa_table = self.query_one("#sa-summary-table", DataTable)
        sa_table.clear()
        for r in sa_rows:
            sa_table.add_row(
                r["name"], r["email"],
                str(r["tenants"]), str(r["students"]),
                str(r["tutors"]), str(r["admins"]),
                r["total_cost"],
            )

        tier_table = self.query_one("#tier-table", DataTable)
        tier_table.clear()
        for r in tier_rows:
            tier_table.add_row(r["tier"], str(r["count"]), r["avg_cost"])
