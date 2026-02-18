"""Reusable cost breakdown DataTable widget."""

from textual.widgets import DataTable


COST_COLUMNS = [
    ("Institution", 22),
    ("Tier", 8),
    ("Status", 10),
    ("Students", 9),
    ("Tutors", 7),
    ("Admins", 7),
    ("Tier Fee", 10),
    ("Stu Cost", 10),
    ("Tut Cost", 10),
    ("Adm Cost", 10),
    ("Total", 12),
]


class CostTable(DataTable):
    """DataTable pre-configured for tenant cost breakdown."""

    def on_mount(self) -> None:
        for label, width in COST_COLUMNS:
            self.add_column(label, width=width)

    def load_costs(self, tenant_costs: list, currency_symbol: str = "$") -> None:
        self.clear()
        cs = currency_symbol
        for tc in tenant_costs:
            self.add_row(
                tc["institution_name"][:20],
                tc["tier"],
                tc.get("status", ""),
                str(tc["student_count"]),
                str(tc["tutor_count"]),
                str(tc["admin_count"]),
                f"{cs}{tc['flat_fee']:.2f}",
                f"{cs}{tc['student_surcharge']:.2f}",
                f"{cs}{tc['tutor_surcharge']:.2f}",
                f"{cs}{tc['admin_surcharge']:.2f}",
                f"{cs}{tc['total_cost']:.2f}",
            )
