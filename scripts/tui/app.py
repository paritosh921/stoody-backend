"""
Skiller Platform TUI - Terminal management interface.

Run: cd backend && python -m scripts.tui.app
"""

import os
import sys
from pathlib import Path

# Ensure backend/ is on sys.path so dotenv can find .env
_backend_dir = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from .db import DB
from .screens.billing import BillingScreen
from .screens.bug_reports import BugReportsScreen
from .screens.dashboard import DashboardScreen
from .screens.diagnostics import DiagnosticsScreen
from .screens.pricing import PricingScreen
from .screens.stats import StatsScreen
from .screens.superadmins import SuperAdminScreen
from .screens.tenants import TenantOverviewScreen

CSS_PATH = str(Path(__file__).parent / "styles.tcss")


class SkillerTUI(App):
    """Skiller Platform Management TUI."""

    TITLE = "Skiller Platform Manager"
    SUB_TITLE = "d=Dashboard  s=Super-Admins  p=Pricing  t=Tenants  a=Stats  b=Billing  i=Diagnostics  m=Messages  q=Quit"
    CSS_PATH = CSS_PATH

    BINDINGS = [
        ("d", "switch_screen('dashboard')", "Dashboard"),
        ("s", "switch_screen('superadmins')", "Super-Admins"),
        ("p", "switch_screen('pricing')", "Pricing"),
        ("t", "switch_screen('tenants')", "Tenants"),
        ("a", "switch_screen('stats')", "Stats"),
        ("b", "switch_screen('billing')", "Billing"),
        ("i", "switch_screen('diagnostics')", "Diagnostics"),
        ("m", "switch_screen('bug_reports')", "Messages"),
        ("q", "quit", "Quit"),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "superadmins": SuperAdminScreen,
        "pricing": PricingScreen,
        "tenants": TenantOverviewScreen,
        "stats": StatsScreen,
        "billing": BillingScreen,
        "diagnostics": DiagnosticsScreen,
        "bug_reports": BugReportsScreen,
    }

    def __init__(self) -> None:
        super().__init__()
        self.db = DB()

    def on_mount(self) -> None:
        self.push_screen("dashboard")

    def on_unmount(self) -> None:
        self.db.close()


def main():
    app = SkillerTUI()
    app.run()


if __name__ == "__main__":
    main()
