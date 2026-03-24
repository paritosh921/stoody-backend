"""Main menu screen — hub TUI landing page with 8 navigation options."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.containers import Vertical
from textual.widgets import Header, Static

from src.widgets.footer import HubFooter

MENU_ITEMS = [
    ("1", "Setup", "Initial configuration"),
    ("2", "Status", "Live dashboard"),
    ("3", "WiFi", "Network configuration"),
    ("4", "Dongles", "BLE dongle management"),
    ("5", "Exams", "Session history"),
    ("6", "Diagnostics", "Test suite runner"),
    ("7", "Logs", "Log viewer"),
    ("8", "Shutdown", "Safe power off"),
]


class MenuScreen(Screen):
    """Main menu with numbered navigation to each sub-screen."""

    BINDINGS = [
        Binding("1", "goto('setup')", "Setup", show=False),
        Binding("2", "goto('status')", "Status", show=False),
        Binding("3", "goto('wifi')", "WiFi", show=False),
        Binding("4", "goto('dongles')", "Dongles", show=False),
        Binding("5", "goto('exams')", "Exams", show=False),
        Binding("6", "goto('diagnostics')", "Diagnostics", show=False),
        Binding("7", "goto('logs')", "Logs", show=False),
        Binding("8", "goto('shutdown')", "Shutdown", show=False),
    ]

    DEFAULT_CSS = """
    MenuScreen {
        align: center middle;
    }
    MenuScreen #menu-container {
        width: 50;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }
    MenuScreen .menu-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    MenuScreen .menu-item {
        margin: 0 1;
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="menu-container"):
            yield Static("ExamPen Hub TUI", classes="menu-title")
            yield Static("")
            for key, name, desc in MENU_ITEMS:
                yield Static(f"  [{key}] {name:<16}{desc}", classes="menu-item")
            yield Static("")
        yield HubFooter()

    def action_goto(self, screen_name: str) -> None:
        self.app.push_screen(screen_name)
