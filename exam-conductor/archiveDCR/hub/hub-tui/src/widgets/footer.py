"""Hub info footer widget — displays hub ID, state, uptime, and IP."""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class HubFooter(Widget):
    """Persistent footer showing hub identification and status summary."""

    DEFAULT_CSS = """
    HubFooter {
        dock: bottom;
        height: 2;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    HubFooter .footer-line {
        width: 1fr;
    }
    """

    hub_id: reactive[str] = reactive("EPH-XXXXX")
    state: reactive[str] = reactive("INIT")
    uptime: reactive[str] = reactive("0h 0m")
    ip_addr: reactive[str] = reactive("---.---.---.---")

    def compose(self) -> ComposeResult:
        yield Static(self._render_line(), id="footer-text", classes="footer-line")

    def _render_line(self) -> str:
        return (
            f"  Hub ID: {self.hub_id}  "
            f"State: {self.state}  "
            f"Uptime: {self.uptime}  "
            f"IP: {self.ip_addr}"
        )

    def watch_hub_id(self) -> None:
        self._refresh_text()

    def watch_state(self) -> None:
        self._refresh_text()

    def watch_uptime(self) -> None:
        self._refresh_text()

    def watch_ip_addr(self) -> None:
        self._refresh_text()

    def _refresh_text(self) -> None:
        try:
            self.query_one("#footer-text", Static).update(self._render_line())
        except Exception:
            pass
