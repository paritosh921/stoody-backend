"""Live status dashboard — hub state, dongles, sync progress, storage.

Wired to ``HubIpcBridge`` for 1 Hz live updates from hub-supervisor,
hub-ble-mgr, hub-uplink, and hub-store via IPC.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Header, Static

from src.ipc_bridge import HubIpcBridge
from src.widgets.footer import HubFooter
from src.widgets.progress_bar import SyncProgressBar
from src.widgets.status_table import StatusTable

_DONGLE_COLS = ["Dongle", "hci", "MAC", "Pens", "Health"]
_DISCONNECTED = "[red]Disconnected[/red]"


def _fmt_timer(remaining_sec: int) -> str:
    """Format seconds into ``MM:SS remaining``."""
    if remaining_sec <= 0:
        return "--:--"
    mins, secs = divmod(remaining_sec, 60)
    return f"{mins:02d}:{secs:02d} remaining"


def _dongle_rows(dongles: list[dict]) -> list[list[str]]:
    """Convert bridge dongle dicts to table rows."""
    rows: list[list[str]] = []
    for i, d in enumerate(dongles, start=1):
        rows.append([
            d.get("dongle_id", f"D{i}"),
            d.get("hci_path", f"hci{i - 1}"),
            d.get("dongle_mac", "??:??:??:??:??:??"),
            d.get("pens", "0/0"),
            d.get("status", "unknown").upper(),
        ])
    return rows


def _total_pens(dongles: list[dict]) -> str:
    """Sum connected / total from dongle pens field (``'n/m'``)."""
    connected = 0
    total = 0
    for d in dongles:
        pens_str = d.get("pens", "0/0")
        try:
            c, t = pens_str.split("/")
            connected += int(c)
            total += int(t)
        except (ValueError, AttributeError):
            pass
    return f"Total: {connected}/{total} pens connected"


class StatusScreen(Screen):
    """Live dashboard — refreshed at 1 Hz via IPC bridge."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    DEFAULT_CSS = """
    StatusScreen #status-container {
        padding: 1 2;
    }
    StatusScreen .section-title {
        text-style: bold;
        margin-top: 1;
    }
    StatusScreen .info-line {
        margin: 0 1;
    }
    StatusScreen .screen-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        bridge: HubIpcBridge | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._bridge = bridge

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="status-container"):
            yield Static(
                "[2] Hub Status - Live Dashboard", classes="screen-title",
            )
            # Hub state summary (updated live)
            yield Static("State: INIT", id="state-line", classes="info-line")
            yield Static(
                "WiFi: --  Backend: --", id="wifi-line", classes="info-line",
            )
            yield Static(
                "Invigilator: --", id="invig-line", classes="info-line",
            )

            # Dongle table
            yield Static("Dongles", classes="section-title")
            yield StatusTable(
                columns=_DONGLE_COLS,
                rows=[],
                status_column=4,
                id="dongle-table",
            )
            yield Static("Total: 0/0 pens connected", id="pen-total", classes="info-line")

            # Sync progress
            yield Static("Sync Progress", classes="section-title")
            yield SyncProgressBar(id="sync-bar")

            # Storage
            yield Static("Storage", classes="section-title")
            yield Static("SD: --  USB: --", id="storage-line", classes="info-line")

        yield HubFooter()

    def on_mount(self) -> None:
        """Start 1 Hz refresh timer."""
        self.set_interval(1.0, self._refresh)

    # -- refresh -------------------------------------------------------------

    def _refresh(self) -> None:
        """Pull cached state from bridge and update all widgets."""
        bridge = self._bridge
        if bridge is None:
            bridge = getattr(self.app, "_ipc_bridge", None)
        if bridge is None:
            self._show_disconnected()
            return
        self._update_state(bridge)
        self._update_dongles(bridge)
        self._update_sync(bridge)
        self._update_storage(bridge)

    def _show_disconnected(self) -> None:
        self.query_one("#state-line", Static).update(
            f"State: {_DISCONNECTED}",
        )
        self.query_one("#wifi-line", Static).update(
            f"WiFi: {_DISCONNECTED}  Backend: {_DISCONNECTED}",
        )

    def _update_state(self, bridge: HubIpcBridge) -> None:
        snap = bridge.supervisor
        if not snap.connected:
            state_text = f"State: {_DISCONNECTED}"
        else:
            timer = _fmt_timer(snap.timer_remaining_sec)
            state_text = f"State: {snap.state}    Timer: {timer}"
        self.query_one("#state-line", Static).update(state_text)

        wifi = bridge.wifi
        if not wifi.connected:
            wifi_text = f"WiFi: {_DISCONNECTED}  Backend: {_DISCONNECTED}"
        else:
            sig = f"{wifi.signal} dBm" if wifi.signal else "--"
            backend = (
                f"Reachable ({wifi.latency_ms} ms)"
                if wifi.backend_reachable
                else "[red]Unreachable[/red]"
            )
            wifi_text = (
                f"WiFi: {wifi.ssid} ({wifi.band}, Ch {wifi.channel}, {sig})"
                f"  Backend: {backend}"
            )
        self.query_one("#wifi-line", Static).update(wifi_text)

    def _update_dongles(self, bridge: HubIpcBridge) -> None:
        dongles = bridge.dongles
        if not dongles.connected:
            rows = []
        else:
            rows = _dongle_rows(dongles.dongles)
        table = self.query_one("#dongle-table", StatusTable)
        table.set_data(rows, status_column=4)
        total_text = (
            _total_pens(dongles.dongles) if dongles.connected
            else f"Total: {_DISCONNECTED}"
        )
        self.query_one("#pen-total", Static).update(total_text)

    def _update_sync(self, bridge: HubIpcBridge) -> None:
        s = bridge.sync
        bar = self.query_one("#sync-bar", SyncProgressBar)
        bar.total = s.total
        bar.complete = s.complete
        bar.in_progress = s.in_progress
        bar.failed = s.failed
        bar.pending = s.pending

    def _update_storage(self, bridge: HubIpcBridge) -> None:
        store = bridge.store
        if not store.connected:
            text = f"SD: {_DISCONNECTED}  USB: {_DISCONNECTED}"
        else:
            sd = store.sd_free if store.sd_free else "--"
            usb = store.usb_free if store.usb_free else "--"
            text = f"SD: {sd} free    USB: {usb} free"
            if store.degraded:
                text += "  [yellow]DEGRADED[/yellow]"
        self.query_one("#storage-line", Static).update(text)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
