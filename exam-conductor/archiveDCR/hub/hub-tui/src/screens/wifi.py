"""WiFi management screen — scan, connect, status display.

Uses ``nmcli`` subprocess calls for network scanning and connection
management.  Current connection status comes from the IPC bridge
(hub-uplink module).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Static

from src.ipc_bridge import HubIpcBridge
from src.widgets.footer import HubFooter
from src.widgets.status_table import StatusTable

logger = logging.getLogger(__name__)

_NETWORK_COLS = ["SSID", "Band", "Channel", "Signal", "Security", "Status"]


# ---------------------------------------------------------------------------
# nmcli helpers (run in executor to avoid blocking the event loop)
# ---------------------------------------------------------------------------

async def _nmcli_scan() -> list[list[str]]:
    """Run ``nmcli device wifi list`` and parse results into table rows."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nmcli", "-t", "-f", "SSID,FREQ,CHAN,SIGNAL,SECURITY,ACTIVE",
            "device", "wifi", "list", "--rescan", "yes",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        rows: list[list[str]] = []
        seen: set[str] = set()
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            parts = line.split(":")
            if len(parts) < 6:
                continue
            ssid = parts[0].strip()
            if not ssid or ssid in seen:
                continue
            seen.add(ssid)
            freq_mhz = parts[1].strip()
            band = "5 GHz" if freq_mhz.startswith("5") else "2.4 GHz"
            channel = parts[2].strip()
            signal = f"-{100 - int(parts[3].strip())} dBm" if parts[3].strip().isdigit() else parts[3].strip()
            security = parts[4].strip() or "Open"
            active = "Connected" if parts[5].strip().lower() == "yes" else ""
            rows.append([ssid, band, channel, signal, security, active])
        return rows
    except Exception:
        logger.warning("nmcli scan failed", exc_info=True)
        return []


async def _nmcli_connect(ssid: str, password: str) -> str:
    """Connect to a WiFi network via ``nmcli``."""
    try:
        cmd = ["nmcli", "device", "wifi", "connect", ssid]
        if password:
            cmd += ["password", password]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        return out or err or "Done"
    except asyncio.TimeoutError:
        return "Connection timed out"
    except Exception as exc:
        return f"Error: {exc}"


async def _nmcli_forget(ssid: str) -> str:
    """Forget (delete) a saved WiFi connection."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nmcli", "connection", "delete", ssid,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        return out or err or "Removed"
    except Exception as exc:
        return f"Error: {exc}"


def _current_connection_lines(bridge: HubIpcBridge | None) -> list[str]:
    """Build display lines for the current WiFi connection."""
    if bridge is None or not bridge.wifi.connected:
        return [
            "  SSID: [red]Disconnected[/red]",
            "  No connection details available",
            "",
            "",
        ]
    w = bridge.wifi
    return [
        f"  SSID: {w.ssid}" if w.ssid else "  SSID: [dim]none[/dim]",
        f"  Band: {w.band}  Channel: {w.channel}  Signal: {w.signal} dBm",
        f"  IP: {w.ip}  Gateway: {w.gateway}  DNS: {w.dns}",
        f"  Backend: {'Reachable' if w.backend_reachable else '[red]Unreachable[/red]'}"
        f"  Latency: {w.latency_ms} ms",
    ]


class WiFiScreen(Screen):
    """WiFi management: scan, connect/forget, current status."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    DEFAULT_CSS = """
    WiFiScreen #wifi-container {
        padding: 1 2;
    }
    WiFiScreen .section-title {
        text-style: bold;
        margin-top: 1;
    }
    WiFiScreen .info-line {
        margin: 0 1;
    }
    WiFiScreen .screen-title {
        text-style: bold;
        margin-bottom: 1;
    }
    WiFiScreen .action-row {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }
    WiFiScreen .action-row Button {
        margin-right: 1;
    }
    WiFiScreen #wifi-password {
        width: 40;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        bridge: HubIpcBridge | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._bridge = bridge
        self._scan_rows: list[list[str]] = []
        self._selected_ssid: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="wifi-container"):
            yield Static("[3] WiFi Management", classes="screen-title")

            # Current connection status
            yield Static("Current Connection", classes="section-title")
            yield Static("  SSID: --", id="conn-ssid", classes="info-line")
            yield Static("  --", id="conn-detail", classes="info-line")
            yield Static("  --", id="conn-ip", classes="info-line")
            yield Static("  --", id="conn-backend", classes="info-line")

            # Network list
            yield Static("Available Networks", classes="section-title")
            yield StatusTable(
                columns=_NETWORK_COLS,
                rows=[],
                status_column=5,
                id="network-table",
            )

            # Password input
            yield Input(
                placeholder="WiFi password (leave blank for open)",
                id="wifi-password",
                password=True,
            )

            # Actions
            with Horizontal(classes="action-row"):
                yield Button("Scan", id="btn-scan", variant="default")
                yield Button("Connect", id="btn-connect", variant="primary")
                yield Button("Forget", id="btn-forget", variant="warning")

            yield Static("", id="wifi-action-result")
        yield HubFooter()

    def on_mount(self) -> None:
        """Populate connection status and trigger initial scan."""
        self._update_connection()
        self.set_interval(5.0, self._update_connection)

    def _get_bridge(self) -> HubIpcBridge | None:
        if self._bridge is not None:
            return self._bridge
        return getattr(self.app, "_ipc_bridge", None)

    def _update_connection(self) -> None:
        bridge = self._get_bridge()
        lines = _current_connection_lines(bridge)
        ids = ["#conn-ssid", "#conn-detail", "#conn-ip", "#conn-backend"]
        for wid, text in zip(ids, lines):
            try:
                self.query_one(wid, Static).update(text)
            except Exception:
                pass

    # -- button handlers -----------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-scan":
            self._do_scan()
        elif event.button.id == "btn-connect":
            self._do_connect()
        elif event.button.id == "btn-forget":
            self._do_forget()

    def _do_scan(self) -> None:
        result = self.query_one("#wifi-action-result", Static)
        result.update("Scanning...")
        self.run_worker(self._scan_worker(), exclusive=True)

    async def _scan_worker(self) -> None:
        rows = await _nmcli_scan()
        self._scan_rows = rows
        table = self.query_one("#network-table", StatusTable)
        table.set_data(rows, status_column=5)
        result = self.query_one("#wifi-action-result", Static)
        if rows:
            self._selected_ssid = rows[0][0]
            result.update(
                f"Found {len(rows)} network(s). First selected: {self._selected_ssid}",
            )
        else:
            result.update("No networks found (nmcli unavailable or no WiFi adapter)")

    def _do_connect(self) -> None:
        if not self._selected_ssid and self._scan_rows:
            self._selected_ssid = self._scan_rows[0][0]
        if not self._selected_ssid:
            self.query_one("#wifi-action-result", Static).update(
                "Scan first, then connect",
            )
            return
        password = self.query_one("#wifi-password", Input).value
        self.query_one("#wifi-action-result", Static).update(
            f"Connecting to {self._selected_ssid}...",
        )
        self.run_worker(self._connect_worker(self._selected_ssid, password))

    async def _connect_worker(self, ssid: str, password: str) -> None:
        msg = await _nmcli_connect(ssid, password)
        self.query_one("#wifi-action-result", Static).update(msg)
        self._update_connection()

    def _do_forget(self) -> None:
        if not self._selected_ssid and self._scan_rows:
            self._selected_ssid = self._scan_rows[0][0]
        if not self._selected_ssid:
            self.query_one("#wifi-action-result", Static).update(
                "Scan first, then forget",
            )
            return
        self.query_one("#wifi-action-result", Static).update(
            f"Forgetting {self._selected_ssid}...",
        )
        self.run_worker(self._forget_worker(self._selected_ssid))

    async def _forget_worker(self, ssid: str) -> None:
        msg = await _nmcli_forget(ssid)
        self.query_one("#wifi-action-result", Static).update(msg)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
