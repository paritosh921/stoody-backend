"""Dongle management screen — enumerate, reset, remove BLE dongles.

Pulls live dongle inventory from the IPC bridge (hub-ble-mgr) and
renders health with colour coding: green = healthy, yellow = degraded,
red = failed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Static

from src.ipc_bridge import HubIpcBridge
from src.widgets.footer import HubFooter
from src.widgets.status_table import StatusTable

logger = logging.getLogger(__name__)

_DONGLE_COLS = ["Dongle ID", "hci", "USB Port", "Firmware", "Pens", "Health"]


def _build_rows(dongles: list[dict[str, Any]]) -> list[list[str]]:
    """Convert bridge dongle dicts into display table rows."""
    rows: list[list[str]] = []
    for i, d in enumerate(dongles, start=1):
        rows.append([
            d.get("dongle_id", f"D{i}"),
            d.get("hci_path", f"hci{i - 1}"),
            d.get("usb_port", ""),
            d.get("firmware", ""),
            d.get("pens", "0/0"),
            d.get("status", "unknown").upper(),
        ])
    return rows


async def _lsusb_rescan() -> str:
    """Trigger a USB rescan via ``lsusb`` and report results."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lsusb",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
        bt_lines = [ln for ln in lines if "Bluetooth" in ln or "bluetooth" in ln]
        if bt_lines:
            return f"Found {len(bt_lines)} BLE device(s): " + "; ".join(
                ln.split(":")[-1].strip() for ln in bt_lines
            )
        return f"lsusb returned {len(lines)} device(s), none identified as BLE"
    except FileNotFoundError:
        return "lsusb not available on this platform"
    except Exception as exc:
        return f"Rescan error: {exc}"


class DonglesScreen(Screen):
    """BLE dongle management: view, reset, remove, rescan."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    DEFAULT_CSS = """
    DonglesScreen #dongles-container {
        padding: 1 2;
    }
    DonglesScreen .screen-title {
        text-style: bold;
        margin-bottom: 1;
    }
    DonglesScreen .action-row {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }
    DonglesScreen .action-row Button {
        margin-right: 1;
    }
    DonglesScreen .info-line {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        bridge: HubIpcBridge | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._bridge = bridge
        self._dongle_macs: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="dongles-container"):
            yield Static("[4] Dongle Management", classes="screen-title")

            yield StatusTable(
                columns=_DONGLE_COLS,
                rows=[],
                status_column=5,
                id="dongle-table",
            )
            yield Static("", id="dongle-summary", classes="info-line")

            with Horizontal(classes="action-row"):
                yield Button("Reset Dongle", id="btn-reset", variant="warning")
                yield Button("Remove Dongle", id="btn-remove", variant="error")
                yield Button("Rescan USB", id="btn-rescan", variant="default")

            yield Static("", id="dongle-action-result")
        yield HubFooter()

    def on_mount(self) -> None:
        """Start 1 Hz dongle table refresh."""
        self.set_interval(1.0, self._refresh)

    def _get_bridge(self) -> HubIpcBridge | None:
        if self._bridge is not None:
            return self._bridge
        return getattr(self.app, "_ipc_bridge", None)

    # -- refresh -------------------------------------------------------------

    def _refresh(self) -> None:
        bridge = self._get_bridge()
        table = self.query_one("#dongle-table", StatusTable)
        summary = self.query_one("#dongle-summary", Static)

        if bridge is None or not bridge.dongles.connected:
            table.set_data([])
            summary.update("[red]Disconnected[/red] — IPC to hub-ble-mgr unavailable")
            self._dongle_macs = []
            return

        dongles = bridge.dongles.dongles
        rows = _build_rows(dongles)
        table.set_data(rows, status_column=5)
        self._dongle_macs = [d.get("dongle_mac", "") for d in dongles]

        healthy = sum(
            1 for d in dongles if d.get("status", "").lower() in ("ok", "healthy")
        )
        degraded = sum(
            1 for d in dongles if d.get("status", "").lower() == "degraded"
        )
        failed = sum(
            1 for d in dongles if d.get("status", "").lower() == "failed"
        )
        summary.update(
            f"{len(dongles)} dongle(s): "
            f"[green]{healthy} healthy[/green]  "
            f"[yellow]{degraded} degraded[/yellow]  "
            f"[red]{failed} failed[/red]"
        )

    # -- button handlers -----------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        result = self.query_one("#dongle-action-result", Static)
        if event.button.id == "btn-reset":
            self._do_reset(result)
        elif event.button.id == "btn-remove":
            result.update("Remove: select dongle first (not yet implemented)")
        elif event.button.id == "btn-rescan":
            self._do_rescan(result)

    def _do_reset(self, result: Static) -> None:
        bridge = self._get_bridge()
        if bridge is None:
            result.update("[red]IPC bridge not available[/red]")
            return
        if not self._dongle_macs:
            result.update("No dongles to reset")
            return
        # Reset first dongle as default; a full UX would let user select.
        mac = self._dongle_macs[0]
        result.update(f"Resetting {mac}...")
        self.run_worker(self._reset_worker(bridge, mac, result))

    async def _reset_worker(
        self, bridge: HubIpcBridge, mac: str, result: Static,
    ) -> None:
        resp = await bridge.request_dongle_reset(mac)
        if "error" in resp:
            result.update(f"[red]Reset failed: {resp['error']}[/red]")
        else:
            result.update(f"[green]Reset OK[/green] for {mac}")

    def _do_rescan(self, result: Static) -> None:
        result.update("Rescanning USB...")
        self.run_worker(self._rescan_worker(result))

    async def _rescan_worker(self, result: Static) -> None:
        msg = await _lsusb_rescan()
        result.update(msg)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
