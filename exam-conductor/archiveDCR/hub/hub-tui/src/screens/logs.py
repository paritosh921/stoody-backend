"""Log viewer screen — tabbed log sources with live reading and filtering."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Select, Static, TabbedContent, TabPane

from src.widgets.footer import HubFooter

_LOG_LEVELS = ("debug", "info", "warn", "error", "critical")

# Map TUI tab names to systemd unit / log-file basenames.
_SERVICE_MAP: dict[str, str] = {
    "Supervisor": "exampen-supervisor",
    "BLE": "exampen-ble-mgr",
    "Pen Sync": "exampen-pen-sync",
    "Uplink": "exampen-uplink",
    "Invigilator": "exampen-invig-ble",
}

_LOG_DIR = Path(os.getenv("EXAMPEN_LOG_DIR", "/var/log/exampen"))
_MAX_LINES = 100
_REFRESH_SECONDS = 5


# ---------------------------------------------------------------------------
# Log-fetching helpers
# ---------------------------------------------------------------------------

def _read_journalctl(service: str) -> list[str] | None:
    """Try reading logs via journalctl.  Returns None if unavailable."""
    try:
        result = subprocess.run(
            [
                "journalctl",
                f"-u{service}",
                "--no-pager",
                f"-n{_MAX_LINES}",
                "--output=cat",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines() or []
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _read_logfile(service: str) -> list[str] | None:
    """Fallback: read the last *_MAX_LINES* from a plain log file."""
    path = _LOG_DIR / f"{service}.log"
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-_MAX_LINES:]
    except OSError:
        return None


def _fetch_logs(service: str) -> list[str]:
    """Return log lines for *service*, trying journalctl then file fallback."""
    lines = _read_journalctl(service)
    if lines is not None:
        return lines
    lines = _read_logfile(service)
    if lines is not None:
        return lines
    return ["Log source unavailable"]


def _apply_level_filter(
    lines: list[str],
    level: str,
) -> list[str]:
    """Keep only lines at or above *level* severity."""
    if level == "debug":
        return lines  # show everything

    idx = _LOG_LEVELS.index(level) if level in _LOG_LEVELS else 0
    allowed = set(_LOG_LEVELS[idx:])

    filtered: list[str] = []
    for line in lines:
        low = line.lower()
        if any(tag in low for tag in allowed):
            filtered.append(line)
    return filtered or ["(no lines match the selected level)"]


def _colorize(line: str) -> str:
    """Add Rich markup colour based on severity keyword."""
    low = line.lower()
    if "error" in low or "critical" in low:
        return f"[red]{line}[/red]"
    if "warn" in low:
        return f"[yellow]{line}[/yellow]"
    if "debug" in low:
        return f"[dim]{line}[/dim]"
    return f"[green]{line}[/green]"


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------


class LogsScreen(Screen):
    """Log viewer with tabbed sources, level filtering, and auto-refresh."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("r", "refresh_logs", "Refresh", show=True),
    ]

    DEFAULT_CSS = """
    LogsScreen #logs-container {
        padding: 1 2;
    }
    LogsScreen .screen-title {
        text-style: bold;
        margin-bottom: 1;
    }
    LogsScreen .log-content {
        height: auto;
        max-height: 20;
        overflow-y: auto;
    }
    LogsScreen #controls {
        height: 3;
        margin-bottom: 1;
    }
    LogsScreen #controls Button {
        margin-right: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._level = "debug"

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="logs-container"):
            yield Static("[7] Log Viewer", classes="screen-title")

            with Horizontal(id="controls"):
                yield Button("Refresh", id="btn-refresh", variant="primary")
                yield Select(
                    [(lv.upper(), lv) for lv in _LOG_LEVELS],
                    value="debug",
                    id="level-select",
                    allow_blank=False,
                )

            with TabbedContent():
                for source in _SERVICE_MAP:
                    tab_id = f"tab-{source.lower().replace(' ', '-')}"
                    with TabPane(source, id=tab_id):
                        yield Static(
                            "Loading...",
                            classes="log-content",
                            id=f"log-{tab_id}",
                        )
        yield HubFooter()

    def on_mount(self) -> None:
        self._load_all_tabs()
        self.set_interval(_REFRESH_SECONDS, self._load_all_tabs)

    # -- Event handlers -----------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh":
            self._load_all_tabs()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "level-select":
            self._level = str(event.value)
            self._load_all_tabs()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_refresh_logs(self) -> None:
        self._load_all_tabs()

    # -- Internal -----------------------------------------------------------

    def _load_all_tabs(self) -> None:
        for source, service in _SERVICE_MAP.items():
            tab_id = f"tab-{source.lower().replace(' ', '-')}"
            widget_id = f"log-{tab_id}"
            try:
                widget = self.query_one(f"#{widget_id}", Static)
            except Exception:
                continue
            raw = _fetch_logs(service)
            filtered = _apply_level_filter(raw, self._level)
            coloured = [_colorize(ln) for ln in filtered]
            widget.update("\n".join(coloured))
