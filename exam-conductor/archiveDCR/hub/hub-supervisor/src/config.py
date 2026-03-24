"""Supervisor configuration constants.

Paths follow ``HUB_DEPLOYMENT_SPEC.md`` and ``hub/ipc-protocol.md``
Section 1.  All values are overridable via environment variables so
that the test suite runs without real hardware paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------
MODULE_ID: str = "hub-supervisor"

# ---------------------------------------------------------------------------
# Hub config file (first-boot detection key)
# ---------------------------------------------------------------------------
HUB_CONF_PATH: str = os.environ.get(
    "EXAMPEN_CONFIG", "/etc/exampen/hub.conf"
)

# ---------------------------------------------------------------------------
# SQLite database (shared hub.db — WAL mode)
# ---------------------------------------------------------------------------
SQLITE_DB_PATH: Path = Path(
    os.environ.get("EXAMPEN_SUPERVISOR_DB", "/var/lib/exampen/hub.db")
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
SD_DATA_PATH: str = os.environ.get("EXAMPEN_DATA", "/var/lib/exampen")
USB_DATA_PATH: str = os.environ.get("EXAMPEN_BACKUP", "/mnt/exampen-backup")

# ---------------------------------------------------------------------------
# IPC sockets — supervisor's own socket and child module sockets
# ---------------------------------------------------------------------------
SOCKET_DIR: str = os.environ.get("EXAMPEN_SOCKET_DIR", "/run/exampen")

SUPERVISOR_SOCKET: str = os.environ.get(
    "EXAMPEN_SUPERVISOR_SOCK",
    os.path.join(SOCKET_DIR, "supervisor.sock"),
)

# Child module socket paths (ipc-protocol.md Section 1)
CHILD_SOCKETS: dict[str, str] = {
    "hub-ble-mgr": os.path.join(SOCKET_DIR, "ble-mgr.sock"),
    "hub-pen-sync": os.path.join(SOCKET_DIR, "pen-sync.sock"),
    "hub-timer": os.path.join(SOCKET_DIR, "timer.sock"),
    "hub-store": os.path.join(SOCKET_DIR, "store.sock"),
    "hub-uplink": os.path.join(SOCKET_DIR, "uplink.sock"),
    "hub-invig-ble": os.path.join(SOCKET_DIR, "invig-ble.sock"),
    "hub-tui": os.path.join(SOCKET_DIR, "tui.sock"),
}

# ---------------------------------------------------------------------------
# Child module executables (systemd dependency tree — HUB_DEPLOYMENT_SPEC §6)
# ---------------------------------------------------------------------------
# Maps module name to the command used to spawn it.
# In production these are absolute paths; overridable for tests.
CHILD_COMMANDS: dict[str, list[str]] = {
    "hub-ble-mgr": ["/opt/exampen/bin/hub-ble-mgr"],
    "hub-pen-sync": ["/opt/exampen/bin/hub-pen-sync"],
    "hub-timer": ["/opt/exampen/bin/hub-timer"],
    "hub-store": ["/opt/exampen/bin/hub-store"],
    "hub-uplink": ["/opt/exampen/bin/hub-uplink"],
    "hub-invig-ble": ["/opt/exampen/bin/hub-invig-ble"],
    "hub-tui": ["/opt/exampen/bin/hub-tui"],
}

# hub-tui is optional — only spawned if HDMI/serial is connected.
OPTIONAL_MODULES: frozenset[str] = frozenset({"hub-tui"})

# Modules that are always required.
REQUIRED_MODULES: tuple[str, ...] = (
    "hub-ble-mgr",
    "hub-pen-sync",
    "hub-timer",
    "hub-store",
    "hub-uplink",
    "hub-invig-ble",
)

# ---------------------------------------------------------------------------
# Process manager tunables
# ---------------------------------------------------------------------------
MAX_RESTART_COUNT: int = 3
WATCHDOG_INTERVAL_SEC: float = 10.0
HEALTH_CHECK_TIMEOUT_SEC: float = 5.0

# ---------------------------------------------------------------------------
# systemd notify (HUB_DEPLOYMENT_SPEC §6 — Type=notify, WatchdogSec=30)
# ---------------------------------------------------------------------------
SYSTEMD_NOTIFY: bool = os.environ.get("NOTIFY_SOCKET", "") != ""
SYSTEMD_WATCHDOG_SEC: float = 30.0
