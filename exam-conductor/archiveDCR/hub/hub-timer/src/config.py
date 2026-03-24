"""Timer module configuration constants.

All hub-timer tunables live here. Paths follow the HUB_DEPLOYMENT_SPEC layout
(/var/lib/exampen/ on real hardware, overridable via env for testing).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
# Default path per HUB_DEPLOYMENT_SPEC $3.1 — overridable via env for tests.
SQLITE_DB_PATH: Path = Path(
    os.environ.get("EXAMPEN_TIMER_DB", "/var/lib/exampen/hub.db")
)

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
# How often (seconds) the countdown writes remaining_sec to SQLite.
# Per FAILURE_MITIGATION_REGISTER F4: up to 10 s accuracy loss on crash is
# acceptable for exam timing.
PERSIST_INTERVAL_SEC: float = 10.0

# ---------------------------------------------------------------------------
# Tick broadcast
# ---------------------------------------------------------------------------
# Interval between timer.tick IPC broadcasts.
TICK_BROADCAST_INTERVAL_SEC: float = 1.0

# ---------------------------------------------------------------------------
# IPC socket
# ---------------------------------------------------------------------------
IPC_SOCKET_PATH: str = os.environ.get(
    "EXAMPEN_TIMER_SOCK", "/run/exampen/timer.sock"
)

# ---------------------------------------------------------------------------
# Module identity (used in IPC envelope 'source' field)
# ---------------------------------------------------------------------------
MODULE_ID: str = "hub-timer"
