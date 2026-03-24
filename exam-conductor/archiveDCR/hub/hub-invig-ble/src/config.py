"""Invigilator BLE module configuration constants.

All hub-invig-ble tunables live here.  Paths follow the HUB_DEPLOYMENT_SPEC
layout (/var/lib/exampen/ on real hardware, overridable via env for testing).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Module identity (used in IPC envelope 'source' field)
# ---------------------------------------------------------------------------
MODULE_ID: str = "hub-invig-ble"

# ---------------------------------------------------------------------------
# IPC socket
# ---------------------------------------------------------------------------
IPC_SOCKET_PATH: str = os.environ.get(
    "EXAMPEN_INVIG_BLE_SOCK", "/run/exampen/invig-ble.sock"
)

# Socket paths for outbound IPC connections to other modules.
SUPERVISOR_SOCKET_PATH: str = os.environ.get(
    "EXAMPEN_SUPERVISOR_SOCK", "/run/exampen/supervisor.sock"
)

# ---------------------------------------------------------------------------
# SQLite (read-only access to invig_codes table)
# ---------------------------------------------------------------------------
SQLITE_DB_PATH: Path = Path(
    os.environ.get("EXAMPEN_INVIG_DB", "/var/lib/exampen/hub.db")
)

# ---------------------------------------------------------------------------
# BLE peripheral configuration
# ---------------------------------------------------------------------------
# Advertised peripheral name visible to the invigilator mobile app.
BLE_PERIPHERAL_NAME: str = os.environ.get(
    "EXAMPEN_BLE_INVIG_NAME", "ExamPen-Hub"
)

# Invigilator GATT service UUID (ble-gatt-spec.md Section 2).
GATT_SERVICE_UUID: str = "6f5f0002-4d8b-4d8d-9d7d-000000000002"

# Characteristic UUIDs.
CHAR_AUTH_UUID: str = "6f5f2001-4d8b-4d8d-9d7d-000000000002"
CHAR_COMMAND_UUID: str = "6f5f2002-4d8b-4d8d-9d7d-000000000002"
CHAR_STATUS_FEED_UUID: str = "6f5f2003-4d8b-4d8d-9d7d-000000000002"
CHAR_MAC_LIST_UUID: str = "6f5f2004-4d8b-4d8d-9d7d-000000000002"

# ---------------------------------------------------------------------------
# Authentication lockout policy (ble-gatt-spec.md Section 6 + S3 mitigation)
# ---------------------------------------------------------------------------
# Maximum consecutive failed auth attempts before lockout.
AUTH_MAX_ATTEMPTS: int = 5

# Duration (seconds) of lockout after exceeding max attempts.
AUTH_LOCKOUT_DURATION_SEC: float = 300.0  # 5 minutes

# ---------------------------------------------------------------------------
# Status feed cadence
# ---------------------------------------------------------------------------
STATUS_FEED_INTERVAL_SEC: float = 1.0
