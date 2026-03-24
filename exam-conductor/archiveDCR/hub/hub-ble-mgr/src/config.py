"""BLE manager configuration constants.

Tunables for dongle capacity, scan timing, and connection lifecycle.
Values derived from HUB_DEPLOYMENT_SPEC.md, FAILURE_MITIGATION_REGISTER (A1.1,
H5), and ble-gatt-spec.md retry semantics.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Dongle capacity (FAILURE_MITIGATION_REGISTER A1.1)
# ---------------------------------------------------------------------------
# Hard limit per BLE dongle — BlueZ practical connection ceiling.
MAX_PENS_PER_DONGLE: int = 8

# Maximum number of USB BLE dongles on the hub.
MAX_DONGLES: int = 5

# Total system capacity: MAX_DONGLES * MAX_PENS_PER_DONGLE = 40.

# ---------------------------------------------------------------------------
# Scan timing (FAILURE_MITIGATION_REGISTER H5)
# ---------------------------------------------------------------------------
# Delay (seconds) between activating successive dongles for scanning.
# Reduces RF collision when all 5 dongles scan simultaneously.
SCAN_STAGGER_DELAY_SEC: float = 0.5

# Default scan timeout for a single scan session (seconds).
DEFAULT_SCAN_TIMEOUT_SEC: int = 60

# ---------------------------------------------------------------------------
# Connection (ble-gatt-spec.md Section 6 retry rules)
# ---------------------------------------------------------------------------
# Timeout for a single pen connection attempt.
CONNECTION_TIMEOUT_SEC: float = 30.0

# Max retries for pen reconnection (per pen, per session).
CONNECTION_MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------
# Interval between periodic dongle health checks (seconds).
HEALTH_CHECK_INTERVAL_SEC: float = 10.0

# Threshold: if hciconfig query takes longer than this, dongle is degraded.
HEALTH_SLOW_RESPONSE_SEC: float = 3.0

# ---------------------------------------------------------------------------
# BLE GATT service UUID (ble-gatt-spec.md Section 1)
# ---------------------------------------------------------------------------
PEN_GATT_SERVICE_UUID: str = "6f5f0001-4d8b-4d8d-9d7d-000000000001"

# ---------------------------------------------------------------------------
# IPC
# ---------------------------------------------------------------------------
IPC_SOCKET_PATH: str = os.environ.get(
    "EXAMPEN_BLE_MGR_SOCK", "/run/exampen/ble-mgr.sock"
)

MODULE_ID: str = "hub-ble-mgr"
