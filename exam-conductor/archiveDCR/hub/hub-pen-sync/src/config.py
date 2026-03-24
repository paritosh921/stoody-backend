"""Configuration for hub-pen-sync.

Constants match FAILURE_MITIGATION_REGISTER.md A1.7 (3 retries, 30s each)
and ble-gatt-spec.md Section 6 (chunk retry semantics).
"""

from __future__ import annotations

from dataclasses import dataclass

# GATT UUIDs from ble-gatt-spec.md Section 1
STROKE_BUFFER_UUID = "6f5f1001-4d8b-4d8d-9d7d-000000000001"
BUFFER_STATUS_UUID = "6f5f1002-4d8b-4d8d-9d7d-000000000001"
PEN_METADATA_UUID = "6f5f1003-4d8b-4d8d-9d7d-000000000001"
SYNC_CONTROL_UUID = "6f5f1004-4d8b-4d8d-9d7d-000000000001"

# Sync control commands (ble-gatt-spec.md Section 1)
SYNC_CMD_START = 0x01
SYNC_CMD_ABORT = 0x02
SYNC_CMD_CLEAR_BUFFER = 0x03

# P05 offline data commands (P05_pen_SDK.md)
P05_CMD_CHECK_OFFLINE_SIZE = 0x08
P05_CMD_REQUEST_OFFLINE_DATA = 0x09
P05_CMD_OFFLINE_DATA_PACKET = 0x07
P05_CMD_TRANSFER_COMPLETE = 0x0B
P05_CMD_DELETE_OFFLINE = 0x0A

# Chunk wire format (ble-gatt-spec.md Section 3)
CHUNK_HEADER_SIZE = 20  # bytes before payload
CHUNK_FLAG_FIRST = 0x01
CHUNK_FLAG_LAST = 0x02
CHUNK_FLAG_RETRANSMIT = 0x04


@dataclass(slots=True, frozen=True)
class PenSyncConfig:
    """Tunable parameters for pen sync operations."""

    # FAILURE_MITIGATION_REGISTER.md A1.7: 3 retries, 30s each
    max_retries: int = 3
    retry_timeout_sec: float = 30.0

    # Per-chunk retry (ble-gatt-spec.md Section 6)
    chunk_retries: int = 3
    chunk_timeout_sec: float = 5.0

    # IPC request timeout for store writes
    store_write_timeout_sec: float = 10.0

    # BLE connect timeout
    ble_connect_timeout_sec: float = 15.0
