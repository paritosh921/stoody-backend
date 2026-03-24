"""Payload dataclasses for all IPC message types.

Each dataclass corresponds to the ``payload`` field of an
:class:`~hub_common.ipc_protocol.IpcEnvelope` whose ``msg_type``
matches one of the constants in :mod:`hub_common.message_types`.

Organized by IPC domain (matches ``ipc-protocol.md`` Section 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ===================================================================
# 3.1 Supervisor / FSM
# ===================================================================

@dataclass(slots=True)
class FsmTransitionRequestPayload:
    exam_id: str
    from_state: str
    to_state: str
    reason: str
    actor: str


@dataclass(slots=True)
class FsmTransitionResultPayload:
    exam_id: str
    state: str
    persisted: bool = True


@dataclass(slots=True)
class FsmSnapshotRequestPayload:
    exam_id: str


@dataclass(slots=True)
class FsmSnapshotResultPayload:
    exam_id: str
    state: str
    timer: dict[str, object] = field(default_factory=dict)
    dongles: dict[str, object] = field(default_factory=dict)
    storage: dict[str, object] = field(default_factory=dict)
    upload: dict[str, object] = field(default_factory=dict)


# ===================================================================
# 3.2 Timer
# ===================================================================

@dataclass(slots=True)
class TimerArmRequestPayload:
    exam_id: str
    duration_sec: int
    armed_by: str


@dataclass(slots=True)
class TimerCancelRequestPayload:
    exam_id: str
    reason: str


@dataclass(slots=True)
class TimerSnapshotRequestPayload:
    exam_id: str


@dataclass(slots=True)
class TimerSnapshotResultPayload:
    exam_id: str
    state: str
    remaining_sec: int
    started_at: str | None = None
    expires_at: str | None = None


@dataclass(slots=True)
class TimerExpiredEventPayload:
    exam_id: str
    expired_at: str


# ===================================================================
# 3.3 BLE Manager
# ===================================================================

@dataclass(slots=True)
class BleScanStartRequestPayload:
    exam_id: str
    mode: str  # "registration" | "sync"
    timeout_sec: int


@dataclass(slots=True)
class BleScanStopRequestPayload:
    exam_id: str
    reason: str


@dataclass(slots=True)
class BleScanResultEventPayload:
    exam_id: str
    pen_mac: str
    dongle_mac: str
    rssi: int
    battery_pct: int


@dataclass(slots=True)
class BleDongleHealthEventPayload:
    dongle_mac: str
    status: str  # "healthy" | "degraded" | "failed"
    detail: str


@dataclass(slots=True)
class BleConnectRequestPayload:
    exam_id: str
    pen_mac: str
    dongle_mac: str


@dataclass(slots=True)
class BleConnectResultPayload:
    exam_id: str
    pen_mac: str
    dongle_mac: str
    connection_id: str


# ===================================================================
# 3.4 Pen Sync
# ===================================================================

@dataclass(slots=True)
class PenSyncRequestPayload:
    exam_id: str
    pen_mac: str
    dongle_mac: str


@dataclass(slots=True)
class PenSyncProgressEventPayload:
    exam_id: str
    pen_mac: str
    chunk_index: int
    total_chunks: int
    bytes_received: int
    status: str


@dataclass(slots=True)
class PenSyncCompleteEventPayload:
    exam_id: str
    pen_mac: str
    total_chunks: int
    checksum_crc32: str
    status: str  # "complete" | "failed" | "timeout"


@dataclass(slots=True)
class PenSyncAbortRequestPayload:
    exam_id: str
    pen_mac: str
    reason: str


# ===================================================================
# 3.5 Store
# ===================================================================

@dataclass(slots=True)
class StoreWriteRequestPayload:
    exam_id: str
    pen_mac: str
    chunk_index: int
    chunk_b64: str
    checksum_crc32: str


@dataclass(slots=True)
class StoreWriteResultPayload:
    exam_id: str
    pen_mac: str
    chunk_index: int
    sd_persisted: bool
    usb_persisted: bool


@dataclass(slots=True)
class StoreReadRequestPayload:
    exam_id: str
    pen_mac: str
    chunk_index: int


@dataclass(slots=True)
class StoreReadResultPayload:
    exam_id: str
    pen_mac: str
    chunk_index: int
    chunk_b64: str
    checksum_crc32: str


@dataclass(slots=True)
class StoreHealthEventPayload:
    sd_ok: bool
    usb_ok: bool
    degraded: bool
    free_bytes: int


# ===================================================================
# 3.6 Uplink
# ===================================================================

@dataclass(slots=True)
class UplinkUploadRequestPayload:
    exam_id: str
    path: str  # "wifi" | "mobile" | "auto"


@dataclass(slots=True)
class UplinkUploadProgressEventPayload:
    exam_id: str
    pen_mac: str
    chunk_index: int
    acked_chunks: int
    total_chunks: int
    path: str


@dataclass(slots=True)
class UplinkUploadCompleteEventPayload:
    exam_id: str
    pen_mac: str
    complete: bool = True


@dataclass(slots=True)
class UplinkUploadErrorPayload:
    exam_id: str
    pen_mac: str
    code: str
    message: str
    retryable: bool


@dataclass(slots=True)
class UplinkStatusRequestPayload:
    exam_id: str


@dataclass(slots=True)
class UplinkStatusResultPayload:
    exam_id: str
    pens: list[dict[str, object]] = field(default_factory=list)


# -- Supervisor health (cross-module) -----------------------------------

@dataclass(slots=True)
class SupervisorHealthRequestPayload:
    pass


@dataclass(slots=True)
class SupervisorHealthResultPayload:
    module: str
    healthy: bool
    detail: dict[str, object] = field(default_factory=dict)


# ===================================================================
# 3.7 Invigilator BLE / TUI
# ===================================================================

@dataclass(slots=True)
class InvigAuthStateEventPayload:
    invig_id: str
    connected: bool
    authenticated: bool


@dataclass(slots=True)
class InvigCommandEventPayload:
    cmd_id: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class UiSnapshotRequestPayload:
    screen: str


@dataclass(slots=True)
class UiSnapshotResultPayload:
    screen: str
    data: dict[str, object] = field(default_factory=dict)


# ===================================================================
# Error envelope payload (shared across all domains)
# ===================================================================

@dataclass(slots=True)
class ErrorPayload:
    code: str
    message: str
