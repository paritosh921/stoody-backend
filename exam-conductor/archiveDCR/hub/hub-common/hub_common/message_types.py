"""IPC message type constants and re-exported payload dataclasses.

Every ``msg_type`` string used between hub modules is defined here.
Payload dataclasses live in :mod:`hub_common.payloads` and are
re-exported so that callers can ``from hub_common.message_types import
STORE_WRITE_REQUEST, StoreWriteRequestPayload``.

The constants match the catalog in ``new-docs/hub/ipc-protocol.md``
Section 3.
"""

from __future__ import annotations

# Re-export all payload dataclasses for single-import convenience.
from hub_common.payloads import (  # noqa: F401
    BleConnectRequestPayload,
    BleConnectResultPayload,
    BleDongleHealthEventPayload,
    BleScanResultEventPayload,
    BleScanStartRequestPayload,
    BleScanStopRequestPayload,
    ErrorPayload,
    FsmSnapshotRequestPayload,
    FsmSnapshotResultPayload,
    FsmTransitionRequestPayload,
    FsmTransitionResultPayload,
    InvigAuthStateEventPayload,
    InvigCommandEventPayload,
    PenSyncAbortRequestPayload,
    PenSyncCompleteEventPayload,
    PenSyncProgressEventPayload,
    PenSyncRequestPayload,
    StoreHealthEventPayload,
    StoreReadRequestPayload,
    StoreReadResultPayload,
    StoreWriteRequestPayload,
    StoreWriteResultPayload,
    TimerArmRequestPayload,
    TimerCancelRequestPayload,
    TimerExpiredEventPayload,
    TimerSnapshotRequestPayload,
    TimerSnapshotResultPayload,
    UiSnapshotRequestPayload,
    UiSnapshotResultPayload,
    UplinkUploadCompleteEventPayload,
    UplinkUploadErrorPayload,
    UplinkUploadProgressEventPayload,
    UplinkUploadRequestPayload,
    UplinkStatusRequestPayload,
    UplinkStatusResultPayload,
    SupervisorHealthRequestPayload,
    SupervisorHealthResultPayload,
)

# ===================================================================
# 3.1 Supervisor / FSM
# ===================================================================

FSM_TRANSITION_REQUEST = "fsm.transition.request"
FSM_TRANSITION_RESULT = "fsm.transition.result"
FSM_TRANSITION_ERROR = "fsm.transition.error"
FSM_SNAPSHOT_REQUEST = "fsm.snapshot.request"
FSM_SNAPSHOT_RESULT = "fsm.snapshot.result"

# ===================================================================
# 3.2 Timer
# ===================================================================

TIMER_ARM_REQUEST = "timer.arm.request"
TIMER_CANCEL_REQUEST = "timer.cancel.request"
TIMER_SNAPSHOT_REQUEST = "timer.snapshot.request"
TIMER_SNAPSHOT_RESULT = "timer.snapshot.result"
TIMER_EXPIRED_EVENT = "timer.expired.event"

# ===================================================================
# 3.3 BLE Manager
# ===================================================================

BLE_SCAN_START_REQUEST = "ble.scan.start.request"
BLE_SCAN_STOP_REQUEST = "ble.scan.stop.request"
BLE_SCAN_RESULT_EVENT = "ble.scan.result.event"
BLE_DONGLE_HEALTH_EVENT = "ble.dongle.health.event"
BLE_CONNECT_REQUEST = "ble.connect.request"
BLE_CONNECT_RESULT = "ble.connect.result"

# ===================================================================
# 3.4 Pen Sync
# ===================================================================

PEN_SYNC_REQUEST = "pen.sync.request"
PEN_SYNC_PROGRESS_EVENT = "pen.sync.progress.event"
PEN_SYNC_COMPLETE_EVENT = "pen.sync.complete.event"
PEN_SYNC_ABORT_REQUEST = "pen.sync.abort.request"

# ===================================================================
# 3.5 Store
# ===================================================================

STORE_WRITE_REQUEST = "store.write.request"
STORE_WRITE_RESULT = "store.write.result"
STORE_READ_REQUEST = "store.read.request"
STORE_READ_RESULT = "store.read.result"
STORE_HEALTH_EVENT = "store.health.event"

# ===================================================================
# 3.6 Uplink
# ===================================================================

UPLINK_UPLOAD_REQUEST = "uplink.upload.request"
UPLINK_UPLOAD_PROGRESS_EVENT = "uplink.upload.progress.event"
UPLINK_UPLOAD_COMPLETE_EVENT = "uplink.upload.complete.event"
UPLINK_UPLOAD_ERROR = "uplink.upload.error"
UPLINK_STATUS_REQUEST = "uplink.status.request"
UPLINK_STATUS_RESULT = "uplink.status.result"

# Supervisor health (module → supervisor heartbeat / supervisor → module poll)
SUPERVISOR_HEALTH_REQUEST = "supervisor.health.request"
SUPERVISOR_HEALTH_RESULT = "supervisor.health.result"

# ===================================================================
# 3.7 Invigilator BLE / TUI
# ===================================================================

INVIG_AUTH_STATE_EVENT = "invig.auth.state.event"
INVIG_COMMAND_EVENT = "invig.command.event"
UI_SNAPSHOT_REQUEST = "ui.snapshot.request"
UI_SNAPSHOT_RESULT = "ui.snapshot.result"
