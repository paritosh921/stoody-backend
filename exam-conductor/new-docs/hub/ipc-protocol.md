# Hub IPC Protocol
# ExamPen — Hub Module Message Envelope & Message Catalog

Status: ACTIVE
Authoritative for: message envelope, routing rules, and payload shapes between `hub-supervisor`, `hub-ble-mgr`, `hub-pen-sync`, `hub-store`, `hub-timer`, `hub-uplink`, `hub-invig-ble`, and `hub-tui`.

Path note: this document is authoritative during planning. A runtime implementation may later live in `hub/hub-common/ipc_protocol.py`, but that implementation must match this doc.

---

## 1. Transport

- Transport: Unix domain sockets using JSON-lines.
- Encoding: UTF-8.
- Framing: one complete JSON object per line.
- Reliability: request/response messages require an acknowledgement or timeout; fire-and-forget events do not.
- Time format: ISO 8601 UTC with `Z` suffix.

Recommended socket paths:

- `/run/exampen/supervisor.sock`
- `/run/exampen/ble-mgr.sock`
- `/run/exampen/pen-sync.sock`
- `/run/exampen/store.sock`
- `/run/exampen/timer.sock`
- `/run/exampen/uplink.sock`
- `/run/exampen/invig-ble.sock`
- `/run/exampen/tui.sock`

---

## 2. Envelope

Every IPC message uses this envelope:

```json
{
  "msg_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "msg_type": "store.write.request",
  "source": "hub-pen-sync",
  "target": "hub-store",
  "sent_at": "2026-03-18T10:20:30Z",
  "correlation_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "expects_reply": true,
  "payload": {}
}
```

Field rules:

| Field | Type | Required | Notes |
|---|---|---|---|
| `msg_id` | string | Yes | Unique ULID/UUID per message |
| `msg_type` | string | Yes | Namespaced message identifier |
| `source` | string | Yes | Sending module ID |
| `target` | string | Yes | Receiving module ID |
| `sent_at` | string | Yes | ISO 8601 UTC timestamp |
| `correlation_id` | string | No | Present on replies/events related to an originating request |
| `expects_reply` | boolean | Yes | `true` for request/response, `false` for events |
| `payload` | object | Yes | Message-specific body |

Reply envelope rules:

- Replies reuse the request `msg_id` in `correlation_id`.
- Successful replies use `*.result`.
- Failed replies use `*.error`.

---

## 3. Core Message Types

### 3.1 Supervisor and FSM

| Message | Source | Target | Payload |
|---|---|---|---|
| `fsm.transition.request` | Any module | `hub-supervisor` | `{exam_id, from_state, to_state, reason, actor}` |
| `fsm.transition.result` | `hub-supervisor` | Caller | `{exam_id, state, persisted: true}` |
| `fsm.transition.error` | `hub-supervisor` | Caller | `{code, message}` |
| `fsm.snapshot.request` | `hub-tui`, `hub-invig-ble` | `hub-supervisor` | `{exam_id}` |
| `fsm.snapshot.result` | `hub-supervisor` | Caller | `{exam_id, state, timer, dongles, storage, upload}` |

### 3.2 Timer

| Message | Source | Target | Payload |
|---|---|---|---|
| `timer.arm.request` | `hub-supervisor` | `hub-timer` | `{exam_id, duration_sec, armed_by}` |
| `timer.cancel.request` | `hub-supervisor` | `hub-timer` | `{exam_id, reason}` |
| `timer.snapshot.request` | `hub-tui`, `hub-invig-ble` | `hub-timer` | `{exam_id}` |
| `timer.snapshot.result` | `hub-timer` | Caller | `{exam_id, state, remaining_sec, started_at, expires_at}` |
| `timer.expired.event` | `hub-timer` | `hub-supervisor` | `{exam_id, expired_at}` |

### 3.3 BLE Manager

| Message | Source | Target | Payload |
|---|---|---|---|
| `ble.scan.start.request` | `hub-supervisor`, `hub-invig-ble` | `hub-ble-mgr` | `{exam_id, mode: "registration"|"sync", timeout_sec}` |
| `ble.scan.stop.request` | `hub-supervisor` | `hub-ble-mgr` | `{exam_id, reason}` |
| `ble.scan.result.event` | `hub-ble-mgr` | `hub-supervisor`, `hub-invig-ble` | `{exam_id, pen_mac, dongle_mac, rssi, battery_pct}` |
| `ble.dongle.health.event` | `hub-ble-mgr` | `hub-supervisor`, `hub-tui` | `{dongle_mac, status, detail}` |
| `ble.connect.request` | `hub-pen-sync` | `hub-ble-mgr` | `{exam_id, pen_mac, dongle_mac}` |
| `ble.connect.result` | `hub-ble-mgr` | `hub-pen-sync` | `{exam_id, pen_mac, dongle_mac, connection_id}` |

### 3.4 Pen Sync

| Message | Source | Target | Payload |
|---|---|---|---|
| `pen.sync.request` | `hub-supervisor` | `hub-pen-sync` | `{exam_id, pen_mac, dongle_mac}` |
| `pen.sync.progress.event` | `hub-pen-sync` | `hub-supervisor`, `hub-tui`, `hub-invig-ble` | `{exam_id, pen_mac, chunk_index, total_chunks, bytes_received, status}` |
| `pen.sync.complete.event` | `hub-pen-sync` | `hub-supervisor`, `hub-uplink` | `{exam_id, pen_mac, total_chunks, checksum_crc32, status: "complete"|"failed"|"timeout"}` |
| `pen.sync.abort.request` | `hub-supervisor` | `hub-pen-sync` | `{exam_id, pen_mac, reason}` |

### 3.5 Store

| Message | Source | Target | Payload |
|---|---|---|---|
| `store.write.request` | `hub-pen-sync` | `hub-store` | `{exam_id, pen_mac, chunk_index, chunk_b64, checksum_crc32}` |
| `store.write.result` | `hub-store` | `hub-pen-sync` | `{exam_id, pen_mac, chunk_index, sd_persisted, usb_persisted}` |
| `store.read.request` | `hub-uplink` | `hub-store` | `{exam_id, pen_mac, chunk_index}` |
| `store.read.result` | `hub-store` | `hub-uplink` | `{exam_id, pen_mac, chunk_index, chunk_b64, checksum_crc32}` |
| `store.health.event` | `hub-store` | `hub-supervisor`, `hub-tui` | `{sd_ok, usb_ok, degraded, free_bytes}` |

### 3.6 Uplink

| Message | Source | Target | Payload |
|---|---|---|---|
| `uplink.upload.request` | `hub-supervisor`, `hub-invig-ble` | `hub-uplink` | `{exam_id, path: "wifi"|"mobile"|"auto"}` |
| `uplink.upload.progress.event` | `hub-uplink` | `hub-supervisor`, `hub-tui`, `hub-invig-ble` | `{exam_id, pen_mac, chunk_index, acked_chunks, total_chunks, path}` |
| `uplink.upload.complete.event` | `hub-uplink` | `hub-supervisor` | `{exam_id, pen_mac, complete: true}` |
| `uplink.upload.error` | `hub-uplink` | Caller | `{exam_id, pen_mac, code, message, retryable}` |

### 3.7 Invigilator BLE and TUI

| Message | Source | Target | Payload |
|---|---|---|---|
| `invig.auth.state.event` | `hub-invig-ble` | `hub-supervisor`, `hub-tui` | `{invig_id, connected, authenticated}` |
| `invig.command.event` | `hub-invig-ble` | `hub-supervisor` | `{cmd_id, payload}` |
| `ui.snapshot.request` | `hub-tui` | `hub-supervisor` | `{screen}` |
| `ui.snapshot.result` | `hub-supervisor` | `hub-tui` | `{screen, data}` |

---

## 4. Error Codes

Common error codes:

| Code | Meaning |
|---|---|
| `invalid_state_transition` | Requested FSM transition is not allowed |
| `unknown_exam` | Exam session does not exist locally |
| `unknown_pen` | Pen is not known to the current exam |
| `dongle_unavailable` | Requested dongle is unhealthy or missing |
| `storage_write_failed` | SD and/or USB write failed |
| `storage_read_failed` | Requested chunk cannot be read |
| `timeout` | Target module did not reply in time |
| `validation_failed` | Payload missing or malformed |

Retry rules:

- `timeout`, `dongle_unavailable`, and retryable uplink/storage errors may be retried.
- `invalid_state_transition` and `validation_failed` are not retryable without changing input.

---

## 5. Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Added the authoritative hub IPC envelope, message catalog, and error model for P2 implementation planning. | Codex |
