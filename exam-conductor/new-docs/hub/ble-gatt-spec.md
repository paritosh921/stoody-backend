# BLE GATT Specification
# ExamPen — BLE Service & Characteristic Definitions

Status: ACTIVE
Authoritative for: all BLE GATT UUIDs, payload formats, MTU handling, error codes, and retry semantics for pen sync and invigilator control.

---

## 1. Pen GATT Service (Peripheral Role on Pen MCU)

Service UUID: `6f5f0001-4d8b-4d8d-9d7d-000000000001`

| Characteristic | UUID | Properties | Payload Format | Notes |
|---|---|---|---|---|
| Stroke buffer | `6f5f1001-4d8b-4d8d-9d7d-000000000001` | Read, Notify | Chunk payload defined in §3 | Hub reads sequentially and may subscribe for chunk-ready notifications. |
| Buffer status | `6f5f1002-4d8b-4d8d-9d7d-000000000001` | Read | `{total_bytes: u32, bytes_remaining: u32, checksum_crc32: u32}` | Little-endian, 12 bytes. |
| Pen metadata | `6f5f1003-4d8b-4d8d-9d7d-000000000001` | Read | `{fw_version: u16, battery_pct: u8, pen_serial: u32, page_count: u8}` | Little-endian, 8 bytes. |
| Sync control | `6f5f1004-4d8b-4d8d-9d7d-000000000001` | Write | `u8`: `0x01` start, `0x02` abort, `0x03` clear buffer | `0x03` only after hub confirms durable dual-write. |

---

## 2. Invigilator GATT Service (Peripheral Role on Hub)

Service UUID: `6f5f0002-4d8b-4d8d-9d7d-000000000002`

| Characteristic | UUID | Properties | Payload Format | Notes |
|---|---|---|---|---|
| Auth | `6f5f2001-4d8b-4d8d-9d7d-000000000002` | Write, Indicate | Write: 12-byte ASCII code. Indicate: one-byte result (`0x01` accept, `0x00` reject). | No command writes accepted until auth succeeds. |
| Command | `6f5f2002-4d8b-4d8d-9d7d-000000000002` | Write | Command header + payload defined in §4 | Idempotent commands must reuse `request_id`. |
| Status feed | `6f5f2003-4d8b-4d8d-9d7d-000000000002` | Notify | UTF-8 JSON object defined in §5 | 1 Hz update cadence. |
| MAC list | `6f5f2004-4d8b-4d8d-9d7d-000000000002` | Read, Notify | UTF-8 JSON array of pen discovery rows | Sent after registration scan updates. |
| Data relay | `6f5f2005-4d8b-4d8d-9d7d-000000000002` | Notify | Same chunk frame as §3 | Enabled only when hub upload path is `mobile`. |

---

## 3. Chunk Wire Format

All binary chunk transfers use the same frame:

| Offset | Size | Field | Type |
|---|---|---|---|
| 0 | 1 | `version` | `u8`, currently `0x01` |
| 1 | 1 | `flags` | bitfield |
| 2 | 2 | `header_len` | `u16`, little-endian |
| 4 | 4 | `chunk_index` | `u32`, little-endian |
| 8 | 4 | `total_chunks` | `u32`, little-endian |
| 12 | 4 | `payload_len` | `u32`, little-endian |
| 16 | 4 | `payload_crc32` | `u32`, little-endian |
| 20 | N | `payload` | raw bytes |

Flag bits:

- `0x01`: first chunk
- `0x02`: last chunk
- `0x04`: retransmission
- `0x08`: relay chunk (hub to mobile relay)

MTU and fragmentation rules:

- Minimum negotiated ATT MTU: 185 bytes.
- Preferred ATT MTU: 247 bytes.
- A single GATT read/notify may carry only part of a chunk frame.
- Receiver reassembles by byte length until `header_len + payload_len` is satisfied.
- Receiver verifies `payload_crc32` before acknowledging the chunk.

Acknowledgement rule:

- Pen advances to the next chunk only after the hub sends the next read or notify confirmation.
- Hub sends `Sync control = 0x03` only after all chunks are read, dual-written, and whole-buffer checksum matches the value from `Buffer status`.

---

## 4. Command IDs

Command characteristic payload:

| Offset | Size | Field | Type |
|---|---|---|---|
| 0 | 1 | `cmd_id` | `u8` |
| 1 | 16 | `request_id` | UTF-8/ASCII request token |
| 17 | N | `payload` | command-specific |

Command catalog:

| `cmd_id` | Name | Payload |
|---|---|---|
| `0x01` | `start_exam` | `{exam_id, duration_sec}` as UTF-8 JSON |
| `0x02` | `stop_exam` | `{exam_id, reason}` as UTF-8 JSON |
| `0x03` | `start_registration_scan` | `{exam_id}` as UTF-8 JSON |
| `0x04` | `manual_register` | `{exam_id, pen_mac, student_id}` as UTF-8 JSON |
| `0x05` | `start_upload` | `{exam_id, path}` as UTF-8 JSON |
| `0x06` | `request_snapshot` | `{screen}` as UTF-8 JSON |

Command semantics:

- `request_id` is required for idempotency.
- Replayed `request_id` values return the last known outcome instead of re-running a side effect.
- Unknown command IDs return `error_code = unsupported_command`.

---

## 5. Status Feed Schema

`Status feed` characteristic sends one UTF-8 JSON object per notification:

```json
{
  "exam_id": "9a3d24a8-bf38-4b09-9c93-a31ad087d7f9",
  "state": "pen_sync",
  "timer_remaining_sec": 0,
  "wifi": { "connected": true, "band": "5GHz", "signal_dbm": -42 },
  "storage": { "sd_ok": true, "usb_ok": true, "degraded": false },
  "sync": { "complete": 34, "in_progress": 3, "failed": 0, "pending": 3 }
}
```

Required JSON fields:

- `exam_id`
- `state`
- `timer_remaining_sec`
- `wifi.connected`
- `storage.degraded`
- `sync.complete`
- `sync.in_progress`
- `sync.failed`
- `sync.pending`

---

## 6. Error Codes and Retry Semantics

Error codes:

| Code | Meaning | Retry |
|---|---|---|
| `auth_failed` | Invigilator auth code rejected | No, unless code changes |
| `unsupported_command` | Unknown `cmd_id` | No |
| `invalid_payload` | JSON or binary payload malformed | No |
| `busy` | Hub currently executing an incompatible action | Yes |
| `chunk_crc_mismatch` | Payload CRC mismatch after reassembly | Yes |
| `buffer_checksum_mismatch` | Whole-pen checksum mismatch | Yes, full re-sync |
| `storage_not_durable` | Dual-write not yet confirmed | Yes |
| `pen_unreachable` | BLE device disconnected or timed out | Yes |

Retry rules:

- Chunk read retries: 3 attempts per chunk before escalating to pen-level sync failure.
- Pen reconnect retries: 3 attempts per pen, 30 seconds per attempt.
- Busy-command retries: caller may retry after 1 second backoff.
- Auth failures: lock out for 5 minutes after 5 consecutive failed codes.

---

## 7. Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Replaced the placeholder BLE draft with concrete UUIDs, chunk framing, command IDs, status-feed schema, and retry/error semantics. | Codex |
