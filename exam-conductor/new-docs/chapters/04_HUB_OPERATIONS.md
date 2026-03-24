# Chapter 04: Hub Operations

## Status
- **Phase:** W6 — Documentation
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6.A6.1)
- **Build status:** DRAFT

## Overview

The ExamPen hub is a Raspberry Pi 4B/5 running Ubuntu Server 24.04 LTS (arm64) that manages BLE pen communication, local stroke storage, exam timing, and data upload. This chapter covers first-boot provisioning, the exam flow lifecycle, TUI screens, and failure recovery.

## Architecture Context

The hub operates autonomously during exams. The server does NOT override hub FSM mid-exam. Post-exam, the server reconciles. The hub communicates with the invigilator via BLE and with the backend via WiFi or mobile BLE relay.

```
+-------------------+     BLE      +-----------+     WiFi/BLE     +---------+
| 5 USB BLE Dongles |<------------>| RPi Hub   |<--------------->| Backend |
| (40 pens max)     |     GATT     | Supervisor|     HTTP        | Cloud   |
+-------------------+              +-----+-----+                 +---------+
                                         |
                                    BLE  |  HDMI/Serial
                                         v
                                   +-----------+    +-----+
                                   | Invig App |    | TUI |
                                   | (Flutter) |    |     |
                                   +-----------+    +-----+
```

## First-Boot Provisioning

**Sequence:**

1. Power on RPi with golden image SD card + USB thumb drive.
2. systemd boots -> `exampen-supervisor` starts -> detects first boot (no `/etc/exampen/hub.conf`).
3. TUI launches -> Setup Screen forced.
4. Operator enters: hub unique code (12-char alphanumeric), backend URL (HTTPS), uplink mode (wifi/mobile/auto).
5. WiFi credentials entered via WiFi screen.
6. Hub connects -> verifies backend -> sends provisioning request: `POST /api/v1/hubs/provision {hub_code}`.
7. Backend responds: `{hub_id, institute_id, invig_codes: [...], pen_inventory: [...]}`.
8. Hub stores config in `/etc/exampen/hub.conf`, caches invig codes, populates `pen_inventory` table.
9. Hub state -> `PROVISIONED` -> TUI shows "Ready" on status screen.
10. USB thumb drive formatted and mounted at `/mnt/exampen-backup` (if not already).

**Golden image contents:** OS base (headless), BlueZ 5.72+, Python 3.12, SQLite3, NetworkManager, chrony, ExamPen hub software as systemd services. WiFi regulatory domain locked to US.

## Exam Flow Lifecycle

### Hub FSM States

```
created -> armed -> timer_running -> dongle_activation -> pen_sync
                                                            |
                                          +-----------------+
                                          v
                                    sync_complete -----> uploading -> upload_complete
                                    sync_partial -----> uploading -> upload_complete
                                    cancelled
```

| State | Description | Trigger |
|---|---|---|
| `created` | Exam session created on hub via invig BLE command | `start_exam` command (cmd_id `0x01`) |
| `armed` | Exam configured, awaiting timer start | Invigilator confirms |
| `timer_running` | Countdown active, pens are offline writing | Timer armed via IPC |
| `dongle_activation` | Timer expired, dongles activated for BLE scan | `timer.expired.event` from `hub-timer` |
| `pen_sync` | Pens connecting and syncing stroke data | Dongles scanning, pens connecting |
| `sync_complete` | All registered pens synced successfully | All pens in `complete` status |
| `sync_partial` | Some pens failed to sync | Timeout reached with incomplete pens |
| `uploading` | Chunks being sent to backend | Upload started via WiFi or mobile relay |
| `upload_complete` | All chunks ACKed by backend | Upload ledger fully complete |
| `cancelled` | Exam cancelled by invigilator | `stop_exam` command (cmd_id `0x02`) |

**Ownership rule:** Hub FSM state is persisted to SQLite `exam_sessions.state` BEFORE side effects execute. If a side effect fails, state is already updated and the side effect is retried on the next supervisor tick.

### Exam Flow: Step by Step

**1. Arm (Invigilator starts exam)**
- Invigilator authenticates via BLE auth characteristic (12-byte rotating code).
- Sends `start_exam` command: `{exam_id, duration_sec}`.
- Hub creates `exam_sessions` row with `state = 'created'`.
- Pre-exam WiFi verification runs (signal, backend reachable, NTP synced).
- FSM transitions: `created` -> `armed` -> `timer_running`.

**2. Timer (Exam in progress)**
- `hub-timer` uses `CLOCK_MONOTONIC` (immune to NTP adjustments during countdown).
- Timer state persisted to SQLite `active_timer` table every 10 seconds.
- NTP sync only needed at timer start, not during.
- Hub is autonomous during exam. WiFi NOT required.

**3. Dongle Activation (Timer expires)**
- `timer.expired.event` fires from `hub-timer` to `hub-supervisor`.
- Supervisor activates all 5 BLE dongles with 500ms stagger (mitigation H5).
- Passive scan first (lower RF interference), then active scan for connection.
- FSM: `timer_running` -> `dongle_activation`.

**4. Pen Sync (Post-exam data collection)**
- Pens advertise, dongles discover and connect (up to 8 pens per dongle).
- `hub-pen-sync` reads chunks via GATT, passes to `hub-store` for dual-write.
- Per-pen status tracked in `pen_sync_status` table.
- FSM: `dongle_activation` -> `pen_sync`.

**5. Upload (Data to backend)**
- After sync, `hub-uplink` uploads chunks via WiFi (primary) or mobile BLE relay (fallback).
- Per-chunk backend ACK, ledger updated.
- FSM: `sync_complete`/`sync_partial` -> `uploading` -> `upload_complete`.

## TUI Screens Overview

| # | Screen | Purpose | Key Information |
|---|---|---|---|
| 1 | **Setup** | Initial configuration | Hub code, backend URL, uplink mode |
| 2 | **Status** | Live dashboard (1 Hz refresh) | FSM state, timer, dongles, sync progress, storage |
| 3 | **WiFi** | Network management | Scan, connect, band preference, signal strength |
| 4 | **Dongles** | BLE dongle management | Per-dongle: MAC, HCI path, pen count, health |
| 5 | **Exams** | Session history | Past exams: ID, date, pens synced, upload status |
| 6 | **Diagnostics** | Test suite runner | Hardware/software/BLE tests with pass/fail |
| 7 | **Logs** | Log viewer | Per-module, filterable by severity |
| 8 | **Shutdown** | Safe power off | Warns if active exam or pending uploads |

**Technology:** Python 3.12, Textual framework (rich-based, async-native). Minimum terminal: 80x24. Connected via HDMI console or USB serial (115200 baud).

### Status Screen Detail

```
+------------------------------------------+
| State: EXAM_ARMED  Timer: 47:23 remain   |
| WiFi: Connected (5GHz, Ch 36, -42 dBm)   |
| Backend: Reachable (latency: 34ms)        |
| Invigilator: Connected (BLE, auth OK)     |
|                                           |
| Dongles:                                  |
|  D1: 8/8  D2: 7/8  D3: 8/8              |
|  D4: 8/8  D5: 6/8  Total: 37/40         |
|                                           |
| Sync: ████████████░░░░ 37/40 (92%)       |
| Storage: SD 2.1/16 GB  USB 2.1/14 GB     |
+------------------------------------------+
```

## Failure Recovery

### Timer Reboot Recovery (Mitigation F4)

1. Timer state persisted to `active_timer` table every 10 seconds: `{start_epoch, duration_sec, remaining_sec}`.
2. On reboot, `hub-supervisor` checks `active_timer` table.
3. If active timer found: resume countdown from last persisted `remaining_sec`.
4. Maximum timer accuracy loss: 10 seconds (between last persist and crash).
5. Interaction log records `hub_boot` event with `{timer_recovered: true, lost_seconds: N}`.

### Dongle Failure (Mitigation H3)

1. `hub-ble-mgr` detects dongle drop via D-Bus signal `org.bluez.Adapter1.Removed`.
2. Pens from failed dongle are re-queued to other dongles with capacity.
3. If no capacity: pens marked as timeout.
4. TUI shows dongle status change.
5. Re-assigned pens must re-advertise and reconnect (10-30s delay).

### SD/USB Dual-Write Failure (Mitigation S4)

| Scenario | Behavior |
|---|---|
| USB write fails | Degrade to SD-only, TUI amber warning, continue |
| SD write fails | Critical alert, no fallback (data at risk) |
| Both fail | Data loss. Copy image upload is ultimate fallback |
| USB missing at boot | Hub boots normally (`nofail` fstab flag), logs warning |

### WiFi Unavailable Post-Exam

1. Hub checks WiFi before upload.
2. If unavailable: fall back to mobile BLE relay path via `hub-invig-ble`.
3. Estimated time for 40 pens via BLE relay: ~12 minutes (mitigation U1).
4. If neither available: data retained locally indefinitely.

## Interfaces

- **IPC protocol:** Unix domain sockets, JSON-lines encoding (`hub/ipc-protocol.md`).
- **BLE GATT (pen):** Service `6f5f0001-...`, characteristics for stroke buffer, buffer status, pen metadata, sync control.
- **BLE GATT (invigilator):** Service `6f5f0002-...`, characteristics for auth, command, status feed, MAC list, data relay.
- **Backend API:** `POST /api/v1/hubs/provision`, `POST /api/v1/strokes/upload`, `GET /api/v1/exams/{exam_id}/upload-status`.

## Configuration

| File | Contents |
|---|---|
| `/etc/exampen/hub.conf` | Hub ID, backend URL, uplink mode, region (US, locked) |
| `/var/lib/exampen/hub.db` | SQLite database (WAL mode): config, exams, pens, sync, uploads, logs |
| `/var/lib/exampen/data/` | SD primary stroke data |
| `/mnt/exampen-backup/` | USB secondary stroke data |

## Testing

- **Hardware:** HW-H1 (dongle enumeration), HW-H2 (dongle hot-plug), HW-B1 (BLE scan + connect), HW-B2 (multi-pen sync), HW-B3 (dual-write integrity), HW-T1 (timer accuracy), HW-W1 (WiFi connectivity), HW-I1 (invigilator BLE), HW-P1 (power failure recovery)
- **TUI diagnostics:** S1 (SQLite integrity), S2 (service health), S3 (IPC connectivity), S4 (backend reachability), S5 (invigilator code cache)

## Failure Modes & Mitigations

| ID | Failure | Mitigation |
|---|---|---|
| F1 | Timer drift (no RTC) | NTP sync at start, CLOCK_MONOTONIC during exam |
| F4 | Hub reboot during timer | SQLite persist every 10s, auto-resume |
| H1 | USB bus power brownout | Powered USB hub mandatory (5V/3A+) |
| H3 | Dongle failure mid-sync | Re-queue pens to other dongles |
| H5 | BLE advertising collision | Stagger dongle scan (500ms), passive scan first |
| S3 | BLE MITM on invigilator | Rotating 24h auth codes, BLE 4.2 LESC |
| S4 | SD card failure | Dual-write to USB, degraded mode |
| A1.1 | BLE connection limit exceeded | 5 dongles x 8 = 40 cap, overflow rejected |
| A1.5 | Pen battery death mid-exam | Low-battery warning at registration, copy image fallback |

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Initial draft: provisioning, exam flow, TUI, failure recovery | Claude Agent (W6.A6.1) |
