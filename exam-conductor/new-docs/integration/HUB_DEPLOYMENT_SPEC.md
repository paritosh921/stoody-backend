# HUB_DEPLOYMENT_SPEC.md
# ExamPen Hub — Deployment, Configuration & Operations Specification

**Status:** ACTIVE
**Authority:** ExamPen hub hardware, edge software, local storage, provisioning, and upload behavior.

Reference: `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/TAMPER_PROOF_SPEC.md`

---

## Codebase Location — CRITICAL

| What | Location | Notes |
|------|----------|-------|
| **ExamPen hub code** | `stoody-pen-multi/exam-hub/` | **NEW directory — to be created.** All ExamPen hub implementation goes here. |
| **Stoody smartboard hub** | `stoody-pen-multi/edge_hub/` | **DO NOT MODIFY for ExamPen.** This is the existing Stoody smartboard hub. It has its own BLE manager, TUI, and PWA server for the teacher monitoring dashboard. |
| **Shared mobile app** | `stoody-pen-multi/mobile-app/` | The invigilator mobile app extends this for ExamPen BLE commands. Shared between smartboard and ExamPen. |
| **Backend ingest API** | `backend/api/v1/evalpen_submissions_async.py`, `backend/api/v1/hub.py` | Hub uploads go to `POST /api/v1/hub/exam-upload` or `POST /api/v1/evalpen/submissions`. |
| **Hub provisioning API** | `backend/api/v1/superadmin_async.py` (to be added) | See `integration/SUPERADMIN_SPEC.md`. |

### Hard Boundary Rule

> **The ExamPen hub is a SEPARATE edge device from the Stoody smartboard hub.**
>
> - `stoody-pen-multi/edge_hub/` = Stoody smartboard hub (teacher monitoring, PWA, multi-pen dashboard). **DO NOT TOUCH.**
> - `stoody-pen-multi/exam-hub/` = ExamPen conducted-exam hub (artifact collection, dual-write, exam timer, invigilator BLE). **BUILD HERE.**
>
> They may share the same Raspberry Pi hardware in some deployments, but the software stacks are independent. The exam-hub has its own systemd services, SQLite DB, TUI, and BLE manager.
>
> The `stoody-pen-multi/mobile-app/` is shared — it connects to whichever hub (smartboard or exam) is nearby via BLE.

---

## Role in Current Architecture

The hub is part of the **shared ingest substrate**. Its job is collection, local durability, and upload of canonical conducted-exam artifacts. It does **not** perform DCR or PCR evaluation.

## 1. Base OS & Image

### 1.1 OS Choice

| Parameter | Value | Rationale |
|---|---|---|
| OS | Ubuntu Server 24.04 LTS (Noble Numbat) | Long-term support, headless, minimal footprint, BlueZ 5.72+ |
| Architecture | arm64 | RPi 4B/5 native |
| Image type | Pre-built golden image (`.img.xz`) | Reproducible fleet deployment |
| Kernel | linux-raspi (Ubuntu-maintained) | Hardware-specific patches for RPi |
| Init system | systemd | Service management, watchdog, journal |

### 1.2 Region & Locale Configuration — CRITICAL

```
LANG=en_US.UTF-8
LC_ALL=en_US.UTF-8
TIMEZONE=UTC
COUNTRY=US
```

**WiFi regulatory domain MUST be set to US:**

```bash
# /etc/default/crda
REGDOMAIN=US

# iw reg set US (runtime)
# Persisted via /etc/modprobe.d/cfg80211.conf:
#   options cfg80211 ieee80211_regdom=US
```

**US WiFi band availability:**

| Band | Channels | Max EIRP | Use Case |
|---|---|---|---|
| 2.4 GHz | 1–11 | 30 dBm | Fallback, crowded, longer range |
| 5 GHz (UNII-1) | 36–48 | 23 dBm (indoor) | Primary for hub uplink — less interference |
| 5 GHz (UNII-3) | 149–165 | 30 dBm | Alternative if UNII-1 congested |
| 6 GHz (WiFi 6E) | Not available on RPi 4/5 | — | Not applicable |

**US region is the locked configuration.** This setting must not be changed. TUI setup screen does not expose region selection — it is baked into the golden image.

### 1.3 Golden Image Contents

```
ubuntu-server-24.04-arm64-raspi.img
├── OS base (headless, no desktop)
├── Packages pre-installed:
│   ├── bluez (5.72+)
│   ├── bluez-tools
│   ├── python3.12, python3-pip, python3-venv
│   ├── sqlite3
│   ├── NetworkManager (nmcli for WiFi)
│   ├── chrony (NTP client)
│   ├── htop, iotop, lsof, strace (debug)
│   ├── usbutils (lsusb for dongle enum)
│   └── openssh-server
├── ExamPen hub software (systemd services)
├── /etc/exampen/ (config directory)
├── /var/lib/exampen/ (data directory — SD primary)
└── cloud-init disabled (golden image, not cloud-init provisioned)
```

### 1.4 Partition Layout

| Partition | Mount | Size | Filesystem | Purpose |
|---|---|---|---|---|
| boot | /boot/firmware | 512 MB | FAT32 | RPi bootloader + kernel |
| rootfs | / | 8 GB | ext4 | OS + hub software |
| data-primary | /var/lib/exampen | 16 GB | ext4, noatime | Pen data primary store |
| swap | — | 1 GB | swap | OOM safety |

**USB thumb drive (secondary store):**

| Mount | Filesystem | Purpose |
|---|---|---|
| /mnt/exampen-backup | ext4, noatime | Pen data secondary copy (independent failure domain) |

Auto-mount via `/etc/fstab` with `nofail` flag — hub must boot even if USB drive is missing. `hub-store` module detects missing secondary and logs warning to TUI + invigilator app.

---

## 2. TUI Specification

### 2.1 Technology

| Parameter | Value |
|---|---|
| Language | Python 3.12 |
| TUI framework | Textual (rich-based, async-native, testable) |
| Terminal | HDMI console or USB serial (115200 baud) |
| Minimum terminal | 80×24 |

### 2.2 Screen Map

```
┌─────────────────────────────────────────┐
│  ExamPen Hub TUI                        │
│                                         │
│  [1] Setup          → Initial config    │
│  [2] Status         → Live dashboard    │
│  [3] WiFi           → Network config    │
│  [4] Dongles        → BLE dongle mgmt   │
│  [5] Exams          → Session history   │
│  [6] Diagnostics    → Test suite runner │
│  [7] Logs           → Log viewer        │
│  [8] Shutdown       → Safe power off    │
│                                         │
│  Hub ID: EPH-00042  State: READY        │
│  Uptime: 4h 23m     IP: 192.168.1.105  │
└─────────────────────────────────────────┘
```

### 2.3 Screen Details

#### [1] Setup Screen

| Field | Input | Validation | Persists To |
|---|---|---|---|
| Hub unique code | Alphanumeric, 12 chars | Backend verification on first WiFi connection | `/etc/exampen/hub.conf` |
| Backend URL | HTTPS URL | DNS resolve + TLS handshake test | `/etc/exampen/hub.conf` |
| Uplink mode | `wifi` / `mobile` / `auto` | Enum | `/etc/exampen/hub.conf` |

#### [2] Status Screen (Live Dashboard)

```
┌─ Hub Status ────────────────────────────┐
│ State: EXAM_ARMED  Timer: 47:23 remain  │
│ WiFi: Connected (5GHz, Ch 36, -42 dBm)  │
│ Backend: Reachable (latency: 34ms)       │
│ Invigilator: Connected (BLE, auth OK)    │
│                                          │
│ ┌─ Dongles ──────────────────────────┐   │
│ │ D1 (hci0/AA:BB:CC:DD:EE:01): 8/8  │   │
│ │ D2 (hci1/AA:BB:CC:DD:EE:02): 7/8  │   │
│ │ D3 (hci2/AA:BB:CC:DD:EE:03): 8/8  │   │
│ │ D4 (hci3/AA:BB:CC:DD:EE:04): 8/8  │   │
│ │ D5 (hci4/AA:BB:CC:DD:EE:05): 6/8  │   │
│ │ Total: 37/40 pens connected        │   │
│ └────────────────────────────────────┘   │
│                                          │
│ ┌─ Sync Progress ────────────────────┐   │
│ │ ████████████░░░░ 37/40 (92%)       │   │
│ │ Complete: 34  In-progress: 3       │   │
│ │ Failed: 0     Pending: 3           │   │
│ └────────────────────────────────────┘   │
│                                          │
│ Storage: SD 2.1/16 GB  USB 2.1/14 GB    │
└──────────────────────────────────────────┘
```

Refresh: 1 Hz via async polling of `hub-supervisor` IPC (see `hub/ipc-protocol.md`).

#### [3] WiFi Screen

| Action | Method |
|---|---|
| Scan networks | `nmcli device wifi list` |
| Connect | `nmcli device wifi connect <SSID> password <pass>` |
| Forget | `nmcli connection delete <SSID>` |
| Band preference | `nmcli connection modify <SSID> 802-11-wireless.band a` (force 5 GHz) |
| Status | Signal strength, channel, band, IP, gateway, DNS |

**Auto-connect on boot:** NetworkManager handles this. Hub software waits for `NetworkManager-wait-online.service` before attempting backend sync.

#### [4] Dongle Management Screen

| Column | Source |
|---|---|
| Dongle ID | Stable MAC from `hciconfig` |
| hci path | `/sys/class/bluetooth/hciX` |
| USB port | `udevadm info --query=path` |
| Firmware | `hciconfig hciX version` |
| Connected pens | Count from `hub-ble-mgr` IPC |
| Health | OK / DEGRADED / FAILED |

Actions: Reset dongle (`hciconfig hciX reset`), Remove dongle, Rescan USB.

#### [5] Exam History Screen

| Column | Source |
|---|---|
| Exam ID | Local SQLite |
| Date | Timestamp |
| Duration | Minutes |
| Pens synced | Count |
| Upload status | `complete` / `partial` / `pending` |
| Invigilator | ID |

Tap exam → detail view with per-pen breakdown.

#### [6] Diagnostics Screen → Test Suite TUI

See `TEST_SUITE_SPEC.md` for full specification.

Quick summary: runs hardware checks (dongles, WiFi, USB, SD, NTP), software checks (services, IPC, DB), and BLE connectivity checks. Results displayed in TUI with pass/fail/warn per test.

#### [7] Log Viewer

| Log Source | Path | Viewer |
|---|---|---|
| Hub supervisor | `journalctl -u exampen-supervisor` | Scrollable, filterable by level |
| BLE manager | `journalctl -u exampen-ble-mgr` | Per-dongle filter |
| Pen sync | `/var/log/exampen/sync.log` | Per-pen filter |
| Uplink | `/var/log/exampen/uplink.log` | Per-upload filter |
| Invigilator BLE | `/var/log/exampen/invig.log` | Session filter |

All logs also forwarded to `journald` for unified access.

#### [8] Shutdown

Sequence:
1. Check for active exam session → warn if active
2. Check for pending uploads → warn if pending
3. Sync filesystem (`sync`)
4. Unmount USB (`umount /mnt/exampen-backup`)
5. `systemctl poweroff`

---

## 3. Local Data Storage Schema

### 3.1 SQLite Database

Path: `/var/lib/exampen/hub.db`

**Tables:**

```sql
-- Hub identity and config (singleton)
CREATE TABLE hub_config (
    hub_id          TEXT PRIMARY KEY,
    backend_url     TEXT NOT NULL,
    uplink_mode     TEXT NOT NULL DEFAULT 'wifi' CHECK (uplink_mode IN ('wifi','mobile','auto')),
    region          TEXT NOT NULL DEFAULT 'US',
    provisioned_at  TEXT NOT NULL,  -- ISO 8601
    last_backend_sync TEXT          -- ISO 8601
);

-- Cached invigilator codes (rotated daily, pre-cached for N days)
CREATE TABLE invig_codes (
    code            TEXT PRIMARY KEY,
    valid_from      TEXT NOT NULL,  -- ISO 8601
    valid_until     TEXT NOT NULL,  -- ISO 8601
    fetched_at      TEXT NOT NULL
);

-- Pen inventory (registered to this hub's institute)
CREATE TABLE pen_inventory (
    pen_mac         TEXT PRIMARY KEY,
    pen_serial      TEXT,
    fw_version      TEXT,
    registered_at   TEXT NOT NULL,
    last_seen       TEXT,
    battery_pct     INTEGER
);

-- Exam sessions
CREATE TABLE exam_sessions (
    exam_id         TEXT PRIMARY KEY,
    invig_id        TEXT NOT NULL,
    started_at      TEXT,           -- NULL until timer starts
    duration_min    INTEGER NOT NULL,
    timer_expires   TEXT,           -- computed: started_at + duration_min
    state           TEXT NOT NULL DEFAULT 'created'
                    CHECK (state IN ('created','armed','timer_running',
                           'dongle_activation','pen_sync','sync_complete',
                           'sync_partial','uploading','upload_complete','cancelled')),
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);

-- Pen-student bindings per exam
CREATE TABLE pen_bindings (
    exam_id         TEXT NOT NULL REFERENCES exam_sessions(exam_id),
    pen_mac         TEXT NOT NULL,
    student_id      TEXT,           -- NULL until resolved from backend
    student_name    TEXT,
    student_roll    TEXT,
    status          TEXT NOT NULL DEFAULT 'discovered'
                    CHECK (status IN ('discovered','provisional','confirmed','rejected')),
    source          TEXT NOT NULL DEFAULT 'scan'
                    CHECK (source IN ('scan','manual_register','server_sync')),
    server_confirmed_at TEXT,
    rejection_reason TEXT,
    bound_at        TEXT NOT NULL,
    PRIMARY KEY (exam_id, pen_mac)
);

-- Per-pen sync status per exam
CREATE TABLE pen_sync_status (
    exam_id         TEXT NOT NULL,
    pen_mac         TEXT NOT NULL,
    dongle_mac      TEXT,           -- which dongle handled this pen
    sync_started    TEXT,
    sync_completed  TEXT,
    bytes_expected  INTEGER,
    bytes_received  INTEGER,
    checksum_expected TEXT,
    checksum_actual TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','connecting','syncing',
                           'complete','failed','timeout')),
    error_detail    TEXT,
    PRIMARY KEY (exam_id, pen_mac)
);

-- Per-pen upload ledger
CREATE TABLE upload_ledger (
    exam_id         TEXT NOT NULL,
    pen_mac         TEXT NOT NULL,
    total_chunks    INTEGER NOT NULL,
    acked_chunks    TEXT NOT NULL DEFAULT '[]',  -- JSON array of indices
    upload_path     TEXT CHECK (upload_path IN ('wifi','mobile')),
    complete        INTEGER NOT NULL DEFAULT 0,
    started_at      TEXT,
    completed_at    TEXT,
    PRIMARY KEY (exam_id, pen_mac)
);

-- Dongle registry (persisted across reboots)
CREATE TABLE dongle_registry (
    dongle_mac      TEXT PRIMARY KEY,
    hci_path        TEXT,           -- updated on each boot
    usb_port_path   TEXT,           -- sysfs path for stable identification
    first_seen      TEXT NOT NULL,
    last_healthy    TEXT,
    status          TEXT NOT NULL DEFAULT 'unknown'
                    CHECK (status IN ('unknown','healthy','degraded','failed'))
);

-- Interaction log (every hub action, for forensic audit)
CREATE TABLE interaction_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,  -- ISO 8601 with ms
    source          TEXT NOT NULL,  -- module name
    event_type      TEXT NOT NULL,  -- e.g., 'invig_auth', 'exam_start', 'pen_sync_complete'
    exam_id         TEXT,
    pen_mac         TEXT,
    invig_id        TEXT,
    detail          TEXT,           -- JSON blob for event-specific data
    severity        TEXT NOT NULL DEFAULT 'info'
                    CHECK (severity IN ('debug','info','warn','error','critical'))
);

-- Timer persistence (for reboot recovery)
CREATE TABLE active_timer (
    exam_id         TEXT PRIMARY KEY,
    start_epoch     INTEGER NOT NULL,  -- Unix epoch seconds
    duration_sec    INTEGER NOT NULL,
    remaining_sec   INTEGER NOT NULL,  -- updated every 10s
    last_updated    INTEGER NOT NULL   -- epoch
);
```

### 3.2 File Storage Layout

```
/var/lib/exampen/
├── hub.db                          # SQLite database
├── hub.db-wal                      # WAL mode enabled
├── hub.db-shm
├── data/
│   └── {exam_id}/
│       └── {pen_mac}/
│           ├── strokes_raw.bin     # Raw bytes from pen GATT read
│           ├── strokes.meta.json   # {bytes, checksum_crc32, pages, sync_ts}
│           └── chunks/
│               ├── chunk_000.bin   # Pre-chunked for upload
│               ├── chunk_001.bin
│               └── ...
└── logs/
    ├── sync.log
    ├── uplink.log
    └── invig.log

/mnt/exampen-backup/                # USB thumb drive (secondary copy)
├── data/
│   └── {exam_id}/
│       └── {pen_mac}/
│           ├── strokes_raw.bin     # Byte-identical copy
│           └── strokes.meta.json
└── hub.db.backup                   # Periodic SQLite backup
```

### 3.3 Dual-Write Protocol

1. `hub-pen-sync` receives chunk from pen GATT.
2. Write chunk to SD path (`/var/lib/exampen/data/{exam_id}/{pen_mac}/strokes_raw.bin`, append mode).
3. `fsync()` SD file descriptor.
4. Write identical chunk to USB path (`/mnt/exampen-backup/data/{exam_id}/{pen_mac}/strokes_raw.bin`, append mode).
5. `fsync()` USB file descriptor.
6. Only after both `fsync()` succeed → ACK pen to send next chunk.
7. If USB write fails → log warning, continue with SD-only, set `hub-store` degraded flag → TUI shows amber warning.

### 3.4 Interaction Log Protocol

Every state transition, command, and event is logged to `interaction_log` table. This is the forensic audit trail.

**Events logged:**

| Event Type | Source | Detail |
|---|---|---|
| `hub_boot` | supervisor | `{uptime, os_version, sw_version}` |
| `invig_connect` | invig-ble | `{invig_id, ble_addr}` |
| `invig_auth_ok` | invig-ble | `{invig_id, code_used}` |
| `invig_auth_fail` | invig-ble | `{attempted_code, reason}` |
| `pen_registration_start` | ble-mgr | `{exam_id}` |
| `pen_discovered` | ble-mgr | `{pen_mac, dongle_mac, rssi}` |
| `exam_timer_start` | timer | `{exam_id, duration_min}` |
| `exam_timer_expire` | timer | `{exam_id}` |
| `dongle_activated` | ble-mgr | `{dongle_mac, hci_path}` |
| `pen_connected` | pen-sync | `{pen_mac, dongle_mac}` |
| `pen_sync_start` | pen-sync | `{pen_mac, bytes_expected}` |
| `pen_sync_chunk` | pen-sync | `{pen_mac, chunk_idx, bytes}` |
| `pen_sync_complete` | pen-sync | `{pen_mac, checksum_match: bool}` |
| `pen_sync_fail` | pen-sync | `{pen_mac, error, bytes_received}` |
| `upload_start` | uplink | `{pen_mac, path: wifi/mobile}` |
| `upload_chunk_ack` | uplink | `{pen_mac, chunk_idx}` |
| `upload_complete` | uplink | `{pen_mac}` |
| `upload_fail` | uplink | `{pen_mac, error, last_acked_chunk}` |
| `dongle_health_change` | ble-mgr | `{dongle_mac, old_status, new_status}` |
| `wifi_state_change` | uplink | `{connected: bool, ssid, signal, channel}` |
| `usb_storage_state` | store | `{mounted: bool, free_bytes}` |
| `hub_shutdown` | supervisor | `{reason, pending_uploads}` |

---

## 4. Pen Management

### 4.1 Pen Lifecycle on Hub

```
Unknown → Discovered (registration scan) → Bound (exam session) → Syncing → Synced → Uploaded → Released
```

### 4.2 Pen State Transitions

| From | To | Trigger | Hub Action |
|---|---|---|---|
| Unknown | Discovered | Registration scan detects MAC | Insert `pen_inventory`, insert `pen_bindings` with `status='discovered'`, `source='scan'` |
| Discovered / Provisional | Bound | Backend resolves MAC→student | Update `pen_bindings.student_id/name/roll`, set `status='confirmed'`, set `server_confirmed_at` |
| Bound | Syncing | Post-exam, pen connects to dongle | Start GATT read, update `pen_sync_status` |
| Syncing | Synced | All chunks received, checksum match | Mark `pen_sync_status.status = 'complete'` |
| Syncing | Failed | Timeout or checksum mismatch | Mark `pen_sync_status.status = 'failed'`, log error |
| Synced | Uploaded | All chunks ACKd by backend | Mark `upload_ledger.complete = 1` |
| Uploaded | Released | Exam session closed | Pen available for next session |

### 4.3 Manual Pen Registration (Mid-Exam)

When a pen wasn't detected during pre-exam registration:

1. Invigilator selects "Manual Register" on mobile app.
2. App sends command to hub via BLE: `{cmd: 'manual_register', pen_mac: 'XX:XX:XX:XX:XX:XX', student_id: 'S123'}`.
3. Hub inserts into `pen_bindings` with `status = 'provisional'`. This is a display hint only — NOT an authoritative binding.
4. Mobile app queues a server-side binding request to the exam orchestration owner. Sent immediately if network available, queued otherwise.
5. Server validates (pen MAC in inventory, student in roster, no conflicting binding) and responds with `status = 'confirmed'` or rejects.
6. Hub updates `pen_bindings.status` to `confirmed` or `rejected` when server response arrives (via mobile relay or WiFi).
7. During post-exam sync, hub includes this MAC in the expected pen list regardless of status — stroke data is always captured. Scoring pipeline uses server-confirmed bindings only.
8. If pen connects but binding is still `provisional` (server unreachable), strokes are synced and stored tagged with pen_mac. Server resolves MAC→student when binding is eventually confirmed.

**Ownership rule:** Hub never creates authoritative bindings. The exam orchestration owner remains the single writable owner. Hub holds provisional records for display and local workflow continuity.

---

## 5. WiFi Management

### 5.1 Connection Priority

1. **Primary:** 5 GHz network (less BLE interference, higher throughput).
2. **Fallback:** 2.4 GHz network (if 5 GHz unavailable).
3. **Band selection:** Enforced via `nmcli connection modify <conn> 802-11-wireless.band a` for 5 GHz preference.

### 5.2 Pre-Exam WiFi Verification

Before invigilator can start exam, hub runs:

| Check | Method | Pass Criteria |
|---|---|---|
| WiFi associated | `nmcli -t -f WIFI g` | `enabled` |
| IP assigned | `nmcli -t -f IP4.ADDRESS device show wlan0` | Non-empty |
| Gateway reachable | `ping -c 3 <gateway>` | ≥2/3 success |
| Backend reachable | `curl -s -o /dev/null -w '%{http_code}' https://<backend>/health` | 200 |
| NTP synced | `chronyc tracking` | `Leap status: Normal` |
| Band | `iw dev wlan0 info` | Prefer `5180–5825 MHz` |
| Signal | `iw dev wlan0 link` | ≥ -70 dBm |

Results displayed on TUI WiFi screen and pushed to invigilator app.

### 5.3 WiFi During Exam

WiFi is NOT required during exam (pens are offline). Hub may or may not be connected. Timer runs locally.

Post-exam, hub checks WiFi before upload. If unavailable, falls back to mobile relay path.

### 5.4 Captive Portal Handling

School WiFi may have captive portals. Hub cannot handle browser-based portals.

**Mitigation:** TUI WiFi screen shows "Connected but no internet" if gateway is reachable but backend is not. Invigilator must whitelist hub MAC on school network or use a mobile hotspot.

---

## 6. systemd Service Definitions

```ini
# /etc/systemd/system/exampen-supervisor.service
[Unit]
Description=ExamPen Hub Supervisor
After=network-online.target bluetooth.target
Wants=network-online.target bluetooth.target

[Service]
Type=notify
ExecStart=/opt/exampen/bin/hub-supervisor
Restart=always
RestartSec=5
WatchdogSec=30
StandardOutput=journal
StandardError=journal
Environment=EXAMPEN_DATA=/var/lib/exampen
Environment=EXAMPEN_BACKUP=/mnt/exampen-backup

[Install]
WantedBy=multi-user.target
```

**Service dependency tree:**

```
exampen-supervisor (Type=notify, watchdog=30s)
├── exampen-ble-mgr (forked by supervisor)
├── exampen-pen-sync (forked by supervisor)
├── exampen-timer (forked by supervisor)
├── exampen-store (forked by supervisor)
├── exampen-uplink (forked by supervisor)
├── exampen-invig-ble (forked by supervisor)
└── exampen-tui (forked by supervisor, optional — only if HDMI/serial connected)
```

Supervisor manages all child processes. If a child crashes, supervisor restarts it and logs to `interaction_log`.

---

## 7. First-Boot Provisioning Sequence

1. Power on RPi with golden image SD card + USB thumb drive.
2. systemd boots → `exampen-supervisor` starts → detects first-boot (no `/etc/exampen/hub.conf`).
3. TUI launches → Setup Screen forced.
4. Operator enters: hub unique code, backend URL, WiFi credentials, uplink mode.
5. Hub connects to WiFi → verifies backend → sends provisioning request: `POST /api/v1/hubs/provision {hub_code}`.
6. Backend responds: `{hub_id, institute_id, invig_codes: [...], pen_inventory: [...]}`.
7. Hub stores config, caches invig codes, populates `pen_inventory`.
8. Hub state → `PROVISIONED` → TUI shows "Ready" on status screen.
9. USB thumb drive formatted and mounted (if not already).
10. First-boot complete. Hub ready for invigilator connection.

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Removed mutable region setup, added explicit `pen_bindings` workflow fields, and aligned first-boot/setup flow with the locked US regulatory configuration. | Codex |
