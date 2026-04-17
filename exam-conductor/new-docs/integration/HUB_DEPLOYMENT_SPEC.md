# HUB_DEPLOYMENT_SPEC.md
# ExamPen Hub — Deployment, Configuration & Operations Specification

**Status:** ACTIVE
**Authority:** ExamPen hub hardware, edge software, local storage, provisioning, and upload behavior.

Reference: `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/TAMPER_PROOF_SPEC.md`

---

## Codebase Location — CRITICAL

| What | Location | Notes |
|------|----------|-------|
| **ExamPen hub code** | `stoody-multi-pen/HUB-exam-conductor/` | Partial runtime implemented: supervisor, store, timer, TUI, BLE/uplink scaffolds, and provisioning cache exist. Supervisor now owns in-process wiring for BLE→pen-sync→uplink data path. Production packaging, systemd install artifacts, and hardware validation are pending. |
| **Stoody smartboard hub** | `stoody-multi-pen/edge_hub/` | **DO NOT MODIFY for ExamPen.** This is the existing Stoody smartboard hub. It has its own BLE manager, TUI, and PWA server for the teacher monitoring dashboard. |
| **Shared mobile app** | `stoody-multi-pen/mobile-app/` | The invigilator mobile app extends this for ExamPen BLE commands. Shared between smartboard and ExamPen. |
| **Backend ingest API** | `backend/api/v1/stroke_ingest_async.py` | Hub uploads go to `POST /api/v1/ingest/strokes/{exam_id}/{pen_mac}` (primary). Legacy bridge at `POST /api/v1/hub/exam-upload` is a compatibility surface only. |
| **Hub provisioning + operations API** | `backend/api/v1/hub_ops_async.py` | Provisioning at `POST /api/v1/hubs/provision`. See §7 for full contract. |
| **Super-admin hub provisioning** | `backend/api/v1/superadmin_async.py` | Admin generates provisioning codes. Hub consumes them via `POST /api/v1/hubs/provision`. See `integration/SUPERADMIN_SPEC.md`. |

### Hard Boundary Rule

> **The ExamPen hub is a SEPARATE edge device from the Stoody smartboard hub.**
>
> - `stoody-multi-pen/edge_hub/` = Stoody smartboard hub (teacher monitoring, PWA, multi-pen dashboard). **DO NOT TOUCH.**
> - `stoody-multi-pen/HUB-exam-conductor/` = ExamPen conducted-exam hub (artifact collection, dual-write, exam timer, invigilator BLE). **BUILD HERE.**
>
> They may share the same Raspberry Pi hardware in some deployments, but the software stacks are independent. The exam-hub has its own systemd services, SQLite DB, TUI, and BLE manager.
>
> The `stoody-multi-pen/mobile-app/` is shared — it connects to whichever hub (smartboard or exam) is nearby via BLE.

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
| /media/exampen-usb | ext4, noatime | Pen data secondary copy (independent failure domain) |

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
4. Unmount USB (`umount /media/exampen-usb`)
5. `systemctl poweroff`

---

## 3. Local Data Storage Schema

### 3.1 SQLite Database

Path: `/var/lib/exampen/hub.db`

WAL mode enabled. Foreign keys enabled via `PRAGMA foreign_keys=ON`.

#### Config Persistence Split

Hub config is stored in **two locations**:

| Location | Format | Contents | Authority |
|---|---|---|---|
| `/etc/exampen/hub.conf` | JSON | `HubConfig` model: `hub_id`, `hub_code`, `backend_url`, `institute_id`, `region`, `uplink_mode`, `provisioned_at`, `provisioning_state`, `hub_token`, `invig_codes`, `pen_inventory` | Primary config load on boot |
| `hub.db` → `hub_config` table | SQLite KV | Generic key-value rows (currently unused by runtime; kept for schema compatibility) | — |

`hub_token` is persisted in the JSON config. It is the hub's long-lived JWT for API calls and is **offline-critical** — without it the hub cannot authenticate to the backend after provisioning.

#### Tables

```sql
-- Hub identity and config (key-value singleton rows)
-- NOTE: Runtime config is loaded from /etc/exampen/hub.conf (JSON).
-- This table exists for schema compatibility.
CREATE TABLE IF NOT EXISTS hub_config (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Cached invigilator codes (offline-critical for exam auth)
-- Written by ConfigStore.apply_provision_response() from provisioning response.
CREATE TABLE IF NOT EXISTS invig_codes (
    code       TEXT PRIMARY KEY,
    valid_from TEXT,       -- NULL if not provided by backend
    expires_at TEXT,       -- NULL if not provided by backend
    fetched_at TEXT        -- when this code was cached from provisioning
);

-- Pen inventory (registered to this hub's institute)
-- Written by ConfigStore.apply_provision_response() from provisioning response.
CREATE TABLE IF NOT EXISTS pen_inventory (
    pen_mac      TEXT PRIMARY KEY,
    pen_id       TEXT,
    student_name TEXT,
    student_id   TEXT
);

-- Exam sessions (session_id is a surrogate PK; exam_id is a regular column
-- to allow re-arming the same exam)
CREATE TABLE IF NOT EXISTS exam_sessions (
    session_id   TEXT PRIMARY KEY,
    exam_id      TEXT,
    exam_type    TEXT,          -- 'dcr' or 'pcr'
    state        TEXT,          -- created, armed, timer_running, dongle_activation,
                                -- pen_sync, sync_complete, sync_partial,
                                -- uploading, upload_complete, cancelled
    started_at   TEXT,
    ended_at     TEXT,
    duration_sec INTEGER
);

-- Pen-student bindings per exam session
-- Future work: add status, source, server_confirmed_at, rejection_reason
-- columns for the full binding workflow described in §4.3.
CREATE TABLE IF NOT EXISTS pen_bindings (
    pen_mac           TEXT,
    exam_session_id   TEXT,
    student_id        TEXT,
    bound_at          TEXT,
    PRIMARY KEY (pen_mac, exam_session_id)
);

-- Per-pen sync status per exam session
CREATE TABLE IF NOT EXISTS pen_sync_status (
    pen_mac           TEXT,
    exam_session_id   TEXT,
    bytes_expected    INTEGER DEFAULT 0,
    bytes_received    INTEGER DEFAULT 0,
    checksum_expected TEXT,
    checksum_received TEXT,
    status            TEXT DEFAULT 'pending',  -- pending, connecting, syncing,
                                                -- complete, failed, timeout
    PRIMARY KEY (pen_mac, exam_session_id)
);

-- Upload ledger (per-artifact upload tracking with retry)
CREATE TABLE IF NOT EXISTS upload_ledger (
    upload_id        TEXT PRIMARY KEY,
    exam_session_id  TEXT,
    exam_id          TEXT,
    exam_type        TEXT DEFAULT '',
    pen_mac          TEXT,
    student_id       TEXT,
    artifact_path    TEXT,
    status           TEXT DEFAULT 'pending',   -- pending, uploading, complete, failed
    attempts         INTEGER DEFAULT 0,
    last_attempt_at  TEXT,
    completed_at     TEXT
);

-- BLE dongle registry (persisted across reboots)
CREATE TABLE IF NOT EXISTS dongle_registry (
    dongle_id    TEXT PRIMARY KEY,
    hci_path     TEXT,
    status       TEXT DEFAULT 'ok',  -- ok, error, degraded
    last_seen_at TEXT
);

-- Interaction log (forensic audit trail)
-- Future work: add exam_id, pen_mac, invig_id, severity columns for
-- the full event catalog described in §3.4.
CREATE TABLE IF NOT EXISTS interaction_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source    TEXT,
    action    TEXT,
    detail    TEXT
);

-- Timer persistence (for reboot recovery)
CREATE TABLE IF NOT EXISTS active_timer (
    exam_id       TEXT PRIMARY KEY,
    start_epoch   REAL,          -- Unix epoch seconds (REAL for sub-second)
    duration_sec  INTEGER,
    remaining_sec REAL           -- updated every ~10s by timer module
);
```

#### Schema Versioning

Not yet implemented. Future work: add a `schema_version` table and migration
helpers to handle schema evolution across software updates. Current approach:
`CREATE TABLE IF NOT EXISTS` is idempotent and safe for new installs. Schema
changes require a new golden image or manual migration.

### 3.2 File Storage Layout

```
/var/lib/exampen/
├── hub.db                          # SQLite database
├── hub.db-wal                      # WAL mode enabled
├── hub.db-shm
├── artifacts/                      # DualWriteStorage primary root
│   └── {exam_id}/
│       └── {pen_mac}/
│           ├── strokes_raw.bin     # Raw bytes from pen GATT read
│           └── strokes.meta.json   # {bytes, checksum_crc32, pages, sync_ts}
└── logs/
    ├── sync.log
    ├── uplink.log
    └── invig.log

/media/exampen-usb/                  # USB thumb drive (secondary copy)
├── artifacts/                       # DualWriteStorage secondary root
│   └── {exam_id}/
│       └── {pen_mac}/
│           ├── strokes_raw.bin     # Byte-identical copy
│           └── strokes.meta.json
```

### 3.3 Dual-Write Protocol

1. `hub-pen-sync` receives chunk from pen GATT.
2. Write chunk to SD path (`/var/lib/exampen/artifacts/{exam_id}/{pen_mac}/strokes_raw.bin`, append mode).
3. `fsync()` SD file descriptor.
4. Write identical chunk to USB path (`/media/exampen-usb/artifacts/{exam_id}/{pen_mac}/strokes_raw.bin`, append mode).
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

> **Note:** The current `pen_bindings` schema uses a simplified shape
> (`pen_mac`, `exam_session_id`, `student_id`, `bound_at`). The full
> binding workflow with `status`, `source`, `server_confirmed_at`,
> `rejection_reason`, `student_name`, and `student_roll` columns is
> **future work** (see §3.1 schema note). The transitions below describe
> the intended end state; current runtime uses a subset.

| From | To | Trigger | Hub Action |
|---|---|---|---|
| Unknown | Discovered | Registration scan detects MAC | Insert `pen_inventory`, insert `pen_bindings` with `student_id` if known |
| Discovered | Bound | Backend resolves MAC→student | Update `pen_bindings.student_id` |
| Bound | Syncing | Post-exam, pen connects to dongle | Start GATT read, upsert `pen_sync_status` |
| Syncing | Synced | All chunks received, checksum match | Mark `pen_sync_status.status = 'complete'` |
| Syncing | Failed | Timeout or checksum mismatch | Mark `pen_sync_status.status = 'failed'`, log error |
| Synced | Uploaded | All chunks ACKd by backend | Mark `upload_ledger.status = 'complete'` |
| Uploaded | Released | Exam session closed | Pen available for next session |

### 4.3 Manual Pen Registration (Mid-Exam) — Local Implementation

The supervisor now handles `manual_register` commands locally against cached
pen inventory and the simplified `pen_bindings` schema:

1. Invigilator selects "Manual Register" on mobile app.
2. App sends command to hub via BLE: `{cmd: 'manual_register', exam_id, pen_mac, student_id}`.
3. Supervisor validates:
   - `exam_id`, `pen_mac`, `student_id` are all present.
   - `exam_id` resolves to an active or persisted session.
   - `pen_mac` exists in cached `pen_inventory` (provisioned from institute data).
   - If `pen_inventory.student_id` is set and conflicts with the payload, returns `student_mismatch`.
4. Hub inserts/updates `pen_bindings` row via `INSERT OR REPLACE` (same `pen_mac` + `exam_session_id` replaces cleanly).
5. Response includes `binding: "local"` — no server-confirmed status.

> **Scope note:** This does not add `status`, `source`, `server_confirmed_at`, or `rejection_reason`
> columns. The binding is local-only. Server-confirmed binding workflow remains future work. The current
> implementation does not claim provisional or confirmed status.

> **BLE registration scan** is now implemented: `START_REGISTRATION_SCAN` calls `DongleDiscovery.scan_for_pens()`
> via `BLEManager.scan_for_pens()`, cross-references against cached `pen_inventory`, and returns
> `{known, unknown}` device lists. The scan is synchronous (blocking the BLE command thread for the
> scan duration, default 10s).

> **Ownership rule (future):** Once server-confirmed bindings are implemented, the exam orchestration
> owner will remain the single authoritative binding source. Hub local bindings are for display and
> local workflow continuity only.

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
Type=simple
WorkingDirectory=/opt/exampen
ExecStart=/usr/bin/python3 -m hub_supervisor
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Service dependency tree:**

```
exampen-supervisor (Type=simple)
├── exampen-ble-mgr (threaded by supervisor)
├── exampen-pen-sync (threaded by supervisor)
├── exampen-timer (threaded by supervisor)
├── exampen-uplink (threaded by supervisor)
├── exampen-invig-ble (threaded by supervisor)
└── exampen-tui (threaded by supervisor, optional — only if HDMI/serial connected)

hub_store — library/repository layer used by the above services, not a supervisor thread
```

Supervisor manages all service threads. If a service thread crashes, supervisor restarts it and logs to `interaction_log`.

> **In-process wiring (current runtime):** The supervisor owns all shared infrastructure — `HubRepository`, `DualWriteStorage`, `ExamTimer`, `ConfigStore` — and injects them into service threads at startup. `hub_ble_mgr` receives a supervisor-provided `on_pen_data` callback that forwards BLE notifications to `PenSyncManager.handle_pen_data`. `hub_pen_sync` and `hub_uplink` share the same `HubRepository` and `DualWriteStorage`. `hub_uplink` reads `backend_url`, `hub_id`, and `hub_token` from the supervisor-owned `ConfigStore`. Module-level `run()` fallbacks exist in each module for standalone/dev operation but are not used when the supervisor is active.

> **Future work:** Upgrade to `Type=notify` with `sd_notify` readiness and `WatchdogSec` once the supervisor implements `READY=1` and `WATCHDOG=1` notifications.

---

## 7. First-Boot Provisioning Sequence

### 7.1 Provisioning Contract

**Endpoint:** `POST /api/v1/hubs/provision`
**Authentication:** Admin or B2C-admin Bearer token (the admin who enters the hub code on the TUI Setup screen).
**Caller type:** `admin` or `b2c_admin` (NOT the hub itself — the hub has no token yet).

**Required input:**

| Field | Type | Description |
|---|---|---|
| `hub_code` | string (min 4 chars) | Provisioning code generated by the admin or super-admin |

**Response fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `hub_id` | string | Yes | System-assigned hub identifier (e.g., `hub-<hex>`) |
| `institute_id` | string | Yes | Tenant identifier derived from the admin's JWT (`db_name`) |
| `hub_token` | string | Yes | Long-lived JWT (365 days) with `user_type: "hub"` for subsequent hub API calls |
| `invig_codes` | string[] | Yes | Pre-generated invigilator auth codes for local caching (minimum 3) |
| `pen_inventory` | object[] | Yes | Known pens for this institute from `pen_registry` (may be empty array) |
| `backend_url` | string | Yes | Absolute URL for hub-to-backend communication (e.g., `https://api.stoody.in`) |
| `provisioned_at` | string (ISO 8601) | Yes | Timestamp of provisioning |

**`backend_url` is absolute.** The hub needs a fully-qualified URL to reach the backend after WiFi connects.

**`invig_codes` and `pen_inventory` are required on first boot.** The hub caches them locally in SQLite (`invig_codes` and `pen_inventory` tables) because WiFi may be unavailable during exam sessions.

### 7.2 Provisioning Flow

1. Power on RPi with golden image SD card + USB thumb drive.
2. systemd boots → `exampen-supervisor` starts → detects first-boot (no `/etc/exampen/hub.conf`).
3. TUI launches → Setup Screen forced.
4. Admin authenticates to the Stoody backend via the TUI (or a pre-obtained admin token is configured).
5. Admin enters hub provisioning code on TUI.
6. TUI calls `POST /api/v1/hubs/provision {hub_code}` with admin Bearer token.
7. Backend validates code, assigns `hub_id`, returns provisioning response (§7.1).
8. Hub stores config to `/etc/exampen/hub.conf`, caches invig codes and pen inventory in SQLite.
9. Hub state → `PROVISIONED` → TUI shows "Ready" on status screen.
10. USB thumb drive formatted and mounted (if not already).
11. First-boot complete. Hub ready for invigilator connection.

---

## 8. Hub Upload Path — Authority

The **authoritative** hub upload route family is:

```
POST   /api/v1/ingest/strokes/{exam_id}/{pen_mac}           — chunk upload
POST   /api/v1/ingest/strokes/{exam_id}/{pen_mac}/complete   — finalize with checksum
GET    /api/v1/ingest/strokes/{exam_id}/{pen_mac}/status     — per-pen status
POST   /api/v1/ingest/strokes/{exam_id}/{pen_mac}/dedup      — dedup check
```

Contract authority: `api/stroke-ingest.openapi.yaml` (version 3.0.0+).

Hub runtime (`HUB-exam-conductor/hub_uplink`) MUST use this route family for all pen artifact uploads.

**Legacy bridge path:** `POST /api/v1/hub/exam-upload` exists in `backend/api/v1/hub.py` as a backward-compatibility surface. It is NOT the primary hub upload path. Do not use it for new implementations.

**Non-hub submission path:** `POST /api/v1/evalpen/submissions` (in `backend/api/v1/evalpen_submissions_async.py`) is for direct client/camera submissions to the ingest substrate, NOT for hub-originated uploads.

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-04-15 | Step 13-14: wired `connected_pen_count` from BLEManager into UplinkManager heartbeat. Implemented `start_registration_scan`: `BLEManager.scan_for_pens()` calls `DongleDiscovery.scan_for_pens()` per dongle, cross-references against cached `pen_inventory`, returns `{known, unknown}` device lists. Updated §4.3 scope note. | Claude |
| 2026-04-15 | Step 12: supervisor-owned in-process wiring for all managed services. `hub_ble_mgr` → `PenSyncManager.handle_pen_data`, `hub_pen_sync` and `hub_uplink` share supervisor-owned `HubRepository`/`DualWriteStorage`/`ConfigStore`. `start_upload` requires `exam_id`, resolves session, updates state, returns upload ledger counts. `request_snapshot` includes `upload`, `storage`, `dongles`. Added in-process wiring note to §6. | Claude |
| 2026-04-15 | Step 11: implemented `manual_register` against cached `pen_inventory` + simplified `pen_bindings` (local binding, no server-confirmed status). Added `get_cached_pen()` to repository. Updated §4.3 to describe current local implementation scope. `start_registration_scan` validates exam_id but hardware scan remains pending. `request_snapshot` now includes `bindings`. | Claude |
| 2026-04-13 | Reconciled §3 local persistence contract with hub_store implementation: updated all table schemas to match code reality (session_id PK, simpler binding/sync/upload shapes, KV hub_config), added config persistence split documentation, added schema versioning future-work note, updated file storage paths to match DualWriteStorage defaults. | Claude |
| 2026-04-09 | Resolved authority conflicts: provisioning contract specified with exact endpoint, response fields, and caller type (§7). Upload path authority section added (§8) — `/api/v1/ingest/strokes/` is the primary hub upload route; legacy bridge paths explicitly marked as non-authoritative. Backend ingest API references updated to match `api/stroke-ingest.openapi.yaml` v3.0.0. | Claude |
| 2026-03-18 | Removed mutable region setup, added explicit `pen_bindings` workflow fields, and aligned first-boot/setup flow with the locked US regulatory configuration. | Codex |
