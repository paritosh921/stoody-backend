-- 001_initial.sql — Complete hub SQLite schema
-- Source: HUB_DEPLOYMENT_SPEC.md §3.1 (AUTHORITATIVE)

-- Hub identity and config (singleton)
CREATE TABLE IF NOT EXISTS hub_config (
    hub_id          TEXT PRIMARY KEY,
    backend_url     TEXT NOT NULL,
    uplink_mode     TEXT NOT NULL DEFAULT 'wifi' CHECK (uplink_mode IN ('wifi','mobile','auto')),
    region          TEXT NOT NULL DEFAULT 'US',
    provisioned_at  TEXT NOT NULL,  -- ISO 8601
    last_backend_sync TEXT          -- ISO 8601
);

-- Cached invigilator codes (rotated daily, pre-cached for N days)
CREATE TABLE IF NOT EXISTS invig_codes (
    code            TEXT PRIMARY KEY,
    valid_from      TEXT NOT NULL,  -- ISO 8601
    valid_until     TEXT NOT NULL,  -- ISO 8601
    fetched_at      TEXT NOT NULL
);

-- Pen inventory (registered to this hub's institute)
CREATE TABLE IF NOT EXISTS pen_inventory (
    pen_mac         TEXT PRIMARY KEY,
    pen_serial      TEXT,
    fw_version      TEXT,
    registered_at   TEXT NOT NULL,
    last_seen       TEXT,
    battery_pct     INTEGER
);

-- Exam sessions
CREATE TABLE IF NOT EXISTS exam_sessions (
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
CREATE TABLE IF NOT EXISTS pen_bindings (
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
CREATE TABLE IF NOT EXISTS pen_sync_status (
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
CREATE TABLE IF NOT EXISTS upload_ledger (
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
CREATE TABLE IF NOT EXISTS dongle_registry (
    dongle_mac      TEXT PRIMARY KEY,
    hci_path        TEXT,           -- updated on each boot
    usb_port_path   TEXT,           -- sysfs path for stable identification
    first_seen      TEXT NOT NULL,
    last_healthy    TEXT,
    status          TEXT NOT NULL DEFAULT 'unknown'
                    CHECK (status IN ('unknown','healthy','degraded','failed'))
);

-- Interaction log (every hub action, for forensic audit)
CREATE TABLE IF NOT EXISTS interaction_log (
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
CREATE TABLE IF NOT EXISTS active_timer (
    exam_id         TEXT PRIMARY KEY,
    start_epoch     INTEGER NOT NULL,  -- Unix epoch seconds
    duration_sec    INTEGER NOT NULL,
    remaining_sec   INTEGER NOT NULL,  -- updated every 10s
    last_updated    INTEGER NOT NULL   -- epoch
);
