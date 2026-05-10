# ExamPen Upstream Stack — Implementation Plan

**Status:** ACTIVE  
**Supersedes:** BUILD_STATUS.md Wave 3+ for upstream ingest/hub/mobile work  
**Authority:** `proposed_plan` section of the upstream stack proposal

> **Authority boundary:** This document is a sequencing and work-packaging plan for the upstream conducted-exam stack (invigilator → hub → backend ingest → `ready_for_eval`). It does **not** define runtime behavior, wire contracts, API shapes, schema shapes, storage layouts, or lifecycle rules. When this document conflicts with a root architecture spec (`architecture/*.md`), an integration spec (`integration/*.md`), an OpenAPI file (`api/*.openapi.yaml`), or an event schema (`contracts/events/*.schema.json`), the authoritative spec wins and this document must be corrected to match.

---

## Scope of This Plan

This plan covers the **upstream conducted-exam stack** — the path from invigilator authentication through hub collection, sync, and canonical ingest, ending when an exam is `ready_for_eval`.

```
invigilator (mobile)
    └── hub (edge_hub shared runtime / ExamPen mode)
            └── backend (exam orchestration + ingest)
                    └── ready_for_eval
```

> **2026-05-09 authority alignment:** Earlier task packs name `stoody-multi-pen/HUB-exam-conductor/` as the hub implementation boundary. That directory is now a reference/decomposition donor. New implementation work should fold the ExamPen behavior into `stoody-multi-pen/edge_hub/` as an independent ExamPen mode/service group over shared runtime services. Backend contracts, OpenAPI paths, lifecycle rules, and storage authority remain unchanged.
>
> Any later "Files to create" entries under `stoody-multi-pen/HUB-exam-conductor/` are historical module-shape references, not current create-path instructions. Map those responsibilities into `edge_hub` shared services or `edge_hub` ExamPen mode services before implementation.

This plan does **NOT** include:
- DCR execution
- PCR execution
- Review queue population
- Score publication

---

## Workstream Summary

| Workstream | Status | Boundary | Can Run In Parallel With |
|---|---|---|---|
| Backend exam orchestration + invigilator APIs | COMPLETE | backend | W2, W3, W4 |
| Hub provisioning + heartbeat contract | COMPLETE | backend + hub | W1, W3 |
| Hub runtime: config, supervisor, store | COMPLETE | `edge_hub` converged runtime; `HUB-exam-conductor` reference | W2 |
| Mobile: multi-hub pairing + session model | COMPLETE | mobile-app | — (separate codebase) |
| Hub runtime: BLE + pen sync + uplink | COMPLETE | `edge_hub` converged runtime; `HUB-exam-conductor` reference | W1 |
| Mobile: camera fallback upload | COMPLETE | mobile-app | W5 |
| Hub runtime: TUI + diagnostics | COMPLETE | `edge_hub` converged runtime; `HUB-exam-conductor` reference | W4, W5 |

---

## Swarm Task Packs

### Wave UP-1 — Backend Exam Orchestration

> **Owner:** `backend/`  
> **Depends on:** Existing exam-conductor ingest (SWM-002, SWM-012)  
> **Read first:** `architecture/DUAL_MODE_ARCHITECTURE.md`, `integration/HUB_DEPLOYMENT_SPEC.md`, `integration/STOODY_INTEGRATION_SPEC.md`

#### UP-001 — Exam Orchestration API

- **Status:** COMPLETE
- **Objective:** Implement exam lifecycle management as the single writable owner.
- **Files to create:**
  - `backend/api/v1/exam_orch_async.py` — exam create/list/view/transition endpoints
- **Files to modify:**
  - `backend/main_async.py` — mount router
- **API contract authority:** `api/exam-orch.openapi.yaml`
- **Key behaviors:** Exam lifecycle transitions (`draft → armed → in_progress → collection_closed → uploading → ready_for_eval`), hub assignment, progress tracking. Exact paths and schemas defined in the OpenAPI contract.
- **Validation:**
  - Lifecycle transitions are valid (e.g., cannot skip states)
  - Hub assignment requires hub to be registered and healthy
  - Only authorized roles can transition lifecycle

#### UP-002 — Invigilator Console API

- **Status:** COMPLETE
- **Objective:** Expose session state and sync status visible to invigilator and teacher.
- **Files to create:**
  - `backend/api/v1/invig_console_async.py`
- **API contract authority:** `api/invig-console.openapi.yaml`
- **Key behaviors:** Session state, per-hub connectivity, connected pens, upload sync progress, and alerts. Read-only operational view — no scoring or review actions.

#### UP-003 — Hub Operations API

- **Status:** COMPLETE
- **Objective:** Hub provisioning, registration, heartbeat, and assignment contract.
- **Files to create:**
  - `backend/api/v1/hub_ops_async.py`
- **Provisioning contract authority:** `integration/HUB_DEPLOYMENT_SPEC.md` §7
- **Key behaviors:**
  - First-boot provisioning: admin consumes provisioning code, receives `hub_id`, `hub_token`, `invig_codes`, `pen_inventory`, `backend_url`
  - Hub self-registration with capabilities/dongles
  - Periodic heartbeat with health status
  - Exam assignment and current assignment query
  - Session start/end reporting
- **Validation:**
  - Hub must be provisioned before receiving assignments
  - Heartbeat must arrive within 90s or hub is marked offline
  - Session start/end must match lifecycle transitions in UP-001

#### UP-004 — Stroke Ingest API

- **Status:** COMPLETE
- **Objective:** Accept pen-originated artifacts from hubs with idempotent deduplication.
- **Files to create:**
  - `backend/api/v1/stroke_ingest_async.py`
- **API contract authority:** `api/stroke-ingest.openapi.yaml` (v3.0.0+)
- **Route family:** `/api/v1/ingest/strokes/{exam_id}/{pen_mac}` with sub-paths for chunk upload, finalize, status, and dedup
- **Key behaviors:**
  - Chunked upload with content-hash deduplication
  - Finalization with SHA-256 checksum verification over all chunks
  - Bridge to IngestService for canonical persistence (`evalpen_submissions`, `evalpen_answer_pages`)
  - Pre-upload dedup check for reconnect/restart scenarios
- **Validation:**
  - Exam must be in `uploading` lifecycle state
  - Pen must be in registered inventory for this hub/institute

#### UP-005 — Camera Fallback Upload API

- **Status:** COMPLETE
- **Objective:** Accept photographed answer pages from mobile with correct provenance; PCR-only route.
- **Files to create:**
  - `backend/api/v1/camera_upload_async.py`
- **API contract authority:** camera upload route mounted at `/api/v1/ingest/camera` (see `backend/api/v1/camera_upload_async.py`)
- **Key behaviors:**
  - Upload photographed pages with provenance (`exam_id`, `student_id`, `page_num`)
  - Rejected for DCR exams (camera fallback is PCR-only)
  - Canonical write to `evalpen_answer_pages` with `source: "camera"`
- **Validation:**
  - Exam must be in `uploading` lifecycle state
  - Exam type must be `"pcr"`
  - Student ID must be in exam roster

---

### Wave UP-2 — Hub Runtime Skeleton

> **Owner:** `stoody-multi-pen/edge_hub/` shared runtime and ExamPen mode services. Use `stoody-multi-pen/HUB-exam-conductor/` as reference material for this task pack.  
> **Depends on:** UP-003 (provisioning contract defined in backend)  
> **Read first:** `integration/HUB_DEPLOYMENT_SPEC.md`, `hub/ipc-protocol.md`

#### UP-006 — Hub Config Module

- **Status:** COMPLETE
- **Objective:** Persisted hub configuration, provisioning state, and identity.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_config/models.py` — HubConfig, ProvisioningState
  - `stoody-multi-pen/HUB-exam-conductor/hub_config/store.py` — config file read/write
  - `stoody-multi-pen/HUB-exam-conductor/hub_config/__init__.py`
- **Key behaviors:**
  - Load/store `/etc/exampen/hub.conf`
  - Fields: `hub_id`, `hub_code`, `backend_url`, `uplink_mode`, `region`, `provisioned_at`
  - Bootstrap detection (first-boot forces Setup TUI)
  - Provisioning flow per `integration/HUB_DEPLOYMENT_SPEC.md` §7: consume code at `POST /api/v1/hubs/provision`, cache `invig_codes` + `pen_inventory`

#### UP-007 — Hub SQLite Store

- **Status:** COMPLETE
- **Objective:** Local SQLite schema and data access layer per HUB_DEPLOYMENT_SPEC.md §3.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_store/database.py` — schema creation, migrations
  - `stoody-multi-pen/HUB-exam-conductor/hub_store/models.py` — ExamSession, PenBinding, PenSyncStatus, UploadLedger, DongleRegistry, InteractionLog, ActiveTimer
  - `stoody-multi-pen/HUB-exam-conductor/hub_store/repository.py` — CRUD operations
  - `stoody-multi-pen/HUB-exam-conductor/hub_store/file_storage.py` — dual-write to SD + USB
  - `stoody-multi-pen/HUB-exam-conductor/hub_store/__init__.py`
- **Key behaviors:**
  - SQLite WAL mode at `/var/lib/exampen/hub.db`
  - Dual-write: every stroke chunk written to SD then USB, fsync on both before ACK
  - Tables: `hub_config`, `invig_codes`, `pen_inventory`, `exam_sessions`, `pen_bindings`, `pen_sync_status`, `upload_ledger`, `dongle_registry`, `interaction_log`, `active_timer`
  - Degraded mode detection when USB backup is unavailable

#### UP-008 — Hub Supervisor

- **Status:** COMPLETE
- **Objective:** Process supervision, watchdog, lifecycle state machine for the hub.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_supervisor/process_mgr.py` — managed service thread lifecycle
  - `stoody-multi-pen/HUB-exam-conductor/hub_supervisor/state_machine.py` — HubStateMachine (provisioned → idle → armed → exam_in_progress → collection_closed → uploading → idle)
  - `stoody-multi-pen/HUB-exam-conductor/hub_supervisor/watchdog.py` — watchdog timer, health aggregation
  - `stoody-multi-pen/HUB-exam-conductor/hub_supervisor/__init__.py`
- **Key behaviors:**
  - Manage service threads (not child processes): `hub_ble_mgr`, `hub_pen_sync`, `hub_timer`, `hub_uplink`, `hub_invig_ble`, `hub_tui`
  - Supervisor owns shared infrastructure (`HubRepository`, `DualWriteStorage`, `ExamTimer`, `ConfigStore`) and injects into service threads
  - If a service thread crashes, supervisor restarts it and logs to `interaction_log`
  - State transitions driven by: mobile commands, timer expiry, uplink events

---

### Wave UP-3 — Hub BLE + Pen Sync + Uplink

> **Owner:** `stoody-multi-pen/edge_hub/` shared runtime and ExamPen mode services. Use `stoody-multi-pen/HUB-exam-conductor/` as reference material for this task pack.  
> **Depends on:** UP-006, UP-007, UP-008 (hub store and supervisor exist)  
> **Read first:** `hub/ble-gatt-spec.md`, `hub/ipc-protocol.md`, `references/P05_pen_SDK.md`

#### UP-009 — Hub BLE Manager

- **Status:** COMPLETE
- **Objective:** Multi-dongle BLE management with per-dongle worker ownership.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_ble_mgr/discovery.py` — dongle enumeration, pen scan
  - `stoody-multi-pen/HUB-exam-conductor/hub_ble_mgr/connection.py` — BLE connection per pen
  - `stoody-multi-pen/HUB-exam-conductor/hub_ble_mgr/dongle_worker.py` — per-dongle worker loop
  - `stoody-multi-pen/HUB-exam-conductor/hub_ble_mgr/__init__.py`
- **Key behaviors:**
  - Discover up to 7 BLE dongles via BlueZ HCI
  - Assign pens to specific dongle workers (load balancing)
  - Handle dongle hot-plug: enumerate on connect, mark unhealthy on disconnect
  - Report connected pen count to supervisor state machine

#### UP-010 — Hub Pen Sync

- **Status:** COMPLETE
- **Objective:** Pen stroke chunk ingestion, local persistence, completion detection.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_pen_sync/ingestion.py` — chunk receiver from BLE GATT
  - `stoody-multi-pen/HUB-exam-conductor/hub_pen_sync/persistence.py` — write to hub_store file_storage
  - `stoody-multi-pen/HUB-exam-conductor/hub_pen_sync/completion.py` — checksum verification, completion signal
  - `stoody-multi-pen/HUB-exam-conductor/hub_pen_sync/__init__.py`
- **Key behaviors:**
  - Receive GATT chunks from `hub_ble_mgr`
  - Dual-write to SD and USB before ACK to pen
  - Track `bytes_expected`, `bytes_received`, `checksum_expected` in `pen_sync_status`
  - Emit completion event to `hub_uplink` and supervisor

#### UP-011 — Hub Uplink

- **Status:** COMPLETE
- **Objective:** Backend upload client with idempotent retry and upload ledger.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_uplink/client.py` — HTTP/HTTPS upload client
  - `stoody-multi-pen/HUB-exam-conductor/hub_uplink/retry.py` — retry queue with exponential backoff
  - `stoody-multi-pen/HUB-exam-conductor/hub_uplink/reconciliation.py` — upload-status polling and ACK tracking
  - `stoody-multi-pen/HUB-exam-conductor/hub_uplink/__init__.py`
- **Upload route authority:** `api/stroke-ingest.openapi.yaml` — uses the `/api/v1/ingest/strokes/` route family
- **Key behaviors:**
  - Chunked upload via the authoritative stroke ingest route family
  - Idempotency key: `exam_id + pen_mac + chunk_index`
  - On network loss: persist to upload queue, retry on reconnect
  - On restart: resume pending uploads from `upload_ledger`
  - Deduplication check before upload to avoid re-sending acknowledged chunks

---

### Wave UP-4 — Hub Timer + Invigilator BLE

> **Owner:** `stoody-multi-pen/edge_hub/` shared runtime and ExamPen mode services. Use `stoody-multi-pen/HUB-exam-conductor/` as reference material for this task pack.  
> **Depends on:** UP-007, UP-008  
> **Read first:** `integration/HUB_DEPLOYMENT_SPEC.md` §4, §5

#### UP-012 — Hub Timer

- **Status:** COMPLETE
- **Objective:** Exam timer with persistence across hub restarts.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_timer/timer.py` — countdown logic, persistence
  - `stoody-multi-pen/HUB-exam-conductor/hub_timer/__init__.py`
- **Key behaviors:**
  - Store `active_timer` in SQLite: `exam_id`, `start_epoch`, `duration_sec`, `remaining_sec`
  - On hub restart: supervisor calls `_restore_active_session()` → `ExamTimer.restore_from_db()` to reconstitute in-memory timer and FSM state
  - On timer expiry: callback to supervisor → `_on_timer_expired()` → trigger `collection_closed` state
  - Timer runs locally during exam (WiFi may be disconnected)

#### UP-013 — Hub Invigilator BLE

- **Status:** COMPLETE
- **Objective:** BLE command channel from mobile app for exam arm/stop/collect.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_invig_ble/channel.py` — BLE GATT service for commands
  - `stoody-multi-pen/HUB-exam-conductor/hub_invig_ble/commands.py` — command parser (arm, start, stop, collect)
  - `stoody-multi-pen/HUB-exam-conductor/hub_invig_ble/__init__.py`
- **Key behaviors:**
  - Expose BLE GATT service with characteristics for: command write, response read, status notify
  - Commands: `START_EXAM` (0x01), `STOP_EXAM` (0x02), `START_REGISTRATION_SCAN` (0x03), `MANUAL_REGISTER` (0x04), `START_UPLOAD` (0x05), `REQUEST_SNAPSHOT` (0x06)
  - Validate invigilator identity via cached `invig_codes`
  - Log all commands to `interaction_log`

---

### Wave UP-5 — Mobile Multi-Hub + Camera

> **Owner:** `stoody-multi-pen/mobile-app/`  
> **Depends on:** UP-001, UP-002, UP-003 (backend APIs exist)  
> **Read first:** `integration/HUB_DEPLOYMENT_SPEC.md`, `integration/STOODY_INTEGRATION_SPEC.md`

#### UP-014 — Mobile: Hub List

- **Status:** COMPLETE
- **Objective:** List provisioned ExamPen hubs from backend.
- **Files created:**
  - `stoody-multi-pen/mobile-app/src/services/exampenHubService.ts` — hub list via `GET /api/v1/hubs`
  - `stoody-multi-pen/mobile-app/src/types/exampen.ts` — ExamHub type
- **Key behaviors:**
  - Reuse existing Stoody JWT auth via `authService`
  - Display hub status: online/offline, connected pens, storage health
  - Note: BLE hub discovery is not in scope — hubs are registered through backend provisioning

#### UP-015 — Mobile: Exam Selection + Hub Assignment

- **Status:** COMPLETE
- **Objective:** Select an exam and assign hubs to it.
- **Files created:**
  - `stoody-multi-pen/mobile-app/src/screens/ExamPenHubSelectScreen.tsx` — exam list + hub selection
  - `stoody-multi-pen/mobile-app/src/services/exampenSessionService.ts` — exam lifecycle + hub assignment
- **Key behaviors:**
  - Fetch exams from `GET /api/v1/exams` (exam orchestration API)
  - Assign hubs one at a time via `POST /api/v1/exams/{exam_id}/hubs`
  - Transition lifecycle via `PATCH /api/v1/exams/{exam_id}/lifecycle`

#### UP-016 — Mobile: Session Dashboard + Per-Hub Status

- **Status:** COMPLETE
- **Objective:** Live status dashboard showing per-hub sync and upload progress.
- **Files created:**
  - `stoody-multi-pen/mobile-app/src/screens/ExamPenSessionDashboardScreen.tsx` — live session view
  - `stoody-multi-pen/mobile-app/src/components/HubStatusCard.tsx` — per-hub status card
  - `stoody-multi-pen/mobile-app/src/services/invigConsoleService.ts` — poll invigilator APIs
- **Key behaviors:**
  - Poll invigilator console API (`/api/v1/invig/sessions/{exam_id}/*`) for session state
  - Per-hub card: connection state, pen count, upload counts, alerts
  - Pull-to-refresh + 5s auto-poll

#### UP-017 — Mobile: Camera Fallback Upload

- **Status:** COMPLETE
- **Objective:** Upload photographed answer pages for PCR exams.
- **Files created:**
  - `stoody-multi-pen/mobile-app/src/services/cameraUploadService.ts` — multipart upload
  - `stoody-multi-pen/mobile-app/src/screens/CameraFallbackScreen.tsx` — capture + upload UI
- **Key behaviors:**
  - Capture photo of answer page
  - Enter student ID manually (roster-backed selector is future work)
  - Enter page number (validated >= 1)
  - Upload via camera ingest API at `POST /api/v1/ingest/camera/{exam_id}/{student_id}/{page_num}` (multipart)
  - Show upload progress and confirmation
  - **Guard:** if exam_type is DCR, disable camera upload and show message

---

### Wave UP-6 — Hub TUI

> **Owner:** `stoody-multi-pen/edge_hub/` shared runtime and ExamPen mode services. Use `stoody-multi-pen/HUB-exam-conductor/` as reference material for this task pack.  
> **Depends on:** UP-006, UP-007, UP-008, UP-009, UP-010, UP-012  
> **Read first:** `integration/HUB_DEPLOYMENT_SPEC.md` §2

#### UP-018 — Hub TUI

- **Status:** COMPLETE
- **Objective:** Textual TUI for setup, status dashboard, WiFi, dongles, exam history, diagnostics, logs.
- **Files to create:**
  - `stoody-multi-pen/HUB-exam-conductor/hub_tui/screens.py` — all TUI screens
  - `stoody-multi-pen/HUB-exam-conductor/hub_tui/app.py` — Textual application
  - `stoody-multi-pen/HUB-exam-conductor/hub_tui/__init__.py`
- **Screens to implement:**
  1. **Setup Screen** — hub code entry, backend URL, WiFi, uplink mode
  2. **Status Dashboard** — live state, dongle table, sync progress, storage
  3. **WiFi Screen** — scan, connect, signal, band
  4. **Dongles Screen** — health per dongle, reset action
  5. **Exam History Screen** — past sessions with per-pen breakdown
  6. **Diagnostics Screen** — hardware/software checks via hub IPC
  7. **Logs Screen** — filterable log viewer via journalctl
  8. **Shutdown Screen** — safe poweroff with pending upload warning

---

## Post-UP-18 Status: Downstream Frontend / Super-Admin / Mobile (2026-05-02)

All UP-001 through UP-018 tasks are COMPLETE. The following downstream surfaces have been built since the upstream plan was completed. These are tracked here for visibility but were not part of the original UP task packs.

### Frontend Teacher Workspace

| Item | Status | Key Files |
|---|---|---|
| 5-tab teacher shell (Exams, Workspace, Results, Recheck, Conversations) | **DONE** | `ExamPenTeacher.tsx` |
| IDE-style workspace (StudentExplorer + QuestionPaper + StudentCopy + QuestionInspector) | **DONE** | `TeacherWorkspace.tsx` |
| Collection Monitor in workspace | **DONE** | `CollectionMonitor.tsx` (Review/Collection tab switcher) |
| Results table with publish | **DONE** | `ExamResults.tsx`, `PublishSummaryBar.tsx`, `PublishAuditLog.tsx` |
| Recheck tab with request management | **DONE** | `RecheckTab.tsx`, `RecheckRequestsPanel.tsx` |
| Conversation tab | **DONE** | `ConversationTab.tsx` |
| Setup/readiness surface | **DONE** | `WorkspaceSetupPanel.tsx`, `WorkspaceHeader.tsx` |
| ExamPen readiness indicator in Document Manager | **DONE** | `ExamPenReadinessIndicator.tsx` mounted in `DocumentDetailPanel.tsx` |
| Shared status module | **DONE** | `examPenStatus.ts` adopted across 6 components |

### Frontend Student Portal

| Item | Status | Key Files |
|---|---|---|
| Published exam list with score breakdown | **DONE** | `ExamPenStudent.tsx` |
| Per-question detail with reference answers | **DONE** | Inline `EnhancedScoreCard` |
| Recheck request dialog | **DONE** | `RecheckRequestDialog.tsx` |
| Student conversation threads | **DONE** | `StudentConversationList.tsx` |

### Super-Admin ExamPen Management

| Item | Status | Key Files |
|---|---|---|
| Feature gate toggle per tenant | **DONE** | `ExamPenManagementPage.tsx` |
| Hub fleet listing + provision code generation | **DONE** | `listProvisionedHubs()`, `generateHubProvisionCode()` |
| Token usage (today) | **DONE** | `getEvalPenUsageAggregate()` |
| Per-tenant exam/submission counts | **PENDING** | Requires new backend endpoint |

### Mobile Camera Fallback

| Item | Status | Key Files |
|---|---|---|
| Offline retry queue | **DONE** | `CameraFallbackScreen.tsx` |
| Roster-backed student selector | **PENDING** | Manual student ID entry still required |

### Still Pending (Backend)

| Item | Status | Notes |
|---|---|---|
| Recheck request router | **Not mounted** | Frontend contract defined; backend endpoints needed |
| Conversation thread router | **Not mounted** | Frontend contract defined; backend endpoints needed |
| Plagiarism detection router | **Not mounted** | Spec exists in `api/plagiarism.openapi.yaml` |
| Analytics router | **Not mounted** | Spec exists in `api/analytics.openapi.yaml` |
| Hub detail endpoint | **Not mounted** | Spec exists in `SUPERADMIN_SPEC.md` §5.2 |
| Hub decommission endpoint | **Not mounted** | Spec exists in `SUPERADMIN_SPEC.md` §5.2 |
| Roster-backed student selector (mobile) | **Not mounted** | Mobile needs roster API for camera fallback |

---

## Validation Tests

| Test ID | What to Verify | Owner |
|---|---|---|
| VAL-UP-01 | Backend: create exam, assign hubs, transition lifecycle through all non-eval states | backend |
| VAL-UP-02 | Backend: hub provisions and heartbeats; assignment appears in hub API | backend + hub |
| VAL-UP-03 | Backend: pen upload lands in `evalpen_submissions` with correct provenance | backend + hub |
| VAL-UP-04 | Backend: camera upload to PCR exam succeeds; camera upload to DCR exam is rejected | backend + mobile |
| VAL-UP-05 | Backend: invigilator console APIs return correct per-hub and per-pen status | backend |
| VAL-UP-06 | Hub: provisioning survives restart; cached invig codes are valid | `edge_hub` ExamPen mode |
| VAL-UP-07 | Hub: multi-dongle pen sync; dual-write to SD+USB; upload retry on network loss | `edge_hub` shared runtime + ExamPen mode |
| VAL-UP-08 | Hub: timer persists across restart and fires correctly on expiry | `edge_hub` ExamPen mode |
| VAL-UP-09 | Mobile: pair multiple hubs; start exam on each independently | mobile-app |
| VAL-UP-10 | Mobile: camera fallback upload with correct exam/student/page provenance | mobile-app |
| VAL-UP-11 | End-to-end: prepared exam → invigilator arms hubs → pen data uploaded → backend shows `ready_for_eval` | all |

---

## Spawn Order

1. **UP-001, UP-002, UP-003** in parallel (backend APIs — no hub/mobile deps)
2. **UP-004, UP-005** after UP-001 (ingest APIs need exam lifecycle)
3. **UP-006, UP-007, UP-008** after UP-003 (hub needs provisioning contract)
4. **UP-009, UP-010, UP-011** after UP-007, UP-008 (BLE/sync need store + supervisor)
5. **UP-012, UP-013** after UP-007, UP-008 (timer and invig BLE need supervisor)
6. **UP-014, UP-015, UP-016** after UP-001, UP-002, UP-003 (mobile needs backend APIs)
7. **UP-017** after UP-004, UP-005 (camera needs backend upload endpoint)
8. **UP-018** after UP-006 through UP-013 (TUI needs all modules)

---

## OpenAPI Contract Alignment

These OpenAPI contracts are the authoritative wire-format definitions for the upstream stack. Paths and schemas in this section reference the contract files; any discrepancy between this table and the contract file is resolved in favor of the contract file.

| Contract | File | Status |
|---|---|---|
| Exam orchestration | `api/exam-orch.openapi.yaml` | COMPLETE |
| Invigilator console | `api/invig-console.openapi.yaml` | COMPLETE |
| Stroke ingest (hub upload) | `api/stroke-ingest.openapi.yaml` (v3.0.0+) | COMPLETE |
| Camera upload | `backend/api/v1/camera_upload_async.py` — mounted at `/api/v1/ingest/camera` | COMPLETE |
| Hub operations (provisioning, heartbeat, assignment) | Defined in `integration/HUB_DEPLOYMENT_SPEC.md` §7 | COMPLETE |

Provisioning contract details: see `integration/HUB_DEPLOYMENT_SPEC.md` §7.1 for exact endpoint, caller type, required inputs, and response fields.
Upload path authority: see `integration/HUB_DEPLOYMENT_SPEC.md` §8 for the authoritative hub upload route family.

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-05-09 | Authority alignment: `HUB-exam-conductor` task-pack paths are now historical/reference module shapes. New edge implementation work targets independent ExamPen mode services inside `stoody-multi-pen/edge_hub`. | Codex |
| 2026-04-17 | Step 24 — restored backend startup. `pip install -r requirements.txt` resolved missing deps (pandas, pyotp, etc.). `import main_async` now succeeds with `_evalpen_available = True` — all 18 ExamPen routers and all other routes load cleanly. Updated BUILD_STATUS.md. | Claude |
| 2026-04-17 | Step 23 — backend router import validation. All 16 ExamPen router modules pass py_compile and direct import. All 23 exam-conductor sub-packages import cleanly. ExamPen try/except block (main_async.py:303-351) succeeds in isolation. Pre-existing issue: `student_bulk_upload.py:17` requires `pandas` (not in venv) which blocks full `main_async.py` load — unrelated to ExamPen. Updated BUILD_STATUS.md deployment checklist (now 18 routers, not 11). | Claude |
| 2026-04-16 | Step 22 — backend/mobile contract alignment check. Found one mismatch: `HubListItem` uses `last_heartbeat_at` / `connected_pen_count` while mobile `ExamHub` expects `last_heartbeat` / `connected_pens`. Fixed with field mapping in `exampenHubService.ts`. Verified all other surfaces aligned: exam list always returns arrays, 409 means same-exam duplicate only, invig console fields match exactly, camera multipart field names match, DCR rejection works both sides. npm run typecheck = 0 errors. | Claude |
| 2026-04-16 | UP-014 through UP-017 marked COMPLETE. Mobile hub list, exam selection + hub assignment, session dashboard, and camera fallback upload all implemented with zero typecheck errors. All 18 upstream tasks now COMPLETE. | Claude |
| 2026-04-09 | Removed inline API path definitions from task descriptions. All paths now reference authoritative OpenAPI contracts and integration specs. Replaced "OpenAPI Contracts to Create or Align" with "OpenAPI Contract Alignment" section that defers to contract files. Added explicit references to `HUB_DEPLOYMENT_SPEC.md` §7 (provisioning) and §8 (upload path). Aligned UP-003, UP-004, UP-005, UP-011, UP-014, UP-015, UP-016, UP-017 with authority model from Step 1. | Claude |
| 2026-04-04 | Created upstream implementation plan with 18 tasks across backend, HUB-exam-conductor, and mobile-app boundaries. | Claude |
| 2026-04-04 | UP-001 through UP-005 COMPLETE. Backend exam orch, invig console, hub ops, stroke ingest, camera upload all implemented and mounted (18 routers). | Claude |
| 2026-04-04 | UP-006 through UP-008 COMPLETE. Hub config (models+store), SQLite store (database+repo+file_storage), supervisor (state_machine+process_mgr+watchdog+supervisor). | Claude |
| 2026-04-04 | UP-009 through UP-013 COMPLETE. Hub BLE mgr (discovery+connection+dongle_worker), pen sync (ingestion+completion), uplink (client+retry+reconciliation), timer, invig BLE (commands+channel). 30 hub files total. | Claude |
| 2026-04-04 | UP-018 COMPLETE. Hub TUI (app+screens with 8 Textual screens). All hub runtime modules implemented — 32 files, all parse OK. | Claude |
