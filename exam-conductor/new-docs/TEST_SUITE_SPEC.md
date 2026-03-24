# TEST_SUITE_SPEC.md
# ExamPen — Test Suite Specification & TUI Test Runner

Reference: R4-EXAMPEN-DEVSTACK
Doctrine source: SOFTWARE_DEVELOPMENT_DOCTRINE.md §6

---

## 1. Validation Evidence Hierarchy

Per doctrine, these are NOT interchangeable:

| Level | Meaning | Sufficient For |
|---|---|---|
| **L1: Build verified** | Code compiles, Docker image builds | Nothing beyond "it parses" |
| **L2: Typecheck/lint verified** | No type errors, no lint violations | Code shape correctness only |
| **L3: Unit test verified** | Domain logic tested in isolation (no I/O) | Business rule correctness |
| **L4: Integration test verified** | Service tested with real DB/NATS/S3 (Docker) | Interface contract compliance |
| **L5: E2E test verified** | Multi-service flow tested end-to-end | Pipeline correctness |
| **L6: Hardware-in-loop verified** | Hub + real BLE dongles + pen simulator | Physical layer correctness |
| **L7: Field trial verified** | Real exam with real students and pens | Production readiness |

**Every PR must state which levels were achieved.** "Tests pass" without level is rejected.

---

## 2. Test Categories Per Service

### 2.1 Unit Tests (L3) — Every Service Must Have

| ID | Service | Test | What It Proves |
|---|---|---|---|
| U-AUTH-01 | `svc-auth` | Stoody JWT validation + claim normalization | Stoody-issued bearer token becomes normalized ExamPen claims without local re-issuance |
| U-AUTH-02 | `svc-auth` | RBAC role hierarchy + role mapping enforcement | Parent/tutor/student roles map correctly, no escalation |
| U-AUTH-03 | `svc-auth` | Tenant isolation (RLS policy check) | Cross-tenant query returns empty, not error |
| U-AUTH-04 | `svc-auth` | Stoody JWKS cache expiry + refresh | Cached keyset used when fresh, re-fetched on `kid` miss |
| U-AUTH-05 | `svc-auth` | Parent-child scope resolution | Parent sees only linked child records |
| U-ORCH-01 | `svc-exam-orch` | Exam FSM valid transitions | `created→armed→timer_running→...→locked` all accepted |
| U-ORCH-02 | `svc-exam-orch` | Exam FSM invalid transitions rejected | `created→locked` returns error |
| U-ORCH-03 | `svc-exam-orch` | Pen binding CRUD + uniqueness constraint | Duplicate MAC in same exam rejected |
| U-STRK-01 | `svc-stroke-ingest` | Stroke packet schema validation (valid) | Well-formed packet accepted |
| U-STRK-02 | `svc-stroke-ingest` | Stroke packet schema validation (invalid) | Negative coordinates, missing fields rejected |
| U-PROC-01 | `svc-stroke-proc` | Coordinate normalization (DPI transform) | Raw→normalized coords correct within ±0.01 |
| U-PROC-02 | `svc-stroke-proc` | Dedup by idempotency key | Same key twice → single output |
| U-PROC-03 | `svc-stroke-proc` | Page assignment from stroke coordinates | Strokes in Q3 region assigned to Q3 |
| U-DOC-01 | `svc-doc-assembly` | Stroke→SVG render | 100-point stroke renders valid SVG path |
| U-DOC-02 | `svc-doc-assembly` | Miss indicator auto-state: no strokes in region | Returns `miss_no_strokes` |
| U-DOC-03 | `svc-doc-assembly` | Miss indicator auto-state: sync failure | Returns `miss_sync_failure` when sync metadata incomplete |
| U-AI-01 | `svc-ai-pipeline` | HWR inference (English test corpus) | CER <10% on test set |
| U-AI-02 | `svc-ai-pipeline` | HWR inference (Devanagari test corpus) | CER <10% on test set |
| U-AI-03 | `svc-ai-pipeline` | Step detection | 4-step math answer segmented into 4 steps |
| U-AI-04 | `svc-ai-pipeline` | Diagram vs text classifier | Known diagram image classified as diagram |
| U-SCR-01 | `svc-score-engine` | Score FSM valid transitions | `ai_draft→teacher_reviewed→finalized→locked` |
| U-SCR-02 | `svc-score-engine` | Score FSM invalid transitions | `ai_draft→locked` rejected |
| U-SCR-03 | `svc-score-engine` | Rubric evaluation (step marking) | 2+1+1 = 4 marks computed correctly |
| U-SCR-04 | `svc-score-engine` | Override with audit trail | Override appends event, materialized view updates |
| U-REV-01 | `svc-review` | Objection FSM | `filed→assigned→reviewing→resolved` |
| U-REV-02 | `svc-review` | Objection rejection requires reason | Empty reason → validation error |
| U-PLAG-01 | `svc-plagiarism` | TF-IDF cosine similarity | Known similar pair → score >0.85 |
| U-PLAG-02 | `svc-plagiarism` | Structural similarity (edit distance) | Known same-error pair weighted higher |
| U-PLAG-03 | `svc-plagiarism` | Question-type threshold adjustment | MCQ pair with identical correct answers → no flag |
| U-ANLY-01 | `svc-analytics` | Percentile calculation | 40 students, known scores → correct percentiles |
| U-ANLY-02 | `svc-analytics` | Leaderboard generation | Sorted by score, ties broken by name |
| U-CHAT-01 | `svc-chat` | Message append (append-only contract) | Insert succeeds, UPDATE/DELETE → error |
| U-CHAT-02 | `svc-chat` | Read receipts | Mark read → timestamp recorded |

Location: `{service}/tests/test_*.py`
Runner: `pytest {service}/tests/ -m unit`

### 2.2 Integration Tests (L4) — Per Service

| ID | Service | Test | Infrastructure |
|---|---|---|---|
| I-AUTH-01 | `svc-auth` | Stoody JWT introspection via REST → normalized claims | Docker: svc-auth + PostgreSQL + Stoody mock |
| I-AUTH-02 | `svc-auth` | Stoody JWKS fetch + token validation (mock Stoody) | Docker: svc-auth + PG + Stoody mock |
| I-AUTH-03 | `svc-auth` | Multi-tenant RLS: tenant A cannot read tenant B data | Docker: svc-auth + PG (two tenants seeded) |
| I-ORCH-01 | `svc-exam-orch` | Exam CRUD via REST + FSM transitions | Docker: svc-exam-orch + PG + Redis |
| I-ORCH-02 | `svc-exam-orch` | Pen binding: create + resolve via Stoody mock | Docker: svc-exam-orch + PG + Stoody mock |
| I-ORCH-03 | `svc-exam-orch` | Concurrent exam session isolation | Docker: two concurrent exam creates, verify no bleed |
| I-STRK-01 | `svc-stroke-ingest` | WebSocket stroke packet → NATS publish | Docker: svc-stroke-ingest + NATS |
| I-STRK-02 | `svc-stroke-ingest` | Duplicate chunk rejection (idempotency) | Docker: same chunk sent twice, verify single NATS publish |
| I-STRK-03 | `svc-stroke-ingest` | Backpressure: NATS slow consumer | Docker: NATS with artificial delay, verify no data loss |
| I-PROC-01 | `svc-stroke-proc` | NATS event → dedup → TimescaleDB commit | Docker: svc-stroke-proc + NATS + TimescaleDB |
| I-PROC-02 | `svc-stroke-proc` | Duplicate NATS event → single DB commit | Docker: same event published twice |
| I-DOC-01 | `svc-doc-assembly` | Stroke event → page image in MinIO | Docker: svc-doc-assembly + MinIO + PG |
| I-DOC-02 | `svc-doc-assembly` | Miss indicator computed and stored | Docker: feed strokes with known gaps |
| I-AI-01 | `svc-ai-pipeline` | Page image → HWR output stored in PG | Docker: svc-ai-pipeline + PG + MinIO |
| I-AI-02 | `svc-ai-pipeline` | Model version recorded with every result | Docker: verify model_version column populated |
| I-SCR-01 | `svc-score-engine` | AI result event → score event appended | Docker: svc-score-engine + NATS + PG |
| I-SCR-02 | `svc-score-engine` | Override via REST → event appended + view updated | Docker: PATCH /scores/{id} |
| I-SCR-03 | `svc-score-engine` | Score.updated NATS event published after commit | Docker: verify NATS consumer receives event |
| I-REV-01 | `svc-review` | Objection filed via REST → state = filed | Docker: svc-review + PG |
| I-REV-02 | `svc-review` | Resolution → re-score NATS command emitted | Docker: svc-review + NATS + PG |
| I-PLAG-01 | `svc-plagiarism` | Exam AI results → flag generation → PG insert | Docker: svc-plagiarism + PG + NATS |
| I-PLAG-02 | `svc-plagiarism` | Teacher verdict via REST → stored | Docker: PATCH /plagiarism/flags/{id} |
| I-BFF-T01 | `svc-teacher-bff` | Aggregated score query (mocked backing services) | Docker: svc-teacher-bff + mock services |
| I-BFF-T02 | `svc-teacher-bff` | RBAC enforcement: student JWT → 403 on teacher endpoints | Docker: svc-teacher-bff + svc-auth |
| I-BFF-S01 | `svc-student-bff` | Score view query (mocked backing services) | Docker: svc-student-bff + mock services |
| I-BFF-S02 | `svc-student-bff` | Objection submit → forwarded to svc-review | Docker: svc-student-bff + svc-review mock |
| I-BFF-S03 | `svc-student-bff` | Parent JWT → child score view allowed only for linked children | Docker: svc-student-bff + svc-auth + relationship fixture |
| I-INVIG-01 | `svc-invig-console` | WebSocket status feed from exam-orch | Docker: svc-invig-console + svc-exam-orch |

Location: `{service}/tests/test_integration_*.py`
Runner: `pytest {service}/tests/ -m integration`
Requires: `docker compose -f docker-compose.test.yml up`

### 2.3 Pipeline Tests (L5) — Cross-Service

| ID | Test | Services Involved | What It Proves |
|---|---|---|---|
| E2E-01 | Stroke ingestion → processing → storage | `svc-stroke-ingest`, `svc-stroke-proc`, TimescaleDB | Stroke data flows through pipeline, deduplication works |
| E2E-02 | Page assembly → AI recognition | `svc-doc-assembly`, `svc-ai-pipeline`, MinIO | Page images generated, HWR produces output |
| E2E-03 | AI result → score generation | `svc-ai-pipeline`, `svc-score-engine` | Scores created from AI output, FSM initialized |
| E2E-04 | Score override → analytics update | `svc-score-engine`, `svc-analytics` | Override propagates to percentile recalculation |
| E2E-05 | Objection → re-score → notification | `svc-review`, `svc-score-engine`, `svc-notify` | Full objection lifecycle |
| E2E-06 | Plagiarism detection end-to-end | `svc-ai-pipeline`, `svc-plagiarism` | Known plagiarism pairs detected, false positives below threshold |
| E2E-07 | Copy image → OCR → score | `svc-copy-upload`, `svc-ai-pipeline`, `svc-score-engine` | Fallback image path produces scores |
| E2E-08 | Full 40-student exam simulation | All pipeline services | Simulated stroke data for 40 students × 10 questions processed end-to-end |
| E2E-09 | Miss indicator propagation | `svc-doc-assembly`, `svc-score-engine`, `svc-teacher-bff` | Auto-detected misses visible in teacher BFF response |
| E2E-10 | Teacher BFF score aggregation | `svc-teacher-bff`, all backing services | Class score overview returns correct aggregated data |
| E2E-11 | Student BFF objection lifecycle | `svc-student-bff`, `svc-review`, `svc-score-engine` | File objection → teacher resolves → student sees updated score |
| E2E-12 | Stoody webhook delivery | `svc-score-engine`, `svc-notify`, Stoody mock | Score publication triggers webhook to Stoody with correct payload |

Location: `test-suite/pipeline-tests/`
Runner: `pytest test-suite/pipeline-tests/`
Requires: Full Docker Compose stack running.

### 2.4 Hub Hardware Tests (L6) — TUI Runner

| ID | Test | Hardware Required | What It Proves |
|---|---|---|---|
| HW-H1 | Dongle enumeration | 5 USB BLE dongles | All dongles detected, stable MAC identification |
| HW-H2 | Dongle hot-plug | 5 dongles, unplug/replug 1 | Graceful degradation + recovery |
| HW-B1 | BLE scan + connect | 1+ BLE pen (or nRF52840-DK simulator) | Pen discovered, GATT service readable |
| HW-B2 | Multi-pen sync | 8 pens per dongle (or simulators) | Concurrent sync, throughput measurement |
| HW-B3 | Dual-write integrity | SD card + USB drive | Both copies byte-identical after sync |
| HW-T1 | Timer accuracy | NTP-synced RPi | 90-minute timer drift < 1 second |
| HW-W1 | WiFi connectivity | WiFi AP | Connect, verify band, check backend reachability |
| HW-I1 | Invigilator BLE | Mobile phone (or BLE test app) | Auth flow, command relay, status feed |
| HW-P1 | Power failure recovery | Kill power during sync, reboot | Timer resumes, partial sync data preserved, no corruption |

Location: `test-suite/hub-tests/`
Runner: Hub TUI → Diagnostics screen (see §3).

---

## 3. Hub TUI Test Runner

### 3.1 Diagnostics Screen Layout

```
┌─ Hub Diagnostics ───────────────────────┐
│                                          │
│  Hardware Tests                          │
│  [H1] Dongle enumeration      ● PASS    │
│  [H2] Dongle hot-plug         ○ SKIP    │
│  [H3] USB storage mount       ● PASS    │
│  [H4] SD card health          ● PASS    │
│  [H5] NTP sync status         ● PASS    │
│  [H6] WiFi connectivity       ● PASS    │
│  [H7] WiFi band check         ● PASS    │
│                                          │
│  Software Tests                          │
│  [S1] SQLite integrity        ● PASS    │
│  [S2] Service health          ● PASS    │
│  [S3] IPC connectivity        ● PASS    │
│  [S4] Backend reachability    ● PASS    │
│  [S5] Invigilator code cache  ● PASS    │
│                                          │
│  BLE Tests (requires pens/simulator)     │
│  [B1] Pen discovery           ◐ RUNNING │
│  [B2] GATT read test          ○ PENDING │
│  [B3] Multi-pen stress        ○ PENDING │
│  [B4] Sync + dual-write       ○ PENDING │
│                                          │
│  ──────────────────────────────          │
│  [R] Run all    [S] Run selected         │
│  [E] Export results    [Q] Back          │
│                                          │
│  Last run: 2026-03-18 14:23:07           │
│  Pass: 11  Fail: 0  Skip: 1  Pending: 3 │
└──────────────────────────────────────────┘
```

### 3.2 Test Status Icons

| Icon | Meaning |
|---|---|
| ● (green) | PASS |
| ✗ (red) | FAIL |
| ◐ (yellow) | RUNNING |
| ○ (gray) | PENDING / SKIP |

### 3.3 Test Result Export

`[E] Export results` → writes JSON to `/var/lib/exampen/diagnostics/{timestamp}.json`:

```json
{
  "hub_id": "EPH-00042",
  "timestamp": "2026-03-18T14:23:07Z",
  "sw_version": "0.4.2",
  "os_version": "Ubuntu 24.04",
  "tests": [
    {"id": "H1", "name": "Dongle enumeration", "status": "PASS", "duration_ms": 1234, "detail": {"dongles_found": 5}},
    {"id": "B1", "name": "Pen discovery", "status": "FAIL", "duration_ms": 30000, "detail": {"error": "Timeout: 0 pens found in 30s"}}
  ]
}
```

Exportable to backend via `hub-uplink` for fleet-wide diagnostics dashboard.

### 3.4 Individual Test Specifications

#### H1: Dongle Enumeration

```
Procedure:
  1. List all Bluetooth HCI adapters: hciconfig -a
  2. For each adapter, read BD_ADDR (MAC)
  3. Cross-reference with dongle_registry table
  4. Verify count matches expected (5)
Pass: 5 dongles detected, all MACs stable from last boot
Fail: <5 dongles, or MAC changed (re-enumeration)
Duration: <5s
```

#### S1: SQLite Integrity

```
Procedure:
  1. PRAGMA integrity_check on hub.db
  2. PRAGMA foreign_key_check
  3. Verify WAL mode: PRAGMA journal_mode
  4. Check file size vs expected range
Pass: All PRAGMAs return 'ok', WAL mode active
Fail: Any integrity error
Duration: <2s
```

#### B3: Multi-Pen Stress

```
Procedure:
  1. Activate all dongles
  2. Wait for N pens to connect (or simulators)
  3. Trigger sync on all connected pens simultaneously
  4. Measure: connection time, throughput per pen, total sync time
  5. Verify: all data dual-written, checksums match
Pass: All pens synced, checksums match, no dongle crashes
Fail: Any pen fails to sync, checksum mismatch, dongle crash
Duration: 2-5 minutes depending on pen count
```

---

## 4. CI/CD Test Pipeline

```
PR opened →
  Stage 1: Build all affected services (L1) .................. ~2 min
  Stage 2: Lint + typecheck (L2) ............................. ~1 min
  Stage 3: Unit tests for affected services (L3) ............. ~3 min
  Stage 4: Integration tests for affected services (L4) ...... ~5 min
  Stage 5: Pipeline tests if pipeline services changed (L5) .. ~10 min

Merge to main →
  Stage 6: Full E2E test suite (L5) .......................... ~15 min
  Stage 7: Hub image build + basic smoke test ................ ~5 min

Release tag →
  Stage 8: Hub hardware-in-loop tests (L6) ................... manual, ~30 min
  Stage 9: Field trial sign-off (L7) ......................... manual
```

### 4.1 Test Evidence in PR

Every PR description must include:

```markdown
## Validation Evidence
- [x] L1: Build verified (Docker images built)
- [x] L2: Lint + typecheck clean
- [x] L3: Unit tests — {list specific tests that ran}
- [x] L4: Integration tests — {list specific tests}
- [ ] L5: E2E — not applicable / deferred to merge pipeline
- **Not verified:** {explicitly list what was NOT tested}
- **Residual risks:** {list any known risks}
```

---

## 5. Test Data Management

### 5.1 Fixtures

| Fixture | Location | Contents |
|---|---|---|
| Sample stroke data | `test-suite/fixtures/strokes/` | Binary stroke files from 5 pen models, various handwriting styles |
| Sample page images | `test-suite/fixtures/pages/` | Rendered pages + camera-captured images for OCR comparison |
| Sample exam config | `test-suite/fixtures/exams/` | Exam definitions with rubrics, question maps, variants |
| Known plagiarism pairs | `test-suite/fixtures/plagiarism/` | Paired answers with known similarity scores |
| BLE pen simulator data | `test-suite/fixtures/ble/` | GATT characteristic dumps for simulated pens |

### 5.2 Seed Script

```bash
# Seed a local dev environment with realistic test data
./scripts/seed-data.sh --students 40 --exams 3 --questions-per-exam 10
```

Creates: 40 students, 3 exams, 400 stroke files, AI results, scores, 5 objections, 2 plagiarism flags.

### 5.3 Pen Simulator

For hub testing without physical pens:

```
nRF52840-DK flashed with pen-simulator firmware
  - Advertises with configurable MAC
  - Serves GATT pen service (0xEP01)
  - Pre-loaded with configurable stroke data (0–2 MB)
  - Supports: normal sync, slow sync, checksum failure, mid-sync disconnect
```

Software alternative: `test-suite/hub-tests/ble_pen_sim.py` using `bleak` library on a second BLE adapter.

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Removed local JWT issuance assumptions, added parent access coverage, and kept plagiarism verdict testing aligned with `svc-plagiarism` ownership. | Codex |
