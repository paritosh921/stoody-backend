# ExamPen — Implementation Task List

> **Purpose**: Maximum-parallelism task breakdown for multi-agent implementation.
> Tasks are organized in **Waves** — all tasks within a wave can run concurrently on independent agents.
> A wave starts only when its listed dependencies from prior waves are complete.
>
> **Coordination rule**: Agents MUST NOT read another agent's service `src/` code. Build against contracts only.
> After each wave, a brief stitch checkpoint verifies integration before the next wave begins.

---

## Legend

| Field | Meaning |
|-------|---------|
| **ID** | Unique task identifier: `W{wave}.{agent}.{seq}` |
| **Agent** | Suggested agent lane (A1–A8). Tasks on the same lane are sequential. |
| **Deps** | Task IDs that must be COMPLETE before this task starts |
| **Contract** | The authoritative spec(s) the agent reads — not source code |
| **Outputs** | Deliverables the agent produces |
| **Validation** | Minimum evidence level (L1–L7) required before marking COMPLETE |
| **Status** | `NOT_STARTED` / `IN_PROGRESS` / `COMPLETE` / `BLOCKED` |

---

## Wave 0 — Foundations (Zero Dependencies, All Parallel)

Everything in this wave can start immediately. Eight independent tracks.

| ID | Task | Agent | Deps | Contract | Outputs | Validation | Status |
|----|------|-------|------|----------|---------|------------|--------|
| W0.A1.1 | **Shared proto/schema definitions** — Create `libs/exampen-proto/` with all Protobuf/JSON Schema types: stroke, exam, score, event envelopes. Derive from `contracts/events/*.schema.json` and `api/*.openapi.yaml` shared `#/components/schemas`. | A1 | — | All `api/*.openapi.yaml`, `contracts/events/*.schema.json` | `libs/exampen-proto/` with generated Python + TS types | L2 (typecheck) | NOT_STARTED |
| W0.A2.1 | **Python shared utilities** — Create `libs/exampen-common-py/`: JWT validation helper (Stoody JWKS), NATS connection factory, PostgreSQL connection factory, structured logging setup. No business logic. | A2 | — | `STOODY_INTEGRATION_SPEC.md` §1.1, `COMPONENT_INDEPENDENCE_MAP.md` §6.2 | `libs/exampen-common-py/` with `auth.py`, `nats_client.py`, `db.py`, `logging.py` + tests | L3 (unit) | NOT_STARTED |
| W0.A3.1 | **TypeScript shared utilities** — Create `libs/exampen-common-ts/`: auth helper, typed API client generator, shared TS types. | A3 | — | `api/*.openapi.yaml` (response shapes) | `libs/exampen-common-ts/` with `auth.ts`, `api-client.ts`, `types.ts` + tests | L2 (typecheck) | NOT_STARTED |
| W0.A4.1 | **hub-store module** — Dual-write engine: SD fsync → USB fsync → ACK. SQLite WAL mode. File layout per `HUB_DEPLOYMENT_SPEC.md` §3.2. Read/integrity-check functions. Degraded mode when USB missing. | A4 | — | `HUB_DEPLOYMENT_SPEC.md` §3, `hub/ipc-protocol.md` (store messages), `STATE_OWNERSHIP_MAP.md` (hub-store rows) | `hub/hub-store/` with `src/`, `tests/`, IPC message handlers | L3 (unit) | NOT_STARTED |
| W0.A5.1 | **hub-timer module** — Countdown timer using `CLOCK_MONOTONIC`. Persist to SQLite every 10s. Reboot recovery from `active_timer` table. Arm/cancel via IPC messages. | A5 | — | `HUB_DEPLOYMENT_SPEC.md` §3.1 (`active_timer` table), `hub/ipc-protocol.md` (timer messages), `FAILURE_MITIGATION_REGISTER.md` F1/F4 | `hub/hub-timer/` with `src/`, `tests/` | L3 (unit) | NOT_STARTED |
| W0.A6.1 | **hub-tui shell** — Textual TUI framework: 8-screen layout (Setup, Status, WiFi, Dongles, Exams, Diagnostics, Logs, Shutdown). Screen shells with placeholder content. Status screen refreshes at 1 Hz via IPC polling stub. | A6 | — | `HUB_DEPLOYMENT_SPEC.md` §2 (full screen specs) | `hub/hub-tui/` with screen modules, `tests/` | L2 (lint + snapshot tests) | NOT_STARTED |
| W0.A7.1 | **Docker Compose dev stack** — `infra/docker-compose.yml` with: PostgreSQL 16, TimescaleDB, NATS JetStream, MinIO, Redis, Traefik. Per-service placeholder entries. Health checks. `infra/docker-compose.test.yml` for test isolation. | A7 | — | `COMPONENT_INDEPENDENCE_MAP.md` §2, service Dockerfiles (placeholder) | `infra/docker-compose.yml`, `infra/docker-compose.test.yml`, `infra/traefik/` config | L1 (`docker compose config` valid) | NOT_STARTED |
| W0.A8.1 | **CI/CD pipeline skeleton** — GitHub Actions (or equivalent): build → lint → unit → integration → E2E stages. Per-service change detection (only test affected services). PR template with validation evidence checklist. | A8 | — | `TEST_SUITE_SPEC.md` §4 (CI stages), `COMPONENT_INDEPENDENCE_MAP.md` §5.1 | `.github/workflows/ci.yml`, PR template, pre-commit hook for file-size enforcement | L1 (pipeline syntax valid) | NOT_STARTED |
| W0.A7.2 | **Monitoring stack** — `infra/monitoring/`: Grafana, Loki, Tempo, Prometheus configs. Dashboard stubs for API latency, NATS lag, hub fleet. Alert rules for error rate, latency. | A7 | — | `COMPONENT_INDEPENDENCE_MAP.md` §2 (`infra/monitoring/`), `FAILURE_MITIGATION_REGISTER.md` (detection mechanisms) | `infra/monitoring/` configs, Grafana dashboard JSONs | L1 (configs parse) | NOT_STARTED |
| W0.A4.2 | **hub-common IPC library** — Shared IPC envelope, message types, Unix domain socket client/server, JSON-lines encoding. This is the `hub/hub-common/` package used by all hub modules. | A4 | — | `hub/ipc-protocol.md` (full spec) | `hub/hub-common/` with `ipc_protocol.py`, `config.py`, `types.py`, tests | L3 (unit) | NOT_STARTED |
| W0.A8.2 | **Seed data script** — `scripts/seed-data.sh` and Python helpers: generate 40 students, 3 exams, 10 questions/exam, stroke fixtures, AI result fixtures, score fixtures, objections, plagiarism flags. | A8 | — | `TEST_SUITE_SPEC.md` §5 (fixtures), `api/*.openapi.yaml` (data shapes) | `scripts/seed-data.sh`, `test-suite/fixtures/` | L1 (script runs without error) | NOT_STARTED |

### Wave 0 Stitch Checkpoint

Before Wave 1 starts:
- [ ] `libs/exampen-proto/` generates valid Python and TS types
- [ ] `libs/exampen-common-py/` JWT helper validates test JWTs
- [ ] `hub-common` IPC envelope round-trips correctly
- [ ] Docker Compose stack starts with `docker compose up -d` and all infra services healthy
- [ ] `hub-store` dual-write passes unit tests
- [ ] `hub-timer` arm/cancel/resume passes unit tests

---

## Wave 1 — Core Services & Hub BLE (Depends on Wave 0)

| ID | Task | Agent | Deps | Contract | Outputs | Validation | Status |
|----|------|-------|------|----------|---------|------------|--------|
| W1.A1.1 | **svc-auth** — Stoody JWT validation (JWKS fetch + cache + rotation). Claim normalization. ExamPen role mapping table. Revocation store. RLS middleware (`SET app.current_tenant`). Stoody profile enrichment (graceful degradation). Parent-child scope resolution. Build with **Stoody mock server**. | A1 | W0.A1.1, W0.A2.1, W0.A7.1 | `api/auth.openapi.yaml`, `STOODY_INTEGRATION_SPEC.md`, `STATE_OWNERSHIP_MAP.md` (auth rows) | `services/svc-auth/` with full layer split (domain/storage/routes/adapters), Stoody mock, tests | L4 (integration) | NOT_STARTED |
| W1.A1.2 | **Stoody mock server** — Lightweight FastAPI app that simulates: JWKS endpoint, user profile API, student roster API, class/subject APIs, parent-children API. Configurable responses. Used by all services needing Stoody data. | A1 | W0.A1.1 | `STOODY_INTEGRATION_SPEC.md` §4 (all consumed APIs) | `test-suite/stoody-mock/` with Dockerfile, canned responses | L1 (starts, serves JWKS) | NOT_STARTED |
| W1.A2.1 | **svc-exam-orch** — Exam CRUD, lifecycle FSM (`created→armed→timer_running→...→locked`), pen binding (create/confirm/reject), invigilator assignment, scheduling. Integration with svc-auth for JWT. | A2 | W0.A1.1, W0.A2.1, W0.A7.1, W1.A1.1 | `api/exam-orch.openapi.yaml`, `contracts/events/exam.lifecycle.schema.json`, `STATE_OWNERSHIP_MAP.md` (exam + binding rows) | `services/svc-exam-orch/` with full layer split, tests, NATS `exam.lifecycle` publisher | L4 (integration) | NOT_STARTED |
| W1.A3.1 | **svc-stroke-ingest** — Hub chunk upload endpoint (`POST /chunks`), per-chunk CRC32 validation, per-chunk ACK, idempotency key (`{exam_id, pen_mac, chunk_index}`), NATS `stroke.raw` publisher, upload status reconciliation endpoint. | A3 | W0.A1.1, W0.A2.1, W0.A7.1, W1.A1.1 | `api/stroke-ingest.openapi.yaml`, `contracts/events/stroke.raw.schema.json`, `STATE_OWNERSHIP_MAP.md` (stroke ingest rows) | `services/svc-stroke-ingest/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W1.A4.1 | **hub-ble-mgr** — Enumerate 5 USB BLE dongles, manage connections (8 pens/dongle), stagger scan activation (500ms between dongles), handle dongle hot-plug/failure, D-Bus integration for BlueZ, pen discovery + GATT service detection. IPC messages to supervisor. | A4 | W0.A4.2 | `hub/ble-gatt-spec.md` (pen GATT service), `hub/ipc-protocol.md` (BLE manager messages), `FAILURE_MITIGATION_REGISTER.md` A1.1/H1/H3/H5 | `hub/hub-ble-mgr/` with `src/`, `tests/` | L3 (unit + mocked BLE) | NOT_STARTED |
| W1.A5.1 | **hub-invig-ble** — BLE peripheral for invigilator mobile app: auth characteristic (rotating codes), command relay (exam start/stop, manual register), status feed (1 Hz JSON), MAC list read. | A5 | W0.A4.2 | `hub/ble-gatt-spec.md` (invigilator GATT service), `hub/ipc-protocol.md` (invig messages), `FAILURE_MITIGATION_REGISTER.md` S3 | `hub/hub-invig-ble/` with `src/`, `tests/` | L3 (unit + mocked BLE) | NOT_STARTED |
| W1.A6.1 | **hub-supervisor** — Process manager for all hub child modules. FSM orchestration (`created→armed→timer_running→dongle_activation→pen_sync→...`). Crash recovery + restart. Watchdog (`WatchdogSec=30`). IPC router. Interaction log writer. First-boot detection. | A6 | W0.A4.2, W0.A4.1, W0.A5.1 | `hub/ipc-protocol.md` (supervisor messages), `HUB_DEPLOYMENT_SPEC.md` §6 (systemd), §7 (first-boot), `STATE_OWNERSHIP_MAP.md` (hub FSM) | `hub/hub-supervisor/` with `src/`, `tests/`, systemd unit file | L3 (unit + IPC integration with stubs) | NOT_STARTED |
| W1.A7.1 | **Hub SQLite schema** — Create migration script for all hub tables: `hub_config`, `invig_codes`, `pen_inventory`, `exam_sessions`, `pen_bindings`, `pen_sync_status`, `upload_ledger`, `dongle_registry`, `interaction_log`, `active_timer`. WAL mode. Foreign key enforcement. | A7 | — | `HUB_DEPLOYMENT_SPEC.md` §3.1 (full DDL) | `hub/hub-common/migrations/001_initial.sql`, migration runner | L3 (schema loads, FK checks pass) | NOT_STARTED |
| W1.A8.1 | **Mock generation tooling** — Script that reads `api/*.openapi.yaml` and generates Prism-based mock servers per service. Each mock returns canned responses matching the spec. `scripts/generate-mocks.sh`. | A8 | W0.A1.1 | All `api/*.openapi.yaml` | `scripts/generate-mocks.sh`, `infra/docker-compose.mock.yml` | L1 (mocks start and serve spec-compliant responses) | NOT_STARTED |

### Wave 1 Stitch Checkpoint

- [ ] `svc-auth` introspects a Stoody-mock JWT and returns normalized ExamPen claims
- [ ] `svc-exam-orch` creates an exam, transitions FSM, and publishes `exam.lifecycle` to NATS
- [ ] `svc-stroke-ingest` accepts a chunk upload, validates CRC, publishes `stroke.raw` to NATS
- [ ] `hub-ble-mgr` enumerates mock dongles and reports via IPC
- [ ] `hub-supervisor` boots, spawns stub children, responds to IPC health queries
- [ ] Mock generation script produces working mocks for all 12 services

---

## Wave 2 — Pipeline Mid-Tier & Hub Sync (Depends on Wave 1)

| ID | Task | Agent | Deps | Contract | Outputs | Validation | Status |
|----|------|-------|------|----------|---------|------------|--------|
| W2.A2.1 | **svc-stroke-proc** — NATS `stroke.raw` consumer, dedup by idempotency key (`SELECT FOR UPDATE`), coordinate normalization (DPI transform), page assignment from question regions, atomic batch commit to TimescaleDB, publish `stroke.processed` event. | A2 | W1.A3.1, W0.A7.1 | `contracts/events/stroke.raw.schema.json`, `contracts/events/stroke.processed.schema.json`, `STATE_OWNERSHIP_MAP.md` (stroke-proc rows) | `services/svc-stroke-proc/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W2.A3.1 | **svc-doc-assembly** — NATS `stroke.processed` consumer, stroke-to-SVG/PNG page rendering, miss indicator auto-detection (no strokes / sync failure / possible miss), page image upload to MinIO, metadata to PostgreSQL (S3 first, PG second), publish `page.ready` event. | A3 | W2.A2.1 (can stub events), W0.A7.1 | `contracts/events/stroke.processed.schema.json`, `contracts/events/page.ready.schema.json`, `STATE_OWNERSHIP_MAP.md` (doc-assembly rows) | `services/svc-doc-assembly/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W2.A4.1 | **hub-pen-sync** — GATT read from pen (chunk-by-chunk), pass chunks to `hub-store` for dual-write, ACK pen only after store confirms, checksum verification (whole buffer CRC32), per-pen sync state machine, 3 retries on disconnect (30s timeout each). IPC messages to supervisor. | A4 | W1.A4.1, W0.A4.1 | `hub/ble-gatt-spec.md` (pen sync protocol), `hub/ipc-protocol.md` (pen-sync messages), `FAILURE_MITIGATION_REGISTER.md` A1.7/S4 | `hub/hub-pen-sync/` with `src/`, `tests/` | L3 (unit + mocked GATT) | NOT_STARTED |
| W2.A5.1 | **hub-uplink** — WiFi/mobile upload path selection, per-pen chunk upload to `svc-stroke-ingest` API, resume from `upload_ledger` (acked_chunks tracking), backend reachability check, upload progress reporting via IPC. | A5 | W0.A4.1, W1.A5.1 | `api/stroke-ingest.openapi.yaml` (upload endpoint), `hub/ipc-protocol.md` (uplink messages), `FAILURE_MITIGATION_REGISTER.md` U1/U4 | `hub/hub-uplink/` with `src/`, `tests/` | L3 (unit + mocked HTTP) | NOT_STARTED |
| W2.A6.1 | **svc-copy-upload** — Multipart image upload, S3 write → PG metadata write (ordered), per-page per-student storage, publish `copy.ready` event, serve images for teacher/student BFFs. | A6 | W1.A1.1, W0.A7.1 | `api/copy-upload.openapi.yaml`, `contracts/events/copy.ready.schema.json` | `services/svc-copy-upload/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W2.A7.1 | **svc-chat** — Append-only message store (no UPDATE, no DELETE), per-objection thread, teacher-student messaging, read receipts. RBAC enforcement (own students only). | A7 | W1.A1.1, W0.A7.1 | `api/chat.openapi.yaml`, `STATE_OWNERSHIP_MAP.md` (chat rows) | `services/svc-chat/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W2.A8.1 | **Hub golden image build script** — `infra/hub-image/` scripts: base Ubuntu 24.04 arm64 RPi image, pre-install packages (BlueZ, Python, SQLite, NM, chrony, etc.), partition layout (boot 512MB, rootfs 8GB, data 16GB, swap 1GB), US WiFi regulatory domain baked in, systemd unit files, `/etc/exampen/` config directory. | A8 | W1.A6.1, W1.A7.1 | `HUB_DEPLOYMENT_SPEC.md` §1 (full image spec), §6 (systemd) | `infra/hub-image/build-image.sh`, systemd unit files, partition config | L1 (script runs, image structure valid) | NOT_STARTED |

### Wave 2 Stitch Checkpoint

- [ ] `stroke.raw` → `svc-stroke-proc` → `stroke.processed` → `svc-doc-assembly` → `page.ready` — full pipeline path verified in Docker
- [ ] `hub-pen-sync` reads mock GATT, writes via `hub-store`, verifies checksum
- [ ] `hub-uplink` uploads chunks to `svc-stroke-ingest` mock and tracks acked chunks
- [ ] `svc-chat` append + read-receipt cycle works
- [ ] Hub golden image script produces valid partition structure

---

## Wave 3 — AI, Scoring, & Analysis (Depends on Wave 2)

| ID | Task | Agent | Deps | Contract | Outputs | Validation | Status |
|----|------|-------|------|----------|---------|------------|--------|
| W3.A1.1 | **svc-ai-pipeline** — Consume `page.ready` events, ONNX model inference (HWR English + Devanagari, step detection, diagram classification), store results in PG with model version, publish `ai.result` event. Support re-run with new model version (new result row, no overwrite). | A1 | W2.A3.1 (page images available) | `contracts/events/page.ready.schema.json`, `contracts/events/ai.result.schema.json`, `STATE_OWNERSHIP_MAP.md` (AI rows) | `services/svc-ai-pipeline/` with full layer split, model registry, tests | L4 (integration with test corpus) | NOT_STARTED |
| W3.A2.1 | **svc-score-engine** — Consume `ai.result` events, rubric evaluation (step-level marking), event-sourced score store (append-only), materialized view for current score, score FSM (`ai_draft→teacher_reviewed→finalized→objection_window→locked`), teacher override with audit trail, score publication trigger, publish `score.updated` event. Rubric versioning. | A2 | W3.A1.1 (can stub AI events) | `api/score-engine.openapi.yaml`, `contracts/events/ai.result.schema.json`, `contracts/events/score.updated.schema.json`, `STATE_OWNERSHIP_MAP.md` (score rows) | `services/svc-score-engine/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W3.A3.1 | **svc-plagiarism** — Consume AI results when `plagiarism.check` triggered (all AI results ready for exam), TF-IDF cosine similarity, structural similarity (edit distance), temporal correlation, seating proximity weighting, question-type threshold adjustment (MCQ exclusion), composite scoring, teacher verdict persistence, publish `plagiarism.result` event. | A3 | W3.A1.1 (can stub AI results) | `api/plagiarism.openapi.yaml`, `contracts/events/plagiarism.check.schema.json`, `contracts/events/plagiarism.result.schema.json`, `STATE_OWNERSHIP_MAP.md` (plagiarism rows) | `services/svc-plagiarism/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W3.A4.1 | **Hub supervisor integration** — Wire all hub modules (store, timer, ble-mgr, pen-sync, uplink, invig-ble, tui) through real IPC. End-to-end hub flow: first-boot → provisioning → invigilator connect → exam arm → timer → dongle activation → pen sync → upload. | A4 | W2.A4.1, W2.A5.1, W1.A4.1, W1.A5.1, W1.A6.1, W0.A5.1, W0.A6.1 | `hub/ipc-protocol.md` (all messages), `HUB_DEPLOYMENT_SPEC.md` §7 (first-boot sequence) | Integrated `hub/` with all modules communicating via IPC, integration tests | L4 (full hub IPC integration) | NOT_STARTED |
| W3.A5.1 | **svc-review** — Objection filing (student), FSM (`filed→assigned→reviewing→resolved`), assignment to evaluator, side-by-side context (answer image + AI + score + rubric + objection text), approve (triggers re-score via NATS to score-engine), reject (mandatory reason), escalate to HOD. | A5 | W3.A2.1 (score context needed) | `api/review.openapi.yaml`, `contracts/events/objection.schema.json`, `STATE_OWNERSHIP_MAP.md` (objection rows) | `services/svc-review/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W3.A6.1 | **svc-analytics** — Consume `score.updated` events, percentile calculation (idempotent recomputation), leaderboard generation (configurable scope: section/grade/institute), class stats (mean, median, std dev, pass %), question-wise difficulty analysis, export metadata (PDF/CSV links). | A6 | W3.A2.1 (score events) | `api/analytics.openapi.yaml`, `contracts/events/score.updated.schema.json`, `STATE_OWNERSHIP_MAP.md` (analytics rows) | `services/svc-analytics/` with full layer split, tests | L4 (integration) | NOT_STARTED |
| W3.A7.1 | **svc-notify** — Event-driven notification triggers: score published → student email/push, objection status change → both parties notified, exam scheduled → student reminder. Template engine. Email + push + SMS adapters (pluggable). | A7 | W1.A1.1 (auth for user lookup) | Consumes: `score.updated`, `objection.*`, `exam.lifecycle` events | `services/svc-notify/` with full layer split, tests | L3 (unit + mocked adapters) | NOT_STARTED |
| W3.A8.1 | **E2E pipeline test suite** — Tests E2E-01 through E2E-12 from `TEST_SUITE_SPEC.md`. Full Docker Compose stack. 40-student exam simulation. Verify stroke → AI → score → analytics propagation. | A8 | W3.A2.1, W3.A1.1, W2.A2.1, W2.A3.1 | `TEST_SUITE_SPEC.md` §2.3 (all E2E test IDs), seed fixtures | `test-suite/pipeline-tests/` with all E2E-* tests | L5 (E2E verified) | NOT_STARTED |

### Wave 3 Stitch Checkpoint

- [ ] Full pipeline: `stroke.raw → stroke.processed → page.ready → ai.result → score.updated` verified end-to-end
- [ ] Score override → `score.updated` → analytics percentile recalculation verified
- [ ] Objection → re-score → notification chain verified
- [ ] Plagiarism detection produces flags for known similar pair in test fixtures
- [ ] Hub integration: boot → provision → arm → sync → upload — all via IPC, no module stubs
- [ ] E2E-08 (40-student simulation) passes

---

## Wave 4 — BFF Aggregators (Depends on Wave 3 Backing Services)

BFFs are thin read-only aggregation layers. They can be built against mocks of backing services and then integration-tested when real services are available.

| ID | Task | Agent | Deps | Contract | Outputs | Validation | Status |
|----|------|-------|------|----------|---------|------------|--------|
| W4.A1.1 | **svc-teacher-bff** — Aggregated class score overview, student drill-down, AI analysis viewer proxy, score edit relay (to score-engine), bulk approve relay, step-level marking relay, miss indicator review, plagiarism review proxy, objection inbox proxy, leaderboard proxy, export trigger. RBAC enforcement (teacher/HOD/principal scopes). | A1 | W3.A2.1, W3.A3.1, W3.A5.1, W3.A6.1, W2.A7.1 (or mocks via W1.A8.1) | `api/teacher-bff.openapi.yaml`, backing service APIs | `services/svc-teacher-bff/` with routes + adapter per backing service, tests | L4 (integration with mocked backing services) | NOT_STARTED |
| W4.A2.1 | **svc-student-bff** — Score summary, question-wise breakdown, answer image viewer proxy, AI analysis read-only proxy, feedback view, miss indicators, percentile chart, objection filing relay (to review), objection status, chat proxy. Parent JWT support (only linked children). | A2 | W3.A2.1, W3.A6.1, W3.A5.1, W2.A7.1 (or mocks) | `api/student-bff.openapi.yaml`, backing service APIs | `services/svc-student-bff/` with routes + adapters, tests | L4 (integration with mocked backing services) | NOT_STARTED |
| W4.A3.1 | **svc-invig-console** — Real-time WebSocket status feed (exam state, per-pen sync progress, dongle health, hub connectivity). REST endpoints for session details. Relay hub status from `hub-uplink` via backend. | A3 | W1.A2.1 (exam-orch for session data), W2.A5.1 (hub-uplink status relay) | `api/invig-console.openapi.yaml`, `hub/ipc-protocol.md` (status shape) | `services/svc-invig-console/` with WebSocket handler, REST routes, tests | L4 (integration) | NOT_STARTED |
| W4.A4.1 | **Hub TUI full integration** — Wire TUI screens to live IPC data: Status screen (real dongle/sync data at 1 Hz), WiFi screen (nmcli integration), Dongle management (health, reset), Exam history (SQLite queries), Diagnostics (test runner integration), Log viewer (journald tailing). | A4 | W3.A4.1 (hub fully integrated) | `HUB_DEPLOYMENT_SPEC.md` §2 (all screen specs), `hub/ipc-protocol.md` | Updated `hub/hub-tui/` with live data bindings, integration tests | L4 (TUI renders real IPC data) | NOT_STARTED |
| W4.A5.1 | **Hub diagnostics test runner** — TUI Diagnostics screen implementation: H1-H7 hardware tests, S1-S5 software tests, B1-B4 BLE tests. Status icons, run-all/run-selected, export to JSON. | A5 | W3.A4.1 (hub integrated) | `TEST_SUITE_SPEC.md` §3 (TUI test runner spec), `HUB_DEPLOYMENT_SPEC.md` §2.3 screen [6] | `hub/hub-tui/diagnostics/` with test implementations, JSON export | L3 (unit + mocked hardware) | NOT_STARTED |
| W4.A6.1 | **Webhook delivery to Stoody** — `svc-notify` or dedicated module: consume `score.updated` (published), `exam.lifecycle` (created/completed) events → POST to Stoody webhook endpoints with correct payloads. Retry with exponential backoff. | A6 | W3.A7.1, W3.A2.1 | `STOODY_INTEGRATION_SPEC.md` §4.2 (webhook payloads) | Webhook publisher in `svc-notify` or standalone, Stoody mock webhook receiver in test suite | L4 (integration with Stoody mock) | NOT_STARTED |

### Wave 4 Stitch Checkpoint

- [ ] Teacher BFF returns aggregated class score overview with correct data from all backing services
- [ ] Student BFF returns own scores, supports objection filing, respects parent-child scoping
- [ ] Invigilator console WebSocket streams real-time sync progress
- [ ] Hub TUI Status screen shows live dongle/pen/sync data from IPC
- [ ] Stoody webhook receives `score.published` event with correct payload

---

## Wave 5 — Frontends & Mobile (Depends on Wave 4 BFFs)

All frontend tasks build against BFF OpenAPI contracts. They can run fully in parallel.

| ID | Task | Agent | Deps | Contract | Outputs | Validation | Status |
|----|------|-------|------|----------|---------|------------|--------|
| W5.A1.1 | **teacher-dashboard: Exam Management** — Create exam form, define rubric (rich editor), question region editor (visual bounding boxes on uploaded answer sheet), assign invigilators/evaluators, set schedule, manage question paper upload, exam list with status filters. | A1 | W4.A1.1 (teacher BFF available or mocked) | `api/teacher-bff.openapi.yaml` (exam endpoints) | `frontend/teacher-dashboard/` exam management pages | L2 (typecheck + lint) | NOT_STARTED |
| W5.A1.2 | **teacher-dashboard: Score Review** — Class score overview table (sort/filter), student drill-down (side-by-side answer + AI), AI analysis viewer, inline score edit (mandatory reason), bulk approve, step-level marking, manual feedback entry, miss indicator grid, plagiarism review (side-by-side + evidence), score finalization. | A1 | W4.A1.1 | `api/teacher-bff.openapi.yaml` (score/plagiarism endpoints) | `frontend/teacher-dashboard/` score review pages | L2 | NOT_STARTED |
| W5.A2.1 | **teacher-dashboard: Objections & Analytics** — Objection inbox, detail view, approve/reject/escalate, per-objection chat thread. Leaderboard, historical performance, class analytics, question-wise analysis, export (PDF/CSV). | A2 | W4.A1.1 | `api/teacher-bff.openapi.yaml` (objection/analytics endpoints) | `frontend/teacher-dashboard/` objection + analytics pages | L2 | NOT_STARTED |
| W5.A3.1 | **student-portal** — Upcoming exams, past exams, score summary, question-wise breakdown, answer image viewer (pinch-zoom), AI analysis (read-only), feedback view, miss indicators, percentile chart. Objection filing (during window only), status tracking, chat with tutor. Score history, trend charts, strength/weakness analysis. | A3 | W4.A2.1 (student BFF available or mocked) | `api/student-bff.openapi.yaml` | `frontend/student-portal/` all pages | L2 | NOT_STARTED |
| W5.A4.1 | **invigilator-console** — Real-time WebSocket dashboard: exam state, per-pen sync progress bars, dongle health indicators, hub connectivity status. Session list. Alert display for failures. | A4 | W4.A3.1 (invig-console service available or mocked) | `api/invig-console.openapi.yaml` | `frontend/invigilator-console/` dashboard page | L2 | NOT_STARTED |
| W5.A5.1 | **exampen-mobile: Hub Control (Invigilator Mode)** — Flutter BLE: connect to hub, authenticate (rotating code), register pens (scan + manual), start/stop exam, monitor sync progress (real-time BLE status feed), trigger upload, camera capture (copy images). Offline-capable for BLE commands. | A5 | W1.A5.1 (hub invig-ble spec) | `hub/ble-gatt-spec.md` (invigilator GATT), `api/copy-upload.openapi.yaml` | `mobile/exampen-mobile/lib/hub_control/` | L2 | NOT_STARTED |
| W5.A6.1 | **exampen-mobile: Teacher View** — Score management: class overview, student drill-down, inline score edit, step-level marking, feedback entry, miss indicator review, plagiarism review (simplified). Objection handling: inbox, review, approve/reject, chat. Analytics: leaderboard, class stats. | A6 | W4.A1.1 (teacher BFF) | `api/teacher-bff.openapi.yaml` | `mobile/exampen-mobile/lib/teacher_view/` | L2 | NOT_STARTED |
| W5.A7.1 | **exampen-mobile: Student View** — Score view: summary, question breakdown, answer images, AI analysis, feedback, miss indicators, percentile. Objections: file, track, chat. Performance: history, trends, strength/weakness. | A7 | W4.A2.1 (student BFF) | `api/student-bff.openapi.yaml` | `mobile/exampen-mobile/lib/student_view/` | L2 | NOT_STARTED |
| W5.A8.1 | **exampen-mobile: Core & Auth** — Flutter shared layer: Stoody JWT auth flow, secure token storage, network service (API client from BFF OpenAPI), offline storage (Hive/drift), push notification registration. Shared between all mobile modes. | A8 | W0.A3.1 (TS types for reference), W1.A1.1 (auth spec) | `api/auth.openapi.yaml`, `STOODY_INTEGRATION_SPEC.md` §5 | `mobile/exampen-mobile/lib/core/` | L2 | NOT_STARTED |

### Wave 5 Stitch Checkpoint

- [ ] Teacher dashboard renders class score overview from BFF mock
- [ ] Student portal renders score summary and files objection against BFF mock
- [ ] Invigilator console shows live WebSocket sync progress
- [ ] Mobile hub-control connects to BLE mock and completes auth flow
- [ ] Mobile teacher + student views render score data from BFF

---

## Wave 6 — Integration, Hardening, & Field Readiness (Depends on Wave 5)

| ID | Task | Agent | Deps | Contract | Outputs | Validation | Status |
|----|------|-------|------|----------|---------|------------|--------|
| W6.A1.1 | **Full E2E integration test** — Wire all services in Docker Compose. Run E2E-01 through E2E-12. Run 40-student simulation (E2E-08). Verify Stoody webhook delivery (E2E-12). Fix integration issues. | A1 | All Wave 3–4 services | `TEST_SUITE_SPEC.md` §2.3 | All E2E tests passing in `test-suite/pipeline-tests/` | L5 (E2E verified) | NOT_STARTED |
| W6.A2.1 | **Frontend ↔ BFF integration test** — Wire teacher-dashboard + student-portal + invigilator-console to real BFF services. Verify all CRUD flows, WebSocket real-time updates, RBAC enforcement (student can't access teacher endpoints). | A2 | W5.A1.1, W5.A1.2, W5.A2.1, W5.A3.1, W5.A4.1 | BFF OpenAPI specs | Integration test results, bug fixes | L5 | NOT_STARTED |
| W6.A3.1 | **Mobile ↔ Hub BLE integration test** — Test mobile hub-control against real hub (or hub-in-Docker with BLE simulator). Auth flow, pen registration, exam lifecycle, sync monitoring, upload trigger. | A3 | W5.A5.1, W3.A4.1 | `hub/ble-gatt-spec.md`, `TEST_SUITE_SPEC.md` §2.4 (HW tests) | Hub hardware test results (HW-I1) | L6 (hardware-in-loop) | NOT_STARTED |
| W6.A4.1 | **Hub hardware-in-loop test suite** — Run HW-H1 through HW-P1 on real RPi with 5 BLE dongles. Pen simulator (nRF52840-DK or `ble_pen_sim.py`). Timer accuracy test. Power failure recovery test. | A4 | W3.A4.1, W4.A5.1 | `TEST_SUITE_SPEC.md` §2.4 (all HW-* tests) | `test-suite/hub-tests/` results, L6 evidence | L6 (hardware-in-loop) | NOT_STARTED |
| W6.A5.1 | **Security hardening** — RLS policy audit (every table has tenant_id policy), RBAC matrix enforcement (test all 7 roles × all endpoints from `STOODY_INTEGRATION_SPEC.md` §6), DPDPA compliance check (data minimization, retention cron, encryption audit), penetration test prep. | A5 | All services complete | `STOODY_INTEGRATION_SPEC.md` §6 (RBAC matrix), `FAILURE_MITIGATION_REGISTER.md` A8.1/A8.2 | Security audit report, RLS test in CI, retention cron job | L4 | NOT_STARTED |
| W6.A6.1 | **Documentation chapters** — Write chapters 01–25 per `DOCUMENTATION_PLAN.md` template. Each chapter references specific test IDs, failure mitigation IDs, and interface specs. Update `BUILD_STATUS.md` to all COMPLETE. | A6 | All prior tasks | `DOCUMENTATION_PLAN.md` (template + quality gates) | `docs/chapters/01_*.md` through `25_*.md` | Review pass | NOT_STARTED |
| W6.A7.1 | **Production Docker Compose + deployment** — `infra/docker-compose.prod.yml` with resource limits, secrets management, TLS everywhere, backup configuration (pgBackRest → S3), monitoring alerts tuned, hub golden image finalized. | A7 | All services complete | `FAILURE_MITIGATION_REGISTER.md` A8.5 (backup), A8.8 (cost) | Production-ready infrastructure | L4 | NOT_STARTED |
| W6.A8.1 | **Performance / load test** — Simulate peak load: 10K students simultaneously (A8.4 from failure register). Measure NATS consumer lag, ingestion throughput, TimescaleDB write performance. Tune rate limits, connection pools, NATS stream sizes. | A8 | All pipeline services | `FAILURE_MITIGATION_REGISTER.md` A8.4 | Load test results, tuning recommendations | L5 | NOT_STARTED |

---

## Agent Lane Summary

Each agent lane is a sequential chain. Lanes run in parallel.

```
A1: proto schemas → svc-auth + Stoody mock → svc-ai-pipeline → svc-teacher-bff → teacher-dashboard (exam mgmt + score review) → full E2E
A2: common-py → svc-exam-orch → svc-stroke-proc → svc-score-engine → svc-student-bff → teacher-dashboard (objections + analytics) → frontend integration
A3: common-ts → svc-stroke-ingest → svc-doc-assembly → svc-plagiarism → svc-invig-console → student-portal → mobile BLE integration
A4: hub-store + hub-common → hub-ble-mgr → hub-pen-sync → hub full integration → hub TUI live data → invigilator-console frontend → hub HW tests
A5: hub-timer → hub-invig-ble → hub-uplink → svc-review → hub diagnostics TUI → mobile hub-control → security hardening
A6: hub-tui shell → hub-supervisor → svc-copy-upload → svc-analytics → webhook delivery → mobile teacher-view → documentation
A7: Docker Compose + monitoring → hub SQLite schema → svc-chat → hub golden image → mobile student-view → production deployment
A8: CI/CD + seed data → mock generation → E2E test suite → svc-notify → mobile core/auth → load testing
```

---

## Coordination Rules

1. **Contract-first**: Never read another agent's service `src/`. Build against `api/*.openapi.yaml`, `contracts/events/*.schema.json`, `hub/ble-gatt-spec.md`, `hub/ipc-protocol.md`.
2. **Stitch checkpoints**: After each wave, run cross-agent integration verification before proceeding.
3. **BUILD_STATUS.md**: Every agent updates this file when claiming, completing, or blocking on a task.
4. **Mock mode**: Every service must support `MOCK_MODE=true` returning canned responses from its OpenAPI spec.
5. **Ownership declarations**: Every service `README.md` must include the ownership block from `STATE_OWNERSHIP_MAP.md` §5.
6. **Test IDs**: Reference specific test IDs (e.g., `U-SCR-01`, `I-ORCH-02`) in PR descriptions.
7. **File limits**: Pre-commit hook enforces 300 lines Python / 250 lines TS per file. Use `# EXEMPT: <reason>` only with documentation.
8. **Domain purity**: `domain/` layer in any service must have ZERO I/O imports. CI linter enforces this.

---

## Task Count Summary

| Wave | Tasks | Parallel Agents | Focus |
|------|-------|----------------|-------|
| W0 | 12 | 8 | Foundations: libs, hub primitives, infra, CI |
| W1 | 9 | 8 | Core services (auth, exam-orch, stroke-ingest), hub BLE, mocks |
| W2 | 8 | 8 | Pipeline mid-tier, hub sync/upload, chat, golden image |
| W3 | 8 | 8 | AI, scoring, plagiarism, review, analytics, notifications, E2E tests |
| W4 | 6 | 6 | BFF aggregators, hub TUI integration, webhooks |
| W5 | 8 | 8 | All frontends + mobile (fully parallel) |
| W6 | 8 | 8 | Integration, hardening, docs, production, load testing |
| **Total** | **59** | **8 max concurrent** | |
