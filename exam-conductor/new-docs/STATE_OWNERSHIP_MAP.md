# STATE_OWNERSHIP_MAP.md
# ExamPen — Critical State Ownership, Read/Write Boundaries & Transactional Boundaries

Reference: R4-EXAMPEN-DEVSTACK
Doctrine source: SOFTWARE_DEVELOPMENT_DOCTRINE.md §4 Rule 1

---

## Purpose

This document enforces the doctrine rule: **every critical state must have exactly one writable owner.** All other layers may only read, derive, cache, or project that state.

Any violation of this map is a design bug, not an implementation detail.

---

## 1. Critical State Ownership Table

| Critical State | Writable Owner | Readers / Derivers / Caches | Transactional Boundary | Notes |
|---|---|---|---|---|
| Exam session lifecycle (FSM) | `svc-exam-orch` | `svc-stroke-ingest` (reads gate), `svc-invig-console` (reads for display), hub (receives commands) | Exam state transitions are single-writer atomic updates in PostgreSQL with row-level locking | Hub timer is a LOCAL projection of server-set duration. Hub does NOT own exam state — it executes commands from the orchestrator via invigilator relay. |
| Hub FSM state | `hub-supervisor` | `hub-tui` (display), `hub-invig-ble` (relay to app), all hub modules (read via IPC) | FSM transitions persisted to SQLite `exam_sessions.state` BEFORE side effects execute | Hub is autonomous during exam. Server does NOT override hub FSM mid-exam. Post-exam, server reconciles. |
| Pen ↔ student binding | `svc-exam-orch` (server-side) | Hub caches in `pen_bindings` table (projection + provisional local workflow state). Mobile app displays. | Binding is created or confirmed server-side. Hub NEVER creates authoritative bindings. | **Offline/manual registration:** Hub may store a PROVISIONAL local binding (`pen_bindings.status = 'provisional'`) when invigilator uses manual-register BLE command. This provisional record is: (a) NOT treated as authoritative for scoring — strokes are tagged with pen_mac only, (b) synced to server when connectivity is available (via mobile relay or WiFi), (c) server validates and promotes to `confirmed` or rejects. Until server confirms, hub treats the binding as a display hint only. Scoring pipeline uses server-confirmed bindings exclusively. |
| Raw stroke data (pen-side) | Pen firmware (flash buffer) | Hub reads via GATT (destructive read — pen clears buffer after hub ACKs) | Pen buffer clear (`0x03` write to Sync Control char) ONLY after hub confirms dual-write to SD + USB | This is the most critical data custody transfer in the system. Pen data is irreplaceable. |
| Stroke data (hub-side) | `hub-store` | `hub-uplink` (reads for upload), `hub-pen-sync` (writes via `hub-store` API) | Dual-write fsync protocol: SD fsync → USB fsync → ACK pen | `hub-pen-sync` does NOT write directly. It passes data to `hub-store` which owns the write path. |
| Stroke data (server-side) | `svc-stroke-ingest` → `svc-stroke-proc` (pipeline) | TimescaleDB (persisted), `svc-doc-assembly` (reads) | `svc-stroke-ingest` validates + publishes to NATS. `svc-stroke-proc` is the durable write owner into TimescaleDB. | Idempotency key: `{exam_id, pen_mac, chunk_index}`. Duplicate chunks are safe to replay. |
| Processed strokes + page assignment | `svc-stroke-proc` | `svc-doc-assembly` (reads from TimescaleDB) | Atomic batch write per pen per exam: all chunks for one pen committed in single transaction | No partial pen data in TimescaleDB. Either all chunks committed or none. |
| Page images | `svc-doc-assembly` | `svc-ai-pipeline` (reads from S3), teacher/student BFFs (read for display) | Image written to S3 + metadata written to PostgreSQL in same logical operation. S3 write first, PG metadata second. | If PG write fails after S3 write, orphaned S3 object is acceptable (garbage collected). Reverse order (PG first, S3 fail) would create dangling reference — unacceptable. |
| Copy images (photographed) | `svc-copy-upload` | `svc-ai-pipeline` (reads if no stroke data), teacher BFF (display) | Multipart upload → S3 write → PG metadata write. Same order as page images. | Copy images are fallback data source. Never overwrite stroke-derived page images. |
| AI recognition results | `svc-ai-pipeline` | `svc-score-engine` (reads), `svc-plagiarism` (reads), teacher BFF (display) | AI results written to PostgreSQL per-question per-student. Published to NATS after PG commit. | Model version stored with every result. Re-running AI with new model creates new version, does not overwrite old. |
| Scores (per question per student) | `svc-score-engine` | `svc-analytics` (reads for aggregation), teacher/student BFFs (display), `svc-review` (reads for objection context) | Event-sourced. Score events appended (never updated in place). Materialized view for current score. | Score lifecycle: `ai_draft → teacher_reviewed → finalized → objection_window → locked`. Only `svc-score-engine` writes score events. |
| Score overrides (teacher edits) | `svc-score-engine` | Audit log (read-only projection) | Override = new event appended with `{old_value, new_value, teacher_id, reason, timestamp}`. Materialized view updated atomically. | Teacher BFF sends override request. `svc-score-engine` validates and appends. BFF does NOT write scores directly. |
| Objection state | `svc-review` | `svc-score-engine` (reads to trigger re-score), teacher/student BFFs (display) | Objection lifecycle: `filed → assigned → reviewing → resolved`. State transitions are single-writer in `svc-review`. | Resolution triggers re-score command to `svc-score-engine` via NATS. `svc-review` does NOT modify scores directly. |
| Miss indicators | `svc-doc-assembly` (auto-detection), `svc-score-engine` (teacher override) | Teacher BFF (display) | Auto-state written by `svc-doc-assembly`. Override-state written by `svc-score-engine` (teacher action). Two columns, not competing writes. | `auto_state` is computed, never manually edited. `override_state` is teacher-set. Display logic: show `override_state` if non-NULL, else `auto_state`. |
| Plagiarism flags + teacher verdicts | `svc-plagiarism` | Teacher BFF (display), `svc-score-engine` (reads for context) | Flags written in bulk per exam after all AI results ready. Teacher verdicts update the same service-owned row. | `svc-plagiarism` owns both flag generation and teacher verdict persistence. `svc-score-engine` reads flags and verdicts as context only. |
| Chat messages | `svc-chat` | Teacher/student BFFs (display) | Messages appended (never edited or deleted). Each message has sender_id, recipient_id, exam_id, timestamp. | No message editing. No deletion. Append-only for audit safety (DPDPA: minor's data). |
| Normalized auth claims + ExamPen role mapping + revocation state | `svc-auth` | All services (read normalized JWT claims), BFFs (read for RBAC gating) | Stoody issues the upstream user JWT. `svc-auth` validates it, enriches it with Stoody profile data, maps Stoody roles to ExamPen roles, and stores ExamPen-side revocation state. | Stoody remains the source of truth for primary identity. `svc-auth` does not become the primary issuer of end-user session tokens. |
| Leaderboard + percentiles | `svc-analytics` | Student/teacher BFFs (read) | Materialized view recomputed on `score.updated` events. Recomputation is idempotent. | `svc-analytics` is the ONLY writer of percentile data. No other service computes percentiles. |
| Hub invigilator codes | Backend `svc-auth` (generates) | Hub caches in `invig_codes` table (read-only cache) | Codes generated server-side, pushed to hub during provisioning and daily sync. Hub NEVER generates codes. | If hub is offline, it uses cached codes. Codes have `valid_until` — expired codes rejected even from cache. |

---

## 2. Read/Write Boundary Declarations

### 2.1 Functions That MUST Be Read-Only

These functions must NEVER mutate durable state. If any of these mutate, it is a doctrine violation.

| Service | Function | Contract |
|---|---|---|
| `hub-pen-sync` | `read_pen_buffer()` | Reads GATT characteristic. Does NOT write to local storage. Returns bytes to caller. |
| `hub-store` | `get_pen_data(exam_id, pen_mac)` | Reads from SD/USB. Returns bytes. No side effects. |
| `hub-uplink` | `check_wifi_status()` | Queries NetworkManager. Returns status struct. No mutations. |
| `hub-uplink` | `check_backend_reachable()` | HTTP HEAD to backend health endpoint. Returns bool. No mutations. |
| `svc-stroke-ingest` | `validate_stroke_packet()` | Schema validation. Returns valid/invalid. Does NOT write to any store. |
| `svc-score-engine` | `get_current_score(exam_id, student_id)` | Reads materialized view. Returns score struct. No mutations. |
| `svc-analytics` | `get_percentile(exam_id, student_id)` | Reads materialized view. Returns percentile. No mutations. |
| `svc-review` | `get_objection(objection_id)` | Reads objection record. Returns struct. No mutations. |
| Teacher BFF | All query endpoints | BFF is a read-aggregator. It NEVER writes directly to any data store. All mutations go through service APIs. |
| Student BFF | All query endpoints | Same as teacher BFF. Read-only aggregator. |

### 2.2 Functions That ARE Allowed to Write

| Service | Function | What It Writes | Transactional Boundary |
|---|---|---|---|
| `hub-store` | `write_pen_chunk(exam_id, pen_mac, chunk)` | SD file + USB file | Dual fsync. Caller blocked until both succeed (or USB degraded). |
| `hub-store` | `clear_pen_buffer_command(pen_mac)` | BLE GATT write to pen | Only called AFTER `write_pen_chunk` confirms dual-write for ALL chunks + checksum match. |
| `svc-stroke-ingest` | `publish_strokes(exam_id, pen_mac, data)` | NATS JetStream | Publish is the write. No DB write here. |
| `svc-stroke-proc` | `commit_processed_strokes(exam_id, pen_mac, strokes)` | TimescaleDB | Single transaction per pen per exam. All-or-nothing. |
| `svc-score-engine` | `append_score_event(event)` | PostgreSQL (event store) + materialized view update | Event append + view update in same DB transaction. |
| `svc-score-engine` | `apply_override(teacher_id, score_id, new_value, reason)` | PostgreSQL (event store) | Validates teacher RBAC, appends override event. |
| `svc-review` | `transition_objection(objection_id, new_state, actor_id)` | PostgreSQL | State machine validation + write in single transaction. |
| `svc-plagiarism` | `publish_flags(exam_id, flags)` | PostgreSQL + NATS | Bulk insert flags, then publish event. |
| `svc-plagiarism` | `record_teacher_verdict(flag_id, teacher_id, verdict, reason)` | PostgreSQL | Validates teacher RBAC, writes verdict fields on the plagiarism flag row, then publishes updated projection/event as needed. |
| `svc-chat` | `append_message(sender, recipient, exam_id, content)` | PostgreSQL | Single insert. No update, no delete. |

### 2.3 DANGEROUS Functions — Side-Effectful Reads (Doctrine §4 Rule 2 Violations to Watch)

These patterns are the most common source of bugs. Flag any implementation that does these:

| Anti-Pattern | Why It's Dangerous | Correct Split |
|---|---|---|
| `hub-pen-sync.sync_pen()` that reads GATT AND writes to storage AND sends ACK to pen | Three side effects in one call. If storage write fails after GATT read, data is read but not persisted. If ACK sent before storage confirmed, pen clears buffer prematurely. | `read_chunk()` → `store.write_chunk()` → only then `pen.ack_chunk()` |
| `svc-stroke-ingest.ingest()` that validates AND publishes AND logs | If publish fails after log, log says success but data is lost. | `validate()` → `publish()` → only then `log_success()` |
| `svc-score-engine.get_or_compute_score()` that reads score and creates AI draft if missing | Read function creates state. Concurrent calls create duplicate drafts. | `get_score()` returns NULL if missing. Separate `create_ai_draft()` called explicitly by pipeline. |

---

## 3. Transactional Boundaries

### 3.1 Hub-Side Transactions

| Boundary | What Must Be Atomic | Mechanism |
|---|---|---|
| Pen chunk receive | GATT read → SD write → USB write → pen ACK | Sequential with fsync gates. No ACK until both writes confirmed. |
| Timer expiry → dongle activation | Timer fires → all 5 dongles activated → scan starts | Supervisor orchestrates. If any dongle fails to activate, log error + continue with remaining dongles. |
| Hub FSM transition | State change in SQLite → side effect execution | SQLite write first. If side effect fails, state is already updated — side effect retried on next supervisor tick. |
| Upload chunk | Read chunk from SD → HTTP POST → receive ACK → update ledger | Ledger updated ONLY after backend ACK. If app crashes between POST and ledger update, chunk re-sent (idempotent). |

### 3.2 Server-Side Transactions

| Boundary | What Must Be Atomic | Mechanism |
|---|---|---|
| Stroke ingestion | Validate → NATS publish | NATS JetStream acknowledged publish. If publish fails, HTTP 503 returned to hub, hub retries. |
| Stroke processing | Dedup → normalize → TimescaleDB commit | Single PostgreSQL transaction. Dedup checked within transaction (SELECT FOR UPDATE on idempotency key). |
| Score event | Append event → update materialized view | Same PostgreSQL transaction. View update is a triggered function within the transaction. |
| Score override | Validate RBAC → append override event → update view → publish NATS event | PG transaction for first three. NATS publish after commit. If NATS fails, event is in PG — async retry picks it up. |
| Objection resolution | Transition objection state → emit re-score command | PG write first. NATS publish after commit. If NATS fails, background worker polls for unprocessed resolutions. |
| Plagiarism flag generation | Compute all flags → bulk insert → publish NATS | PG bulk insert in single transaction. NATS after commit. |

### 3.3 Cross-System Transactional Boundaries

| Boundary | Participants | Strategy |
|---|---|---|
| Pen data custody: pen → hub | Pen firmware, hub BLE, hub storage | Pen does NOT clear buffer until hub sends explicit `0x03` clear command. Hub sends clear ONLY after dual-write confirmed. This is a two-phase commit with the pen as the transaction coordinator. |
| Pen data custody: hub → server | Hub uplink, backend ingest | Hub marks pen as "uploaded" ONLY after backend ACK per chunk. Backend idempotency key prevents duplicates. Hub retries indefinitely until all chunks ACKd. |
| Score publication | Score engine, analytics, notification | Score engine publishes `score.updated` event. Analytics and notification are eventually consistent consumers. No two-phase commit needed — scores are the source of truth. |

---

## 4. Ownership Violation Detection Rules

Codify these as linter rules or code review gates:

1. **No service writes to another service's database.** Each PostgreSQL schema is service-scoped. Cross-service access is via API or event only.
2. **BFF services have zero write access to any database.** All mutations via service APIs.
3. **Hub modules write to local storage ONLY via `hub-store`.** No module writes files directly.
4. **Score modifications ONLY via `svc-score-engine.append_score_event()`.** No direct SQL UPDATE on score tables from any other service.
5. **Objection state transitions ONLY via `svc-review.transition_objection()`.** No direct SQL UPDATE from BFF or score engine.
6. **Chat messages ONLY via `svc-chat.append_message()`.** Append-only, no UPDATE, no DELETE at the database level.

---

## 5. Ownership Map Diagram Key

For code review and CI enforcement, every service's README must declare:

```markdown
## Ownership Declaration
- **Writes:** [list of state this service owns]
- **Reads from:** [list of services/stores this service reads]
- **Never writes to:** [explicit exclusions]
- **Transactional boundaries:** [list]
```

Any PR that adds a write path not declared in the ownership declaration must be flagged for architecture review.

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Aligned auth ownership to Stoody-issued JWT validation, clarified provisional hub bindings, and made `svc-plagiarism` the single writer for verdict state. | Codex |
