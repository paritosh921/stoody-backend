# FAILURE_MITIGATION_REGISTER.md
# ExamPen — Failure Mode Mitigation Register

Reference: R4-EXAMPEN-DEVSTACK
Doctrine source: SYSTEM_DESIGN_GUIDELINE.md §7

---

## Purpose

The design document (R1/R2/R3) lists problems. This register assigns explicit mitigations, owners, residual risks, and detection mechanisms for each.

**Every entry must have:** mitigation (what prevents it), detection (how you know it happened), owner (who is responsible), residual risk (what remains after mitigation).

---

## 1. BLE & Hub Layer Failures

| ID | Failure Mode | Mitigation | Detection | Owner | Residual Risk |
|---|---|---|---|---|---|
| A1.1 | BLE connection limit exceeded (>8 per dongle) | 5 dongles × 8 = 40 capacity. `hub-ble-mgr` enforces hard limit per dongle, redirects overflow to next available dongle. | `hub-ble-mgr` logs rejection if all slots full. TUI shows "capacity full" warning. | `hub-ble-mgr` | >40 pens in a room requires second hub. No automatic multi-hub coordination. |
| A1.5 | Pen battery death mid-exam | Pen firmware: low-battery warning at 15% via BLE advertisement flag (visible during registration). Stroke data in flash survives power loss. Post-exam: if pen won't power on, data is lost. | Registration scan captures battery level. Missing pen in sync phase flagged as timeout. | Pen firmware + `hub-pen-sync` | If pen dies AND flash is corrupted, data is irrecoverable. Mitigation: copy image upload path. |
| A1.7 | BLE disconnect during post-exam sync | `hub-pen-sync` implements per-chunk checkpointing. If connection drops, reconnect and resume from last confirmed chunk index. 3 retry attempts per pen, 30s timeout each. | `pen_sync_status.status = 'failed'` after 3 retries. Interaction log records each disconnect. | `hub-pen-sync` | If pen is physically moved out of range, retries will fail. Invigilator alerted to bring pen closer. |
| H1 | USB bus power brownout (5 dongles) | Powered USB hub (7-port, 5V/3A+) mandatory in BOM. Hub software checks dongle count on boot — warns if <5. | `hub-ble-mgr` health check: if dongle responds to `hciconfig` but fails scan, flag as power issue. | `hub-supervisor` | Cheap USB hubs may still brownout under simultaneous scan + connect. Specify minimum hub quality in procurement. |
| H3 | Dongle failure mid-sync | `hub-ble-mgr` detects dongle drop (D-Bus disconnect event). Pens from failed dongle are re-queued. If another dongle has capacity, reassign. If not, mark as timeout. | D-Bus signal `org.bluez.Adapter1.Removed`. TUI shows dongle status change. | `hub-ble-mgr` | Reassigned pens must re-advertise and re-connect — adds 10-30s delay. Some pens may miss the window. |
| H5 | BLE advertising collision (40 pens simultaneously) | Stagger dongle scan activation: 500ms delay between dongles. Use passive scan first (lower interference), switch to active scan for connection. | If discovery rate is <1 pen/sec after 60s, log warning suggesting RF congestion. | `hub-ble-mgr` | Dense RF environment (school with many APs) may still cause slow discovery. Mitigation: extend sync timeout to 5 min. |
| F1 | Timer drift (no RTC, no NTP) | Pre-exam WiFi verification includes NTP sync check. Timer uses `CLOCK_MONOTONIC` (immune to NTP adjustments during countdown). NTP sync only needed at timer start, not during. | `chronyc tracking` checked before timer start. If offset >2s, warn invigilator. | `hub-timer` | If hub boots without network and cached NTP offset is stale, timer may be off by minutes. Mitigation: invigilator verifies wall clock matches TUI timer. |
| F4 | Hub reboot during timer | Timer state persisted to SQLite every 10 seconds: `{start_epoch, duration_sec, remaining_sec}`. On boot, supervisor checks `active_timer` table, resumes countdown. | Interaction log: `hub_boot` event with `{timer_recovered: true, lost_seconds: N}`. | `hub-timer` + `hub-supervisor` | Up to 10s of timer accuracy lost (between last persist and crash). Acceptable for exam timing. |
| S3 | BLE MITM on invigilator channel | Rotating 24h auth codes (not static pairing). Command characteristic requires auth before accepting commands. BLE 4.2 LE Secure Connections (LESC) with numeric comparison if hardware supports. | Auth failure logged. Multiple failed auth attempts → lockout for 5 min. | `hub-invig-ble` | BLE LESC requires both devices to support it. Fallback to Just Works pairing is vulnerable. Accept risk for V1; add app-level challenge-response in V2. |
| S4 | SD card failure (primary storage) | Dual-write to SD + USB. If SD write fails, hub degrades to USB-only mode. Hub TUI shows critical warning. | `fsync()` failure on SD path triggers immediate alert. | `hub-store` | If both SD and USB fail simultaneously (power surge), all local data lost. Extremely unlikely with independent media. Copy image upload is the ultimate fallback. |

## 2. Data Pipeline Failures

| ID | Failure Mode | Mitigation | Detection | Owner | Residual Risk |
|---|---|---|---|---|---|
| U1 | BLE relay too slow (17 min for 40 pens) | WiFi is always primary. BLE relay is last-resort. Mobile app shows estimated time before confirming upload path. | Upload progress timer displayed on mobile app and hub TUI. If BLE path selected, clear "~12 minutes estimated" warning. | `hub-uplink` | If WiFi is truly unavailable AND school is in a hurry, 12-min BLE relay is the hard floor. No mitigation other than reducing data size (compression on pen). |
| U4 | Partial upload resume failure | Per-pen upload ledger tracks `acked_chunks[]`. Resume sends only missing indices. Backend rejects duplicates (idempotency key: `{exam_id, pen_mac, chunk_index}`). | Ledger mismatch between hub and backend detected by reconciliation endpoint: `GET /api/v1/exams/{exam_id}/upload-status`. | `hub-uplink` + `svc-stroke-ingest` | If SQLite ledger corrupted on hub, worst case is re-upload of all chunks (wasteful but not lossy — backend deduplicates). |
| A8.4 | Peak load: 10K students simultaneously | Stroke ingestion is stateless (scales horizontally). NATS JetStream buffers bursts. TimescaleDB handles high write throughput. Rate limiting at API gateway (Traefik) per hub. | NATS consumer lag metric. If lag >10s, alert. Grafana dashboard for ingestion throughput. | `svc-stroke-ingest` + infra | At 10K students × 336 KB = 3.3 GB burst. NATS must be sized for this. If under-provisioned, backpressure slows hub uploads (acceptable — hub retries). |
| A8.6 | Duplicate processing | Idempotency key at every pipeline stage. Stroke ingest: `{exam_id, pen_mac, chunk_index}`. Stroke proc: dedup by idempotency key before commit. Score engine: event-sourced (duplicate events detected by sequence number). | Duplicate detection logged. Grafana panel for duplicate rate. If >1%, investigate source. | All pipeline services | Idempotency adds a small overhead per write (key lookup). Acceptable. |

## 3. AI & Scoring Failures

| ID | Failure Mode | Mitigation | Detection | Owner | Residual Risk |
|---|---|---|---|---|---|
| A4.6 | AI misrecognition (reads "5" as "3") | Confidence score per character. Below-threshold characters flagged in teacher review UI with amber highlight. Teacher must confirm/override below-threshold answers. | AI confidence < 0.85 per character → flag. Aggregate: if >30% of answer is below-threshold, entire answer flagged. | `svc-ai-pipeline` + teacher UI | Even with flagging, teacher may rubber-stamp AI output. Mitigation: double-blind evaluation option for high-stakes exams. |
| A5.5 | Rubric change after partial scoring | Rubric versioning: every rubric edit creates new version. Scores computed against rubric version at time of scoring. If rubric updated, system offers "re-score affected papers with new rubric" — explicit action, not automatic. | Score records include `rubric_version`. Dashboard shows if any scores were computed against old rubric. | `svc-score-engine` | Re-scoring is expensive (re-runs AI interpretation against new rubric). May delay results. |
| PL5 | Plagiarism false positive | High threshold: composite >0.75 for "review", >0.90 for "strong match". Question-type adjustment: MCQ/objective only use temporal + proximity signals, not text similarity. NEVER auto-penalize — teacher review required. | False positive rate tracked: dismissed flags / total flags. If >50% dismissal rate, model weights need recalibration. | `svc-plagiarism` | Even with high threshold, some false positives will occur for open-ended questions with common phrasing. Teacher review is the ultimate filter. |
| Q1 | Question miss ambiguity (3-way) | Three distinct indicator states with different colors and labels. Teacher override workflow with mandatory reason. Copy image viewer integrated into miss review screen. | Miss indicator auto-state computed. If auto-state is wrong, teacher override logged. Override rate tracked per exam type — high rate indicates poor question region definitions. | `svc-doc-assembly` + `svc-score-engine` | If question regions are poorly defined (A2.2), miss detection will be unreliable. Invest in question region editor UX. |

## 4. Infrastructure Failures

| ID | Failure Mode | Mitigation | Detection | Owner | Residual Risk |
|---|---|---|---|---|---|
| A8.1 | Multi-tenant data leak | PostgreSQL Row-Level Security (RLS) with `tenant_id` on every table. Application-level middleware injects `SET app.current_tenant = '{tenant_id}'` per request. Integration test: attempt cross-tenant access → must fail. | RLS policy violation logged by PostgreSQL. Penetration test includes cross-tenant access attempts. | `svc-auth` + DB schema | RLS is only as good as the policy definitions. Missing RLS on a new table = leak. CI check: every new migration must include RLS policy or explicit exemption comment. |
| A8.2 | DPDPA violation (children's data) | Data minimization: collect only what's needed. Consent: parent consent recorded during student registration in Stoody. Retention: auto-delete stroke data after configurable period (default 2 years). Encryption: data at rest (PostgreSQL TDE) + in transit (TLS everywhere). | Annual compliance audit checklist. Data retention cron job logs deletions. | Compliance officer + `svc-auth` | DPDPA is new and evolving. Legal review needed for specific data categories (biometric? stroke data may be considered behavioral biometric). |
| A8.5 | Backup failure → data loss | PostgreSQL: pgBackRest continuous WAL archival to S3 + daily full backup. MinIO: cross-region replication. SQLite (hub): dual-write + periodic backup to USB. RPO: <1 hour for server, <10 seconds for hub (dual-write). | Backup monitoring: pgBackRest reports success/failure. S3 replication lag metric. Hub USB health check. | Infra + `hub-store` | If both S3 regions fail simultaneously (AWS-level outage), backup is unavailable. Extremely unlikely. |
| A8.8 | Cost exceeds ₹2000/student/year | Self-hosted ONNX for AI (CapEx, not OpEx). MinIO instead of cloud S3. PostgreSQL instead of managed DB. Estimated cost model: server (₹100/student/year) + hub hardware amortized (₹50/student/year) + pens amortized (₹150/student/year) + operational (₹100/student/year) = ~₹400/student/year. | Quarterly cost review per institute. Alert if cost/student exceeds threshold. | Business + infra | Pen hardware cost is the wildcard. If pens cost >₹2000 each with <2 year lifespan, unit economics break for low-volume institutes. |

## 5. Unmitigated Risks (Acknowledged, Not Solved)

| ID | Risk | Why Unmitigated | Impact | Planned Resolution |
|---|---|---|---|---|
| UR1 | Pen hardware vendor discontinues chip | Single-vendor BLE pen MCU. No second source qualified. | Hub firmware + GATT protocol locked to one pen design. | Qualify second pen vendor before V2 launch. Abstract GATT protocol to support multiple pen profiles. |
| UR2 | Devanagari HWR accuracy below 90% | Training data for handwritten Hindi is scarce. Cold start problem. | Scores for Hindi-medium exams unreliable. Teacher override burden high. | Collect labeled data from pilot schools. Fine-tune model iteratively. Accept that V1 Hindi support is "assisted" not "automated." |
| UR3 | Multi-hub coordination for >40 students | Current design: 1 hub per room, max 40 pens. No hub-to-hub communication. | Rooms with 60+ students need 2 hubs. Invigilator must manage 2 mobile app connections. | V2: hub mesh networking or server-coordinated multi-hub sessions. |
| UR4 | Real-time collaborative exam editing (future) | Not in scope. Exam creation is single-user. | If two tutors edit the same exam simultaneously, last-write-wins. | Add optimistic locking or CRDT-based editing if demand arises. Not V1. |
