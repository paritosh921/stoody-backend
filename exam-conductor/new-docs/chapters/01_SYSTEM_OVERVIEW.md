# Chapter 01: System Overview

## Status
- **Phase:** W6 — Documentation
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6.A6.1)
- **Build status:** DRAFT

## Overview

ExamPen is a pen-based exam management subsystem of the Stoody education platform. Physical BLE pens capture handwritten answers on dot-matrix paper, a Raspberry Pi hub collects and uploads stroke data, and a cloud pipeline processes strokes through AI recognition and automated scoring. ExamPen plugs into Stoody via API-driven native embed (Option B, frozen decision).

## Architecture Context

```
                    +-----------+
                    |  Stoody   |  (identity, roster, gradebook)
                    |  Platform |
                    +-----+-----+
                          | JWKS, REST, Webhooks
                          v
+------+  BLE   +-----+  WiFi/   +-------------------+   NATS    +------------------+
| Pens |------->| Hub |--------->| Cloud Services    |---------->| AI + Scoring     |
| x40  |  GATT  | RPi |  HTTP    | (Ingest, Process) | JetStream | (Pipeline, Score)|
+------+        +--+--+          +---------+---------+           +--------+---------+
                   |                       |                              |
                   | BLE                   | REST                        | Events
                   v                       v                              v
            +-----------+       +-------------------+          +------------------+
            | Invig App |       | Teacher Dashboard |          | Student Portal   |
            | (Flutter) |       | + Teacher BFF     |          | + Student BFF    |
            +-----------+       +-------------------+          +------------------+
```

## Service Catalog

| Service | Owner Layer | Responsibility |
|---|---|---|
| `svc-auth` | Auth | Stoody JWT validation, role mapping, RLS, revocation |
| `svc-exam-orch` | Orchestration | Exam lifecycle FSM, pen binding, scheduling |
| `svc-stroke-ingest` | Ingestion | Chunk upload validation, NATS publish |
| `svc-stroke-proc` | Processing | Dedup, normalize, TimescaleDB commit |
| `svc-doc-assembly` | Assembly | Stroke-to-page rendering, miss indicators |
| `svc-ai-pipeline` | AI | HWR/OCR, step detection, diagram classification |
| `svc-score-engine` | Scoring | Event-sourced scoring, rubric eval, overrides |
| `svc-review` | Review | Objection lifecycle FSM |
| `svc-analytics` | Analytics | Percentiles, leaderboards, class stats |
| `svc-plagiarism` | Integrity | TF-IDF + structural similarity, teacher verdicts |
| `svc-chat` | Communication | Append-only messaging |
| `svc-notify` | Notification | Email, push, SMS triggers |
| `svc-copy-upload` | Fallback | Photo-based answer capture |
| `svc-teacher-bff` | Aggregation | Read-only aggregator for teacher UI |
| `svc-student-bff` | Aggregation | Read-only aggregator for student UI |
| `svc-invig-console` | Real-time | WebSocket invigilator dashboard |

### Hub Modules

| Module | Responsibility |
|---|---|
| `hub-supervisor` | Process manager, FSM orchestration, child process lifecycle |
| `hub-ble-mgr` | BLE dongle management (5 dongles x 8 pens = 40 max) |
| `hub-pen-sync` | GATT read, chunk transfer from pens |
| `hub-timer` | Exam countdown, CLOCK_MONOTONIC, reboot recovery |
| `hub-store` | Dual-write (SD + USB), fsync protocol |
| `hub-uplink` | WiFi/mobile upload, resume ledger |
| `hub-invig-ble` | Invigilator mobile BLE relay |
| `hub-tui` | Textual-based TUI (8 screens) |

## Data Flow: End-to-End

```
Pen (BLE GATT)
  |  stroke capture on dot-matrix paper
  v
Hub (RPi)
  |  hub-pen-sync reads chunks via GATT
  |  hub-store dual-writes: SD fsync -> USB fsync -> ACK pen
  v
Hub-Uplink (WiFi or mobile BLE relay)
  |  chunked HTTP POST, per-chunk backend ACK
  |  idempotency key: {exam_id, pen_mac, chunk_index}
  v
svc-stroke-ingest
  |  schema validation -> NATS JetStream publish (stroke.raw)
  v
svc-stroke-proc
  |  dedup by idempotency key -> normalize coordinates -> TimescaleDB commit
  |  publishes stroke.processed
  v
svc-doc-assembly
  |  stroke-to-page rendering (SVG) -> MinIO (S3)
  |  miss indicator auto-detection
  |  publishes page.ready
  v
svc-ai-pipeline
  |  HWR/OCR (ONNX) -> step detection -> diagram classification
  |  publishes ai.result
  v
svc-score-engine
  |  rubric evaluation -> event-sourced score creation (ai_draft)
  |  teacher review -> override -> finalize -> publish -> lock
  |  publishes score.updated
  v
svc-teacher-bff / svc-student-bff
  |  read-only aggregation for dashboards
  v
Teacher Dashboard / Student Portal / Mobile Apps
```

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Backend services | Python 3.12, FastAPI | Per-service schemas, RLS |
| Database (relational) | PostgreSQL | Per-service schemas, Row-Level Security |
| Database (time-series) | TimescaleDB | Stroke storage |
| Event bus | NATS JetStream | Async event pipeline |
| Object store | MinIO (S3-compatible) | Page images, copy images |
| Hub OS | Ubuntu Server 24.04 LTS arm64 | Golden image for RPi 4B/5 |
| Hub runtime | Python 3.12, SQLite (WAL), BlueZ 5.72+ | Textual TUI, systemd |
| Web frontends | TypeScript, React | teacher-dashboard, student-portal, invigilator-console |
| Mobile | Flutter | Single app, dual-mode (hub-control + teacher/student view) |
| AI inference | ONNX Runtime | Self-hosted HWR/OCR, step detection, diagram classification |
| Infrastructure | Docker Compose, Traefik | Grafana + Loki + Tempo + Prometheus |

## Key Architecture Rules

1. **Single writable owner** per critical state (`STATE_OWNERSHIP_MAP.md`).
2. **Read operations must not mutate.** No side-effectful reads.
3. **Contract-first development.** OpenAPI, NATS JSON Schema, GATT spec defined before implementation.
4. **BFF services are read-only aggregators.** Zero write access to any database.
5. **Domain layer is pure.** No I/O imports in `domain/` directories.
6. **One service = one directory = one Dockerfile = one Docker Compose service.**

## Interfaces

- REST contracts: `api/*.openapi.yaml` (12 services)
- Event contracts: `contracts/events/*.schema.json` (10 event types)
- BLE GATT spec: `hub/ble-gatt-spec.md`
- Hub IPC protocol: `hub/ipc-protocol.md`

## Testing

- Validation evidence hierarchy: L1 (build) through L7 (field trial)
- All tests have explicit IDs (e.g., `U-SCR-01`, `I-ORCH-02`, `E2E-08`)
- Full specification: `TEST_SUITE_SPEC.md`

## Failure Modes & Mitigations

Key failure modes are cataloged in `FAILURE_MITIGATION_REGISTER.md` with IDs:
- BLE layer: A1.1, A1.5, A1.7, H1, H3, H5
- Data pipeline: U1, U4, A8.4, A8.6
- AI & scoring: A4.6, A5.5, PL5, Q1
- Infrastructure: A8.1, A8.2, A8.5, A8.8

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Initial draft: architecture diagram, service catalog, data flow, tech stack | Claude Agent (W6.A6.1) |
