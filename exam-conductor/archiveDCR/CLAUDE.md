# ExamPen (exam-conductor) — AI Agent Guide

## Overview

ExamPen is a subsystem of the **Stoody education platform** that adds pen-based exam management. Physical BLE pens capture handwritten answers on dot-matrix paper, a Raspberry Pi hub collects and uploads stroke data, and a cloud pipeline processes strokes through AI recognition and automated scoring. Teachers review AI-generated scores, students view results and file objections.

ExamPen does NOT replace Stoody. It plugs into Stoody's existing tutor and student portals via **API-driven native embed** (Option B, frozen decision). Stoody's frontend calls ExamPen BFF APIs directly and renders data in Stoody's own UI components.

### Relationship to Existing Codebase

This project draws from the existing `backend/`, `frontend/`, and `stoody-ble-agent/` codebases:
- **Pen protocol**: Reuses the P05 pen BLE GATT protocol, frame format, CRC-16/XMODEM, and coordinate processing from `stoody-ble-agent/`
- **Stroke pipeline**: Inspired by the current Stoody pen-to-canvas-to-DB pipeline (see `new-docs/PEN_TO_CANVAS_TO_DB_REFERENCE.md`)
- **Auth model**: Stoody issues JWTs; ExamPen validates them via JWKS (no separate session tokens)
- **Data mapping**: Student/tutor identities, class structure, and subjects come from Stoody (read-only)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend services | Python 3.12, FastAPI, PostgreSQL (per-service schemas, RLS), TimescaleDB (strokes), NATS JetStream (events), MinIO (S3-compatible object store) |
| Hub (RPi) | Python 3.12, SQLite (WAL mode), BlueZ 5.72+, Textual TUI, systemd, Ubuntu Server 24.04 LTS arm64 |
| Web frontends | TypeScript, React (teacher-dashboard, student-portal, invigilator-console) |
| Mobile | Flutter (single app, dual-mode: hub-control + teacher/student view) |
| AI | ONNX Runtime (self-hosted HWR/OCR, step detection, diagram classification) |
| Infrastructure | Docker Compose, Traefik, Grafana+Loki+Tempo+Prometheus |

## Repository Structure

```
exampen/
├── docs/                           # Living documentation (promoted from new-docs/)
│   ├── api/                        # OpenAPI 3.1 specs per service (12 services)
│   ├── contracts/events/           # JSON Schema per NATS event (10 event types)
│   ├── hub/                        # BLE GATT spec, IPC protocol
│   └── chapters/                   # Documentation chapters + BUILD_STATUS.md
├── libs/                           # Shared libraries
│   ├── exampen-proto/              # Protobuf/JSON Schema definitions
│   ├── exampen-common-py/          # Python: auth, nats_client, db, logging
│   └── exampen-common-ts/          # TypeScript: auth, api-client, types
├── services/                       # Backend microservices (each has own Dockerfile)
│   ├── svc-auth/                   # Stoody JWT validation, role mapping, RLS
│   ├── svc-exam-orch/              # Exam lifecycle FSM, pen binding, scheduling
│   ├── svc-stroke-ingest/          # Chunk upload ingestion, NATS publish
│   ├── svc-stroke-proc/            # Dedup, normalize, TimescaleDB commit
│   ├── svc-doc-assembly/           # Stroke-to-page rendering, miss indicators
│   ├── svc-ai-pipeline/            # HWR, step detection, diagram classification
│   ├── svc-score-engine/           # Event-sourced scoring, rubric eval, overrides
│   ├── svc-review/                 # Objection lifecycle FSM
│   ├── svc-analytics/              # Percentiles, leaderboards, class stats
│   ├── svc-plagiarism/             # TF-IDF + structural similarity, teacher verdicts
│   ├── svc-chat/                   # Append-only messaging
│   ├── svc-notify/                 # Email, push, SMS triggers
│   ├── svc-copy-upload/            # Fallback photo-based answer capture
│   ├── svc-teacher-bff/            # Read-only aggregator for teacher UI
│   ├── svc-student-bff/            # Read-only aggregator for student UI
│   └── svc-invig-console/          # Real-time WebSocket invigilator dashboard
├── hub/                            # RPi hub software
│   ├── hub-supervisor/             # Process manager, FSM orchestration
│   ├── hub-ble-mgr/                # BLE dongle management (5 dongles x 8 pens)
│   ├── hub-pen-sync/               # GATT read, chunk transfer
│   ├── hub-timer/                  # Exam countdown, reboot recovery
│   ├── hub-store/                  # Dual-write (SD+USB), fsync protocol
│   ├── hub-uplink/                 # WiFi/mobile upload, resume ledger
│   ├── hub-invig-ble/              # Invigilator mobile BLE relay
│   ├── hub-tui/                    # Textual-based TUI
│   └── hub-common/                 # IPC definitions, shared config
├── frontend/
│   ├── teacher-dashboard/
│   ├── student-portal/
│   └── invigilator-console/
├── mobile/
│   └── exampen-mobile/             # Flutter, dual-mode
├── infra/                          # Docker Compose, Traefik, monitoring
├── test-suite/                     # Integration, E2E, hub hardware tests
└── scripts/                        # Dev setup, seed data, mock generation
```

## Documentation System

**All design documentation is in `new-docs/` (will be promoted to `docs/`).**

### Reading Order
1. **Always start** with `new-docs/agent_ref_index.md` — routes you to the right docs for your task
2. `new-docs/DOCUMENT_REGISTRY.md` — authority map and conflict resolution
3. Task-specific docs per the routing table in agent_ref_index.md

### Document Authority
- **AUTHORITATIVE** docs override supplementary docs on conflicts
- Concrete contracts in `api/`, `contracts/events/`, `hub/` override prose descriptions
- `STATE_OWNERSHIP_MAP.md` is the final word on who writes what
- `COMPONENT_INDEPENDENCE_MAP.md` defines build order and dependencies

### Key Authoritative Specs
| Topic | Document |
|-------|----------|
| State ownership | `STATE_OWNERSHIP_MAP.md` |
| Component deps / build order | `COMPONENT_INDEPENDENCE_MAP.md` |
| Stoody integration | `STOODY_INTEGRATION_SPEC.md` |
| Hub deployment | `HUB_DEPLOYMENT_SPEC.md` |
| BLE protocol | `hub/ble-gatt-spec.md` |
| Hub IPC | `hub/ipc-protocol.md` |
| Tests | `TEST_SUITE_SPEC.md` |
| Failure modes | `FAILURE_MITIGATION_REGISTER.md` |
| REST contracts | `api/{service}.openapi.yaml` (12 files) |
| Event contracts | `contracts/events/{event}.schema.json` (10 files) |

## Architecture Rules

### Core Doctrine (from SOFTWARE_DEVELOPMENT_DOCTRINE.md)

1. **Single writable owner** per critical state. All other layers read, derive, or cache. Violations are design bugs.
2. **Read operations must not mutate.** No side-effectful reads.
3. **Transaction boundaries** for every racing event pair. Define what must be atomic.
4. **Normalize at ingress.** Validation and transformation happen at the entry point.
5. **No hidden derived owners.** If a cache can be written to, it must be declared in `STATE_OWNERSHIP_MAP.md`.

### Service Architecture

- **One service = one directory = one Dockerfile = one Docker Compose service**
- Shared code goes in `libs/` — never import from another service's `src/`
- No circular imports between services
- Database migrations owned by the service that owns the schema
- Tests live next to the code: `{service}/tests/`

### Per-Service Layer Rules

```
routes/    → domain/, storage/, adapters/, libs/     (HTTP I/O)
events/    → domain/, storage/, adapters/, libs/     (NATS I/O)
domain/    → libs/exampen-proto ONLY                  (ZERO I/O — pure logic)
storage/   → libs/                                    (DB I/O)
adapters/  → libs/                                    (external HTTP, NATS I/O)
```

**The `domain/` layer must NEVER import asyncio, aiohttp, sqlalchemy, nats, or any I/O library.**

### BFF Services
- `svc-teacher-bff` and `svc-student-bff` are **read-only aggregators**
- Zero write access to any database
- All mutations go through backing service APIs

### Contract-First Development
- Every cross-component interaction defined by a contract BEFORE implementation
- Agents build against contracts (OpenAPI, NATS schema, GATT spec), not source code
- Mock mode per service: `docker compose up svc-{name} --env MOCK_MODE=true`

## Key State Ownership

| State | Writable Owner | Key Constraint |
|-------|---------------|----------------|
| Exam session FSM | `svc-exam-orch` | Row-level locking in PostgreSQL |
| Hub FSM | `hub-supervisor` | SQLite `exam_sessions.state` persisted BEFORE side effects |
| Pen-student binding | `svc-exam-orch` (authoritative) | Hub holds provisional only; scoring uses server-confirmed |
| Raw pen strokes | Pen firmware flash | Irreplaceable — pen clears only after hub confirms dual-write |
| Hub stroke storage | `hub-store` | Dual-write: SD fsync → USB fsync → ACK pen |
| Server strokes | `svc-stroke-proc` (TimescaleDB) | Idempotency: `{exam_id, pen_mac, chunk_index}` |
| Scores | `svc-score-engine` | Event-sourced, append-only. FSM: `ai_draft→teacher_reviewed→finalized→locked` |
| Objections | `svc-review` | FSM: `filed→assigned→reviewing→resolved` |
| Plagiarism flags + verdicts | `svc-plagiarism` | Single writer for both detection and teacher verdict |
| Chat messages | `svc-chat` | Append-only. No UPDATE, no DELETE. |
| Auth claims + role mapping | `svc-auth` | Stoody is source of truth for identity; ExamPen maps roles |

## Build Phases & Dependencies

All components are currently `NOT_STARTED` (see `chapters/BUILD_STATUS.md`).

### Parallel Build Matrix
```
Agent 1: libs/proto → svc-auth (Stoody stub) → svc-exam-orch → svc-teacher-bff → teacher-dashboard
Agent 2: hub-store → hub-timer → hub-ble-mgr → hub-pen-sync → hub-supervisor integration
Agent 3: libs/common → svc-stroke-ingest → svc-stroke-proc → svc-doc-assembly
Agent 4: hub-tui → hub-invig-ble → hub-uplink → svc-invig-console → mobile (hub control)
Agent 5: svc-ai-pipeline → svc-plagiarism → svc-score-engine → svc-review → svc-student-bff
Agent 6: infra (Docker Compose) → monitoring → test-suite → CI/CD → svc-analytics → svc-chat → svc-notify
```

### Zero-Dependency (start immediately)
`libs/exampen-proto`, `libs/exampen-common-py`, `libs/exampen-common-ts`, `hub-timer`, `hub-store`, `hub-tui`

### Requires Stoody Stub
`svc-auth` — needs JWKS endpoint mock and user profile API mock

## Auth & Access Control

- **SSO**: Stoody issues JWT → ExamPen `svc-auth` validates via Stoody JWKS endpoint
- **User identity**: Stoody's `user_id` is primary key; ExamPen stores `stoody_user_id`
- **Roles**: Stoody provides base role (tutor/student/parent); ExamPen adds exam-specific roles (invigilator, evaluator, reviewer)
- **Multi-tenant**: PostgreSQL RLS with `tenant_id` on every table
- **Parent access**: Parent sees only linked children's data (resolved via Stoody API)
- **DPDPA compliance**: Data minimization, consent via Stoody, auto-delete after retention period

## Pen & BLE Protocol

- **Pen model**: P05, BLE GATT service `0000ae30-...`
- **Commands to AE10** (write-with-response), **responses on AE02** (notify)
- **Frame**: `Head(2) + SerialNum(4) + ID(4) + Cmd(1) + DataFormat(1) + DataLen(2) + Data(N) + CRC16(2)`
- **CRC-16/XMODEM**: polynomial `0x1021`, init `0x0000`, computed over SerialNum through Data
- **Coordinate frame**: 14 bytes — bookType, pageNo, X, Y, pressure, penProp, timestamp
- **Scale**: 10 pen units/mm, 4 canvas px/mm, Y-inverted (pen=bottom-left, canvas=top-left)
- **Hub capacity**: 5 USB BLE dongles x 8 pens = 40 pens max per hub

## Testing Strategy

### Validation Evidence Hierarchy
| Level | Meaning |
|-------|---------|
| L1 | Build verified (Docker image builds) |
| L2 | Typecheck/lint verified |
| L3 | Unit test verified (domain logic, no I/O) |
| L4 | Integration test verified (real DB/NATS/S3 in Docker) |
| L5 | E2E test verified (multi-service pipeline) |
| L6 | Hardware-in-loop verified (hub + BLE dongles + pen simulator) |
| L7 | Field trial verified (real exam) |

**Every PR must state which levels were achieved.** "Tests pass" without level is rejected.

### Test IDs
All tests have explicit IDs (e.g., `U-SCR-01`, `I-ORCH-02`, `E2E-08`). Reference by ID in PRs and docs.

## File Size & Complexity Limits

| Language | Max Lines/File | Exception |
|----------|---------------|-----------|
| Python | 300 | Single data model with 40+ fields (documented) |
| TypeScript/React | 250 | Single complex component with no extractable sub-components |
| SQL migrations | 200 | Initial schema creation only |
| Config (YAML/TOML) | 150 | Split into per-service configs |

| Metric | Max |
|--------|-----|
| Files per module/package | 15 |
| Functions per file | 12 |
| Parameters per function | 6 |
| Nesting depth | 3 levels |
| Cyclomatic complexity per function | 10 |

## Event-Driven Pipeline

Key NATS events (schemas in `contracts/events/`):
- `stroke.raw` → `stroke.processed` → `page.ready` → `ai.result` → `score.updated`
- `exam.lifecycle` (FSM transitions)
- `objection.*` (filed → resolved)
- `plagiarism.check` → `plagiarism.result`
- `copy.ready` (fallback photo path)

## Hub Operations

- **Golden image**: Pre-built `.img.xz` for RPi 4B/5, US WiFi regulatory domain locked
- **Dual-write**: SD fsync → USB fsync → ACK pen. If USB fails, degrade to SD-only with TUI warning
- **Timer**: Uses `CLOCK_MONOTONIC`, persists to SQLite every 10s, survives reboot
- **IPC**: Unix domain sockets, JSON-lines encoding between hub modules
- **TUI**: 8 screens — Setup, Status, WiFi, Dongles, Exams, Diagnostics, Logs, Shutdown

## Commands

```bash
# Local dev stack
docker compose -f infra/docker-compose.yml up

# Run service unit tests
pytest services/svc-score-engine/tests/ -m unit

# Run integration tests (requires Docker Compose test stack)
docker compose -f infra/docker-compose.test.yml up -d
pytest services/svc-score-engine/tests/ -m integration

# Run pipeline E2E tests
pytest test-suite/pipeline-tests/

# Seed test data
./scripts/seed-data.sh --students 40 --exams 3 --questions-per-exam 10

# Start service in mock mode
docker compose up svc-score-engine --env MOCK_MODE=true
```

## Integration with Existing Stoody

### ExamPen Consumes from Stoody
- `GET /api/students?class_id=&section_id=` — student roster
- `GET /api/tutors?subject_id=` — tutor list
- `GET /api/classes`, `GET /api/subjects` — reference data
- `GET /api/users/{user_id}` — profile enrichment
- `GET /.well-known/jwks.json` — JWT signing keys
- `GET /api/parents/{user_id}/children` — parent access scope

### ExamPen Pushes to Stoody
- `POST /api/webhooks/exampen/scores` — score publication
- `POST /api/webhooks/exampen/exams` — exam created/completed

## Unmitigated Risks (V1)

- **UR1**: Single pen hardware vendor — no second source qualified
- **UR2**: Devanagari HWR accuracy may be below 90% (training data scarcity)
- **UR3**: No multi-hub coordination for >40 students per room
- **UR4**: No collaborative exam editing (last-write-wins)
