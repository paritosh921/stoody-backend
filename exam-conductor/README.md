# ExamPen (exam-conductor)

ExamPen is a pen-based exam management subsystem of the Stoody education platform. It supports two operational modes:

## Modes

### DCR — Digital Copy Review

The current implementation. BLE smart pens capture handwritten answers on dot-matrix paper, a Raspberry Pi hub collects and uploads stroke data via WiFi, and a cloud pipeline processes strokes through AI recognition and automated scoring.

**Status**: Implemented (Waves 0-6 complete)

Key features:
- BLE pen data capture via RPi hub (5 dongles x 8 pens = 40 students)
- Stroke pipeline: pen → hub dual-write → server ingestion → processing → page assembly
- AI pipeline: HWR (English + Devanagari), step detection, diagram classification
- Event-sourced scoring with rubric evaluation and teacher overrides
- Plagiarism detection (TF-IDF + structural + temporal + proximity)
- Objection lifecycle with re-scoring
- Teacher dashboard, student portal, invigilator console (web + mobile)
- Multi-tenant with RLS, RBAC (7 roles), Stoody SSO integration

See [`DCR/CLAUDE.md`](DCR/CLAUDE.md) for architecture details and [`DCR/SETUP_GUIDE.md`](DCR/SETUP_GUIDE.md) for setup instructions.

### PCR — Physical Copy Review

Planned. Traditional paper-based exam workflow where physical answer sheets are photographed or scanned, then processed through the same AI scoring pipeline.

**Status**: Not yet implemented

See [`PCR/`](PCR/) for planning documents.

## Shared Documentation

Design specifications, contracts, and architecture docs shared across both modes:

- [`new-docs/`](new-docs/) — All authoritative design documents
  - `api/` — OpenAPI 3.1 specs for all services
  - `contracts/events/` — NATS event JSON schemas
  - `hub/` — BLE GATT spec, IPC protocol
  - `chapters/` — Living documentation chapters
  - Architecture specs: STATE_OWNERSHIP_MAP, COMPONENT_INDEPENDENCE_MAP, etc.

## Directory Structure

```
exam-conductor/
├── README.md              # This file
├── new-docs/              # Shared design documentation
│   ├── api/               # OpenAPI specs (12 services)
│   ├── contracts/events/  # NATS event schemas (10 events)
│   ├── hub/               # BLE GATT spec, IPC protocol
│   ├── chapters/          # Documentation chapters (01-25)
│   └── ...                # Architecture specs, doctrine
├── DCR/                   # Digital Copy Review (implemented)
│   ├── CLAUDE.md          # AI agent guide for DCR
│   ├── SETUP_GUIDE.md     # Setup instructions
│   ├── TASKS.md           # Implementation task list
│   ├── services/          # 16 backend microservices
│   ├── hub/               # 8 RPi hub modules
│   ├── libs/              # Shared libraries
│   ├── frontend/          # 3 web frontends (React)
│   ├── mobile/            # Flutter mobile app
│   ├── infra/             # Docker Compose, monitoring, CI/CD
│   ├── test-suite/        # E2E, security, load, hub HW tests
│   └── scripts/           # Dev tools, seed data
└── PCR/                   # Physical Copy Review (planned)
    └── README.md          # Planning placeholder
```
