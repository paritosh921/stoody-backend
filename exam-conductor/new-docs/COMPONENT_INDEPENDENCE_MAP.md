# COMPONENT_INDEPENDENCE_MAP.md
# ExamPen — Component Independence, Multi-Agent Build Guide & No-Monolith Enforcement

Reference: R4-EXAMPEN-DEVSTACK

---

## Purpose

This document enables multiple AI agents (or developers) to build ExamPen components in parallel without stepping on each other. It defines hard interface contracts, file size limits, module isolation rules, and the exact dependency graph that determines what can be built concurrently.

Path note: all contract paths in this document are relative to this documentation root (`new-docs/` while drafting, `docs/` after promotion).

---

## 1. No-Monolith Rules — Hard Enforcement

### 1.1 File Size Limits

| Language | Max Lines per File | Exception Condition |
|---|---|---|
| Python | 300 | Only if file is a single data model with many fields (e.g., SQLAlchemy model with 40+ columns). Must be documented. |
| TypeScript/React | 250 | Only if file is a single complex component with no extractable sub-components. Requires review. |
| SQL migrations | 200 | Only for initial schema creation. Subsequent migrations must be incremental. |
| Config (YAML/TOML) | 150 | Split into per-service configs if exceeding. |
| Markdown docs | No limit | Documentation is exempt. |

**Enforcement:** Pre-commit hook rejects files exceeding limits without an `# EXEMPT: <reason>` header comment.

### 1.2 Module Size Limits

| Metric | Max | Action if Exceeded |
|---|---|---|
| Files per module/package | 15 | Split into sub-modules |
| Functions per file | 12 | Extract helper module |
| Parameters per function | 6 | Introduce config/context object |
| Nesting depth (if/for/try) | 3 levels | Extract inner logic to named function |
| Cyclomatic complexity per function | 10 | Decompose |

### 1.3 Structural Rules

1. **One service = one directory = one Dockerfile = one Docker Compose service.** No shared source directories between services.
2. **Shared code between services goes in a `libs/` directory** with explicit packages: `libs/exampen-proto/` (protobuf/schema definitions), `libs/exampen-common/` (shared utilities). Shared libs are versioned and pinned.
3. **No circular imports between services.** Service A may import from `libs/`, but never from Service B's source.
4. **Database migrations owned by the service that owns the schema.** `svc-score-engine/migrations/` contains only score-related tables. No cross-service migrations.
5. **Each service has its own `README.md` with ownership declaration** (per `STATE_OWNERSHIP_MAP.md` §5).
6. **Tests live next to the code they test.** `svc-score-engine/tests/` not `tests/score-engine/`.

---

## 2. Repository Structure

```
exampen/
├── docs/                           # All documentation
│   ├── architecture/               # System design docs
│   ├── api/                        # API specs (OpenAPI)
│   ├── hub/                        # Hub-specific docs
│   └── chapters/                   # Living documentation chapters
├── libs/                           # Shared libraries
│   ├── exampen-proto/              # Protobuf/JSON schema definitions
│   │   ├── stroke.proto
│   │   ├── exam.proto
│   │   ├── score.proto
│   │   └── events.proto
│   ├── exampen-common-py/          # Python shared utilities
│   │   ├── auth.py                 # JWT validation helper
│   │   ├── nats_client.py          # NATS connection factory
│   │   ├── db.py                   # PostgreSQL connection factory
│   │   └── logging.py              # Structured logging setup
│   └── exampen-common-ts/          # TypeScript shared utilities
│       ├── auth.ts
│       ├── api-client.ts
│       └── types.ts
├── services/                       # Backend microservices
│   ├── svc-auth/
│   │   ├── Dockerfile
│   │   ├── README.md               # Ownership declaration
│   │   ├── src/
│   │   ├── tests/
│   │   └── migrations/
│   ├── svc-exam-orch/
│   ├── svc-stroke-ingest/
│   ├── svc-stroke-proc/
│   ├── svc-doc-assembly/
│   ├── svc-ai-pipeline/
│   ├── svc-score-engine/
│   ├── svc-review/
│   ├── svc-analytics/
│   ├── svc-notify/
│   ├── svc-plagiarism/
│   ├── svc-chat/
│   ├── svc-copy-upload/
│   ├── svc-teacher-bff/            # Aggregation layer for teacher UI
│   ├── svc-student-bff/            # Aggregation layer for student UI
│   └── svc-invig-console/          # Real-time invigilator dashboard backend
├── hub/                            # Hub software (runs on RPi)
│   ├── hub-supervisor/
│   ├── hub-ble-mgr/
│   ├── hub-pen-sync/
│   ├── hub-timer/
│   ├── hub-store/
│   ├── hub-uplink/
│   ├── hub-invig-ble/
│   ├── hub-tui/
│   └── hub-common/                 # Shared hub utilities (IPC, config)
├── frontend/                       # Web frontends
│   ├── teacher-dashboard/
│   ├── student-portal/
│   └── invigilator-console/
├── mobile/                         # Mobile apps
│   ├── exampen-mobile/             # Single app, dual-mode
│   │   ├── lib/
│   │   │   ├── hub_control/        # BLE invigilator mode
│   │   │   ├── teacher_view/       # Score management mode
│   │   │   ├── camera/             # Copy image capture
│   │   │   └── core/               # Auth, networking, storage
│   │   └── test/
├── infra/                          # Infrastructure
│   ├── docker-compose.yml          # Local dev stack
│   ├── docker-compose.prod.yml
│   ├── traefik/
│   ├── monitoring/                 # Grafana, Loki, Tempo, Prometheus configs
│   └── hub-image/                  # Golden image build scripts
├── test-suite/                     # Integration & E2E tests
│   ├── hub-tests/
│   ├── pipeline-tests/
│   ├── e2e-tests/
│   └── tui-runner/                 # TUI test runner
└── scripts/
    ├── dev-setup.sh
    ├── seed-data.sh
    └── generate-mocks.sh
```

---

## 3. Dependency Graph — What Can Be Built in Parallel

### 3.1 Zero-Dependency Components (Start Immediately, Any Agent)

These have no dependencies on other ExamPen components. Build them first or in parallel.

| Component | Interface Contract | Agent Can Start When |
|---|---|---|
| `libs/exampen-proto` | Schema definitions (protobuf/JSON schema) | Immediately |
| `libs/exampen-common-py` | Utility functions (no business logic) | Immediately |
| `libs/exampen-common-ts` | Utility functions (no business logic) | Immediately |
| `hub-timer` | Internal: countdown, persist, resume | Immediately — uses only SQLite, no IPC |
| `hub-store` | Internal: dual-write, read, integrity check | Immediately — uses only filesystem + SQLite |
| `hub-tui` | Internal: display framework, screen shells | Immediately — uses only terminal I/O |

**Note:** `svc-auth` was previously listed here but has an external dependency on the Stoody platform (JWKS endpoint for SSO token validation, user identity API). See §3.1a.

### 3.1a External-Dependency Components (Start Immediately, But Requires Stoody Stub)

| Component | Internal Deps | External Deps | Agent Can Start When |
|---|---|---|---|
| `svc-auth` | `libs/exampen-proto`, PostgreSQL | Stoody JWKS endpoint (for JWT validation), Stoody `/api/users/{id}` (for profile enrichment) | Immediately — build with Stoody stub/mock. Real integration requires Stoody sandbox access. |

**Stoody failure handling required in `svc-auth`:**

| Stoody Failure | `svc-auth` Behavior |
|---|---|
| JWKS endpoint unreachable | Cache last-known JWKS keyset (TTL 24h). Validate against cache. If cache expired AND Stoody down, reject all tokens with 503. |
| JWKS key rotation | Fetch new keyset on `kid` mismatch. If fetch fails, reject unknown `kid` tokens. Cache supports multiple concurrent keysets. |
| User profile API down | Auth succeeds (JWT valid), but profile enrichment skipped. Return base JWT claims only. BFFs degrade gracefully (show user_id instead of name). |
| Stoody role mapping drift | `svc-auth` maps Stoody roles to ExamPen roles via configurable mapping table. If Stoody adds a new role, it maps to `no_exampen_access` by default until mapping is updated. |

### 3.2 First-Tier Dependencies (Requires `libs/` or `svc-auth`)

| Component | Depends On | Can Parallelize With |
|---|---|---|
| `svc-exam-orch` | `libs/exampen-proto`, `svc-auth` (JWT validation) | Any other first-tier component |
| `svc-stroke-ingest` | `libs/exampen-proto`, `svc-auth` | Any other first-tier component |
| `hub-ble-mgr` | `hub-common` (IPC definitions) | `hub-invig-ble`, `hub-uplink` |
| `hub-invig-ble` | `hub-common` | `hub-ble-mgr`, `hub-uplink` |
| `hub-supervisor` | `hub-common`, all hub modules (but can stub them) | Start with stubs, integrate later |

### 3.3 Second-Tier Dependencies (Requires first-tier services running)

| Component | Depends On | Can Parallelize With |
|---|---|---|
| `svc-stroke-proc` | `svc-stroke-ingest` (NATS events), TimescaleDB | `svc-doc-assembly` (stub events) |
| `svc-doc-assembly` | `svc-stroke-proc` (NATS events), MinIO | `svc-ai-pipeline` (stub events) |
| `hub-pen-sync` | `hub-ble-mgr`, `hub-store` | `hub-uplink` |
| `hub-uplink` | `hub-store`, `hub-invig-ble` | `hub-pen-sync` |

### 3.4 Third-Tier Dependencies (Requires pipeline functional)

| Component | Depends On |
|---|---|
| `svc-ai-pipeline` | `svc-doc-assembly` (page images in S3) |
| `svc-score-engine` | `svc-ai-pipeline` (AI results via NATS) |
| `svc-plagiarism` | `svc-ai-pipeline` (recognized text) |
| `svc-copy-upload` | `svc-auth`, MinIO |
| `svc-review` | `svc-score-engine` (score context for objections) |
| `svc-analytics` | `svc-score-engine` (score.updated events) |
| `svc-chat` | `svc-auth` (user identity), PostgreSQL |

### 3.4a BFF Services (Third-Tier — Requires backing services)

BFF services are read-only aggregators. They have no business logic and own no state. They depend on multiple backing services being at least mockable.

| Component | Depends On | Notes |
|---|---|---|
| `svc-teacher-bff` | `svc-score-engine`, `svc-analytics`, `svc-review`, `svc-plagiarism`, `svc-chat` | Build against mocks of each backing service. Real integration when backing services reach L4. |
| `svc-student-bff` | `svc-score-engine`, `svc-analytics`, `svc-review`, `svc-chat` | Same approach. |
| `svc-invig-console` | `svc-exam-orch`, `hub-uplink` (WebSocket status feed via backend relay) | Real-time WebSocket endpoint. Depends on exam-orch for session data. |

### 3.5 Fourth-Tier (Frontend/Mobile — Requires BFF APIs available)

| Component | Depends On |
|---|---|
| `teacher-dashboard` | `svc-teacher-bff` (OpenAPI contract: `api/teacher-bff.openapi.yaml`) |
| `student-portal` | `svc-student-bff` (OpenAPI contract: `api/student-bff.openapi.yaml`) |
| `invigilator-console` | `svc-invig-console` (OpenAPI + WebSocket contract: `api/invig-console.openapi.yaml`) |
| `exampen-mobile` (hub control) | Hub BLE GATT interface (`hub/ble-gatt-spec.md`) |
| `exampen-mobile` (teacher view) | `svc-teacher-bff` REST API |
| `exampen-mobile` (student view) | `svc-student-bff` REST API |

### 3.6 Parallel Build Matrix

```
Time →
Agent 1: libs/proto → svc-auth (with Stoody stub) → svc-exam-orch → svc-teacher-bff → teacher-dashboard
Agent 2: hub-store → hub-timer → hub-ble-mgr → hub-pen-sync → hub-supervisor integration
Agent 3: libs/common → svc-stroke-ingest → svc-stroke-proc → svc-doc-assembly
Agent 4: hub-tui → hub-invig-ble → hub-uplink → svc-invig-console → mobile (hub control mode)
Agent 5: svc-ai-pipeline → svc-plagiarism → svc-score-engine → svc-review → svc-student-bff → student-portal
Agent 6: infra (docker-compose) → monitoring → test-suite → CI/CD pipeline → svc-analytics → svc-chat → svc-notify
```

---

## 4. Interface Contracts Between Components

Every cross-component interaction MUST be defined by a contract before implementation. Agent building component A must not need to read component B's source code.

### 4.1 Contract Types

| Type | Format | Location |
|---|---|---|
| REST API | OpenAPI 3.1 YAML | `api/{service-name}.openapi.yaml` |
| NATS events | JSON Schema | `contracts/events/{event-name}.schema.json` |
| BLE GATT | Characteristic table (UUID, properties, payload format) | `hub/ble-gatt-spec.md` |
| Hub IPC | JSON-lines message envelope and module message catalog | `hub/ipc-protocol.md` |
| Database schema | SQL migrations | `services/{service}/migrations/` |
| Shared types | Protobuf or JSON Schema | `libs/exampen-proto/` |

Runtime note: generated shared-code artifacts may later be copied into `libs/exampen-proto/` or implementation modules, but the documentation contracts above are authoritative during design and build planning.

### 4.2 Contract-First Rule

1. Agent A defines the contract (e.g., OpenAPI spec for `svc-score-engine`).
2. Agent B implements a mock server from the contract.
3. Agent C builds the consumer against the mock.
4. Integration test verifies real server matches contract.

**No agent should implement a consumer by inspecting the producer's source code.** The contract is the interface.

### 4.3 Mock Generation

Each service provides a mock mode:

```bash
# Start svc-score-engine in mock mode (returns canned responses per OpenAPI spec)
docker compose up svc-score-engine --env MOCK_MODE=true
```

Mock mode is auto-generated from OpenAPI spec using `prism` or equivalent.

---

## 5. Agent Build Instructions

### 5.1 Per-Agent Rules

1. **Read before writing.** Before building any component:
   - Read `STATE_OWNERSHIP_MAP.md` for ownership rules.
   - Read the component's interface contract (OpenAPI, NATS schema, GATT spec).
   - Read `FEATURE_PLANNING_CHECKLIST.md` and answer all questions for your component.

2. **Smallest possible files.** Each file does one thing. If you're writing a function and it needs a helper, the helper goes in a separate file if it's >20 lines.

3. **No cross-service imports.** If you're building `svc-score-engine` and need a type from `svc-ai-pipeline`, that type MUST be in `libs/exampen-proto`. Never import from another service's `src/`.

4. **Tests are not optional.** Every PR must include tests. "I'll add tests later" is not accepted. Tests live in `{component}/tests/`.

5. **Document as you build.** Every component's `README.md` must be updated with:
   - What the component does (one paragraph)
   - Ownership declaration
   - How to run it locally
   - How to run its tests
   - What it depends on
   - What depends on it

### 5.2 Agent Handoff Protocol

When Agent A completes a component that Agent B depends on:

1. Agent A commits code + tests + contract + README.
2. Agent A runs `make test` and confirms green.
3. Agent A updates `chapters/BUILD_STATUS.md` with:
   - Component name
   - Status: `COMPLETE` / `IN_PROGRESS` / `BLOCKED`
   - Contract location
   - Mock availability
   - Known issues
4. Agent B checks `BUILD_STATUS.md` before starting dependent work.
5. If dependency is `IN_PROGRESS`, Agent B uses mock mode.

### 5.3 Integration Points

When two agents' components need to talk:

1. Both agents agree on the contract (OpenAPI, NATS schema, etc.) BEFORE either implements.
2. Contract is committed to `contracts/events/`, `api/`, `hub/`, or another authoritative doc path listed in `DOCUMENT_REGISTRY.md`.
3. Both agents implement against the contract independently.
4. Integration test (in `test-suite/`) verifies both sides comply.
5. Integration test is owned by neither agent — it's in the shared `test-suite/` directory.

---

## 6. Code Organization Rules Per Service

### 6.1 Standard Service Layout

```
svc-score-engine/
├── Dockerfile
├── README.md                  # Ownership declaration, run instructions
├── pyproject.toml             # Dependencies
├── src/
│   ├── __init__.py
│   ├── main.py                # Entry point (FastAPI app creation, ~50 lines)
│   ├── config.py              # Environment config loading (~30 lines)
│   ├── routes/                # HTTP route handlers
│   │   ├── scores.py          # GET/PATCH /scores endpoints
│   │   └── rubrics.py         # CRUD /rubrics endpoints
│   ├── events/                # NATS event handlers
│   │   ├── ai_result.py       # Handle ai.result events
│   │   └── score_updated.py   # Publish score.updated events
│   ├── domain/                # Business logic (no I/O)
│   │   ├── score_fsm.py       # Score lifecycle state machine
│   │   ├── rubric_eval.py     # Rubric evaluation logic
│   │   └── override.py        # Override validation logic
│   ├── storage/               # Database access (I/O boundary)
│   │   ├── score_repo.py      # Score event store queries
│   │   └── rubric_repo.py     # Rubric CRUD queries
│   └── adapters/              # External service clients
│       ├── nats_adapter.py    # NATS publish/subscribe
│       └── auth_adapter.py    # JWT validation client
├── tests/
│   ├── test_score_fsm.py      # Unit: state machine transitions
│   ├── test_rubric_eval.py    # Unit: rubric evaluation
│   ├── test_override.py       # Unit: override validation
│   ├── test_routes_scores.py  # Integration: HTTP endpoints
│   └── test_events_ai.py     # Integration: NATS event handling
└── migrations/
    ├── 001_create_score_events.sql
    └── 002_create_materialized_view.sql
```

### 6.2 Layer Rules

| Layer | May Import From | May NOT Import From | I/O Allowed |
|---|---|---|---|
| `routes/` | `domain/`, `storage/`, `adapters/`, `libs/` | Other services | Yes (HTTP) |
| `events/` | `domain/`, `storage/`, `adapters/`, `libs/` | Other services | Yes (NATS) |
| `domain/` | `libs/exampen-proto` only | `storage/`, `adapters/`, `routes/`, `events/` | **NO** — pure logic only |
| `storage/` | `libs/` | `domain/`, `routes/`, `events/` | Yes (DB) |
| `adapters/` | `libs/` | `domain/`, `routes/`, `events/`, `storage/` | Yes (HTTP, NATS) |

**The `domain/` layer is the most critical.** It contains all business logic and has ZERO I/O. This makes it trivially unit-testable. Any agent building a service must ensure `domain/` never imports `asyncio`, `aiohttp`, `sqlalchemy`, `nats`, or any I/O library.

### 6.3 Hub Module Layout

```
hub-pen-sync/
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py              # Module entry point (~40 lines)
│   ├── config.py
│   ├── gatt_reader.py       # GATT characteristic read logic
│   ├── chunk_manager.py     # Chunk assembly, checksum verification
│   └── sync_state.py        # Per-pen sync state machine
├── tests/
│   ├── test_chunk_manager.py
│   └── test_sync_state.py
└── ipc/
    └── pen_sync_ipc.py      # IPC message definitions for supervisor
```

Same layer rules apply. `sync_state.py` (domain) must not import `gatt_reader.py` (I/O).

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Normalized contract paths to the `new-docs` root, made `contracts/events/` authoritative for event schemas, and aligned hub IPC references to `hub/ipc-protocol.md`. | Codex |
