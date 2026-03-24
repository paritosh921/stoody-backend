# Build Status

Last updated: 2026-03-20 (W6.A1 post-build reconciliation)

This file is the multi-agent coordination surface. Before starting work on any component, check this file.

---

## Component Status

| Component | Phase | Status | Agent | Contract Location | Mock Available | Known Issues |
|---|---|---|---|---|---|---|
| **Shared Libraries** | | | | | | |
| `libs/exampen-proto` | P0 | COMPLETE | Agent-1 | Self-defining | N/A | — |
| `libs/exampen-common-py` | P0 | COMPLETE | Agent-3 | N/A | N/A | — |
| `libs/exampen-common-ts` | P0 | COMPLETE | Agent-3 | N/A | N/A | — |
| **Backend Services** | | | | | | |
| `svc-auth` | P0 | COMPLETE | Agent-1 | `api/auth.openapi.yaml` | Yes | — |
| `svc-exam-orch` | P1 | COMPLETE | Agent-1 | `api/exam-orch.openapi.yaml` | Yes | Question-paper upload endpoint is a stub (returns 501) |
| `svc-stroke-ingest` | P2 | COMPLETE | Agent-3 | `api/stroke-ingest.openapi.yaml` | Yes | — |
| `svc-stroke-proc` | P3 | COMPLETE | Agent-3 | `contracts/events/` | Yes | — |
| `svc-doc-assembly` | P5 | COMPLETE | Agent-3 | `contracts/events/` | Yes | — |
| `svc-ai-pipeline` | P6 | COMPLETE | Agent-5 | `contracts/events/` | Yes | — |
| `svc-score-engine` | P8 | COMPLETE | Agent-5 | `api/score-engine.openapi.yaml` | Yes | — |
| `svc-review` | P11 | COMPLETE | Agent-5 | `api/review.openapi.yaml` | Yes | — |
| `svc-analytics` | P12 | COMPLETE | Agent-6 | `api/analytics.openapi.yaml` | Yes | — |
| `svc-plagiarism` | P8b | COMPLETE | Agent-5 | `api/plagiarism.openapi.yaml` | Yes | — |
| `svc-chat` | P9d | COMPLETE | Agent-6 | `api/chat.openapi.yaml` | Yes | — |
| `svc-copy-upload` | P2i-ext | COMPLETE | Agent-6 | `api/copy-upload.openapi.yaml` | Yes | — |
| `svc-notify` | P14 | COMPLETE | Agent-6 | — | Yes | — |
| `svc-teacher-bff` | P9 | COMPLETE | Agent-1 | `api/teacher-bff.openapi.yaml` | Yes | — |
| `svc-student-bff` | P10 | COMPLETE | Agent-5 | `api/student-bff.openapi.yaml` | Yes | — |
| `svc-invig-console` | P13 | COMPLETE | Agent-4 | `api/invig-console.openapi.yaml` | Yes | — |
| **Hub Modules** | | | | | | |
| `hub-supervisor` | P2a | COMPLETE | Agent-2 | `hub/ipc-protocol.md` | N/A | — |
| `hub-ble-mgr` | P2b | COMPLETE | Agent-2 | `hub/ipc-protocol.md` | N/A | — |
| `hub-invig-ble` | P2c | COMPLETE | Agent-4 | `hub/ble-gatt-spec.md` | N/A | — |
| `hub-pen-sync` | P2f | COMPLETE | Agent-2 | `hub/ble-gatt-spec.md` | N/A | — |
| `hub-timer` | P2e | COMPLETE | Agent-2 | `hub/ipc-protocol.md` | N/A | — |
| `hub-store` | P2a | COMPLETE | Agent-2 | `hub/ipc-protocol.md` | N/A | — |
| `hub-uplink` | P2g/P2h | COMPLETE | Agent-4 | `hub/ipc-protocol.md` + `api/stroke-ingest.openapi.yaml` | N/A | — |
| `hub-tui` | P2a | COMPLETE | Agent-4 | N/A | N/A | — |
| **Frontend** | | | | | | |
| `teacher-dashboard` | P9 | COMPLETE | Agent-1 | `api/teacher-bff.openapi.yaml` | — | — |
| `student-portal` | P10 | COMPLETE | Agent-5 | `api/student-bff.openapi.yaml` | — | — |
| `invigilator-console` | P13 | COMPLETE | Agent-4 | `api/invig-console.openapi.yaml` | — | — |
| **Mobile** | | | | | | |
| `exampen-mobile` (hub control) | P2i | COMPLETE | Agent-4 | `hub/ble-gatt-spec.md` | — | — |
| `exampen-mobile` (teacher view) | P9a | COMPLETE | Agent-1 | `api/teacher-bff.openapi.yaml` | — | — |
| `exampen-mobile` (student view) | P10a | COMPLETE | Agent-5 | `api/student-bff.openapi.yaml` | — | — |
| **Infrastructure** | | | | | | |
| Docker Compose (dev) | P0 | COMPLETE | Agent-6 | N/A | N/A | — |
| Monitoring (Grafana stack) | P0 | COMPLETE | Agent-6 | N/A | N/A | — |
| Hub golden image | P2a | COMPLETE | Agent-2 | `HUB_DEPLOYMENT_SPEC.md` | N/A | — |
| CI/CD pipeline | P0 | COMPLETE | Agent-6 | N/A | N/A | — |
| **Documentation** | | | | | | |
| OpenAPI contracts (12 services) | R6 | COMPLETE | — | `api/` | N/A | Concrete MVP contracts ready |
| Event contracts (10 events) | R6 | COMPLETE | — | `contracts/events/` | N/A | Concrete event schemas ready |
| BLE GATT spec | R6 | COMPLETE | — | `hub/ble-gatt-spec.md` | N/A | BLE command and sync semantics frozen for P2 work |
| Hub IPC protocol | R6 | COMPLETE | — | `hub/ipc-protocol.md` | N/A | — |
| Doctrine files | R5 | COMPLETE | — | Root of `new-docs/` | N/A | Imported and reviewed |
| State ownership map | R5 | COMPLETE | — | `STATE_OWNERSHIP_MAP.md` | N/A | — |
| Component independence map | R5 | COMPLETE | — | `COMPONENT_INDEPENDENCE_MAP.md` | N/A | — |
| Stoody integration spec | R5 | COMPLETE | — | `STOODY_INTEGRATION_SPEC.md` | N/A | — |
| Hub deployment spec | R5 | COMPLETE | — | `HUB_DEPLOYMENT_SPEC.md` | N/A | — |
| Test suite spec | R5 | COMPLETE | — | `TEST_SUITE_SPEC.md` | N/A | — |
| Failure mitigation register | R5 | COMPLETE | — | `FAILURE_MITIGATION_REGISTER.md` | N/A | — |
| Documentation plan | R5 | COMPLETE | — | `DOCUMENTATION_PLAN.md` | N/A | — |
| Document registry | R5 | COMPLETE | — | `DOCUMENT_REGISTRY.md` | N/A | — |
| Chapter: System Overview | W6 | DRAFT | Claude | `chapters/01_SYSTEM_OVERVIEW.md` | N/A | Needs review |
| Chapter: Stroke Pipeline | W6 | DRAFT | Claude | `chapters/02_STROKE_PIPELINE.md` | N/A | Needs review |
| Chapter: Scoring Pipeline | W6 | DRAFT | Claude | `chapters/03_SCORING_PIPELINE.md` | N/A | Needs review |
| Chapter: Hub Operations | W6 | DRAFT | Claude | `chapters/04_HUB_OPERATIONS.md` | N/A | Needs review |
| Chapter: Auth and RBAC | W6 | DRAFT | Claude | `chapters/05_AUTH_AND_RBAC.md` | N/A | Needs review |
| Chapters 06-25 | W6 | DRAFT | Claude | `chapters/06_*` through `chapters/25_*` | N/A | Stub skeletons only |

---

## Summary

- **Documentation & Contracts:** All design docs, API contracts, event schemas, BLE spec, IPC protocol, and supporting specs are COMPLETE from Waves R5-R6. Five documentation chapters drafted in W6; chapters 06-25 created as stub skeletons.
- **Implementation:** All code components across Waves 0-6 are COMPLETE: shared libraries (Wave 0), backend services (Waves 1-3), hub modules (Waves 0-2), BFF services (Wave 4), frontend and mobile (Wave 5), infrastructure (Waves 0, 6). Total test coverage: 1256+ tests across L1-L5 levels.
- **Known gaps:** (1) Question-paper upload endpoint in `svc-exam-orch` is a stub (returns 501). (2) Six security findings from W6 audit remain open — see security audit report.
- **Next steps:** Resolve question-paper upload stub, close security audit findings, promote chapters 06-25 from stubs to full content, field trial preparation (L6/L7).

---

## How to Use This File

1. **Before starting work:** Find your component, check Status and Known Issues.
2. **Claim work:** Change Status to `IN_PROGRESS`, add your Agent ID.
3. **On completion:** Change Status to `COMPLETE`, fill Contract and Mock columns.
4. **On blocking:** Change Status to `BLOCKED`, describe blocker in Known Issues.

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Reconciled BUILD_STATUS with reality: all code components (libs, services, hub, frontend, mobile, infra) marked COMPLETE through Wave 6. Updated summary to reflect 1256+ tests and known gaps (question-paper stub, 6 security findings). Added chapter 06-25 stub entries. | Claude Agent (W6.A1 reconciliation) |
| 2026-03-20 | W6 update: added documentation chapter entries (01-05 DRAFT), promoted all contract/spec docs to COMPLETE, expanded documentation section with individual spec entries, added summary section | Claude Agent (W6.A6.1) |
| 2026-03-18 | Promoted API/event/BLE contract packages to ACTIVE, normalized contract paths to the `new-docs` root, and aligned hub modules on `hub/ipc-protocol.md`. | Codex |
