# Agent Reference Index

Read this file first before reading any other document in `new-docs/`.

## Agent Instructions

1. Identify the current task type before reading docs.
2. Use the routing guide below to choose only the relevant documents.
3. Prefer `AUTHORITATIVE` docs over `REFERENCE`, `PROCESS`, and `TEMPLATE` docs.
4. If two docs appear to conflict, resolve using `DOCUMENT_REGISTRY.md` first.
5. For implementation work, prefer concrete contracts and current-state references over generic guidance.
6. Do not invent behavior that is already specified in a referenced document.
7. Read the minimum set that gives full task context, then start work.

## Precedence Rules

- `DOCUMENT_REGISTRY.md` decides document authority and conflict resolution.
- Concrete interface packages in `api/`, `contracts/events/`, and `hub/` override summary prose.
- Current-state technical references override generic templates and process prompts for "how it works today".
- `GUIDE_RULE_DOCS/` contains canonical reusable process docs. Matching files at the `new-docs/` root are identical mirrors for convenience.
- `exampen-system-design.docx` is high-level context only, not the final word on implementation details.

## Quick Task Router

| Task | Read First |
|---|---|
| New feature planning | `DOCUMENT_REGISTRY.md`, `GUIDE_RULE_DOCS/SOFTWARE_DEVELOPMENT_DOCTRINE.md`, `GUIDE_RULE_DOCS/SYSTEM_DESIGN_GUIDELINE.md`, `GUIDE_RULE_DOCS/FEATURE_PLANNING_CHECKLIST.md` |
| Parallel build / agent coordination | `COMPONENT_INDEPENDENCE_MAP.md`, `chapters/BUILD_STATUS.md`, `TEST_SUITE_SPEC.md` |
| Auth / Stoody / roles / parent access | `STOODY_INTEGRATION_SPEC.md`, `STATE_OWNERSHIP_MAP.md`, `api/auth.openapi.yaml` |
| Hub / BLE / pen sync / offline transfer | `P05_pen_SDK.md`, `hub/ble-gatt-spec.md`, `HUB_DEPLOYMENT_SPEC.md`, `hub/ipc-protocol.md`, `FAILURE_MITIGATION_REGISTER.md` |
| Stroke pipeline / canvas / page persistence | `PEN_TO_CANVAS_TO_DB_REFERENCE.md`, `P05_pen_SDK.md`, `STATE_OWNERSHIP_MAP.md`, relevant files in `api/` and `contracts/events/` |
| Service API implementation | `api/`, `STATE_OWNERSHIP_MAP.md`, `COMPONENT_INDEPENDENCE_MAP.md`, `TEST_SUITE_SPEC.md` |
| Event-driven pipeline work | `contracts/events/`, `STATE_OWNERSHIP_MAP.md`, `FAILURE_MITIGATION_REGISTER.md`, `TEST_SUITE_SPEC.md` |
| Testing / validation / diagnostics | `TEST_SUITE_SPEC.md`, `FAILURE_MITIGATION_REGISTER.md`, `chapters/BUILD_STATUS.md` |
| Documentation changes | `DOCUMENT_REGISTRY.md`, `DOCUMENTATION_PLAN.md`, relevant authoritative spec |

## Document Index

| Document | Location(s) | Class | What It Covers | Read When |
|---|---|---|---|---|
| `DOCUMENT_REGISTRY.md` | `new-docs/DOCUMENT_REGISTRY.md` | REGISTRY | Source-of-truth map, authority levels, conflict resolution, lifecycle status. | Always first if authority is unclear. |
| `COMPONENT_INDEPENDENCE_MAP.md` | `new-docs/COMPONENT_INDEPENDENCE_MAP.md` | AUTHORITATIVE | Build order, dependency graph, module boundaries, contract-first rules, parallel-agent constraints. | Building services, coordinating agents, or deciding sequencing. |
| `STATE_OWNERSHIP_MAP.md` | `new-docs/STATE_OWNERSHIP_MAP.md` | AUTHORITATIVE | Single writable owner rules, read/write boundaries, transactional boundaries, violation detection. | Any task that reads or writes critical state. |
| `STOODY_INTEGRATION_SPEC.md` | `new-docs/STOODY_INTEGRATION_SPEC.md` | AUTHORITATIVE | Stoody auth, role mapping, parent access, embedding model, consumed/pushed APIs, portal behavior. | Stoody, auth, BFF, portal, or parent-access work. |
| `HUB_DEPLOYMENT_SPEC.md` | `new-docs/HUB_DEPLOYMENT_SPEC.md` | AUTHORITATIVE | Hub image, setup, TUI, local schema, pen lifecycle, WiFi policy, provisioning, operations. | Hub runtime, storage, setup, TUI, or ops work. |
| `TEST_SUITE_SPEC.md` | `new-docs/TEST_SUITE_SPEC.md` | AUTHORITATIVE | Unit, integration, pipeline, hardware, TUI diagnostics, CI evidence, test IDs. | Designing tests or checking validation requirements. |
| `FAILURE_MITIGATION_REGISTER.md` | `new-docs/FAILURE_MITIGATION_REGISTER.md` | AUTHORITATIVE | Known failure modes, mitigation expectations, acknowledged unsolved risks. | Error handling, resilience, fallback, and recovery design. |
| `DOCUMENTATION_PLAN.md` | `new-docs/DOCUMENTATION_PLAN.md` | PROCESS | Doc structure, chapter template, update triggers, quality gates, build-status format. | Editing docs or deciding where new docs belong. |
| `chapters/BUILD_STATUS.md` | `new-docs/chapters/BUILD_STATUS.md` | STATUS | Current build coordination table, component status, contract readiness, known blockers. | Before claiming work or checking readiness. |
| `api/` | `new-docs/api/` | AUTHORITATIVE | Concrete REST/OpenAPI contracts for service request/response behavior. | Implementing or consuming service APIs. |
| `contracts/events/` | `new-docs/contracts/events/` | AUTHORITATIVE | Concrete event envelopes and payload schemas for async service communication. | Implementing publishers, consumers, or event validation. |
| `hub/ble-gatt-spec.md` | `new-docs/hub/ble-gatt-spec.md` | AUTHORITATIVE | Hub-side BLE UUIDs, chunk framing, command IDs, MTU, retries, status feed. | Hub BLE implementation or mobile hub-control integration. |
| `hub/ipc-protocol.md` | `new-docs/hub/ipc-protocol.md` | AUTHORITATIVE | Hub internal message envelope, routing rules, module payload shapes, IPC errors. | Hub module-to-module integration. |
| `P05_pen_SDK.md` | `new-docs/P05_pen_SDK.md` | REFERENCE | Verified real pen GATT layout, actual command channel, frame format, CRC, offline sync, OTA flow. | Anything touching physical pen protocol or stroke ingestion. |
| `PEN_TO_CANVAS_TO_DB_REFERENCE.md` | `new-docs/PEN_TO_CANVAS_TO_DB_REFERENCE.md` | REFERENCE | Current end-to-end Stoody pen stack: BLE agent, frontend canvas, backend persistence, sync, message shapes. | Matching existing stroke behavior instead of redesigning it. |
| `AI_AGENT_WORKFLOW_PROMPT.md` | `new-docs/AI_AGENT_WORKFLOW_PROMPT.md`; `new-docs/GUIDE_RULE_DOCS/AI_AGENT_WORKFLOW_PROMPT.md` | PROCESS | Agent operating prompt: explore first, ask high-value questions, lock design, validate honestly. | Guiding agent behavior before coding. |
| `FEATURE_PLANNING_CHECKLIST.md` | `new-docs/FEATURE_PLANNING_CHECKLIST.md`; `new-docs/GUIDE_RULE_DOCS/FEATURE_PLANNING_CHECKLIST.md` | PROCESS | Minimal pre-implementation checklist for problem, ownership, interfaces, timing, failure, and validation. | Quick feature-scoping and readiness checks. |
| `SOFTWARE_DEVELOPMENT_DOCTRINE.md` | `new-docs/SOFTWARE_DEVELOPMENT_DOCTRINE.md`; `new-docs/GUIDE_RULE_DOCS/SOFTWARE_DEVELOPMENT_DOCTRINE.md` | PROCESS | Core engineering rules: ownership, pure reads, transactions, ingress normalization, test integrity. | Any non-trivial design or implementation task. |
| `SYSTEM_DESIGN_GUIDELINE.md` | `new-docs/SYSTEM_DESIGN_GUIDELINE.md`; `new-docs/GUIDE_RULE_DOCS/SYSTEM_DESIGN_GUIDELINE.md` | PROCESS | Deep design interrogation flow for requirements, boundaries, interfaces, NFRs, failures, validation. | Designing a subsystem before implementation. |
| `SYSTEM_DESIGN_TEMPLATE.md` | `new-docs/SYSTEM_DESIGN_TEMPLATE.md`; `new-docs/GUIDE_RULE_DOCS/SYSTEM_DESIGN_TEMPLATE.md` | TEMPLATE | Fill-in template for producing an implementation-ready design document. | Writing a new system design doc. |
| `exampen-system-design.docx` | `new-docs/exampen-system-design.docx` | REFERENCE | Legacy/high-level system overview and background context. | Background orientation only; not for final interface truth. |

## Notes on Duplicate Guide Docs

- The five guide documents in `GUIDE_RULE_DOCS/` are byte-identical to the matching copies at the `new-docs/` root.
- Treat `GUIDE_RULE_DOCS/` as the canonical home for reusable process guidance.
- Treat the root copies as convenience mirrors unless a future update states otherwise.

## Minimal Reading Sets

### If implementing a backend service

- `DOCUMENT_REGISTRY.md`
- `STATE_OWNERSHIP_MAP.md`
- relevant file(s) in `api/` or `contracts/events/`
- `COMPONENT_INDEPENDENCE_MAP.md`
- `TEST_SUITE_SPEC.md`

### If implementing hub or BLE behavior

- `DOCUMENT_REGISTRY.md`
- `HUB_DEPLOYMENT_SPEC.md`
- `hub/ble-gatt-spec.md`
- `hub/ipc-protocol.md`
- `P05_pen_SDK.md`
- `FAILURE_MITIGATION_REGISTER.md`

### If implementing stroke or canvas behavior

- `DOCUMENT_REGISTRY.md`
- `PEN_TO_CANVAS_TO_DB_REFERENCE.md`
- `P05_pen_SDK.md`
- `STATE_OWNERSHIP_MAP.md`
- relevant `api/` or `contracts/events/` docs

### If planning a new feature

- `DOCUMENT_REGISTRY.md`
- `GUIDE_RULE_DOCS/SOFTWARE_DEVELOPMENT_DOCTRINE.md`
- `GUIDE_RULE_DOCS/SYSTEM_DESIGN_GUIDELINE.md`
- `GUIDE_RULE_DOCS/FEATURE_PLANNING_CHECKLIST.md`
- then only the relevant authoritative specs
