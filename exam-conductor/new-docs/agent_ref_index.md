# Agent Reference Index

Read this file first before reading any other document in `new-docs/`.

## Implementation Boundaries

ExamPen spans 4 implementation boundaries. Each boundary has its own codebase location. **Do NOT cross boundaries without reading the relevant integration spec.**

| # | Boundary | Codebase Location | Tech | Notes |
|---|----------|------------------|------|-------|
| 1 | Backend + exam-conductor | `backend/` (includes `backend/exam-conductor/`) | Python 3.11, FastAPI, MongoDB | Core APIs, engines, gate. **BUILT.** 18 routers mounted, `_evalpen_available = True`. |
| 2 | Frontend | `frontend/` | React 18, TS, Vite, Tailwind | Teacher workspace, results, recheck, student portal, conversations **BUILT.** See `frontend/src/components/exam-pen/`. |
| 3 | Super-Admin | `super-admin/` | Electron + React | Feature gate, hub provisioning, usage analytics **PARTIALLY BUILT.** See `super-admin/src/pages/ExamPenManagementPage.tsx`. |
| 4 | Edge hub runtime + ExamPen mode | `stoody-multi-pen/edge_hub/` | Python 3.12, nRF UART transport, local services, systemd | Converged RPi host runtime. Smartboard mode and ExamPen mode must run as independent service groups over shared transport, pen registry, storage, health, logging, and uplink substrate. `stoody-multi-pen/HUB-exam-conductor/` remains a reference/decomposition donor until its ExamPen behavior is folded into the shared runtime. |

### Critical Folder Boundaries

- **`stoody-multi-pen/edge_hub/`** — Converged Stoody edge runtime. New ExamPen hub work should be added here as an independent ExamPen mode/service group, not interleaved into smartboard mode.
- **`stoody-multi-pen/HUB-exam-conductor/`** — Legacy/reference ExamPen hub implementation. Use as a design donor for supervisor, timer, invigilator BLE, pen sync, and uplink behavior while migrating concepts into `edge_hub`.
- **`stoody-multi-pen/mobile-app/`** — The invigilator mobile app will be extended for ExamPen (BLE commands to exam-hub). Shared between smartboard and ExamPen use cases.
- **`backend/exam-conductor/`** — All ExamPen backend modules. Never import from `archiveDCR/` or root `exam-conductor/`.
- **`frontend/`** — ExamPen tutor/student pages will be added here. Follow existing frontend patterns (see `frontend/CLAUDE.md`).
- **`super-admin/`** — ExamPen platform admin surfaces. See `integration/SUPERADMIN_SPEC.md`.

## Agent Instructions

1. Start with `governance/DOCUMENT_REGISTRY.md` if authority is unclear.
2. Read only the minimum authoritative set for the current task.
3. Prefer root specs, concrete contracts, and current-state references over chapters and historical docs.
4. Treat `archiveDCR/` as backup-only and out of scope for implementation decisions.
5. Treat `GUIDE_RULE_DOCS/` as the only canonical home for reusable process guidance.
6. Treat `architecture/unifiedPlan.md`, `references/exampen-system-design.docx`, and `../pcr/eval-engine-plan-v3.md` as historical context only.
7. For new edge work, target `stoody-multi-pen/edge_hub/` shared runtime seams. Keep smartboard and ExamPen mode state machines independent. Use `stoody-multi-pen/HUB-exam-conductor/` as reference material, not as the final deployment boundary.

## Precedence Rules

- `governance/DOCUMENT_REGISTRY.md` resolves all authority conflicts.
- `api/*.openapi.yaml`, `contracts/events/*.schema.json`, and `hub/*.md` override summary prose.
- Root specs override chapters.
- Current-state references override generic templates when the task is about matching existing Stoody behavior.

## Known Drift (2026-05-02)

Some tracker and chapter docs lag behind the current implementation. Active drifts:

- **Frontend is ahead of older index/tracker wording.** The teacher workspace (5-tab model: Exams, Workspace, Results, Recheck, Conversations), student portal, and super-admin ExamPen management page are built and compiling. Some tracker docs still say "NOT BUILT" or omit these surfaces.
- **Some frontend recheck/conversation flows exist before fully confirmed backend parity.** The frontend defines `RecheckStatus` (with required `submission_id`) and `ConversationThread` types, but backend recheck/conversation routers are not yet mounted. These are frontend-only contracts until backend endpoints are implemented.
- **Plagiarism and analytics specs exist in authority docs** (`api/plagiarism.openapi.yaml`, `api/analytics.openapi.yaml`, chapters 13 and 15) but are not yet mounted backend surfaces.
- **DCR question-region population is still pending.** `exampen_question_regions` collection is not populated end-to-end. DCR falls back to whole-page regions. See `BUILD_STATUS.md` Known Gaps.
- **Hub runtime exists, but productionization remains pending.** Packaging, systemd installation, hardware BLE validation, and rollout verification are still open.
- **Super-admin usage analytics is partial.** Tokens Today is real (from `evalpen/gate/usage/aggregate`). Per-tenant exam/submission counts require a new backend endpoint.

## Quick Task Router

| Task | Read First |
|---|---|
| New feature planning | `governance/DOCUMENT_REGISTRY.md`, `GUIDE_RULE_DOCS/SOFTWARE_DEVELOPMENT_DOCTRINE.md`, `GUIDE_RULE_DOCS/SYSTEM_DESIGN_GUIDELINE.md`, `GUIDE_RULE_DOCS/FEATURE_PLANNING_CHECKLIST.md` |
| Shared ingest / hub / canonical exam storage | `architecture/DUAL_MODE_ARCHITECTURE.md`, `integration/HUB_DEPLOYMENT_SPEC.md`, `hub/ble-gatt-spec.md`, `hub/ipc-protocol.md`, `governance/STATE_OWNERSHIP_MAP.md` |
| DCR engine work | `architecture/DUAL_MODE_ARCHITECTURE.md`, `governance/STATE_OWNERSHIP_MAP.md`, relevant `api/` or `contracts/events/` docs |
| PCR engine work | `architecture/PCR_EVAL_ENGINE_SPEC.md`, `architecture/LLM_GATE_SPEC.md`, `architecture/TAMPER_PROOF_SPEC.md`, relevant `api/eval-*.openapi.yaml`, relevant `contracts/events/eval.*.schema.json` |
| LLM gate / token budget work | `architecture/LLM_GATE_SPEC.md`, `api/eval-usage.openapi.yaml`, `governance/STATE_OWNERSHIP_MAP.md` |
| Tamper-proofing / audit integrity | `architecture/TAMPER_PROOF_SPEC.md`, `governance/STATE_OWNERSHIP_MAP.md`, relevant `contracts/events/*.schema.json` |
| Stoody integration / tutor visibility / practice boundaries | `integration/STOODY_INTEGRATION_SPEC.md`, `architecture/DUAL_MODE_ARCHITECTURE.md`, `governance/STATE_OWNERSHIP_MAP.md` |
| Pen stack / frontend canvas / existing sync behavior | `references/PEN_TO_CANVAS_TO_DB_REFERENCE.md`, `references/P05_pen_SDK.md`, `integration/HUB_DEPLOYMENT_SPEC.md` |
| Parallel build / agent coordination | `governance/COMPONENT_INDEPENDENCE_MAP.md`, `chapters/BUILD_STATUS.md`, `governance/TEST_SUITE_SPEC.md` |
| Super-admin / platform control | `integration/SUPERADMIN_SPEC.md`, `governance/STATE_OWNERSHIP_MAP.md` |
| ExamPen hub (edge runtime) work | `integration/HUB_DEPLOYMENT_SPEC.md`, `hub/ble-gatt-spec.md`, `hub/ipc-protocol.md`, plus `stoody-multi-pen/docs/EDGE_HUB_NRF_UNIFIED_RUNTIME_PLAN.md` — **code goes in `stoody-multi-pen/edge_hub/` as independent ExamPen/shared-runtime services.** |
| Hub provisioning / stroke ingest API work | `integration/HUB_DEPLOYMENT_SPEC.md` §7 (provisioning contract), §8 (upload path authority), `api/stroke-ingest.openapi.yaml` (upload wire format), `integration/SUPERADMIN_SPEC.md` §5 (two-party flow) |
| Frontend ExamPen UI | `integration/STOODY_INTEGRATION_SPEC.md`, relevant `api/*.openapi.yaml` — **code goes in `frontend/`** |
| Documentation changes | `governance/DOCUMENT_REGISTRY.md`, `governance/DOCUMENTATION_PLAN.md`, then the relevant authoritative spec |

## Document Index

| Document | Class | What It Covers | Read When |
|---|---|---|---|
| `governance/DOCUMENT_REGISTRY.md` | REGISTRY | Authority map, lifecycle, promotion/supersession rules. | Always first if authority is unclear. |
| `architecture/DUAL_MODE_ARCHITECTURE.md` | AUTHORITATIVE | Shared ingest substrate, engine split, DCR contract, integration boundaries. | Any DCR/PCR architecture or routing work. |
| `architecture/PCR_EVAL_ENGINE_SPEC.md` | AUTHORITATIVE | PCR segmentation, classification, evaluation, flags, and practice boundary. | PCR implementation or planning. |
| `architecture/LLM_GATE_SPEC.md` | AUTHORITATIVE | Shared LLM gate contract, allowed callers, storage, budgets, usage API. | Gate, token usage, or budget work. |
| `architecture/TAMPER_PROOF_SPEC.md` | AUTHORITATIVE | Canonical artifact integrity, server-side fetch, audit model. | Integrity or anti-tampering work. |
| `governance/STATE_OWNERSHIP_MAP.md` | AUTHORITATIVE | Single writable owner rules by subsystem/engine. | Any stateful change. |
| `governance/COMPONENT_INDEPENDENCE_MAP.md` | AUTHORITATIVE | Build ordering and dependency boundaries. | Sequencing or agent coordination. |
| `integration/STOODY_INTEGRATION_SPEC.md` | AUTHORITATIVE | Stoody auth, tutor/student access, tenant integration rules. | Stoody-facing changes. |
| `integration/HUB_DEPLOYMENT_SPEC.md` | AUTHORITATIVE | Hub runtime, provisioning, local storage, upload behavior. | Hub implementation or ops work. |
| `hub/ble-gatt-spec.md` | AUTHORITATIVE | BLE protocol details. | Pen/hub BLE work. |
| `hub/ipc-protocol.md` | AUTHORITATIVE | Hub internal IPC contract. | Hub module integration work. |
| `references/P05_pen_SDK.md` | REFERENCE | Verified pen protocol details. | Physical pen protocol work. |
| `references/PEN_TO_CANVAS_TO_DB_REFERENCE.md` | REFERENCE | Current Stoody pen/canvas/backend flow. | Matching existing ingest behavior. |
| `governance/TEST_SUITE_SPEC.md` | AUTHORITATIVE | Test IDs and validation expectations. | Planning or verifying tests. |
| `governance/FAILURE_MITIGATION_REGISTER.md` | AUTHORITATIVE | Failure modes and mitigation expectations. | Error handling and resilience work. |
| `integration/SUPERADMIN_SPEC.md` | AUTHORITATIVE | Super-admin control surface for ExamPen (gate config, hub provisioning, usage analytics). | Super-admin or platform-level ExamPen work. |
| `governance/DOCUMENTATION_PLAN.md` | PROCESS | Documentation templates, quality gates, chapter rules. | Editing docs or creating new ones. |
| `chapters/09_TEACHER_BFF_DASHBOARD.md` | STATUS | Teacher workspace layout, tab model, pane structure, recheck/conversation UI. | Teacher frontend work. |
| `chapters/10_STUDENT_BFF_PORTAL.md` | STATUS | Student portal: published exams, score breakdown, recheck requests, conversations. | Student frontend work. |
| `chapters/12_MOBILE_APP.md` | STATUS | Mobile invigilator app: hub list, session dashboard, camera fallback. | Mobile app work. |
| `chapters/BUILD_STATUS.md` | STATUS | Swarm task board, execution waves, and file-owned task packs. | Before claiming or sequencing work. |
| `IMPLEMENTATION_PLAN.md` | STATUS | Upstream stack work sequencing (invigilator → hub → ingest → ready_for_eval). | Before claiming upstream tasks. Non-authoritative for contracts. |
| `GUIDE_RULE_DOCS/*` | PROCESS/TEMPLATE | Reusable planning and design guidance. | Generic design/process guidance. |
| `architecture/unifiedPlan.md` | HISTORICAL | Historical transition blueprint. | Only for migration rationale. |
| `../pcr/eval-engine-plan-v3.md` | HISTORICAL | Locked PCR source material preserved for traceability. | Only when porting missing PCR detail into active specs. |
| `references/exampen-system-design.docx` | HISTORICAL | Legacy high-level context. | Background only. |

## Minimal Reading Sets

### If implementing the shared ingestion substrate

- `governance/DOCUMENT_REGISTRY.md`
- `architecture/DUAL_MODE_ARCHITECTURE.md`
- `governance/STATE_OWNERSHIP_MAP.md`
- `integration/HUB_DEPLOYMENT_SPEC.md`
- `hub/ble-gatt-spec.md` or `hub/ipc-protocol.md` as needed

### If implementing DCR

- `governance/DOCUMENT_REGISTRY.md`
- `architecture/DUAL_MODE_ARCHITECTURE.md`
- `governance/STATE_OWNERSHIP_MAP.md`
- relevant `api/` or `contracts/events/` docs
- `governance/TEST_SUITE_SPEC.md`

### If implementing PCR

- `governance/DOCUMENT_REGISTRY.md`
- `architecture/PCR_EVAL_ENGINE_SPEC.md`
- `architecture/LLM_GATE_SPEC.md`
- `architecture/TAMPER_PROOF_SPEC.md`
- relevant `api/eval-*.openapi.yaml`
- relevant `contracts/events/eval.*.schema.json`
- `governance/TEST_SUITE_SPEC.md`

### If planning documentation work

- `governance/DOCUMENT_REGISTRY.md`
- `governance/DOCUMENTATION_PLAN.md`
- `chapters/BUILD_STATUS.md`
- the relevant authoritative spec
