# DOCUMENT_REGISTRY.md
# ExamPen — Document Authority Registry

This file is the single source of truth for document authority inside `backend/exam-conductor/new-docs/`.
If two documents disagree, the document marked authoritative here wins.

---

## Authority Map

| Topic | Authoritative Document | Supplementary Documents | Notes |
|---|---|---|---|
| Platform composition, shared ingest substrate, DCR/PCR engine split | `architecture/DUAL_MODE_ARCHITECTURE.md` | `chapters/01_SYSTEM_OVERVIEW.md` | Root architecture authority. |
| PCR pipeline behavior, segmentation, classification, evaluation flow | `architecture/PCR_EVAL_ENGINE_SPEC.md` | `../pcr/eval-engine-plan-v3.md` | `../pcr/eval-engine-plan-v3.md` is historical source material only after porting. |
| Shared LLM gate contract, allowed callers, budgets, token storage | `architecture/LLM_GATE_SPEC.md` | — | Gate applies to both DCR and PCR. |
| Tamper-proof rules, canonical artifact integrity, audit model | `architecture/TAMPER_PROOF_SPEC.md` | — | Covers conducted-exam flows. Practice persistence remains external. |
| State ownership + read/write boundaries | `governance/STATE_OWNERSHIP_MAP.md` | `architecture/DUAL_MODE_ARCHITECTURE.md` | Ownership is defined by subsystem/engine, not by legacy folder layout. |
| Hub hardware, OS, local storage, provisioning, operations | `integration/HUB_DEPLOYMENT_SPEC.md` | `references/P05_pen_SDK.md`, `references/PEN_TO_CANVAS_TO_DB_REFERENCE.md` | Hub collects and stores canonical artifacts; it does not evaluate them. |
| BLE GATT protocol | `hub/ble-gatt-spec.md` | `references/P05_pen_SDK.md`, `integration/HUB_DEPLOYMENT_SPEC.md` | Concrete BLE authority. |
| Hub IPC protocol | `hub/ipc-protocol.md` | `integration/HUB_DEPLOYMENT_SPEC.md` | Concrete hub module-to-module contract. |
| Current pen/canvas/backend behavior | `references/PEN_TO_CANVAS_TO_DB_REFERENCE.md` | `references/P05_pen_SDK.md` | Use when matching existing Stoody behavior instead of redesigning it. |
| Component dependencies + multi-agent build | `governance/COMPONENT_INDEPENDENCE_MAP.md` | `chapters/BUILD_STATUS.md` | Build ordering and parallelization rules. |
| Stoody integration, auth, tutor/student visibility | `integration/STOODY_INTEGRATION_SPEC.md` | `governance/STATE_OWNERSHIP_MAP.md` | Tutor access follows the admin-owned student visibility model. |
| Super-admin platform control, hub provisioning, gate admin | `integration/SUPERADMIN_SPEC.md` | `integration/HUB_DEPLOYMENT_SPEC.md`, `architecture/LLM_GATE_SPEC.md` | Super-admin does NOT access individual student data. |
| Test strategy + test IDs | `governance/TEST_SUITE_SPEC.md` | `governance/FAILURE_MITIGATION_REGISTER.md` | Documentation and implementation work should reference test IDs from here. |
| Failure modes + mitigations | `governance/FAILURE_MITIGATION_REGISTER.md` | `governance/TEST_SUITE_SPEC.md` | Specs reference failures by ID only. |
| Living documentation structure | `governance/DOCUMENTATION_PLAN.md` | `chapters/BUILD_STATUS.md` | Defines chapter standards and quality gates. |
| REST API contracts | `api/*.openapi.yaml` | Related root specs | OpenAPI owns wire format. |
| Event contracts | `contracts/events/*.schema.json` | Related root specs | Schemas own async payload shape. |
| Reusable process guidance | `GUIDE_RULE_DOCS/*` | — | This folder is the canonical home for generic planning/design docs. |
| Upstream work sequencing and task packaging | `IMPLEMENTATION_PLAN.md` | — | **Status/execution doc only.** Authority limited to work-packaging and spawn ordering for the upstream ingest/hub/mobile stack. Cannot override root architecture docs (`architecture/*.md`), integration specs (`integration/*.md`), OpenAPI files (`api/*.openapi.yaml`), or event schemas (`contracts/events/*.schema.json`). If IMPLEMENTATION_PLAN.md disagrees with any of those, the other document wins. |
| Implementation progress tracking | `chapters/BUILD_STATUS.md` | — | **Status/execution doc only.** Progress tracker for downstream engine/backend work (SWM-* tasks). Cannot override root architecture docs, integration specs, OpenAPI files, or event schemas. If BUILD_STATUS.md disagrees with any of those, the other document wins. |
| Historical background only | `references/exampen-system-design.docx`, `architecture/unifiedPlan.md`, `../pcr/eval-engine-plan-v3.md` | — | Useful for context and rationale only. |

---

## Conflict Resolution FAQ

**Q: If IMPLEMENTATION_PLAN.md disagrees with an OpenAPI file, which one wins?**
A: The OpenAPI file wins. `IMPLEMENTATION_PLAN.md` is a sequencing and work-packaging document. It may describe intended API behavior for task context, but it does not own wire format. If a conflict is found, update IMPLEMENTATION_PLAN.md to match the OpenAPI contract.

**Q: If BUILD_STATUS.md claims a collection or API shape that contradicts a root architecture spec?**
A: The root architecture spec wins. BUILD_STATUS.md reports implementation state; it does not define schema, storage, lifecycle, or API contracts.

**Q: Can a tracker doc (BUILD_STATUS.md, IMPLEMENTATION_PLAN.md) introduce a new API path, collection name, or field that is not in the authoritative specs?**
A: No. Tracker docs may reference or describe existing contracts for task context, but any new contract surface must be defined in the appropriate authoritative document first (architecture spec, OpenAPI, event schema, or integration spec).

---

## Resolution Protocol

When a conflict is found between documents:

1. Check this registry for authority.
2. Update the non-authoritative document to match.
3. If the authoritative document is wrong, fix it first, then propagate.
4. Record the change in the authoritative document changelog.

---

## Document Lifecycle

| Status | Meaning |
|---|---|
| `STUB` | Skeleton placeholder. Not safe to build against. |
| `DRAFT` | Initial content, not fully reviewed. |
| `ACTIVE` | Reviewed and safe to build against. |
| `SUPERSEDED` | Replaced by another document. Keep only for historical reference. |

Promotion to `ACTIVE` requires:
- no unresolved conflicts with authoritative peers
- no TODO/TBD placeholders in the authoritative sections
- concrete schemas/contracts where the document defines an interface
- review against the current `new-docs` authority model

---

## Current Status

| Document | Status | Last Reviewed |
|---|---|---|
| `architecture/unifiedPlan.md` | SUPERSEDED | 2026-03-24 |
| `architecture/DUAL_MODE_ARCHITECTURE.md` | ACTIVE | 2026-03-24 |
| `architecture/PCR_EVAL_ENGINE_SPEC.md` | ACTIVE | 2026-03-24 |
| `architecture/LLM_GATE_SPEC.md` | ACTIVE | 2026-03-24 |
| `architecture/TAMPER_PROOF_SPEC.md` | ACTIVE | 2026-03-24 |
| `governance/STATE_OWNERSHIP_MAP.md` | ACTIVE | R5 fix pass |
| `governance/COMPONENT_INDEPENDENCE_MAP.md` | ACTIVE | R5 fix pass |
| `integration/HUB_DEPLOYMENT_SPEC.md` | ACTIVE | R5 fix pass |
| `hub/ble-gatt-spec.md` | ACTIVE | R6 hardening pass |
| `hub/ipc-protocol.md` | ACTIVE | R6 hardening pass |
| `references/P05_pen_SDK.md` | ACTIVE | current reference pass |
| `references/PEN_TO_CANVAS_TO_DB_REFERENCE.md` | ACTIVE | current reference pass |
| `integration/STOODY_INTEGRATION_SPEC.md` | ACTIVE | R5 fix pass |
| `governance/TEST_SUITE_SPEC.md` | ACTIVE | R5 fix pass |
| `governance/FAILURE_MITIGATION_REGISTER.md` | ACTIVE | R4 |
| `governance/DOCUMENTATION_PLAN.md` | ACTIVE | R4 |
| `chapters/BUILD_STATUS.md` | ACTIVE | 2026-03-24 |
| `IMPLEMENTATION_PLAN.md` | ACTIVE | 2026-04-04 |
| `api/*.openapi.yaml` | ACTIVE | R6 hardening pass |
| `contracts/events/*.schema.json` | ACTIVE | R6 hardening pass |
| `GUIDE_RULE_DOCS/*` | ACTIVE | retained as canonical process guidance |
| `references/exampen-system-design.docx` | SUPERSEDED | 2026-03-24 |
| `../pcr/eval-engine-plan-v3.md` | SUPERSEDED | 2026-03-24 |

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-04-09 | Added explicit registry entries for `chapters/BUILD_STATUS.md` and `IMPLEMENTATION_PLAN.md` as status/execution documents. Added Conflict Resolution FAQ. Both docs marked as non-authoritative for architecture, API shape, schema shape, storage contracts, and lifecycle contracts. | Claude |
| 2026-03-24 | Rebased document authority onto `new-docs`, added root DCR/PCR/gate/tamper specs, demoted legacy `.docx` and PCR v3 plan to historical reference, and made `GUIDE_RULE_DOCS/` the only canonical home for reusable process docs. | Codex |
