# DOCUMENT_REGISTRY.md
# ExamPen — Document Authority Registry

This is the single source of truth for which document is authoritative for which topic.
If two documents disagree, the one marked **AUTHORITATIVE** here wins. The other must be updated to match.

---

## Authority Map

| Topic | Authoritative Document | Supplementary Documents | Notes |
|---|---|---|---|
| **System architecture overview** | `exampen-system-design.docx` (R1/R2/R3) | — | High-level architecture, service catalog, data stores, event bus, build phases. Does NOT define implementation details. On auth, binding authority, BLE, contracts, and service ownership, the satellite docs below override any older `.docx` wording. |
| **State ownership + read/write boundaries** | `STATE_OWNERSHIP_MAP.md` | `exampen-system-design.docx` (for context) | If the .docx implies a different owner than this map, this map wins. |
| **Hub hardware, OS, TUI, local storage** | `HUB_DEPLOYMENT_SPEC.md` | `exampen-system-design.docx` §B-H | Hub SQLite schema, TUI screens, systemd services, WiFi config, pen lifecycle — all in this doc. |
| **BLE GATT protocol** | `hub/ble-gatt-spec.md` | `exampen-system-design.docx` §B-H4, `HUB_DEPLOYMENT_SPEC.md` | Characteristic UUIDs, payload formats, MTU handling, error codes. |
| **Hub IPC protocol** | `hub/ipc-protocol.md` | `HUB_DEPLOYMENT_SPEC.md` §2 | Message envelope and module message catalog. Runtime implementation may later live in `hub/hub-common/ipc_protocol.py`, but this doc is authoritative during planning. |
| **Component dependencies + multi-agent build** | `COMPONENT_INDEPENDENCE_MAP.md` | — | Repo structure, dependency graph, build matrix, file size rules, layer rules. |
| **Test strategy + test IDs** | `TEST_SUITE_SPEC.md` | — | Every test has an explicit ID. Documentation chapters reference these IDs. |
| **Living documentation structure** | `DOCUMENTATION_PLAN.md` | — | Chapter index, template, update triggers, quality gates. |
| **Stoody platform integration** | `STOODY_INTEGRATION_SPEC.md` | — | SSO, API mapping, tutor/student features, RBAC matrix, embedding decision (Option B, frozen). |
| **Failure modes + mitigations** | `FAILURE_MITIGATION_REGISTER.md` | `exampen-system-design.docx` Part A (problem listing) | The .docx lists problems. This register assigns mitigations. If the register says "mitigated" but the .docx says "unsolved", the register is current. |
| **REST API contracts** | `api/{service}.openapi.yaml` | — | OpenAPI 3.1 specs. These are the interface contracts that agents build against. |
| **NATS event contracts** | `contracts/events/` | — | JSON Schema per event. Published and consumed by services. |
| **Development doctrine** | `SOFTWARE_DEVELOPMENT_DOCTRINE.md` | `FEATURE_PLANNING_CHECKLIST.md`, `SYSTEM_DESIGN_GUIDELINE.md`, `SYSTEM_DESIGN_TEMPLATE.md` | These are generic (repo-portable). ExamPen-specific docs above implement the doctrine for this project. |
| **Agent workflow** | `AI_AGENT_WORKFLOW_PROMPT.md` | — | Copy-paste prompt for any agent working in this repo. |

---

## Resolution Protocol

When a conflict is found between documents:

1. Check this registry for which doc is authoritative.
2. Update the non-authoritative doc to match.
3. If the authoritative doc is wrong, update IT first, then propagate.
4. Log the resolution in the authoritative doc's changelog.

---

## Document Lifecycle

| Status | Meaning |
|---|---|
| **STUB** | Skeleton placeholder. Not safe to build against. |
| **DRAFT** | Initial content, not reviewed. May contain placeholders. |
| **ACTIVE** | Reviewed, consistent with other docs, safe to build against. |
| **SUPERSEDED** | Replaced by another doc. Must link to replacement. Do not build against. |

All documents start as STUB or DRAFT. Promotion to ACTIVE requires: (a) no conflicts with authoritative peers, (b) all TODO/TBD placeholders resolved, (c) concrete schemas or message definitions where the document is a contract, (d) reviewed by at least one other agent/person.

---

## Current Status

| Document | Status | Last Reviewed |
|---|---|---|
| `exampen-system-design.docx` | ACTIVE | R3 (March 2026) |
| `HUB_DEPLOYMENT_SPEC.md` | ACTIVE | R5 fix pass |
| `STATE_OWNERSHIP_MAP.md` | ACTIVE | R5 fix pass (pen binding authority resolved) |
| `COMPONENT_INDEPENDENCE_MAP.md` | ACTIVE | R5 fix pass (BFF services added, svc-auth dep fixed) |
| `TEST_SUITE_SPEC.md` | ACTIVE | R5 fix pass (all test IDs assigned) |
| `DOCUMENTATION_PLAN.md` | ACTIVE | R4 |
| `STOODY_INTEGRATION_SPEC.md` | ACTIVE | R5 fix pass (embed decision frozen) |
| `FAILURE_MITIGATION_REGISTER.md` | ACTIVE | R4 |
| `DOCUMENT_REGISTRY.md` | ACTIVE | R5 |
| `api/*.openapi.yaml` | ACTIVE | R6 hardening pass |
| `hub/ble-gatt-spec.md` | ACTIVE | R6 hardening pass |
| `hub/ipc-protocol.md` | ACTIVE | R6 hardening pass |
| `contracts/events/*.schema.json` | ACTIVE | R6 hardening pass |
| `chapters/BUILD_STATUS.md` | ACTIVE | R6 hardening pass |
| Doctrine files (5) | ACTIVE | Imported R5 |

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Normalized contract paths to the `new-docs` root, made the hub IPC doc authoritative during planning, and promoted concrete contract packages from STUB to ACTIVE. | Codex |
