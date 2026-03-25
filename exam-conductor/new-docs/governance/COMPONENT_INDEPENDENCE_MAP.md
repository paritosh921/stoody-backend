# COMPONENT_INDEPENDENCE_MAP.md
# ExamPen — Component Independence and Build Order

Reference: `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/PCR_EVAL_ENGINE_SPEC.md`, `architecture/LLM_GATE_SPEC.md`

---

## Purpose

This document defines the non-monolithic build boundaries for the active ExamPen architecture.

The platform is split into:

- shared ingest substrate
- DCR engine
- PCR engine
- LLM gate
- Stoody integration surfaces

No component should blur these boundaries.

---

## 1. Hard Independence Rules

1. The shared ingest substrate does not own scoring semantics.
2. DCR and PCR do not own raw conducted-exam artifact capture.
3. Practice persistence stays outside ExamPen.
4. The gate is shared and separate from both engines.
5. Reusable process guidance lives only in `GUIDE_RULE_DOCS/`.

---

## 2. Dependency Order

### Tier 0 — Context and authority

- `governance/DOCUMENT_REGISTRY.md`
- `agent_ref_index.md`
- `chapters/BUILD_STATUS.md`

### Tier 1 — Core architecture authorities

- `architecture/DUAL_MODE_ARCHITECTURE.md`
- `architecture/LLM_GATE_SPEC.md`
- `architecture/TAMPER_PROOF_SPEC.md`

### Tier 2 — Engine detail and executable backlog

- `architecture/PCR_EVAL_ENGINE_SPEC.md`
- DCR details within `architecture/DUAL_MODE_ARCHITECTURE.md`
- `chapters/BUILD_STATUS.md`

### Tier 3 — Governance

- `governance/STATE_OWNERSHIP_MAP.md`
- `governance/FAILURE_MITIGATION_REGISTER.md`
- `governance/TEST_SUITE_SPEC.md`

### Tier 4 — Integration and current-state references

- `integration/HUB_DEPLOYMENT_SPEC.md`
- `integration/STOODY_INTEGRATION_SPEC.md`
- `references/P05_pen_SDK.md`
- `references/PEN_TO_CANVAS_TO_DB_REFERENCE.md`

### Tier 5 — Chapters and interface packages

- `chapters/*`
- `api/*`
- `contracts/events/*`

---

## 3. Parallelizable Work

```text
shared ingest substrate docs  ─┐
                               ├─> governance docs ──> chapters
DCR architecture docs        ──┤
                               │
PCR architecture docs        ──┤
                               └─> API/event contracts

LLM gate docs                ─────> usage API + evaluation docs
```

### Can be updated in parallel

- DCR architecture vs PCR engine detail
- gate contract vs Stoody integration
- references vs chapters

### Should not diverge

- `STATE_OWNERSHIP_MAP.md` from `DUAL_MODE_ARCHITECTURE.md`
- `TEST_SUITE_SPEC.md` from `FAILURE_MITIGATION_REGISTER.md`
- `chapters/01-03` from the root architecture docs

---

## 4. No-Monolith Review Questions

Before approving a design or implementation:

1. Did the ingest substrate start owning evaluation semantics?
2. Did DCR or PCR start owning raw artifact collection?
3. Did practice evaluation start creating new persistence?
4. Did a new LLM caller bypass the gate?
5. Did a “summary” doc become a second authority source?
