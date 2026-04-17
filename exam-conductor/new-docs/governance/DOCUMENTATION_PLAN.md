# DOCUMENTATION_PLAN.md
# ExamPen — Living Documentation Plan

Reference: `architecture/unifiedPlan.md`

---

## Principle

Active documentation must mirror the current architecture:

- shared ingest substrate
- DCR engine
- PCR engine
- shared LLM gate

If a chapter or support doc still describes the older single-pipeline or PostgreSQL-centric model, it must be updated or explicitly marked historical.

---

## Folder Model

```text
new-docs/
├── agent_ref_index.md          # single root entrypoint
├── architecture/              # root authority docs
├── governance/                # authority registry, ownership, tests, failures
├── integration/               # Stoody + hub integration docs
├── references/                # current-state and historical references
├── hub/                       # protocol docs
├── api/                       # OpenAPI contracts
├── contracts/events/          # event contracts
└── chapters/                  # living explanatory docs
```

---

## Chapter Rules

Every chapter must:

1. name its authority source
2. avoid contradicting the root authority docs
3. reference specific test IDs and failure IDs when relevant
4. remain explanatory, not become a second contract source

---

## Tracker Document Rules

Documents that track implementation state or work sequencing (e.g., `chapters/BUILD_STATUS.md`, `IMPLEMENTATION_PLAN.md`) are classified as **execution/status documents**. They exist to report progress and organize task assignments.

Tracker documents:

- **may** report which tasks are complete, in progress, or not started
- **may** reference authoritative documents for task context
- **may not** define API shape, endpoint paths, request/response fields, or HTTP status codes (authority: `api/*.openapi.yaml`)
- **may not** define event payload shape or field names (authority: `contracts/events/*.schema.json`)
- **may not** define storage collections, indexes, field names, or write-once rules (authority: root architecture specs)
- **may not** define lifecycle state machines, ownership boundaries, or tamper-proofing rules (authority: root architecture specs + governance docs)

If a tracker document describes an API path, schema field, or storage layout that conflicts with an authoritative document, the tracker is wrong and must be updated to match the authority.

---

## Quality Gates

A documentation unit is complete only when:

- no stale PostgreSQL or legacy monolith assumptions remain
- paths match the current folder structure
- `GUIDE_RULE_DOCS/` is the only guide-doc home
- conduct-exam vs practice boundaries are preserved

---

## Current Priority

1. Root architecture docs
2. Governance docs
3. Integration docs
4. Chapters 01-03
5. Remaining chapters
