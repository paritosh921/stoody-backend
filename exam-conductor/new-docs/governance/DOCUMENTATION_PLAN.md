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
