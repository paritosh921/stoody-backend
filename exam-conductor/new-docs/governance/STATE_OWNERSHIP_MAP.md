# STATE_OWNERSHIP_MAP.md
# ExamPen — State Ownership and Boundaries

Reference: `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/PCR_EVAL_ENGINE_SPEC.md`, `architecture/LLM_GATE_SPEC.md`, `architecture/TAMPER_PROOF_SPEC.md`  
Doctrine source: `GUIDE_RULE_DOCS/SOFTWARE_DEVELOPMENT_DOCTRINE.md` §4 Rule 1

---

## Purpose

Every critical state has one writable owner. In the current architecture the primary owners are:

- shared ingest substrate
- DCR engine
- PCR engine
- LLM gate
- existing Stoody practice backend (for practice persistence only)

---

## 1. Critical State Ownership

| Critical State | Writable Owner | Readers / Consumers | Transactional Boundary | Notes |
|---|---|---|---|---|
| Exam metadata and `exam_type` routing | Exam orchestration / backend exam owner | Ingest substrate, DCR engine, PCR engine | Exam created or updated | Determines DCR vs PCR routing. |
| Canonical conducted-exam pen artifacts | Shared ingest substrate | DCR engine, PCR engine | Artifact persisted with provenance and content hash | Includes `admin_id`, `student_id`, `pen_mac`, timestamps. |
| Canonical conducted-exam camera artifacts | Shared ingest substrate | PCR engine | Artifact persisted with provenance and content hash | Camera path is PCR-only. |
| DCR recognized text | DCR engine | Review/reporting readers | DCR recognition commit | Derived from canonical conducted-exam artifacts. |
| DCR match result + score | DCR engine | Review/reporting readers | DCR scoring commit | Deterministic default path. |
| PCR `PageOCR` normalization | PCR engine | PCR segmentation pipeline | PCR normalization commit | Internal engine state. |
| PCR detected responses and flags | PCR engine | Review/reporting readers | Segmentation/classification commit | Conducted-exam only. |
| PCR evaluations | PCR engine | Review/reporting readers | Evaluation commit | Includes score, feedback, step marks, token refs. |
| LLM gate config | LLM gate | DCR engine, PCR engine, usage APIs | Config update | MongoDB only. |
| LLM gate token usage log | LLM gate | Usage/reporting readers | Append-only gate call log | Shared across DCR and PCR. |
| LLM gate rollups | LLM gate | Usage/reporting readers | Rollup job | Daily/weekly/monthly aggregates. |
| Practice attempt persistence | Existing Stoody practice backend | PCR practice endpoint caller | Existing practice save flow | Explicitly out of ExamPen persistence scope. |
| Tutor visibility over exam data | Existing Stoody/admin visibility model | Tutor-facing readers | Existing tutor scoping rules | Read-only derivation, not a second owner. |

---

## 2. Read/Write Boundary Rules

### Must Be Read-Only

- tutor/student views of conducted-exam results
- agent routing and governance docs
- current-state reference documents
- engines reading canonical conducted-exam artifacts

### Allowed Writers

- shared ingest substrate writes conducted-exam artifacts
- DCR engine writes DCR results
- PCR engine writes PCR detection/evaluation state
- LLM gate writes gate config/log/rollups
- existing practice backend writes practice persistence

---

## 3. Hard Violations

1. PCR practice endpoint creating new conducted-exam collections.
2. DCR or PCR accepting client-supplied answer text as authoritative for conducted exams.
3. Any component calling an LLM provider directly outside the gate.
4. Any layer other than the ingest substrate mutating canonical conducted-exam raw artifacts.
5. Any component treating tutor visibility as a second ownership source.

---

## 4. Transactional Boundaries

| Boundary | What Must Be Atomic |
|---|---|
| Conducted-exam artifact ingest | canonical artifact persistence + provenance metadata |
| DCR result creation | recognized text + match result + score |
| PCR segmentation result creation | detected response + flags + provenance refs |
| PCR evaluation creation | evaluation output + gate usage refs + audit metadata |
| Gate logging | provider response + append-only usage log |

---

## 5. Ownership Declaration Template

Every implementation unit should declare:

```markdown
## Ownership Declaration
- Writes:
- Reads from:
- Never writes to:
- Transactional boundaries:
```
