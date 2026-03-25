# FAILURE_MITIGATION_REGISTER.md
# ExamPen — Failure Mitigation Register

Reference: `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/PCR_EVAL_ENGINE_SPEC.md`, `architecture/LLM_GATE_SPEC.md`, `architecture/TAMPER_PROOF_SPEC.md`  
Doctrine source: `GUIDE_RULE_DOCS/SYSTEM_DESIGN_GUIDELINE.md` §7

---

## 1. Shared Ingest Substrate

| ID | Failure Mode | Mitigation |
|---|---|---|
| `ING-01` | Conducted-exam artifact lost before canonical persistence | write-once canonical persistence with provenance and retryable upload flow |
| `ING-02` | Artifact mapped to wrong student or exam | persist `admin_id`, `student_id`, `pen_mac`, timestamps, exam refs together |
| `ING-03` | Duplicate artifact upload | idempotent ingest keyed by canonical artifact identity |

## 2. DCR Engine

| ID | Failure Mode | Mitigation |
|---|---|---|
| `DCR-01` | HWR confidence too low | route to configured fallback via gate or review path |
| `DCR-02` | Numeric template mismatch for semantically equivalent answers | numeric tolerance and normalization rules |
| `DCR-03` | DCR accidentally depends on deep PCR semantics | keep DCR default path deterministic and contract-bound |

## 3. PCR Engine

| ID | Failure Mode | Mitigation |
|---|---|---|
| `PCR-01` | Boundary/marker detection failure | flags + review queue |
| `PCR-02` | Clubbed responses undetected | multiple heuristics plus optional LLM-assisted topic discontinuity check |
| `PCR-03` | Diagram-heavy answer auto-scored incorrectly | content classification blocks or downweights unsafe auto-eval |
| `PCR-04` | Practice path starts creating persistence | explicit stateless endpoint contract |

## 4. LLM Gate

| ID | Failure Mode | Mitigation |
|---|---|---|
| `GATE-01` | Budget exhaustion | explicit refusal semantics and usage API visibility |
| `GATE-02` | New caller bypasses gate | allowed-caller registration rule |
| `GATE-03` | Token logging missing or inconsistent | append-only log + rollups in tenant MongoDB |

## 5. Integrity and Access

| ID | Failure Mode | Mitigation |
|---|---|---|
| `TAMP-01` | Client substitutes corrected answer text in conducted exam flow | server-side fetch of canonical artifacts/text |
| `TAMP-02` | Raw conducted-exam artifact silently altered | content hash + write-once raw artifact model |
| `STOODY-01` | Tutor sees students outside allowed scope | existing admin-owned tutor visibility model only |

---

## 6. Residual Rule

Any new mitigation must remain consistent with:

- MongoDB-only storage
- independent ingest substrate
- independent DCR/PCR engines
- untouched practice persistence
