# TEST_SUITE_SPEC.md
# ExamPen — Test Suite Specification

Reference: `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/PCR_EVAL_ENGINE_SPEC.md`, `architecture/LLM_GATE_SPEC.md`, `architecture/TAMPER_PROOF_SPEC.md`  
Doctrine source: `GUIDE_RULE_DOCS/SOFTWARE_DEVELOPMENT_DOCTRINE.md` §6

---

## 1. Validation Levels

| Level | Meaning |
|---|---|
| `L1` | Build / parse verified |
| `L2` | Static checks / lint / schema checks |
| `L3` | Unit tests |
| `L4` | Integration tests |
| `L5` | End-to-end tests |
| `L6` | Hardware-in-loop tests |
| `L7` | Field trial validation |

---

## 2. Core Test Families

### 2.1 Shared Ingest Substrate

| ID | What It Covers |
|---|---|
| `U-ING-01` | conducted-exam artifact stored with `admin_id`, `student_id`, `pen_mac`, timestamps |
| `U-ING-02` | content hash generated for conducted-exam artifact |
| `I-ING-01` | hub/upload path writes canonical artifact once |
| `I-ING-02` | duplicate upload is idempotent |
| `E2E-ING-01` | conducted exam artifact becomes available to the correct engine based on `exam_type` |

### 2.2 DCR Engine

| ID | What It Covers |
|---|---|
| `U-DCR-01` | Vision OCR recognizer output normalized into DCR recognition input |
| `U-DCR-02` | exact/partial/numeric/no-match logic |
| `U-DCR-03` | DCR recognition routes through the shared LLM gate (Vision OCR) |
| `I-DCR-01` | canonical artifact -> DCR result commit |
| `I-DCR-02` | gate call logged with dcr_ai caller_id |
| `E2E-DCR-01` | conducted DCR exam -> canonical artifact -> DCR score |

### 2.3 PCR Engine

| ID | What It Covers |
|---|---|
| `U-SEG-01` | boundary detection |
| `U-SEG-02` | question marker parsing |
| `U-SEG-03` | cross-page stitching |
| `U-CCLS-01` | content classification |
| `U-CLUB-01` | clubbed response heuristics |
| `U-EVAL-01` | eval result parsing and scoring envelope |
| `I-PCR-01` | conducted artifact -> PageOCR -> detected responses |
| `I-PCR-02` | blocking flags prevent auto-eval |
| `I-PCR-03` | practice call is synchronous and creates no new PCR persistence |
| `E2E-PCR-01` | conducted PCR exam -> segmentation -> evaluation |

### 2.4 LLM Gate

| ID | What It Covers |
|---|---|
| `U-GATE-01` | allowed caller validation |
| `U-GATE-02` | per-call input/output limits |
| `U-GATE-03` | daily/weekly/monthly budget checks |
| `U-GATE-04` | append-only token log shape |
| `I-GATE-01` | DCR fallback call flows through gate |
| `I-GATE-02` | PCR evaluation call flows through gate |
| `I-GATE-03` | usage API reflects stored config and usage |

### 2.5 Tamper-Proofing

| ID | What It Covers |
|---|---|
| `U-TAMP-01` | conducted-exam evaluation rejects client-submitted authoritative answer text |
| `U-TAMP-02` | canonical artifact hash preserved |
| `I-TAMP-01` | conducted DCR eval fetches server-side artifact |
| `I-TAMP-02` | conducted PCR eval fetches server-side artifact |
| `I-TAMP-03` | append-only audit entries are created for overrides / resolutions |

### 2.6 Stoody / Visibility / Practice Boundary

| ID | What It Covers |
|---|---|
| `U-STOODY-01` | tutor visibility follows existing admin-owned student visibility rules |
| `I-STOODY-01` | practice persistence remains in existing backend path |
| `I-STOODY-02` | exam-conducted data remains tenant/admin scoped |

---

## 3. Mandatory Scenarios

1. Conducted DCR exam with BLE pen artifacts.
2. Conducted PCR exam with BLE pen artifacts.
3. Conducted PCR exam with camera/scan artifacts.
4. PCR live practice evaluation with no new persistence.
5. Gate budget exhaustion mid-flow.
6. Teacher override with preserved audit trail.

---

## 4. Hard Requirements

1. No PostgreSQL-only assumptions in tests.
2. No test should require practice persistence changes.
3. All LLM-mediated tests must flow through the gate.
