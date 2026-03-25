# Dual-Mode Architecture

**Status:** ACTIVE  
**Authority:** Shared ingest substrate, DCR engine boundary, PCR engine boundary, and top-level routing.

---

## 1. Summary

ExamPen is not a monolith. It is composed of four distinct parts:

- **Shared ingest substrate**
- **DCR engine**
- **PCR engine**
- **Shared LLM gate**

The ingest substrate collects and stores canonical conducted-exam artifacts. The engines evaluate those artifacts. The gate controls all LLM use.

---

## 2. System Composition

```text
                     EXAMPEN PLATFORM

  ┌──────────────────────────────────────────────────────────┐
  │                 Shared Ingest Substrate                 │
  │                                                          │
  │  BLE Pen -> Hub -> Upload -> Canonical Exam Artifacts   │
  │  Camera/Scan ---------------> Canonical Exam Artifacts  │
  │                                                          │
  │  Stored in tenant/admin MongoDB with:                   │
  │  admin_id, student_id, pen_mac, timestamps, exam refs   │
  └──────────────────────────┬───────────────────────────────┘
                             │
              exam_type=dcr  │  exam_type=pcr
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
      ┌──────────────────┐      ┌─────────────────────┐
      │    DCR Engine    │      │     PCR Engine      │
      │                  │      │                     │
      │ Vision OCR       │      │ Vision OCR -> PageOCR│
      │ Template match   │      │ Segment + classify  │
      │ Answer key score │      │ Deep evaluation     │
      └────────┬─────────┘      └──────────┬──────────┘
               │                           │
               │ always                    │ always
               └──────────────┬────────────┘
                              ▼
                    ┌──────────────────┐
                    │    LLM Gate      │
                    │  token budgets   │
                    │  usage history   │
                    └──────────────────┘
```

---

## 3. Shared Ingest Substrate

### 3.1 Responsibilities

- Collect BLE pen or camera-originated exam artifacts
- Preserve conducted-exam provenance: `admin_id`, `student_id`, `pen_mac`, timestamps, exam refs
- Store canonical artifacts in tenant/admin MongoDB
- Provide artifact references or server-side fetchable records to the engines
- Remain independent from scoring and evaluation semantics

### 3.2 Non-Responsibilities

- No DCR template matching
- No PCR segmentation/evaluation
- No token budgeting
- No practice persistence redesign

### 3.3 Storage Placement

All conducted-exam artifacts are stored in tenant/admin MongoDB. Tutor visibility is derived from existing Stoody admin-owned student visibility rules.

---

## 4. DCR Engine

### 4.1 Purpose

DCR evaluates structured conducted exams using character/template matching. It is optimized for short answers, MCQs, fill-in-the-blank, and deterministic compare-heavy scoring.

### 4.2 Canonical Input

Server-side conducted-exam artifacts from the shared ingest substrate:

- `exam_id`
- `student_id`
- `admin_id`
- `pen_mac`
- `page_number`
- canonical stroke vectors

Minimum request shape for conducted-exam DCR work:

| Field | Type | Meaning |
|---|---|---|
| `submission_id` | string | Canonical DCR submission record |
| `exam_id` | string | Conducted exam identifier |
| `student_id` | string | Student identity |
| `question_id` | string | Target question |
| `page_number` | integer | Page reference |

### 4.3 Core Flow

```text
Student strokes (canonical)
    │
    ▼
Render to image (PIL)
    │
    ▼
Overlay on answer template
    │
    ▼
LLM Vision OCR (via shared gate, dcr_ai caller)
    │
    ▼
Extract: { Q1: "A", Q2: "B", ... }
    │
    ▼
Template matching (answer key comparison)
    │
    ▼
match_type + score
```

### 4.4 Gate Usage

DCR **always** uses the shared LLM gate for Vision OCR recognition. The `dcr_ai` caller_id is used for all recognition calls.

Gate use in DCR:

- Vision OCR of rendered stroke images overlaid on answer templates
- future language-specific enhancement (e.g., Devanagari via `dcr_devanagari`)

### 4.5 Canonical Output

| Field | Type | Meaning |
|---|---|---|
| `recognized_text` | string | Vision OCR output |
| `confidence` | number | Recognition confidence |
| `match_type` | enum | `exact_match`, `partial_match`, `numeric_match`, `no_match` |
| `score` | number | Awarded marks |
| `max_score` | number | Maximum marks |
| `audit_trail[]` | array | Append-only fallback or override history |

### 4.6 DCR Collections

MongoDB only. Collections live in the tenant/admin DB.

**DCR reads canonical artifacts from the shared ingest substrate** — it does NOT maintain a separate submissions collection. The ingest substrate owns `evalpen_submissions` and `evalpen_answer_pages` (see Section 3). DCR fetches submission metadata and per-page raw stroke data from these shared collections.

DCR owns only its results collection:

`exampen_dcr_results`

- `exam_id`
- `student_id`
- `question_id`
- `recognized_text`
- `confidence`
- `match_type`
- `score`
- `max_score`
- `audit_trail[]`

Indexes:

- `{ exam_id: 1, student_id: 1, question_id: 1 }` unique compound

---

## 5. PCR Engine

### 5.1 Purpose

PCR evaluates paginated subjective responses for conducted exams and exposes a stateless live-practice evaluation endpoint.

### 5.2 Conducted Exam Path

- Consumes canonical server-side artifacts from the ingest substrate
- Normalizes strokes or images into `PageOCR`
- Segments per-question responses
- Classifies content
- Evaluates through the shared gate

### 5.3 Practice Path

- Accepts live student practice input
- Evaluates synchronously
- Does **not** create new persistence inside ExamPen
- Existing practice persistence stays in the current backend

---

## 6. Shared LLM Gate

The gate is a separate cross-cutting module used by both engines.

It owns:

- allowed caller identities
- per-call token limits
- daily/weekly/monthly budgets
- append-only token logging
- usage rollups
- `/v1/evalpen/usage/*`

No engine or endpoint may call an LLM provider directly outside this contract.

---

## 7. Integration Surfaces

The active integration surfaces are:

- shared conducted-exam ingest: `api/stroke-ingest.openapi.yaml`, `api/copy-upload.openapi.yaml`
- conducted-exam orchestration: `api/exam-orch.openapi.yaml`
- PCR conducted-exam work: `api/eval-submissions.openapi.yaml`, `api/eval-evaluate.openapi.yaml`, `api/eval-solutions.openapi.yaml`
- PCR practice work: `api/eval-practice.openapi.yaml`
- shared gate usage: `api/eval-usage.openapi.yaml`
- review and publication: `api/review.openapi.yaml`, `api/score-engine.openapi.yaml`

Conducted-exam flows are reference-driven and server-side. Practice remains synchronous and stateless from the ExamPen side.

---

## 8. Hard Boundaries

1. The ingest substrate is the only owner of canonical conducted-exam artifact persistence.
2. DCR and PCR read canonical artifacts; they do not accept client-submitted answer text as authoritative for conducted exams.
3. Practice persistence remains outside ExamPen.
4. All LLM-mediated work goes through the shared gate.
5. `archiveDCR/` is not an authority source.

---

## 9. Related Authorities

- PCR internals: `architecture/PCR_EVAL_ENGINE_SPEC.md`
- Gate contract: `architecture/LLM_GATE_SPEC.md`
- Integrity/audit rules: `architecture/TAMPER_PROOF_SPEC.md`
- Ownership map: `governance/STATE_OWNERSHIP_MAP.md`
- Conducted-exam ingest contract: `api/stroke-ingest.openapi.yaml`
