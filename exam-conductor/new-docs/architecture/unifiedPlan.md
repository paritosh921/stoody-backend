# ExamPen Unified Architecture Plan — DCR + PCR

**Version:** 2.1
**Date:** 2026-03-24
**Status:** SUPERSEDED — HISTORICAL TRANSITION DOCUMENT
**Supersedes:** unifiedPlan v1.0, v2.0
**Delta from v2.0:** Removed practice-storage contradiction (no new persistence for practice mode); moved usage API under `/v1/evalpen/` namespace; declared this document as a transition document that yields authority once root specs are created; fixed DCR-to-gate diagram to show conditional routing.

**Document Role:** This file is a historical transition plan, not an active spec. The authoritative documents are now `DUAL_MODE_ARCHITECTURE.md`, `PCR_EVAL_ENGINE_SPEC.md`, `LLM_GATE_SPEC.md`, and `TAMPER_PROOF_SPEC.md`, plus the matching API and event contracts. Implementers must build against those root specs, not against this file. Keep this document only for migration rationale and decision traceability.

This plan intentionally avoids a monolithic "one evaluator service owns everything" model. ExamPen is composed of:

- a shared hub/collection substrate that collects and stores canonical exam-conducted artifacts
- a DCR engine that evaluates template-bound structured responses
- a PCR engine that evaluates paginated subjective responses
- a shared LLM Gate used by both engines

The hub/collection substrate and the evaluators are separate concerns with separate ownership boundaries.

---

## 1. System Composition

ExamPen is a modular assessment platform with four distinct parts:

- **Hub / Collection Substrate**: Collects pen-originated or camera-originated exam artifacts, stores them canonically in the tenant/admin MongoDB, and preserves student-to-exam-to-pen provenance.
- **DCR Engine**: Consumes canonical exam artifacts and performs deterministic character/template matching for structured exams.
- **PCR Engine**: Consumes canonical exam artifacts for conducted exams and exposes a stateless live-practice evaluation endpoint for practice mode.
- **LLM Gate**: Shared control module for all LLM-mediated calls across DCR and PCR, with centralized token accounting and budget enforcement.

The hub/substrate is not part of either evaluation engine. DCR and PCR plug into canonical stored artifacts through API and storage contracts.

---

## 2. Constraints

- **MongoDB only** — all storage (including LLM Gate) uses MongoDB collections in the tenant DB. No PostgreSQL.
- **DCR and PCR are engines defined by this architecture** — not by any legacy code or backup folders.
- **Practice-mode persistence stays untouched** — PCR exposes a practice evaluation endpoint; the current backend calls it and continues storing results in its own collections. The practice endpoint creates no new persistent artifacts.
- **Tutor access** uses the existing admin-owned student visibility model.
- **No premature code/packaging decisions** — this document defines contracts, ownership, and storage. Engines may be embedded in the current backend runtime or deployed separately; that choice is deferred.

---

## 3. Document Authority Matrix

Each topic has exactly one authoritative document. When documents conflict, the authority wins.

| Topic | Authoritative Document | Chapters/Prose Role |
|-------|----------------------|---------------------|
| Dual-mode framing, shared collection layer, routing | `architecture/DUAL_MODE_ARCHITECTURE.md` | `chapters/01_SYSTEM_OVERVIEW.md` references it |
| DCR input/output contract, template matching semantics | `architecture/DUAL_MODE_ARCHITECTURE.md` §DCR | — |
| PCR pipeline behavior, segmentation, classification, eval flow | `architecture/PCR_EVAL_ENGINE_SPEC.md` | Chapters explain context, spec owns detail |
| LLM Gate contract, callers, budget, token storage | `architecture/LLM_GATE_SPEC.md` | — |
| Immutability rules, server-side fetch, audit trail | `architecture/TAMPER_PROOF_SPEC.md` | — |
| PCR API request/response shapes | `api/eval-*.openapi.yaml` | Spec describes behavior; OpenAPI owns wire format |
| PCR async event payloads | `contracts/events/eval.*.schema.json` | Spec describes triggers; schemas own payload shape |
| State ownership per subsystem/engine | `governance/STATE_OWNERSHIP_MAP.md` | — |
| Failure modes and mitigations | `governance/FAILURE_MITIGATION_REGISTER.md` | Specs reference by ID only |
| Test identifiers | `governance/TEST_SUITE_SPEC.md` | Specs reference by ID only |

**Rule**: Chapters explain *why* and *when*. Specs and contracts own *what exactly*. If a chapter and a spec disagree on a data shape, field name, or state transition — the spec wins.

---

## 4. Shared Collection Layer

The RPi hub and related upload paths form a shared ingestion substrate. Their job is to collect and persist canonical exam-conducted artifacts under the admin-owned tenant data model. They do not evaluate responses.

The `exam_type` field in exam metadata determines which engine consumes the stored artifacts next.

```
                      SHARED COLLECTION LAYER
  ┌──────────────────────────────────────────────────────┐
  │  BLE Pen → Hub (dual-write SD+USB) → Hub Uplink     │
  │                                                      │
  │  Camera/Scan ─────────────────────────┐              │
  └──────────────────┬────────────────────┼──────────────┘
                     │                    │
           exam_type in metadata          │
                     │                    │
        ┌────────────┴───────────┐        │
        ▼                        ▼        ▼
  ┌───────────┐           ┌───────────────────┐
  │ DCR Mode  │           │     PCR Mode      │
  │           │           │                   │
  │ Strokes   │           │ Strokes OR Images │
  │    │      │           │    │              │
  │    ▼      │           │    ▼              │
  │ ONNX HWR │           │ OCR (ONNX/Paddle) │
  │    │      │           │    │              │
  │    ▼      │           │    ▼              │
  │ Character │           │ PageOCR           │
  │ Template  │           │    │              │
  │ Matching  │           │    ▼              │
  │    │      │           │ Segmentation      │
  │    ▼      │           │    │              │
  │ Score     │           │    ▼              │
  │           │           │ Content Classify  │
  │           │           │    │              │
  │           │           │    ▼              │
  │           │           │ LLM Eval (Gate)   │
  │           │           │    │              │
  │           │           │    ▼              │
  │           │           │ Step Marks +      │
  │           │           │ Feedback + Score  │
  └───────────┘           └───────────────────┘
        │                         │
        │ (default: no gate)      │ (always through gate)
        │                         │
        │  ┌─ ─ ─ ─ ─ ─ ─ ┐      │
        │    fallback only         │
        └──▶ (low HWR conf, ├─────┘
             assisted mode)        │
           └─ ─ ─ ─ ─ ─ ─ ┘      │
                     │             │
                     ▼             ▼
        ╔═══════════════════════════╗
        ║       LLM GATE           ║
        ║                          ║
        ║  PCR: always             ║
        ║  DCR: only on fallback   ║
        ╚═══════════════════════════╝
```

Camera uploads (PCR-only) go directly to `POST /v1/evalpen/submissions` with `source: "camera"`.

### 4.1 Ownership Boundary

| Responsibility | Owner |
|---|---|
| Capture pen strokes, sync, upload, and canonical storage | Hub / collection substrate |
| Map exam artifacts to `admin_id`, `student_id`, `pen_mac`, timestamps, exam context | Hub / collection substrate |
| DCR recognition and structured scoring | DCR engine |
| PCR OCR normalization, segmentation, classification, deep evaluation | PCR engine |
| Token budget enforcement and token history | LLM Gate |

The evaluator engines must consume canonical server-side artifacts or artifact references. They must not take client-supplied "student answer text" as authoritative input for conducted exams.

---

## 5. DCR Mode — Contract

DCR must be defined with the same rigor as PCR. This section establishes DCR's canonical contracts.

### 5.1 Input Artifact

| Field | Type | Description |
|-------|------|-------------|
| `exam_id` | string | Exam identifier |
| `student_id` | string | Student identifier |
| `pen_mac` | string | BLE pen MAC address |
| `page_number` | int | Page within answer booklet |
| `strokes` | array | Canonical stroke vectors `(x, y, t, pressure)` per page |

Source: Hub uplink batch upload. Only BLE pen input — no camera path for DCR.

### 5.2 Processing Pipeline

```
  Raw strokes (canonical vectors)
       │
       ▼
  ┌──────────────────────┐
  │  ONNX HWR Model      │  Handwriting → character sequences
  │                      │  Output: recognized_text + confidence
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Template Matcher     │  Compare recognized_text against
  │                      │  answer_template for question
  │                      │
  │  Match types:        │
  │  ├─ exact_match      │  Text matches template exactly
  │  ├─ partial_match    │  Subset of expected content present
  │  ├─ numeric_match    │  Numeric value within tolerance
  │  └─ no_match         │  No recognizable match
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Score Assignment     │  Based on match type + question marks
  │                      │  exact_match → full marks
  │                      │  partial_match → prorated
  │                      │  numeric_match → full if within tolerance
  │                      │  no_match → 0
  └──────────────────────┘
```

### 5.3 Output Artifact

| Field | Type | Description |
|-------|------|-------------|
| `exam_id` | string | — |
| `student_id` | string | — |
| `question_id` | string | — |
| `recognized_text` | string | HWR output |
| `confidence` | float | 0.0–1.0 |
| `match_type` | enum | `exact_match`, `partial_match`, `numeric_match`, `no_match` |
| `score` | int | Awarded marks |
| `max_score` | int | Maximum possible |

### 5.4 Ownership Boundaries

| State | Writable Owner |
|-------|---------------|
| Raw stroke data | Hub / collection substrate (write-once) |
| HWR recognized text | DCR engine |
| Template match result + score | DCR engine |

### 5.5 LLM Gate Usage

DCR does **not** use the LLM Gate in its default path. Template matching is deterministic (ONNX model + string comparison). The gate is used by DCR only when:
- HWR confidence falls below threshold and an LLM fallback is configured
- A future Devanagari recognition enhancement requires LLM assistance
- Admin explicitly enables LLM-assisted scoring for a DCR exam

When DCR calls the gate, it uses `caller: "dcr_ai"` and is subject to the same budget limits.

### 5.6 DCR Collections

All in the per-tenant DB (`skb_<tenant>`):

| Collection | Purpose |
|-----------|---------|
| `exampen_dcr_submissions` | Raw stroke submissions (write-once, immutable) |
| `exampen_dcr_results` | HWR output + template match + score per question |

---

## 6. PCR End-to-End Pipeline

### 6.1 Dual Input Convergence

```
  ┌─────────────────┐      ┌──────────────────┐
  │ Path A: BLE Pen │      │ Path B: Camera   │
  │ Stroke vectors  │      │ JPEG/PNG images  │
  │ (x, y, t, p)   │      │                  │
  │    │            │      │    │             │
  │    ▼            │      │    ▼             │
  │ ONNX HWR       │      │ Preprocess       │
  │                 │      │ (deskew/crop/    │
  │    │            │      │  binarize)       │
  │    ▼            │      │    │             │
  │ TextBlocks +    │      │    ▼             │
  │ bounding boxes  │      │ PaddleOCR        │
  │ + confidence    │      │    │             │
  └────────┬────────┘      └────┼─────────────┘
           │                    │
           └────────┬───────────┘
                    ▼
        ┌──────────────────────┐
        │  Unified PageOCR     │
        │  List[TextBlock]     │
        │  per page            │
        └──────────┬───────────┘
                   │
                   ▼
```

Both paths converge to a unified `PageOCR` object: list of text blocks with bounding boxes and confidence scores, one per page.

### 6.2 Segmentation Pipeline

```
        ┌──────────────────────┐
        │  Boundary Detector   │  Pen: stroke geometry filter
        │                      │  Cam: Canny → HoughLinesP
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Q Marker Parser     │  Regex: Q.No X.Ans variants
        │                      │  Post-OCR fixes: l→1, O→0
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Segmenter           │  Boundaries + markers
        │                      │  → per-question segments
        │                      │  Cross-page stitching
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Content Classifier  │  Per segment:
        │                      │  TEXT_ONLY (>85% text)
        │                      │  MIXED (40-85% text)
        │                      │  DIAGRAM_HEAVY (<40%)
        │                      │  TABLE_PRESENT (grid)
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Clubbed Detector    │  H1: multiple markers
        │                      │  H2: length anomaly
        │                      │  H3: missing question
        │                      │  H4: topic discontinuity
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Detected Responses  │  With embedded flags[]
        │  stored immutably    │
        └──────────┬───────────┘
                   │
          ┌────────┴─────────┐
          ▼                  ▼
    No blocking         Has blocking
    flags               flags
          │                  │
          ▼                  ▼
    ╔═══════════╗      Teacher
    ║ LLM GATE  ║      Review Queue
    ║ eval call ║
    ╚═════╤═════╝
          │
          ▼
    EvalResult:
    ├─ total_score
    ├─ max_score
    ├─ scoreable_max
    ├─ step_marks[]
    ├─ feedback
    ├─ reference_solution
    ├─ content_flags[]
    └─ token_usage {}
```

### 6.3 Student Instructions (Enforced)

Students delimit responses with double horizontal lines and mark each response with `Q.No X.Ans`:

```
  ════════════════════
  ════════════════════
  Q.No 1.Ans
  [response to Q1]
  ════════════════════     ← shared delimiter
  ════════════════════
  Q.No 2.Ans
  [response to Q2]
  ════════════════════
  ════════════════════
```

### 6.4 Boundary Detection Parameters

**BLE Pen Path**:

| Parameter | Constraint |
|-----------|-----------|
| Slope | < ±10° from horizontal |
| Length | > 40% of page width |
| Y-gap between pair | 2–15mm |
| Temporal proximity | Both drawn within ~3 seconds |
| Horizontal overlap | > 70% |

**Camera Path**: Canny → HoughLinesP → filter horizontal → sort by Y → pair adjacent. `minLineLength = 0.4 × image_width`, slope ≤10°, Y-gap 2–15mm.

### 6.5 Q Marker Detection

Regex (case-insensitive, post-OCR):

```
/Q\.?\s*(?:No|no)\.?\s*(\d{1,3})\s*(?:[\.\(\s]*([a-z]|[ivx]+|[A-Z])[\)\.]?)?\s*\.?\s*(?:Ans|ans|ANS)\.?/i
```

Captures: `\1` = question number, `\2` = sub-part. Output: `Q1`, `Q1a`, `Q3ii`.

Post-OCR fixes in marker context: `l`→`1`, `O`→`0`, `I`→`1`.

### 6.6 Cross-Page Stitching

| Page Transition | Meaning |
|----------------|---------|
| Page N ends without closing boundary | Response continues on N+1 |
| Page N+1 starts with boundary | Previous response closed at N bottom |
| Page N+1 starts with content, no boundary | Continuation |
| Page N+1 has Q marker without boundary | Student forgot lines. Associate, flag. |

### 6.7 Content Classification

| Type | Definition | Eval Path |
|------|-----------|-----------|
| `TEXT_ONLY` | > 85% text coverage, no significant non-text features | Auto-evaluate (normal) |
| `MIXED` | 40–85% text, diagram/figure present alongside text | Evaluate text portion. Flag diagram as unscored. |
| `DIAGRAM_HEAVY` | < 40% text, diagram/figure dominant | Skip auto-eval. Manual review required. |
| `TABLE_PRESENT` | Grid/tabular structure detected | Route to table eval template if available, else flag. |

Diagram weight prorating: `scoreable_marks = max_marks × (1 - diagram_weight)`

### 6.8 Clubbed Response Detection

| Heuristic | Detection | Confidence |
|-----------|----------|-----------|
| H1: Multiple markers | Regex finds >1 Q marker in one segment | Very high |
| H2: Length anomaly | Word count > expected_max × 2.5 for question marks/type | Medium |
| H3: Missing question | Exam manifest question not in any segment | High |
| H4: Topic discontinuity | LLM check (goes through LLM Gate, counts against budget) | Medium-high |

---

## 7. Eval Core

### 7.1 Solution Cache Strategy

```
  Eval request arrives
       │
       ▼
  ┌─────────────────┐
  │ Solution cache   │
  │ lookup by        │
  │ question_id      │
  └────────┬────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
  CACHE HIT    CACHE MISS
    │             │
    ▼             ▼
  Compare-only  Router selects
  (Haiku via    model tier
   Gate)        (Haiku/Sonnet/Opus
    │            via Gate)
    │             │
    │             ▼
    │          Generate solution
    │          + evaluate
    │          + store in cache
    │             │
    └──────┬──────┘
           ▼
     Build prompt
     from template
           │
           ▼
     LLM GATE call
           │
           ▼
     EvalResult
```

### 7.2 Complexity Router

| Complexity | Model | Trigger |
|-----------|-------|---------|
| Cache-hit (any) | Haiku | Always — compare only |
| L1 cache-miss | Haiku | Factual, single-step |
| L2 cache-miss | Sonnet | Multi-step, short-answer |
| L3 cache-miss | Opus/Sonnet | Essay, open-ended |

### 7.3 Subject-Specific Eval Templates

| Template ID | Subjects | Step Marks |
|-------------|----------|-----------|
| `stepwise_numerical` | Physics, Chemistry, Math, Accountancy numericals | Yes |
| `essay_rubric` | English, Hindi essays, History/Geo long answers | No (rubric categories) |
| `factual_recall` | Biology, History, Geography short answers | No |
| `keyword_coverage` | Science definitions, bio processes | No |
| `ledger_tabular` | Accountancy ledger/journal entries | Yes |
| `proof_derivation` | Math proofs, Physics derivations | Yes |

### 7.4 OCR Error Tolerance

Prompt instructs LLM to tolerate: `l`↔`1`, missing superscripts, `rn`→`m`, Devanagari matra errors.

---

## 8. Unified Flag System

### 8.1 Flag Schema

```json
{
  "flag_id": "FLG-001",
  "response_id": "RESP-001",
  "source": "segmenter | content_classifier | clubbed_detector | ocr | eval | llm_gate",
  "flag_type": "...",
  "severity": "blocking | warning | info",
  "reason": "Human-readable explanation",
  "suggested_action": "...",
  "metadata": { }
}
```

### 8.2 Complete Flag Type Registry

| Source | Flag Type | Severity |
|--------|----------|----------|
| segmenter | `no_question_marker` | warning |
| segmenter | `no_boundary_detected` | warning |
| segmenter | `boundary_only_no_marker` | warning |
| segmenter | `low_segmentation_confidence` | warning |
| content_classifier | `diagram_present` | info |
| content_classifier | `diagram_heavy_content` | blocking |
| content_classifier | `table_detected` | info |
| content_classifier | `expected_diagram_missing` | warning |
| clubbed_detector | `clubbed_multiple_markers` | blocking |
| clubbed_detector | `clubbed_length_anomaly` | warning |
| clubbed_detector | `clubbed_missing_question` | warning |
| clubbed_detector | `clubbed_topic_discontinuity` | warning |
| ocr | `low_ocr_confidence` | warning |
| ocr | `ocr_rejected` | blocking |
| eval | `partial_eval_diagram_excluded` | info |
| eval | `llm_score_divergence` | warning |
| llm_gate | `budget_warning_80pct` | warning |
| llm_gate | `budget_exhausted` | blocking |

### 8.3 Flag Severity Behavior

| Severity | Meaning | API Behavior |
|----------|---------|-------------|
| `blocking` | Cannot auto-evaluate | `eval_status: "blocked"` — teacher must resolve |
| `warning` | Can evaluate, result may be unreliable | `eval_status: "evaluated_with_warnings"` |
| `info` | FYI only | `eval_status: "evaluated"` |

---

## 9. Three Ingestion Paths — One Eval Core

```
  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐
  │ OFFLINE EXAM│  │ LIVE PRACTICE│  │ CAMERA UPLOAD    │
  │ (hub sync)  │  │ (student)   │  │ (scan/photo)     │
  │             │  │             │  │                  │
  │ Hub batch   │  │ Canvas/photo│  │ JPEG/PNG pages   │
  │ upload      │  │ per question│  │ per submission   │
  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘
         │               │                   │
         ▼               ▼                   ▼
  POST /submissions  POST /practice/   POST /submissions
  (bulk, async)      evaluate          (single, async)
         │          (sync, stateless)        │
         │               │                   │
         ▼               │                   ▼
  ┌──────────────┐       │           ┌──────────────┐
  │ Full pipeline│       │           │ Full pipeline│
  │ OCR          │       │           │ OCR          │
  │ Segment      │       │           │ Segment      │
  │ Classify     │       │           │ Classify     │
  │ Flag         │       │           │ Flag         │
  └──────┬───────┘       │           └──────┬───────┘
         │               │                  │
         ▼               ▼                  ▼
  ┌──────────────────────────────────────────────┐
  │           SHARED EVAL CORE                    │
  │                                              │
  │  Same solution cache                         │
  │  Same subject templates                      │
  │  Same complexity router                      │
  │  Same content-type routing                   │
  │                                              │
  │  ALL calls through LLM Gate                  │
  │  Offline: inputs stored immutably first      │
  │  Offline: evals fetch text server-side       │
  │  Practice: stateless, no new persistence     │
  └──────────────────────────────────────────────┘
         │               │                  │
         ▼               ▼                  ▼
  Teacher polls    Return result      Teacher polls
  + batch eval     synchronously      + batch eval
```

### 9.1 Offline Exam Sync (Hub Batch Upload)

```
Hub finishes exam → Hub Uplink batches all pages
  → POST /v1/evalpen/submissions (bulk)
  → Store raw immutably (SHA-256 hash + write-once)
  → Async: OCR → Segmentation → Content classify → Flag
  → detected_responses[] stored with flags[]
  → Teacher polls GET /v1/evalpen/submissions/{id}/responses
  → Teacher triggers POST /v1/evalpen/evaluate/batch
  → Results via GET /v1/evalpen/evaluations
```

### 9.2 Live Practice Mode (Synchronous, Stateless)

```
  ┌────────────────────────────────────────────────────────┐
  │  CURRENT BACKEND (practice_async.py)                   │
  │                                                        │
  │  Student submits answer (canvas/photo/text)            │
  │       │                                                │
  │       │  (later: connect to PCR practice endpoint)     │
  │       ▼                                                │
  └───────┬────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────────┐
  │  PCR PRACTICE ENDPOINT                                 │
  │  POST /v1/evalpen/practice/evaluate                    │
  │                                                        │
  │  Input:                                                │
  │  ├─ question_id                                        │
  │  ├─ source: "canvas" | "camera" | "text"               │
  │  ├─ canvas_data (base64) OR image(s) OR text           │
  │  └─ exam context (subject, grade, question metadata)   │
  │                                                        │
  │  Pipeline (synchronous, stateless):                    │
  │  ┌──────────────────────────────────────────────┐      │
  │  │ 1. Receive raw input (no new persistence)    │      │
  │  │    Practice does NOT write to                │      │
  │  │    evalpen_submissions or any new collection │      │
  │  ├──────────────────────────────────────────────┤      │
  │  │ 2. OCR (if image/canvas input)               │      │
  │  │    PaddleOCR or GPT-4o Vision via Gate       │      │
  │  │    → detected_text + confidence              │      │
  │  ├──────────────────────────────────────────────┤      │
  │  │ 3. Skip segmentation                         │      │
  │  │    (single question — no boundaries needed)  │      │
  │  ├──────────────────────────────────────────────┤      │
  │  │ 4. Content classification                    │      │
  │  │    TEXT_ONLY / MIXED / DIAGRAM_HEAVY         │      │
  │  │    (determines eval routing)                 │      │
  │  ├──────────────────────────────────────────────┤      │
  │  │ 5. Eval via LLM Gate                         │      │
  │  │    Same eval core as offline exam mode        │      │
  │  │    Same solution cache, same templates        │      │
  │  │    Same complexity router (Haiku/Sonnet/Opus) │      │
  │  └──────────────────────────────────────────────┘      │
  │                                                        │
  │  Output (synchronous response):                        │
  │  ├─ total_score, max_score                             │
  │  ├─ step_marks[] (per-step breakdown)                  │
  │  ├─ feedback (overall + per-step justification)        │
  │  ├─ reference_solution                                 │
  │  ├─ detected_text (what OCR extracted)                 │
  │  ├─ content_type                                       │
  │  ├─ flags[] (warnings only — blocking flags            │
  │  │   trigger manual_review_required response)          │
  │  └─ token_usage {} (model, tokens, cost)               │
  └────────────────────────────────────────────────────────┘
```

**Key design points**:
- Same eval core, same LLM Gate, same solution cache, same templates as offline exam mode
- Only difference: synchronous (request → wait → response) vs offline async (submit → poll)
- Segmentation skipped (single question per call, no boundary/marker parsing needed)
- Content classification still runs (routes DIAGRAM_HEAVY to `manual_review_required`)
- **No new persistence**: practice calls do not write to `evalpen_submissions` or any new collection. The endpoint is stateless — receive input, evaluate, return result. Token usage is still logged via the gate (that is gate persistence, not practice persistence).
- Existing backend practice persistence completely untouched — the current backend calls this PCR endpoint and continues storing results in its own collections as before

### 9.3 Camera Upload (Async)

Same as offline exam but submitted directly via HTTP (no hub):
- `POST /v1/evalpen/submissions` with `source: "camera"`
- Full pipeline: OCR → Segment → Classify → Flag → Teacher reviews → Batch eval

---

## 10. LLM Gate — Single Door Token Controller

### 10.1 Purpose

Every LLM call in the entire system — both DCR and PCR — goes through one gate. No service calls the LLM client directly. The gate enforces per-call limits, tracks token budgets, and logs all usage.

### 10.2 Architecture

```
  ┌──────────────────────────────────────────┐
  │  ALL LLM CALLERS                         │
  │                                          │
  │  DCR AI Pipeline ──────┐                 │
  │  PCR Eval Core ────────┤                 │
  │  PCR Cache Warmup ─────┤                 │
  │  PCR Clubbed H4 ───────┤                 │
  │  Future: Vision LLM ───┤                 │
  │                        │                 │
  │  NO service calls LLM  │                 │
  │  client directly       │                 │
  └────────────────────────┼─────────────────┘
                           │
                           ▼
  ╔════════════════════════════════════════════╗
  ║              LLM GATE (single door)       ║
  ║                                           ║
  ║  1. Budget check                          ║
  ║     daily/weekly/monthly totals           ║
  ║     vs configured limits                  ║
  ║     │                                     ║
  ║     ├─ EXHAUSTED → HTTP 429               ║
  ║     ▼                                     ║
  ║  2. Per-call limits                       ║
  ║     estimate input tokens (char count/4)  ║
  ║     clamp output tokens                   ║
  ║     │                                     ║
  ║     ├─ EXCEEDED → HTTP 413                ║
  ║     ▼                                     ║
  ║  3. LLM Client → API call                ║
  ║     (LiteLLM, temperature=0)              ║
  ║     │                                     ║
  ║     ▼                                     ║
  ║  4. Token Logger                          ║
  ║     → llm_token_usage_log                 ║
  ║     │                                     ║
  ║     ▼                                     ║
  ║  5. Return GateResponse                   ║
  ║     (content + usage metadata)            ║
  ╚════════════════════════════════════════════╝
```

### 10.3 Contract

```
call(model_id, prompt, caller_id, **kwargs) → GateResponse

GateResponse:
  content: str            # LLM output
  usage: TokenUsageRecord # tokens, cost, model, caller
```

### 10.4 Allowed Callers

Only registered callers may invoke the gate. Any call with an unregistered `caller_id` is rejected.

| caller_id | Pipeline | Purpose |
|-----------|----------|---------|
| `pcr_eval_core` | PCR | Evaluate student response (cache hit or miss) |
| `pcr_cache_warmup` | PCR | Batch-generate reference solutions |
| `pcr_clubbed_h4` | PCR | Topic discontinuity check for clubbed detection |
| `pcr_practice` | PCR | Live practice evaluation |
| `dcr_ai` | DCR | LLM fallback when HWR confidence below threshold |
| `dcr_devanagari` | DCR | Future: LLM-assisted Devanagari recognition |

New callers must be registered in `LLM_GATE_SPEC.md` before use. This prevents undocumented LLM calls from bypassing budget tracking.

### 10.5 Gate Configuration

```
GateConfig:
  max_input_tokens: int | null     # null = unlimited. Per-call ceiling.
  max_output_tokens: int | null    # null = unlimited. Clamped on API param.

BudgetConfig:
  daily_token_limit: int | null    # null = unlimited
  weekly_token_limit: int | null   # null = unlimited
  monthly_token_limit: int | null  # null = unlimited
```

All limits independently settable. `null` = unlimited for that dimension.

### 10.6 Token Usage Record

```
TokenUsageRecord:
  model: str
  input_tokens: int
  output_tokens: int
  cache_read_tokens: int
  cache_creation_tokens: int
  total_tokens: int              # input + output
  estimated_cost_usd: float
  timestamp: datetime
  caller: str                    # must match allowed callers list
```

### 10.7 Token Usage Lifecycle

```
  llm_token_usage_log (append-only, per LLM call)
  │
  │  Kept: 7 days of raw rows (TTL index on called_at)
  │
  │  Every midnight:
  │  └─ Aggregate → llm_token_usage_rollup (period_type: 'daily')
  │
  │  Every Monday:
  │  ├─ Raw rows >7 days: TTL auto-delete
  │  └─ Aggregate dailies → rollup (period_type: 'weekly')
  │
  │  Every 1st of month:
  │  ├─ Delete dailies >1 month
  │  └─ Aggregate weeklies → rollup (period_type: 'monthly')
  │
  │  Every 3 months:
  │  └─ Delete monthlies >3 months
  │
  ▼
  Retention at any point:
  ┌──────────────────────────────────────┐
  │ Raw log        │ 7 days  │ per-call  │
  │ Daily rollups  │ 1 month │ per-day   │
  │ Weekly rollups │ 3 months│ per-week  │
  │ Monthly rollups│ 3 months│ per-month │
  └──────────────────────────────────────┘
```

### 10.8 Error Responses

Budget exhausted:
```json
{
  "error": "budget_exhausted",
  "period": "daily",
  "used_tokens": 498500,
  "limit_tokens": 500000,
  "resets_at": "2026-03-25T00:00:00Z",
  "suggestion": "Increase daily_token_limit via PUT /v1/evalpen/usage/config or wait for reset."
}
```

Per-call limit exceeded:
```json
{
  "error": "token_limit_exceeded",
  "limit_type": "per_call_input",
  "estimated_tokens": 6200,
  "allowed_tokens": 4000,
  "suggestion": "Reduce prompt size or increase max_input_tokens via PUT /v1/evalpen/usage/config."
}
```

Batch evaluate: if budget runs out mid-batch, completed evals returned with partial results + budget error on remaining items.

### 10.9 Usage API

```
GET  /v1/evalpen/usage/current   — current budget status (daily/weekly/monthly totals, limits, remaining)
GET  /v1/evalpen/usage/history   — historical usage by period (with model + caller breakdown)
PUT  /v1/evalpen/usage/config    — update gate + budget config (null = unlimited)
```

Namespaced under `/v1/evalpen/` — the gate is ExamPen-scoped, not a global platform service.

---

## 11. Tamper-Proof Pipeline

### 11.1 Architecture

```
  Student writes answer
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  LAYER 1: RAW IMMUTABILITY                  │
  │                                             │
  │  Submission arrives (strokes or images)      │
  │       │                                     │
  │       ▼                                     │
  │  SHA-256(raw_content) → content_hash        │
  │       │                                     │
  │       ▼                                     │
  │  Write-once store:                          │
  │  ├─ MongoDB: _immutable: true               │
  │  │  (repo rejects update_one on immutable)  │
  │  └─ MinIO: object versioning enabled        │
  │     (originals never overwritten)           │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  LAYER 2: SERVER-SIDE TEXT FETCH            │
  │                                             │
  │  POST /evaluate request body:               │
  │  { response_id, question_id }               │
  │  ← NO student text in request               │
  │                                             │
  │  Eval service internally:                   │
  │  detected_text = db.get(response_id).text   │
  │  ← fetched from server-side storage         │
  │                                             │
  │  Client CANNOT substitute correct text      │
  │  for what was actually written              │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  LAYER 3: APPEND-ONLY AUDIT TRAIL          │
  │                                             │
  │  Every evaluation:                          │
  │  ├─ who triggered, when, which model        │
  │  ├─ prompt sent to LLM                      │
  │  └─ full LLM response received              │
  │                                             │
  │  Every score override:                      │
  │  ├─ original_score → new_score              │
  │  ├─ who overrode, reason (min 5 chars)      │
  │  └─ timestamp                               │
  │                                             │
  │  Every flag resolution:                     │
  │  ├─ flag_id, action taken                   │
  │  ├─ who resolved, timestamp                 │
  │  └─ before/after state                      │
  │                                             │
  │  All stored as append-only sub-documents    │
  │  (no updates, no deletes)                   │
  └─────────────────────────────────────────────┘
```

### 11.2 Scope

Applies to PCR offline exam submissions and DCR submissions. Practice mode is **out of scope** — the PCR practice endpoint evaluates and returns results synchronously without creating new persistent artifacts. No new collections, no immutable storage, no audit subdocuments for practice calls. The current backend's practice persistence is unchanged.

---

## 12. MongoDB Collection Model

### 12.1 Tenant Placement Rule

**All ExamPen collections live in the per-tenant DB** (`skb_<tenant>`). There is no separate "system DB" for ExamPen. Gate collections also live in the tenant DB — each tenant has its own budget tracking.

If cross-tenant budget aggregation is needed (e.g., platform-wide cost monitoring), it is handled by reading across tenant DBs at query time, not by a shared collection.

### 12.2 PCR Collections (per-tenant)

#### `evalpen_submissions`

Used for offline exam and camera submissions only. **Not used by practice mode.**

| Field | Type | Notes |
|-------|------|-------|
| `_id` | ObjectId | — |
| `submission_id` | string | Unique, indexed |
| `student_id` | string | Indexed |
| `exam_id` | string | Indexed |
| `admin_id` | string | Scoping (existing pattern) |
| `source` | enum | `"ble_pen"` / `"camera"` |
| `page_count` | int | — |
| `pages` | array | Embedded: `[{ page_num, raw_asset_ref, ocr_result, boundaries }]` |
| `content_hash` | string | SHA-256 of raw content |
| `_immutable` | bool | `true` — repo rejects updates |
| `submitted_at` | datetime | — |
| `segmentation_status` | enum | `"pending"` / `"complete"` / `"failed"` |

**Indexes**: `{ submission_id: 1 }` unique, `{ exam_id: 1, student_id: 1 }`, `{ admin_id: 1 }`

#### `evalpen_detected_responses`

| Field | Type | Notes |
|-------|------|-------|
| `_id` | ObjectId | — |
| `response_id` | string | Unique, indexed |
| `submission_id` | string | Indexed |
| `question_id` | string | Indexed |
| `detected_text` | string | OCR output (server-side, not client-provided) |
| `source_pages` | array[int] | Which pages this response spans |
| `content_type` | enum | `TEXT_ONLY` / `MIXED` / `DIAGRAM_HEAVY` / `TABLE_PRESENT` |
| `content_classification` | object | Full classification detail |
| `eval_status` | enum | `"pending"` / `"ready"` / `"ready_with_warnings"` / `"blocked"` / `"evaluated"` / `"manual_review"` |
| `segmentation_confidence` | float | 0.0–1.0 |
| `segmentation_method` | string | `"boundary+marker"` / `"marker_only"` / `"boundary_only"` |
| `ocr_confidence` | float | 0.0–1.0 |
| `flags` | array | Embedded: `[{ flag_id, source, flag_type, severity, reason, suggested_action, metadata, resolved, resolved_by, resolved_at }]` |
| `_immutable` | bool | `true` for detected_text; flags sub-array allows appends for resolution |
| `created_at` | datetime | — |

**Indexes**: `{ response_id: 1 }` unique, `{ submission_id: 1 }`, `{ eval_status: 1 }`, `{ "flags.severity": 1 }`

#### `evalpen_evaluations`

| Field | Type | Notes |
|-------|------|-------|
| `_id` | ObjectId | — |
| `evaluation_id` | string | Unique, indexed |
| `response_id` | string | Unique (one eval per response), indexed |
| `question_id` | string | Indexed |
| `student_id` | string | Indexed |
| `eval_path` | string | `"cache_hit"` / `"cache_miss"` |
| `model_used` | string | LLM model ID |
| `content_type` | enum | — |
| `total_score` | int | — |
| `max_score` | int | — |
| `scoreable_max` | int | Adjusted for diagram weight |
| `step_marks` | array | Embedded: `[{ step, marks_awarded, marks_possible, justification }]` |
| `overall_feedback` | string | — |
| `reference_solution` | object | `{ text, source, version }` |
| `ocr_concerns` | array | — |
| `token_usage` | object | Embedded: `{ model, input_tokens, output_tokens, ... }` |
| `raw_llm_response` | object | Full LLM response (audit) |
| `audit_trail` | array | Append-only: `[{ action, actor_id, timestamp, before, after, reason }]` |
| `evaluated_at` | datetime | — |

**Indexes**: `{ evaluation_id: 1 }` unique, `{ response_id: 1 }` unique, `{ student_id: 1 }`, `{ question_id: 1 }`

#### `evalpen_questions`

| Field | Type | Notes |
|-------|------|-------|
| `question_id` | string | Unique, indexed |
| `exam_id` | string | Indexed |
| `subject` | string | — |
| `grade` | int | — |
| `question_text` | string | — |
| `question_type` | string | — |
| `complexity` | enum | `"L1"` / `"L2"` / `"L3"` |
| `eval_template` | string | Template ID |
| `max_marks` | int | — |
| `rubric` | object | Template-specific rubric |
| `has_diagram` | bool | — |
| `expects_diagram` | bool | — |
| `diagram_weight` | float | 0.0–1.0 |
| `expected_word_range` | object | `{ min, max }` |

**Indexes**: `{ question_id: 1 }` unique, `{ exam_id: 1 }`

#### `evalpen_solutions`

| Field | Type | Notes |
|-------|------|-------|
| `question_id` | string | Indexed |
| `version` | int | — |
| `reference_solution` | string | — |
| `solution_source` | enum | `"teacher"` / `"llm"` |
| `model_used` | string | null if teacher-uploaded |
| `created_at` | datetime | — |

**Indexes**: `{ question_id: 1, version: -1 }` unique compound

### 12.3 DCR Collections (per-tenant)

#### `exampen_dcr_submissions`

| Field | Type | Notes |
|-------|------|-------|
| `submission_id` | string | Unique, indexed |
| `exam_id` | string | Indexed |
| `student_id` | string | Indexed |
| `admin_id` | string | Scoping |
| `pen_mac` | string | — |
| `pages` | array | Embedded: `[{ page_number, strokes }]` |
| `content_hash` | string | SHA-256 |
| `_immutable` | bool | `true` |
| `submitted_at` | datetime | — |

**Indexes**: `{ submission_id: 1 }` unique, `{ exam_id: 1, student_id: 1 }`

#### `exampen_dcr_results`

| Field | Type | Notes |
|-------|------|-------|
| `exam_id` | string | Indexed |
| `student_id` | string | Indexed |
| `question_id` | string | Indexed |
| `recognized_text` | string | HWR output |
| `confidence` | float | — |
| `match_type` | enum | `exact_match` / `partial_match` / `numeric_match` / `no_match` |
| `score` | int | — |
| `max_score` | int | — |
| `audit_trail` | array | Append-only (for overrides) |

**Indexes**: `{ exam_id: 1, student_id: 1, question_id: 1 }` unique compound

### 12.4 LLM Gate Collections (per-tenant)

#### `llm_gate_config`

Single document per tenant. Upsert on first use.

| Field | Type | Notes |
|-------|------|-------|
| `_id` | string | Fixed: `"gate_config"` |
| `max_input_tokens` | int/null | — |
| `max_output_tokens` | int/null | — |
| `daily_token_limit` | int/null | — |
| `weekly_token_limit` | int/null | — |
| `monthly_token_limit` | int/null | — |
| `updated_at` | datetime | — |

#### `llm_token_usage_log`

Append-only, one document per LLM call.

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | — |
| `caller` | string | Must match allowed callers |
| `input_tokens` | int | — |
| `output_tokens` | int | — |
| `cache_read_tokens` | int | — |
| `cache_creation_tokens` | int | — |
| `total_tokens` | int | input + output |
| `estimated_cost_usd` | float | — |
| `called_at` | datetime | — |

**Indexes**: `{ called_at: 1 }` with TTL (7 days), `{ caller: 1 }`, `{ model: 1 }`

#### `llm_token_usage_rollup`

| Field | Type | Notes |
|-------|------|-------|
| `period_type` | enum | `"daily"` / `"weekly"` / `"monthly"` |
| `period_start` | date | — |
| `period_end` | date | — |
| `total_tokens` | int | — |
| `total_input` | int | — |
| `total_output` | int | — |
| `total_cost_usd` | float | — |
| `call_count` | int | — |
| `breakdown_by_model` | object | — |
| `breakdown_by_caller` | object | — |

**Indexes**: `{ period_type: 1, period_start: 1 }` unique compound

### 12.5 Unchanged Collections

The following are **not modified** by this architecture:

- All existing `skb_<tenant>` collections (students, questions, practice sessions, etc.)
- All existing practice-mode persistence
- `skb_master` collections

---

## 13. API Endpoints (PCR)

### 13.1 Submissions

```
POST /v1/evalpen/submissions                           — submit pages for segmentation
GET  /v1/evalpen/submissions/{id}/responses             — fetch detected responses
GET  /v1/evalpen/submissions/{id}/responses/{resp_id}   — single response detail
GET  /v1/evalpen/submissions/{id}/flagged               — only blocked/warning items
PATCH /v1/evalpen/submissions/{id}/responses/{resp_id}  — resolve flags, split, manual score
```

### 13.2 Evaluate

```
POST /v1/evalpen/evaluate                  — evaluate single response
POST /v1/evalpen/evaluate/batch            — batch evaluate (budget-aware)
GET  /v1/evalpen/evaluations               — evaluation history
```

### 13.3 Practice

```
POST /v1/evalpen/practice/evaluate         — live synchronous evaluation (stateless, no new persistence)
```

### 13.4 Solutions

```
GET  /v1/evalpen/solutions/{question_id}   — fetch reference solution
PUT  /v1/evalpen/solutions/{question_id}   — upload/update solution
```

### 13.5 Usage (Gate)

```
GET  /v1/evalpen/usage/current             — current budget status
GET  /v1/evalpen/usage/history             — historical usage
PUT  /v1/evalpen/usage/config              — update gate + budget config
```

All endpoints live under `/v1/evalpen/` and are tenant-gated via the `exampen_pcr` feature flag. The gate is an ExamPen-scoped contract — its usage API shares the same namespace as the rest of the PCR surface.

---

## 14. Event Schemas (PCR)

Transport-agnostic event contracts:

```
eval.submission.received      — Raw submission stored
eval.ocr.complete            — OCR finished for a submission
eval.segmentation.complete   — Segmentation done, responses detected
eval.result                  — Evaluation completed (score, feedback)
eval.practice.evaluated      — Practice mode eval complete (ephemeral analytics/logging only)
eval.flag.resolved           — Teacher resolved a flag
```

---

## 15. Cost Model

Assumptions: 500 students × 8 subjects × 20 questions/subject/month = 80,000 evaluations/month.

| Phase | Cache-Hit Rate | Haiku Calls | Sonnet Calls | Monthly Cost |
|-------|---------------|-------------|-------------|-------------|
| Month 1 | 20% (cold) | 16,000 | 64,000 | ~$656 |
| Month 2 | 70% | 68,000 | 12,000 | ~$188 |
| Month 3+ | 90%+ | 76,000 | 4,000 | ~$116 |
| Steady state (warm) | 98% | 79,200 | 800 | ~$87 |

Cost reduction levers: teacher-uploaded solutions (100% cache hit), prompt caching (~50% reduction), batch API (50% off), shared question banks across schools. Gate budget limits prevent runaway costs.

---

## 16. Risk Register (PCR-Specific)

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|-----------|
| OCR too low for Hindi handwriting | Med | High | Benchmark PaddleOCR Devanagari. Fallback: GPT-4o Vision via Gate. |
| LLM inconsistent scores | Med | High | temperature=0. Cache results. Dual-eval sample, flag divergence >1 mark. |
| Students ignore boundary/marker rules | High | Med | Confidence gating. Teacher review. Iterative instruction improvement. |
| Diagram detection false positives | Med | Med | Conservative thresholds. Teacher override via PATCH. |
| Diagram detection false negatives | Med | Med | `expects_diagram` cross-check from question metadata. |
| Clubbed responses undetected | Low | High | H3 (missing question) catches most. H4 (topic discontinuity) as secondary. |
| Cost escalation with low cache-hit | Low | Med | Gate budget limits. Mandatory cache warm-up. Teacher solution upload. |
| Budget exhaustion during batch grading | Med | Med | Gate returns partial results. Alert at 80% utilization. Admin can increase via API. |
| Token rollup cron failure | Low | Low | Raw logs retained 7 days regardless. Rollup idempotent, can rerun. |
| Accountancy ledger OCR fails | Med | Med | Table detection routes to specialized template. |
| Content classifier misclassifies tables as diagrams | Low | Med | Table detection runs first, takes priority. |

---

## 17. Adaptations from eval-engine-plan-v3

When this plan was created from `../pcr/eval-engine-plan-v3.md`, the following adaptations were made:

| Plan v3 says | Unified plan adaptation |
|---|---|
| PostgreSQL tables | MongoDB collections in per-tenant DB (embedded documents for flags[], step_marks[]) |
| `libs/llm-gate-py/` folder structure | Contract spec only (no code layout prescribed) |
| Practice mode migration/replacement | Practice persistence unchanged; PCR exposes stateless endpoint, backend calls it |
| Standalone microservice | Endpoints are tenant-gated; implementation packaging deferred |
| NATS stream `EVALPEN.>` | Transport-agnostic event schemas in `contracts/events/eval.*.schema.json` |
| `gate_config` PG table | `llm_gate_config` MongoDB collection (single document per tenant) |
| `token_usage_log` PG table | `llm_token_usage_log` MongoDB collection (append-only, TTL index) |
| `token_usage_rollup` PG table | `llm_token_usage_rollup` MongoDB collection (compound key) |

---

## 18. Documentation Cleanup Plan

### 18.1 Keep Active

- `architecture/`, `governance/`, `integration/`, `references/`, `chapters/`, `hub/`, `api/`, and `contracts/events/`
- `GUIDE_RULE_DOCS/` as the canonical home for reusable process guidance
- `architecture/unifiedPlan.md` only until the root specs become ACTIVE

### 18.2 Keep as Historical Reference Only

- `../pcr/eval-engine-plan-v3.md` — preserved as the locked PCR source document, but marked `SUPERSEDED` once `PCR_EVAL_ENGINE_SPEC.md` is active

### 18.3 Rewrite as Thin Entry Docs

- `backend/exam-conductor/README.md` — keep as a short entrypoint that sends readers to `new-docs/agent_ref_index.md`

### 18.4 Remove After Migration

- `backend/exam-conductor/evaluation_engine_plan.md` — superseded by the new engine and gate specs
- `backend/exam-conductor/pcr/README.md` — superseded by the new active docs
- root-level guide mirrors in `new-docs/` once routing points only to `GUIDE_RULE_DOCS/`

---

## 19. Documentation Deliverables

### New Root-Level Specs to Create

| Document | Purpose | Initial Status |
|----------|---------|---------------|
| `architecture/DUAL_MODE_ARCHITECTURE.md` | DCR/PCR mode framing, shared collection layer, routing, DCR contract | DRAFT |
| `architecture/PCR_EVAL_ENGINE_SPEC.md` | Full PCR pipeline: segmentation, classification, eval, flags | DRAFT |
| `architecture/LLM_GATE_SPEC.md` | Gate contract, allowed callers, budget, token storage | DRAFT |
| `architecture/TAMPER_PROOF_SPEC.md` | Immutability, server-side fetch, audit trail rules | DRAFT |

### New API Specs

| Document | Purpose | Initial Status |
|----------|---------|---------------|
| `api/eval-submissions.openapi.yaml` | Submission + response + flag resolution endpoints | DRAFT |
| `api/eval-evaluate.openapi.yaml` | Evaluate + batch + evaluation history endpoints | DRAFT |
| `api/eval-practice.openapi.yaml` | Live synchronous practice evaluation endpoint | DRAFT |
| `api/eval-solutions.openapi.yaml` | Solution CRUD endpoints | DRAFT |
| `api/eval-usage.openapi.yaml` | Gate usage monitoring + config endpoints | DRAFT |

### New Event Schemas

| Document | Purpose | Initial Status |
|----------|---------|---------------|
| `contracts/events/eval.submission.received.schema.json` | Submission received | DRAFT |
| `contracts/events/eval.ocr.complete.schema.json` | OCR complete | DRAFT |
| `contracts/events/eval.segmentation.complete.schema.json` | Segmentation complete | DRAFT |
| `contracts/events/eval.result.schema.json` | Evaluation result | DRAFT |
| `contracts/events/eval.practice.evaluated.schema.json` | Practice eval complete | DRAFT |
| `contracts/events/eval.flag.resolved.schema.json` | Flag resolved | DRAFT |

### Existing Documents to Update

| Document | Update |
|----------|--------|
| `governance/DOCUMENT_REGISTRY.md` | Register all new documents with authority rules from §3 |
| `agent_ref_index.md` | Add PCR/DCR/gate task routing to new root specs |
| `governance/STATE_OWNERSHIP_MAP.md` | Add PCR + DCR state ownership entries |
| `governance/FAILURE_MITIGATION_REGISTER.md` | Add PCR failure modes with IDs |
| `governance/TEST_SUITE_SPEC.md` | Add PCR test IDs (U-SEG-*, U-EVAL-*, U-GATE-*, I-EVAL-*, E2E-PCR-*) |
| `governance/COMPONENT_INDEPENDENCE_MAP.md` | Add PCR components to dependency graph |
| `integration/STOODY_INTEGRATION_SPEC.md` | Add PCR integration points, practice mode, feature flag |
| `chapters/01_SYSTEM_OVERVIEW.md` | Reference `DUAL_MODE_ARCHITECTURE.md` for mode introduction |
| `chapters/BUILD_STATUS.md` | Add PCR build phases and tracking |

### Documentation Build Order

1. **Pass 1 — Foundation specs**: `architecture/DUAL_MODE_ARCHITECTURE.md`, `architecture/LLM_GATE_SPEC.md`, `architecture/TAMPER_PROOF_SPEC.md`
2. **Pass 2 — PCR detail**: `architecture/PCR_EVAL_ENGINE_SPEC.md`, 5 OpenAPI specs, 6 event schemas
3. **Pass 3 — Cross-cutting updates**: DOCUMENT_REGISTRY, agent_ref_index, STATE_OWNERSHIP_MAP, FAILURE_MITIGATION_REGISTER, TEST_SUITE_SPEC, COMPONENT_INDEPENDENCE_MAP, STOODY_INTEGRATION_SPEC, 01_SYSTEM_OVERVIEW, BUILD_STATUS
4. **Pass 4 — Quality gate review**: Verify failure mode IDs, test IDs, OpenAPI↔spec consistency, event schema↔spec consistency, promote DRAFT→ACTIVE
5. **Pass 5 — Retire this file**: Once all four root specs are ACTIVE, mark `architecture/unifiedPlan.md` as SUPERSEDED in `governance/DOCUMENT_REGISTRY.md`. It remains readable for historical rationale but is no longer authoritative for any topic.

---

## Verification Checklist

- [ ] All new documents follow DOCUMENTATION_PLAN.md §2 template sections
- [ ] All new documents registered in DOCUMENT_REGISTRY.md with authority rules
- [ ] Document authority matrix (§3) has no overlapping owners
- [ ] agent_ref_index.md correctly routes tasks to root specs (not chapters)
- [ ] DCR contract has same minimum rigor as PCR (input, output, ownership, gate usage)
- [ ] No references to archiveDCR/ anywhere in new documents
- [ ] No PostgreSQL references anywhere
- [ ] All collections placed in per-tenant DB — no hidden system DB
- [ ] Practice mode creates no new persistence — no new collections, no writes to evalpen_submissions
- [ ] Practice endpoint is stateless (receive, evaluate, return) — only gate token logging occurs
- [ ] LLM Gate spec includes allowed callers list
- [ ] LLM Gate spec uses MongoDB collections with defined indexes
- [ ] All collection schemas define `_immutable` field where applicable
- [ ] All collections define audit trail subdocument structure where applicable
- [ ] Failure modes have unique IDs in FAILURE_MITIGATION_REGISTER
- [ ] Test IDs enumerated in TEST_SUITE_SPEC
- [ ] OpenAPI specs and event schemas match spec descriptions (specs own behavior, contracts own wire format)
- [ ] All API endpoints under `/v1/evalpen/` namespace (including usage)
- [ ] No premature code-layout or packaging commitments
- [ ] This file marked SUPERSEDED in DOCUMENT_REGISTRY once root specs are ACTIVE
- [ ] DCR-to-gate routing shown as conditional/fallback in all diagrams
