# PCR Eval Engine Spec

**Status:** ACTIVE  
**Authority:** PCR engine behavior, segmentation, evaluation flow, collections, and practice boundary.

---

## 1. Summary

PCR evaluates paginated subjective responses.

It supports:

- conducted exams from the shared ingest substrate
- live practice evaluation through a stateless endpoint

It does not own:

- hub collection
- practice persistence
- gate policy

---

## 2. Inputs

### 2.1 Conducted Exams

Canonical artifacts from the shared ingest substrate:

- pen-originated stroke pages
- camera or scan pages
- `admin_id`, `student_id`, `exam_id`, timestamps, artifact refs

### 2.2 Practice

Live request payload sent to the PCR practice endpoint. PCR evaluates synchronously and returns a result. Persistence remains in the existing backend.

---

## 3. End-to-End Flow

```text
  ┌─────────────────────────┐      ┌──────────────────────────┐
  │ Path A: BLE Pen         │      │ Path B: Camera           │
  │ Stroke vectors          │      │ JPEG/PNG images          │
  │                         │      │                          │
  │        ▼                │      │         ▼                │
  │ Render to image (PIL)   │      │   Preprocess image       │
  │                         │      │                          │
  │        ▼                │      │         ▼                │
  │ LLM Vision OCR          │      │  LLM Vision OCR          │
  │ (via gate, pcr caller)  │      │  (via gate, pcr caller)  │
  │                         │      │                          │
  │        ▼                │      │         ▼                │
  │ TextBlocks+bbox         │      │  TextBlocks+bbox         │
  └────────────┬────────────┘      └────────────┬─────────────┘
               └──────────────┬─────────────────┘
                              ▼
                         Unified PageOCR
                              │
                              ▼
     boundary detect -> marker parse -> segment -> classify -> clubbed detect
                              │
                              ▼
                      detected responses + flags
                         │                │
                         │                └-> blocking -> review
                         ▼
                   shared LLM gate
                         │
                         ▼
             score + step marks + feedback + token usage
```

Both pen and camera paths use LLM Vision OCR through the shared gate with registered `caller_id`s. The gate is model-agnostic -- the active provider is selected via the `AI_PROVIDER` environment variable.

---

## 4. Segmentation and Classification

### 4.1 Boundary Detection

PCR uses double horizontal response delimiters.

BLE pen path:

| Parameter | Constraint |
|---|---|
| slope | within +/-10 degrees of horizontal |
| length | greater than 40% of page width |
| Y-gap between paired lines | 2-15 mm |
| temporal proximity | both lines drawn within about 3 seconds |
| horizontal overlap | greater than 70% |

Camera path:

- Canny -> HoughLinesP
- `minLineLength = 0.4 * image_width`
- slope within +/-10 degrees
- Y-gap 2-15 mm

### 4.2 Q Marker Parsing

Students are expected to mark responses as `Q.No X.Ans`.

Regex:

```text
/Q\.?\s*(?:No|no)\.?\s*(\d{1,3})\s*(?:[\.\(\s]*([a-z]|[ivx]+|[A-Z])[\)\.]?)?\s*\.?\s*(?:Ans|ans|ANS)\.?/i
```

Captures:

- `\1` -> question number
- `\2` -> sub-part

Post-OCR fixes in marker context:

- `l -> 1`
- `O -> 0`
- `I -> 1`

### 4.3 Cross-Page Stitching

| Transition | Interpretation |
|---|---|
| page N ends without closing boundary | response continues on N+1 |
| page N+1 starts with boundary | previous response closed at prior page bottom |
| page N+1 starts with content and no boundary | continuation |
| page N+1 has marker without boundary | associate to prior response and flag |

### 4.4 Content Classification

| Type | Definition | Eval Path |
|---|---|---|
| `TEXT_ONLY` | more than 85% text coverage | normal auto-eval |
| `MIXED` | 40-85% text with figure content | evaluate text, flag diagram |
| `DIAGRAM_HEAVY` | less than 40% text | block auto-eval |
| `TABLE_PRESENT` | grid or tabular structure | route to table template or flag |

Diagram prorating:

```text
scoreable_marks = max_marks * (1 - diagram_weight)
```

### 4.5 Clubbed Response Heuristics

| Heuristic | Detection | Confidence |
|---|---|---|
| `H1` multiple markers | more than one Q marker in one segment | very high |
| `H2` length anomaly | word count greater than expected max * 2.5 | medium |
| `H3` missing question | manifest question not represented in segments | high |
| `H4` topic discontinuity | LLM-assisted discontinuity check through gate | medium-high |

---

## 5. Evaluation Core

### 5.1 Solution Cache Strategy

Cache key is question-centric. On cache hit, PCR runs compare-only evaluation. On cache miss, PCR routes to a model tier, generates or refreshes a reference solution, stores it, then evaluates.

### 5.2 Complexity Router

| Complexity | Model | Trigger |
|---|---|---|
| cache-hit | Haiku-class compare-only | any cache hit |
| `L1` miss | Haiku-class | factual or single-step |
| `L2` miss | Sonnet-class | multi-step short answer |
| `L3` miss | Sonnet/Opus-class | essay or open-ended |

### 5.3 Template Families

| Template | Use |
|---|---|
| `stepwise_numerical` | math, physics, chemistry, numerical accountancy |
| `essay_rubric` | language and long-form humanities answers |
| `factual_recall` | short factual answers |
| `keyword_coverage` | definitions and process answers |
| `ledger_tabular` | accountancy tables and ledgers |
| `proof_derivation` | proofs and derivations |

### 5.4 Vision OCR Error Tolerance

PCR prompts must tolerate common Vision OCR confusion such as:

- `l` vs `1`
- missing superscripts
- `rn` vs `m`
- Devanagari matra errors

---

## 6. Unified Flag System

### 6.1 Flag Shape

```json
{
  "flag_id": "FLG-001",
  "response_id": "RESP-001",
  "source": "segmenter",
  "flag_type": "no_question_marker",
  "severity": "warning",
  "reason": "Marker missing",
  "suggested_action": "review association",
  "metadata": {}
}
```

### 6.2 Flag Registry

| Source | Flag Type | Severity |
|---|---|---|
| `segmenter` | `no_question_marker` | warning |
| `segmenter` | `no_boundary_detected` | warning |
| `segmenter` | `boundary_only_no_marker` | warning |
| `segmenter` | `low_segmentation_confidence` | warning |
| `content_classifier` | `diagram_present` | info |
| `content_classifier` | `diagram_heavy_content` | blocking |
| `content_classifier` | `table_detected` | info |
| `content_classifier` | `expected_diagram_missing` | warning |
| `clubbed_detector` | `clubbed_multiple_markers` | blocking |
| `clubbed_detector` | `clubbed_length_anomaly` | warning |
| `clubbed_detector` | `clubbed_missing_question` | warning |
| `clubbed_detector` | `clubbed_topic_discontinuity` | warning |
| `ocr` | `low_ocr_confidence` | warning |
| `ocr` | `ocr_rejected` | blocking |
| `eval` | `partial_eval_diagram_excluded` | info |
| `eval` | `llm_score_divergence` | warning |
| `llm_gate` | `budget_warning_80pct` | warning |
| `llm_gate` | `budget_exhausted` | blocking |

### 6.3 Severity Behavior

| Severity | Meaning | Result |
|---|---|---|
| `blocking` | cannot auto-evaluate | `eval_status = blocked` |
| `warning` | can evaluate but may be unreliable | `evaluated_with_warnings` |
| `info` | informational only | `evaluated` |

---

## 7. Conducted-Exam Storage Model

MongoDB only. All conducted-exam PCR collections live in the tenant/admin DB.

### 7.1 `evalpen_submissions`

Offline exam and camera submissions only. Not used by practice mode.

Core fields:

- `submission_id`
- `student_id`
- `exam_id`
- `admin_id`
- `source`
- `page_count`
- `pages[]`
- `content_hash`
- `_immutable = true`
- `submitted_at`
- `segmentation_status`

Indexes:

- `{ submission_id: 1 }` unique
- `{ exam_id: 1, student_id: 1 }`
- `{ admin_id: 1 }`

### 7.2 `evalpen_detected_responses`

Core fields:

- `response_id`
- `submission_id`
- `question_id`
- `detected_text`
- `source_pages[]`
- `content_type`
- `eval_status`
- `segmentation_confidence`
- `ocr_confidence`
- `flags[]`
- `_immutable = true` for detected text

Indexes:

- `{ response_id: 1 }` unique
- `{ submission_id: 1 }`
- `{ eval_status: 1 }`
- `{ "flags.severity": 1 }`

### 7.3 `evalpen_evaluations`

Core fields:

- `evaluation_id`
- `response_id`
- `question_id`
- `student_id`
- `eval_path`
- `model_used`
- `total_score`
- `max_score`
- `scoreable_max`
- `step_marks[]`
- `overall_feedback`
- `reference_solution`
- `token_usage`
- `raw_llm_response`
- `audit_trail[]`

Indexes:

- `{ evaluation_id: 1 }` unique
- `{ response_id: 1 }` unique
- `{ student_id: 1 }`
- `{ question_id: 1 }`

### 7.4 `evalpen_questions`

Core fields:

- `question_id`
- `exam_id`
- `subject`
- `question_type`
- `complexity`
- `eval_template`
- `max_marks`
- `rubric`
- `expects_diagram`
- `diagram_weight`
- `expected_word_range`

Indexes:

- `{ question_id: 1 }` unique
- `{ exam_id: 1 }`

### 7.5 `evalpen_solutions`

Core fields:

- `question_id`
- `version`
- `reference_solution`
- `solution_source`
- `model_used`
- `created_at`

Indexes:

- `{ question_id: 1, version: -1 }` unique compound

Practice evaluation does not create new PCR persistence.

---

## 8. Boundaries

### 8.1 Conducted Exams

- artifacts arrive through the shared ingest substrate
- PCR stores conducted-exam submission, response, and evaluation records
- clients do not supply authoritative answer text

### 8.2 Practice

- request -> evaluate -> return
- no new `evalpen_submissions`
- no immutable practice artifact store
- token logging may still occur through the gate

---

## 9. Related Contracts

- API: `api/eval-submissions.openapi.yaml`
- API: `api/eval-evaluate.openapi.yaml`
- API: `api/eval-practice.openapi.yaml`
- API: `api/eval-solutions.openapi.yaml`
- API: `api/eval-usage.openapi.yaml`
- Events: `contracts/events/eval.*.schema.json`
- Gate: `architecture/LLM_GATE_SPEC.md`
- Integrity: `architecture/TAMPER_PROOF_SPEC.md`
- Historical detail: `../pcr/eval-engine-plan-v3.md`
