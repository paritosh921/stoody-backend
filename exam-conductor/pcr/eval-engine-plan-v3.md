# R-009 | Student Response Evaluation Engine — Consolidated Architecture Plan (Rev 3)

**Version:** 3.0  
**Date:** 2026-03-24  
**Status:** Architecture Locked  
**Supersedes:** R-008, R-007  
**Delta from R-008:** Added LLM Gate (single-door token budget controller), token usage logging, rolling usage aggregation with tiered cleanup, usage monitoring API, per-call and per-period token limits. Gate integrated into Phase 0 as foundational infrastructure.  
**Scope:** School-grade subjective response evaluation — all subjects (Physics, Chemistry, Biology, Math, English, Hindi, History, Geography, Accountancy, etc.)

---

## 1. System Identity

**Name:** Eval Engine (working title)  
**Nature:** Standalone, API-first microservice. No UI. Plug-and-play into any student-facing platform.  
**Inputs:** OCR'd student responses (from BLE pen or camera)  
**Outputs:** Per-question score, step marks, feedback, reference solution, detected text, content flags, token usage metadata

---

## 2. System Flow — Unicode Diagrams

### 2.1 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     STUDENT SUBMISSION                          │
│              (BLE pen strokes OR camera photos)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST /v1/submissions                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
           ┌─────────────┴──────────────┐
           ▼                            ▼
┌─────────────────────┐     ┌──────────────────────┐
│   Path A: BLE Pen   │     │  Path B: Camera/Scan │
│                     │     │                      │
│  Stroke vectors     │     │  JPEG/PNG images     │
│  (x, y, t, p)      │     │                      │
│       │             │     │       │              │
│       ▼             │     │       ▼              │
│  ONNX HWR Model    │     │  Preprocessing       │
│                     │     │  (deskew/crop/binar.) │
│       │             │     │       │              │
│       ▼             │     │       ▼              │
│  Text blocks +      │     │  PaddleOCR           │
│  bounding boxes +   │     │       │              │
│  confidence         │     │       ▼              │
│       │             │     │  Text blocks +       │
│       │             │     │  bounding boxes +    │
│       │             │     │  confidence          │
└───────┬─────────────┘     └──────────┬───────────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   Unified Page OCR Object    │
        │   List[TextBlock] per page   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │     BOUNDARY DETECTOR        │
        │                              │
        │  Pen: stroke geometry filter │
        │  Cam: Hough transform        │
        │         │                    │
        │         ▼                    │
        │  List of boundary Y-pos      │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │     Q MARKER PARSER          │
        │                              │
        │  Regex: Q.No X.Ans variants  │
        │         │                    │
        │         ▼                    │
        │  Question ID per segment     │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │     SEGMENTER                │
        │                              │
        │  Boundaries + Markers        │
        │  → split into segments       │
        │  → cross-page stitching      │
        │  → clubbed-response check    │
        │         │                    │
        │         ▼                    │
        │  List[ResponseSegment]       │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   CONTENT CLASSIFIER         │
        │                              │
        │  Per segment:                │
        │  ├─ text_ratio               │
        │  ├─ diagram_detected (bool)  │
        │  ├─ table_detected (bool)    │
        │  └─ content_type enum        │
        │         │                    │
        │         ▼                    │
        │  Enriched ResponseSegment    │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   RESPONSE STORED IN DB      │
        │   + FLAGS COMPUTED           │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  CLIENT FETCHES RESPONSES    │
        │  GET /v1/submissions/{id}/   │
        │      responses               │
        │                              │
        │  Each response includes:     │
        │  ├─ detected_text            │
        │  ├─ eval_status              │
        │  ├─ content_type             │
        │  └─ flags[]  (unified)       │
        └──────────────┬───────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐        ┌───────────────────┐
  │ NO FLAGS or   │        │ HAS BLOCKING      │
  │ only warnings │        │ FLAGS             │
  │               │        │                   │
  │ Auto-evaluate │        │ Queue for teacher │
  │ via POST      │        │ review            │
  │ /v1/evaluate  │        │                   │
  └───────┬───────┘        └───────────────────┘
          │
          ▼
  ┌──────────────────────────────────────────────────┐
  │        EVAL CORE                                  │
  │                                                   │
  │  ┌─────────────────────┐                         │
  │  │ Solution Cache      │                         │
  │  │ Lookup question_id  │                         │
  │  └─────────┬───────────┘                         │
  │            │                                     │
  │   ┌────────┴─────────┐                           │
  │   ▼                  ▼                           │
  │ CACHE HIT         CACHE MISS                     │
  │   │                  │                           │
  │   ▼                  ▼                           │
  │ Compare-only      Router selects tier            │
  │   │                  │                           │
  │   └──────┬───────────┘                           │
  │          │                                       │
  │          ▼                                       │
  │  ┌───────────────────────────────┐               │
  │  │  Build prompt from template   │               │
  │  └───────────────┬───────────────┘               │
  │                  │                               │
  │                  ▼                               │
  │  ╔═══════════════════════════════════════════╗   │
  │  ║           LLM GATE (single door)          ║   │
  │  ║                                           ║   │
  │  ║  1. Budget check (daily/weekly/monthly)   ║   │
  │  ║     │                                     ║   │
  │  ║     ├─ EXHAUSTED → HTTP 429               ║   │
  │  ║     │                                     ║   │
  │  ║     ▼                                     ║   │
  │  ║  2. Per-call token limits                 ║   │
  │  ║     │                                     ║   │
  │  ║     ├─ EXCEEDED → HTTP 413                ║   │
  │  ║     │                                     ║   │
  │  ║     ▼                                     ║   │
  │  ║  3. LLM Client (LiteLLM) → API call      ║   │
  │  ║     │                                     ║   │
  │  ║     ▼                                     ║   │
  │  ║  4. Token Logger → token_usage_log        ║   │
  │  ║     │                                     ║   │
  │  ║     ▼                                     ║   │
  │  ║  5. Return response + usage metadata      ║   │
  │  ╚═══════════════════════╤═══════════════════╝   │
  │                          │                       │
  │                          ▼                       │
  │    EvalResult:                                   │
  │    ├─ total_score                                │
  │    ├─ max_score                                  │
  │    ├─ step_marks[]                               │
  │    ├─ feedback                                   │
  │    ├─ reference_solution                         │
  │    ├─ content_flags[]                            │
  │    └─ token_usage {}                             │
  └──────────────────────────────────────────────────┘
```

### 2.2 Segmentation Detail — Single Page

```
  Page Top (implicit boundary if no lines here)
  ║
  ║   ┌─────────────────────────────────────────┐
  ║   │  ════════════════════════════════════    │ ← Boundary B0
  ║   │  ════════════════════════════════════    │
  ║   │                                         │
  ║   │  Q.No 1.Ans                             │ ← Marker M0 → question_id = "Q1"
  ║   │  F = ma                                 │
  ║   │  m = 5kg, F = 49N                       │ ← Segment S0 content
  ║   │  a = F/m = 9.8 m/s²                    │
  ║   │                                         │
  ║   │  ════════════════════════════════════    │ ← Boundary B1
  ║   │  ════════════════════════════════════    │
  ║   │                                         │
  ║   │  Q.No 2.Ans                             │ ← Marker M1 → question_id = "Q2"
  ║   │  [diagram of circuit]                   │ ← Content classifier detects diagram
  ║   │  V = IR                                 │ ← Segment S1 content (mixed)
  ║   │  I = V/R = 12/4 = 3A                   │
  ║   │                                         │
  ║   │  ════════════════════════════════════    │ ← Boundary B2
  ║   │  ════════════════════════════════════    │
  ║   │                                         │
  ║   │  Q.No 3.Ans                             │ ← Marker M2 → question_id = "Q3"
  ║   │  The mitochondria is responsible        │
  ║   │  for ATP production...                  │ ← Segment S2 content (text-only)
  ║   │  [labeled diagram of mitochondria]      │ ← diagram region detected
  ║   │                                         │
  ║   └─────────────────────────────────────────┘
  ║
  Page Bottom (implicit boundary if no lines here)


  Segmenter output for this page:

  S0: { question_id: "Q1", text: "F=ma...", content_type: "text_only",    flags: [] }
  S1: { question_id: "Q2", text: "V=IR...", content_type: "mixed",        flags: ["diagram_present"] }
  S2: { question_id: "Q3", text: "The mi..", content_type: "mixed",       flags: ["diagram_present"] }
```

### 2.3 Clubbed Response Detection — Failsafe Flow

```
  ┌───────────────────────────────────────────────────┐
  │  Segment S0 detected between B0 and B1            │
  │  Marker found: Q.No 1.Ans                         │
  └───────────────────────┬───────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────┐
  │  CLUBBED RESPONSE HEURISTIC CHECK                 │
  │                                                   │
  │  1. Scan segment text for ADDITIONAL Q markers    │
  │     beyond the first one                          │
  │     "Q.No 1.Ans ... [content] ... Q.No 2.Ans"    │
  │     → CLUBBED: 2 markers, 0 boundary between     │
  │                                                   │
  │  2. Check segment length vs expected              │
  │     exam_id metadata says Q1 = 3 marks            │
  │     expected_length: "short" (~50-100 words)      │
  │     actual segment: 400+ words                    │
  │     → SUSPICIOUS: length >> expected              │
  │                                                   │
  │  3. Topic discontinuity (optional, LLM-assisted)  │
  │     Q1 is about Newton's laws                     │
  │     Segment has both mechanics AND optics content  │
  │     → SUSPICIOUS: topic shift mid-segment         │
  │     NOTE: This call also goes through LLM Gate    │
  │                                                   │
  │  4. Exam manifest cross-check                     │
  │     Exam has Q1-Q10. Segments found: Q1,Q3-Q10   │
  │     Q2 is missing from segments                   │
  │     → CHECK: Q2 may be clubbed inside Q1 or Q3   │
  └───────────────────────┬───────────────────────────┘
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
     No issues found            Issue detected
              │                        │
              ▼                        ▼
     Proceed to eval            Flag with reason:
                                ├─ "clubbed_multiple_markers"
                                ├─ "clubbed_length_anomaly"
                                ├─ "clubbed_topic_discontinuity"
                                └─ "clubbed_missing_question"
```

### 2.4 Content Classification — Diagram Detection Flow

```
  ┌───────────────────────────────────────────────────┐
  │  ResponseSegment from Segmenter                   │
  │  Has: text_blocks[], bboxes, page image ref       │
  └───────────────────────┬───────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────┐
  │  CONTENT CLASSIFIER (per segment)                 │
  │                                                   │
  │  Input: segment bounding box on page image        │
  │                                                   │
  │  Step 1: Compute text coverage ratio              │
  │    text_pixel_area = sum(bbox areas of text)      │
  │    segment_area = total segment region area        │
  │    text_ratio = text_pixel_area / segment_area    │
  │                                                   │
  │  Step 2: Detect non-text regions                  │
  │    segment_image = crop page to segment region    │
  │    non_text_mask = segment_image − text_bboxes    │
  │    Analyze non_text_mask for:                     │
  │    ├─ Connected components > min_size             │
  │    ├─ Line segments (not boundary lines)          │
  │    ├─ Circles, curves (contour analysis)          │
  │    └─ Arrow-like shapes                           │
  │                                                   │
  │  Step 3: Classify                                 │
  │    ┌──────────────┬───────────────────────┐       │
  │    │ text_ratio   │ non_text_features     │       │
  │    ├──────────────┼───────────────────────┤       │
  │    │ > 0.85       │ few/none              │       │
  │    │ → TEXT_ONLY                           │       │
  │    ├──────────────┼───────────────────────┤       │
  │    │ 0.40–0.85    │ significant           │       │
  │    │ → MIXED (text + diagram)             │       │
  │    ├──────────────┼───────────────────────┤       │
  │    │ < 0.40       │ dominant              │       │
  │    │ → DIAGRAM_HEAVY                      │       │
  │    ├──────────────┼───────────────────────┤       │
  │    │ any          │ grid/table structure   │       │
  │    │ → TABLE_DETECTED                     │       │
  │    └──────────────┴───────────────────────┘       │
  │                                                   │
  │  Step 4: Question metadata cross-check            │
  │    questions table has: has_diagram, expects_diagram│
  │    If expects_diagram AND no diagram detected     │
  │    → flag: "expected_diagram_missing"             │
  └───────────────────────┬───────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────┐
  │  EVAL ROUTING DECISION                            │
  │                                                   │
  │  TEXT_ONLY        → auto-evaluate (normal path)   │
  │  MIXED            → evaluate text portion only,   │
  │                     flag: diagram not scored       │
  │  DIAGRAM_HEAVY    → skip auto-eval entirely,      │
  │                     flag: manual_review_required   │
  │  TABLE_DETECTED   → route to ledger/table eval    │
  │                     template if available,         │
  │                     else flag for manual           │
  └───────────────────────────────────────────────────┘
```

### 2.5 LLM Gate — Single Door Architecture

```
  ┌──────────────────────────────────────────────────────────┐
  │  ALL LLM CALLERS                                         │
  │                                                          │
  │  Eval Core ──────────┐                                   │
  │  Cache Warmup ───────┤                                   │
  │  Clubbed H4 (topic)──┤                                   │
  │  Future: Vision LLM──┤                                   │
  │  Future: Diagram Eval┤                                   │
  │                      │                                   │
  │  NO service calls    │                                   │
  │  LLM Client directly │                                   │
  └──────────────────────┼───────────────────────────────────┘
                         │
                         ▼
  ╔══════════════════════════════════════════════════════════╗
  ║                    LLM GATE                              ║
  ║                 (single door)                            ║
  ║                                                          ║
  ║  ┌────────────────────────────────────────────────────┐  ║
  ║  │  STEP 1: BUDGET CHECK                              │  ║
  ║  │                                                    │  ║
  ║  │  Query token_usage_log:                            │  ║
  ║  │    daily_total   vs daily_token_limit              │  ║
  ║  │    weekly_total  vs weekly_token_limit             │  ║
  ║  │    monthly_total vs monthly_token_limit            │  ║
  ║  │                                                    │  ║
  ║  │  ANY limit exceeded → BudgetExhausted (HTTP 429)   │  ║
  ║  │  All null (unlimited) → always passes              │  ║
  ║  └────────────────────┬───────────────────────────────┘  ║
  ║                       │ OK                                ║
  ║                       ▼                                  ║
  ║  ┌────────────────────────────────────────────────────┐  ║
  ║  │  STEP 2: PER-CALL LIMITS                           │  ║
  ║  │                                                    │  ║
  ║  │  Estimate input tokens (char count / 4)            │  ║
  ║  │  If max_input_tokens set AND est > limit           │  ║
  ║  │    → TokenLimitExceeded (HTTP 413)                 │  ║
  ║  │                                                    │  ║
  ║  │  Clamp max_tokens param to max_output_tokens       │  ║
  ║  │  (null = no clamp, use caller's value)             │  ║
  ║  └────────────────────┬───────────────────────────────┘  ║
  ║                       │ OK                                ║
  ║                       ▼                                  ║
  ║  ┌────────────────────────────────────────────────────┐  ║
  ║  │  STEP 3: LLM CLIENT (LiteLLM)                     │  ║
  ║  │                                                    │  ║
  ║  │  litellm.acompletion(                              │  ║
  ║  │    model=model_id,                                 │  ║
  ║  │    messages=messages,                              │  ║
  ║  │    temperature=0,                                  │  ║
  ║  │    max_tokens=clamped_value                        │  ║
  ║  │  )                                                 │  ║
  ║  │                                                    │  ║
  ║  │  Returns: response + usage object                  │  ║
  ║  └────────────────────┬───────────────────────────────┘  ║
  ║                       │                                  ║
  ║                       ▼                                  ║
  ║  ┌────────────────────────────────────────────────────┐  ║
  ║  │  STEP 4: TOKEN LOGGER                              │  ║
  ║  │                                                    │  ║
  ║  │  Extract from API response.usage:                  │  ║
  ║  │    input_tokens                                    │  ║
  ║  │    output_tokens                                   │  ║
  ║  │    cache_read_input_tokens                         │  ║
  ║  │    cache_creation_input_tokens                     │  ║
  ║  │                                                    │  ║
  ║  │  Compute: estimated_cost_usd via                   │  ║
  ║  │    litellm.completion_cost(response)               │  ║
  ║  │                                                    │  ║
  ║  │  INSERT INTO token_usage_log(...)                  │  ║
  ║  └────────────────────┬───────────────────────────────┘  ║
  ║                       │                                  ║
  ║                       ▼                                  ║
  ║  ┌────────────────────────────────────────────────────┐  ║
  ║  │  STEP 5: RETURN                                    │  ║
  ║  │                                                    │  ║
  ║  │  GateResponse:                                     │  ║
  ║  │    content: str (LLM output)                       │  ║
  ║  │    usage: TokenUsageRecord                         │  ║
  ║  └────────────────────────────────────────────────────┘  ║
  ╚══════════════════════════════════════════════════════════╝
```

### 2.6 Token Usage Lifecycle — Rollup and Cleanup

```
  token_usage_log (append-only, one row per LLM call)
  │
  │  Kept for: 7 days of raw rows
  │
  │  Every midnight:
  │  ├─ Aggregate yesterday's rows → INSERT token_usage_rollup (period_type='daily')
  │  └─ (raw rows kept until weekly cleanup)
  │
  │  Every Monday:
  │  ├─ DELETE token_usage_log rows older than 7 days
  │  └─ Aggregate last week's daily rollups → INSERT token_usage_rollup (period_type='weekly')
  │
  │  Every 1st of month:
  │  ├─ DELETE daily rollups older than 1 month
  │  └─ Aggregate last month's weekly rollups → INSERT token_usage_rollup (period_type='monthly')
  │
  │  Every 3 months:
  │  └─ DELETE monthly rollups older than 3 months
  │
  ▼

  Data retention at any point in time:

  ┌──────────────────────────────────────────────────────┐
  │  Raw log         │ last 7 days    │ per-call detail  │
  │  Daily rollups   │ last 1 month   │ per-day summary  │
  │  Weekly rollups  │ last 3 months  │ per-week summary │
  │  Monthly rollups │ last 3 months  │ per-month summary│
  └──────────────────────────────────────────────────────┘
```

### 2.7 Service Architecture

```
  ┌────────────────────────────────────────────────────────────┐
  │                   API Gateway (FastAPI)                     │
  │                                                            │
  │  /v1/submissions    /v1/evaluate    /v1/solutions          │
  │  /v1/submissions/{id}/responses                            │
  │  /v1/submissions/{id}/flagged                              │
  │  /v1/usage/current  /v1/usage/history  /v1/usage/config    │
  └────┬──────────────────┬─────────────────┬──────────────────┘
       │                  │                 │
       ▼                  ▼                 ▼
  ┌──────────┐     ┌──────────┐     ┌───────────────┐
  │Segmentat.│     │Eval Core │     │Solution Mgr   │
  │Service   │     │Service   │     │Service        │
  └────┬─────┘     └────┬─────┘     └───────┬───────┘
       │                │                   │
       ▼                │                   │
  ┌──────────┐         │                   │
  │OCR / HWR │         │                   │
  │Service   │         │                   │
  └────┬─────┘         │                   │
       │                │                   │
       ▼                │                   │
  ┌──────────┐         │                   │
  │Content   │         │                   │
  │Classifier│         │                   │
  └────┬─────┘         │                   │
       │                │                   │
       ▼                │                   │
  ┌──────────┐         │                   │
  │Clubbed   │         │                   │
  │Detector  │         │                   │
  └──────────┘         │                   │
                       │                   │
       ┌───────────────┘                   │
       │  (ALL LLM calls)                  │
       ▼                                   │
  ╔══════════════════════════╗              │
  ║      LLM GATE           ║              │
  ║   (single door)         ║              │
  ║                         ║              │
  ║  ┌───────────────────┐  ║              │
  ║  │ Budget Check      │  ║              │
  ║  │ daily/weekly/mo.  │  ║              │
  ║  └────────┬──────────┘  ║              │
  ║           │             ║              │
  ║  ┌────────▼──────────┐  ║              │
  ║  │ Per-Call Limits   │  ║              │
  ║  │ input/output caps │  ║              │
  ║  └────────┬──────────┘  ║              │
  ║           │             ║              │
  ║  ┌────────▼──────────┐  ║              │
  ║  │ LLM Client       │  ║              │
  ║  │ (LiteLLM)        │  ║              │
  ║  └────────┬──────────┘  ║              │
  ║           │             ║              │
  ║  ┌────────▼──────────┐  ║              │
  ║  │ Token Logger   ───╫──╫──→ token_usage_log
  ║  └──────────────────┘  ║              │
  ╚═════════════════════════╝              │
                                           ▼
                                    ┌──────────┐
                                    │PostgreSQL│
                                    └──────────┘
                                    ┌──────────┐
                                    │  MinIO   │
                                    └──────────┘

  ─────────────────────────────────────────────
  Plug-and-Play Interfaces (each replaceable):
  ─────────────────────────────────────────────

  OCR/HWR:          process(page) → List[TextBlock]
  Segmentation:     segment(List[PageOCR]) → List[ResponseSegment]
  Content Classif:  classify(segment, page_image) → ContentClassification
  Clubbed Detect:   check(segment, exam_manifest) → List[Flag]
  LLM Gate:         call(model_id, prompt, caller_id) → GateResponse
  Solution Mgr:     get/put(question_id) → Solution
  Eval Core:        evaluate(response_id, question_id) → EvalResult
```

---

## 3. Dual Input Pipeline

### 3.1 Path A — BLE Pen (Stroke Vectors)

| Stage | Detail |
|-------|--------|
| Raw input | Stroke vectors (x, y, t, pressure) per page, with page identity |
| Processing | Flatten to final page state at submission event |
| Recognition | ONNX HWR model (existing C++ plugin + Python bindings) |
| Output | Text blocks with bounding boxes, confidence scores, page number |

**Stroke history** stored separately in TimescaleDB for analytics. **Not** part of eval pipeline.

### 3.2 Path B — Camera Upload (Images)

| Stage | Detail |
|-------|--------|
| Raw input | JPEG/PNG images of notebook pages |
| Preprocessing | Deskew, crop, binarize (OpenCV) |
| Recognition | PaddleOCR (self-hosted, Apache 2.0, supports English + Devanagari) |
| Output | Text blocks with bounding boxes, confidence scores, page number |

### 3.3 Convergence — Unified Page OCR Object

```json
{
  "page_num": 1,
  "source": "ble_pen | camera",
  "raw_asset_ref": "minio://submissions/{submission_id}/page_{N}.png",
  "text_blocks": [
    { "bbox": [x, y, w, h], "text": "Q.No 1.Ans", "confidence": 0.94 },
    { "bbox": [x, y, w, h], "text": "F = ma, m = 5kg...", "confidence": 0.88 }
  ],
  "boundaries": [
    { "y_mid": 120, "type": "double_line", "confidence": 0.91 }
  ]
}
```

---

## 4. Response Segmentation

### 4.1 Student Instructions (Enforced)

1. **Question marker:** `Q.No X.Ans` (with formatting flexibility)
2. **Double-line boundary:** Two horizontal lines drawn between every pair of adjacent responses

Layout:

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

Each double-line pair = **single delimiter** shared between adjacent responses. Two lines, not four.

### 4.2 Boundary Detection

#### BLE Pen Path

| Parameter | Constraint |
|-----------|-----------|
| Slope | < ±10° from horizontal |
| Length | > 40% of page width |
| Y-gap between pair | 2–15mm |
| Temporal proximity | Both drawn within ~3 seconds |
| Horizontal overlap | > 70% |

#### Camera Path

Canny → HoughLinesP → filter horizontal → sort by Y → pair adjacent with small Y-gap. Parameters: `minLineLength = 0.4 × image_width`, slope ~10°, Y-gap 2–15mm in pixels.

### 4.3 Q Marker Detection

Regex (case-insensitive, post-OCR):

```
/Q\.?\s*(?:No|no)\.?\s*(\d{1,3})\s*(?:[\.\(\s]*([a-z]|[ivx]+|[A-Z])[\)\.]?)?\s*\.?\s*(?:Ans|ans|ANS)\.?/i
```

Captures: `\1` = question number, `\2` = sub-part. Output: `Q1`, `Q1a`, `Q3ii`.

Post-OCR fixes in marker context: `l`→`1`, `O`→`0`, `I`→`1`.

### 4.4 Segmentation Algorithm

```
Input:  sorted boundaries + OCR text blocks (one page)
Output: segments with question_id + text

1. Implicit boundaries at page top (Y=0) and bottom (Y=PAGE_HEIGHT)
2. For each consecutive boundary pair [B_i, B_{i+1}]:
     a. Collect text blocks with Y-center between B_i and B_{i+1}
     b. Empty → skip
     c. Scan first block(s) for Q marker
     d. Remaining blocks → response text
     e. Emit segment
```

### 4.5 Cross-Page Stitching

| Page Transition | Meaning |
|----------------|---------|
| Page N ends without closing boundary | Response continues on N+1 |
| Page N+1 starts with boundary | Previous response closed at N bottom |
| Page N+1 starts with content, no boundary | Continuation |
| Page N+1 has Q marker without boundary | Student forgot lines. Associate, flag. |

### 4.6 Edge Cases

| Case | Handling |
|------|---------|
| No opening boundary at page top | Page top = implicit boundary |
| No closing boundary at page bottom | Page bottom = implicit boundary |
| Single line instead of double | Accept. Lower confidence. |
| Triple lines | 2+ horizontal lines within ~20mm = one boundary |
| Duplicate Q markers in segment | Take first. All content → that Q. |
| OCR confidence < 0.60 overall | Reject. Request re-upload. |

---

## 5. Content Classifier — Diagram Detection

### 5.1 Purpose

Classify each response segment's content type **before** sending to eval. Determines auto-evaluate, partial-evaluate, or manual review routing.

### 5.2 Content Types

| Type | Definition | Eval Path |
|------|-----------|-----------|
| `TEXT_ONLY` | > 85% text coverage, no significant non-text features | Auto-evaluate (normal) |
| `MIXED` | 40–85% text, diagram/figure present alongside text | Evaluate text portion. Flag diagram as unscored. |
| `DIAGRAM_HEAVY` | < 40% text, diagram/figure dominant | Skip auto-eval. Manual review required. |
| `TABLE_PRESENT` | Grid/tabular structure detected | Route to table eval template if available, else flag. |

### 5.3 Detection Method — CV Heuristics (No ML)

```python
def classify_content(segment_bbox, page_image, text_blocks):
    segment_crop = page_image[y:y+h, x:x+w]
    segment_area = w * h

    text_pixel_area = sum(tb.w * tb.h for tb in text_blocks)
    text_ratio = text_pixel_area / segment_area

    non_text_mask = np.ones_like(segment_crop)
    for tb in text_blocks:
        non_text_mask[tb.y:tb.y+tb.h, tb.x:tb.x+tb.w] = 0

    edges = cv2.Canny(segment_crop * non_text_mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 0.005 * segment_area]

    horizontal_lines = detect_horizontal_lines(edges, min_length=0.3*w)
    vertical_lines = detect_vertical_lines(edges, min_length=0.3*h)
    has_table = len(horizontal_lines) >= 3 and len(vertical_lines) >= 2

    if has_table:
        return ContentType.TABLE_PRESENT
    elif text_ratio > 0.85 and len(significant_contours) < 3:
        return ContentType.TEXT_ONLY
    elif text_ratio < 0.40 or len(significant_contours) > 10:
        return ContentType.DIAGRAM_HEAVY
    else:
        return ContentType.MIXED
```

### 5.4 Future-Proofing — Plugin Interface

```python
class ContentClassifierInterface(ABC):
    @abstractmethod
    def classify(self, segment_bbox, page_image, text_blocks) -> ContentClassification:
        pass

# Current: services/content_classifier/cv_heuristic.py
# Future:  services/content_classifier/vision_llm.py (same interface, swap in config)
```

Eval core consumes `ContentClassification` object only. Never knows which classifier generated it.

### 5.5 Question Metadata Cross-Check

| `expects_diagram` | Diagram Detected | Action |
|-------------------|-----------------|--------|
| true | yes | Flag text as partial, note diagram unscored |
| true | no | Flag: `expected_diagram_missing` |
| false | yes | Evaluate text only. No penalty. |
| false | no | Normal text eval |

`diagram_weight` (0.0–1.0): fraction of marks attributable to diagram. Used to prorate:

```
scoreable_marks = max_marks × (1 - diagram_weight)
```

---

## 6. Clubbed Response Failsafe

### 6.1 Detection Heuristics — Per Segment

| Heuristic | Detection | Confidence |
|-----------|----------|-----------|
| **H1: Multiple markers** | Regex finds >1 Q marker in one segment | Very high |
| **H2: Length anomaly** | Word count >> expected for question's marks/type | Medium |
| **H3: Missing question** | Exam manifest question not in any segment | High |
| **H4: Topic discontinuity** | LLM check (goes through LLM Gate, counts against budget) | Medium-high |

### 6.2 Expected Length Estimation

| Question Type | Max Marks | Expected Words |
|--------------|-----------|---------------|
| Factual (1-2 marks) | 1–2 | 20–60 |
| Short answer (3 marks) | 3 | 50–120 |
| Numerical (3-5 marks) | 3–5 | 30–80 |
| Long answer (5 marks) | 5 | 100–250 |
| Essay (8-10 marks) | 8–10 | 200–400 |

Anomaly threshold: `actual > expected_max × 2.5`

### 6.3 Flag Severity

| Severity | Meaning | API Behavior |
|----------|---------|-------------|
| `blocking` | Cannot auto-evaluate | `eval_status: "blocked"` |
| `warning` | Can evaluate, result may be unreliable | `eval_status: "evaluated_with_warnings"` |
| `info` | FYI only | `eval_status: "evaluated"` |

---

## 7. Unified Flag System

### 7.1 Flag Schema

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

### 7.2 Complete Flag Type Registry

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

---

## 8. LLM Gate — Single Door Token Controller

### 8.1 Purpose

Every LLM call in the entire system goes through one gate. No service calls the LLM client directly. The gate enforces per-call limits, tracks token budgets, and logs all usage.

### 8.2 Gate Configuration

```python
@dataclass
class GateConfig:
    max_input_tokens: int | None     # None = unlimited. Per-call ceiling.
    max_output_tokens: int | None    # None = unlimited. Clamped on API param.

@dataclass
class BudgetConfig:
    daily_token_limit: int | None    # None = unlimited
    weekly_token_limit: int | None   # None = unlimited
    monthly_token_limit: int | None  # None = unlimited
```

All limits independently settable. `None` = unlimited for that dimension.

### 8.3 Gate Implementation

```python
class LLMGate:
    def __init__(self, gate_config: GateConfig, budget_config: BudgetConfig):
        self.gate = gate_config
        self.budget = budget_config
        self.client = LLMClient()

    async def call(self, model_id: str, prompt: str, **kwargs) -> GateResponse:
        # --- PRE-CALL ---
        estimated_input = self._estimate_tokens(prompt)

        # Per-call input limit
        if self.gate.max_input_tokens and estimated_input > self.gate.max_input_tokens:
            raise TokenLimitExceeded(
                limit_type="per_call_input",
                requested=estimated_input,
                allowed=self.gate.max_input_tokens
            )

        # Budget headroom
        usage_now = await self._get_current_usage()
        if not self._has_budget(usage_now, estimated_input):
            raise BudgetExhausted(
                period=self._exhausted_period(usage_now),
                used=usage_now,
                limit=self._active_limit(usage_now)
            )

        # Clamp output tokens
        call_kwargs = {**kwargs}
        if self.gate.max_output_tokens:
            call_kwargs["max_tokens"] = min(
                kwargs.get("max_tokens", 4096),
                self.gate.max_output_tokens
            )

        # --- CALL ---
        response = await self.client.call(model_id, prompt, **call_kwargs)

        # --- POST-CALL ---
        token_record = TokenUsageRecord(
            model=model_id,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cache_read_tokens=getattr(response.usage, 'cache_read_input_tokens', 0),
            cache_creation_tokens=getattr(response.usage, 'cache_creation_input_tokens', 0),
            estimated_cost_usd=litellm.completion_cost(response),
            timestamp=datetime.utcnow(),
            caller=kwargs.get("caller_id", "unknown")
        )
        await self._record_usage(token_record)

        return GateResponse(content=response.content, usage=token_record)

    async def _has_budget(self, usage: CurrentUsage, estimated_input: int) -> bool:
        total_estimate = usage.daily_total + estimated_input  # conservative
        if self.budget.daily_token_limit is not None:
            if total_estimate > self.budget.daily_token_limit:
                return False
        if self.budget.weekly_token_limit is not None:
            if usage.weekly_total + estimated_input > self.budget.weekly_token_limit:
                return False
        if self.budget.monthly_token_limit is not None:
            if usage.monthly_total + estimated_input > self.budget.monthly_token_limit:
                return False
        return True

    async def _get_current_usage(self) -> CurrentUsage:
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = today_start.replace(day=1)

        row = await db.fetchone("""
            SELECT
                COALESCE(SUM(CASE WHEN called_at >= $1 THEN total_tokens END), 0) as daily_total,
                COALESCE(SUM(CASE WHEN called_at >= $2 THEN total_tokens END), 0) as weekly_total,
                COALESCE(SUM(CASE WHEN called_at >= $3 THEN total_tokens END), 0) as monthly_total
            FROM token_usage_log
            WHERE called_at >= $3
        """, today_start, week_start, month_start)

        return CurrentUsage(**row)
```

### 8.4 Token Usage Record

```python
@dataclass
class TokenUsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    total_tokens: int              # input + output
    estimated_cost_usd: float
    timestamp: datetime
    caller: str                    # 'eval_core', 'cache_warmup', 'clubbed_h4'
```

### 8.5 Error Responses

Budget exhausted:
```json
HTTP 429
{
  "error": "budget_exhausted",
  "period": "daily",
  "used_tokens": 498500,
  "limit_tokens": 500000,
  "resets_at": "2026-03-25T00:00:00Z",
  "suggestion": "Increase daily_token_limit via PUT /v1/usage/config or wait for reset."
}
```

Per-call limit exceeded:
```json
HTTP 413
{
  "error": "token_limit_exceeded",
  "limit_type": "per_call_input",
  "estimated_tokens": 6200,
  "allowed_tokens": 4000,
  "suggestion": "Reduce prompt size or increase max_input_tokens in gate config."
}
```

Batch evaluate: if budget runs out mid-batch, completed evals returned with partial results + budget error on remaining items.

---

## 9. API Design

### 9.1 Design Principles

| Principle | Implementation |
|-----------|---------------|
| Plug-and-play | Standalone service. Any frontend can call it. |
| Stateless | No session. Every request carries full context. |
| Async-capable | Support webhook/polling for long evals. |
| Idempotent | Same (submission_id, question_id) returns cached result. |
| Unified flags | Every response carries flags from all subsystems in one array. |
| Token-aware | Every eval response includes token usage. Usage monitoring API available. |

### 9.2 Endpoints

#### 9.2.1 Submit Pages for Segmentation

```
POST /v1/submissions
```

Request:
```json
{
  "submission_id": "SUB-20260324-A1B2",
  "student_id": "STU-042",
  "exam_id": "EXAM-PHY-2026-03",
  "source": "ble_pen",
  "pages": [
    {
      "page_num": 1,
      "asset_ref": "minio://submissions/SUB-.../page_001.png",
      "strokes": [ ]
    }
  ]
}
```

Response:
```json
{
  "submission_id": "SUB-20260324-A1B2",
  "status": "segmented",
  "responses": [
    {
      "response_id": "RESP-001",
      "question_id": "Q1",
      "sub_part": null,
      "source_pages": [1],
      "detected_text": "F = ma, m = 5kg...",
      "content_type": "text_only",
      "eval_status": "ready",
      "segmentation_confidence": 0.92,
      "segmentation_method": "boundary+marker",
      "ocr_confidence": 0.88,
      "flags": []
    },
    {
      "response_id": "RESP-002",
      "question_id": "Q2",
      "sub_part": "a",
      "source_pages": [1, 2],
      "detected_text": "V = IR, I = V/R = 12/4 = 3A",
      "content_type": "mixed",
      "eval_status": "ready_with_warnings",
      "segmentation_confidence": 0.85,
      "segmentation_method": "boundary+marker",
      "ocr_confidence": 0.90,
      "flags": [
        {
          "flag_id": "FLG-002",
          "source": "content_classifier",
          "flag_type": "diagram_present",
          "severity": "info",
          "reason": "Circuit diagram detected alongside text. Text will be evaluated; diagram portion unscored.",
          "suggested_action": "review_diagram_marks"
        }
      ]
    },
    {
      "response_id": "RESP-003",
      "question_id": "Q3",
      "sub_part": null,
      "source_pages": [3],
      "detected_text": "The mitochondria is...",
      "content_type": "diagram_heavy",
      "eval_status": "blocked",
      "segmentation_confidence": 0.88,
      "segmentation_method": "boundary+marker",
      "ocr_confidence": 0.75,
      "flags": [
        {
          "flag_id": "FLG-003",
          "source": "content_classifier",
          "flag_type": "diagram_heavy_content",
          "severity": "blocking",
          "reason": "Response is primarily a labeled diagram (text_ratio: 0.28). Cannot auto-evaluate.",
          "suggested_action": "manual_review_required"
        }
      ]
    },
    {
      "response_id": "RESP-004",
      "question_id": "Q4",
      "sub_part": null,
      "source_pages": [3, 4],
      "detected_text": "Q.No 4.Ans Newton's first law... Q.No 5.Ans The acceleration...",
      "content_type": "text_only",
      "eval_status": "blocked",
      "segmentation_confidence": 0.60,
      "segmentation_method": "marker_only",
      "ocr_confidence": 0.91,
      "flags": [
        {
          "flag_id": "FLG-004",
          "source": "clubbed_detector",
          "flag_type": "clubbed_multiple_markers",
          "severity": "blocking",
          "reason": "Found 2 question markers (Q4, Q5) in single segment. Student likely missed boundary lines.",
          "suggested_action": "split_segment",
          "metadata": { "detected_markers": ["Q4", "Q5"] }
        }
      ]
    }
  ],
  "submission_flags": [
    {
      "flag_id": "FLG-005",
      "source": "clubbed_detector",
      "flag_type": "clubbed_missing_question",
      "severity": "warning",
      "reason": "Question Q5 from exam manifest not found as separate segment. May be clubbed inside RESP-004.",
      "suggested_action": "review_for_clubbing",
      "metadata": { "missing_question_id": "Q5", "suspect_segments": ["RESP-004"] }
    }
  ]
}
```

**`eval_status` field:**

| Value | Meaning | Client Action |
|-------|---------|--------------|
| `ready` | No blocking flags | Call `POST /v1/evaluate` |
| `ready_with_warnings` | Warning flags present | Evaluate, show warnings to teacher |
| `blocked` | Blocking flags | Teacher must resolve first |
| `evaluated` | Already evaluated | Fetch result |
| `manual_review` | Content prevents auto-eval | Teacher grades manually |

#### 9.2.2 Fetch Detected Responses

```
GET /v1/submissions/{submission_id}/responses
GET /v1/submissions/{submission_id}/responses/{response_id}
GET /v1/submissions/{id}/responses?eval_status=blocked
GET /v1/submissions/{id}/responses?flag_type=clubbed_multiple_markers
GET /v1/submissions/{id}/responses?content_type=diagram_heavy
```

#### 9.2.3 Fetch Only Flagged Items

```
GET /v1/submissions/{submission_id}/flagged
```

Returns only responses with `severity: "blocking"` or `"warning"`, plus submission-level flags.

#### 9.2.4 Resolve Flags / Manual Override

```
PATCH /v1/submissions/{submission_id}/responses/{response_id}
```

Split clubbed segment:
```json
{ "action": "split", "split_at_marker": "Q5", "resolved_by": "teacher_id" }
```

Override eval status:
```json
{ "action": "override_eval_status", "new_eval_status": "ready", "override_reason": "...", "resolved_by": "teacher_id" }
```

Manual score:
```json
{ "action": "manual_score", "total_score": 4, "max_score": 5, "feedback": "...", "resolved_by": "teacher_id" }
```

#### 9.2.5 Evaluate a Response

```
POST /v1/evaluate
```

Request:
```json
{
  "submission_id": "SUB-20260324-A1B2",
  "response_id": "RESP-001",
  "question_id": "PHY-MECH-042"
}
```

`student_text` not in request body. Engine fetches from `detected_responses` using `response_id`. Prevents client-side text manipulation.

Response:
```json
{
  "evaluation_id": "EVAL-001",
  "response_id": "RESP-001",
  "question_id": "PHY-MECH-042",
  "eval_path": "cache_hit",
  "model_used": "claude-haiku-4-5-20251001",
  "content_type": "text_only",
  "reference_solution": {
    "text": "Given: m=5kg, F=49N. Using F=ma, a=F/m=49/5=9.8 m/s²",
    "source": "teacher",
    "version": 1
  },
  "total_score": 4,
  "max_score": 5,
  "scoreable_max": 5,
  "step_marks": [
    { "step": "Identify given values", "marks_awarded": 1, "marks_possible": 1, "justification": "Correctly identified m and F" },
    { "step": "State relevant formula", "marks_awarded": 1, "marks_possible": 1, "justification": "F=ma stated" },
    { "step": "Rearrange for unknown", "marks_awarded": 1, "marks_possible": 1, "justification": "a=F/m derived" },
    { "step": "Substitute and compute", "marks_awarded": 1, "marks_possible": 1, "justification": "49/5=9.8 correct" },
    { "step": "Final answer with unit", "marks_awarded": 0, "marks_possible": 1, "justification": "Unit m/s² missing" }
  ],
  "overall_feedback": "Correct method and computation. Include SI units in final answer.",
  "flags": [],
  "ocr_concerns": [],
  "token_usage": {
    "input_tokens": 1247,
    "output_tokens": 312,
    "cache_read_tokens": 800,
    "cache_creation_tokens": 0,
    "total_tokens": 1559,
    "model": "claude-haiku-4-5-20251001",
    "estimated_cost_usd": 0.00078
  }
}
```

For MIXED content:
```json
{
  "evaluation_id": "EVAL-002",
  "response_id": "RESP-002",
  "content_type": "mixed",
  "total_score": 3,
  "max_score": 5,
  "scoreable_max": 4,
  "step_marks": [...],
  "diagram_score": null,
  "flags": [
    {
      "flag_type": "partial_eval_diagram_excluded",
      "severity": "info",
      "reason": "1 mark allocated to circuit diagram (diagram_weight: 0.20). Diagram detected but not scored."
    }
  ],
  "token_usage": { ... }
}
```

For BLOCKED responses (HTTP 422):
```json
{
  "error": "evaluation_blocked",
  "response_id": "RESP-003",
  "blocking_flags": [ ... ],
  "resolution_options": [ ... ]
}
```

#### 9.2.6 Fetch Reference Solution

```
GET /v1/solutions/{question_id}
```

#### 9.2.7 Upload / Update Reference Solution (Teacher)

```
PUT /v1/solutions/{question_id}
```

#### 9.2.8 Batch Evaluate

```
POST /v1/evaluate/batch
```

Skips blocked responses. If budget exhausts mid-batch, returns completed evals + budget error on remaining.

```json
{
  "completed": [ ... ],
  "skipped_blocked": [
    { "response_id": "RESP-003", "reason": "diagram_heavy_content" }
  ],
  "failed_budget": [
    { "response_id": "RESP-008", "error": "budget_exhausted", "period": "daily" }
  ],
  "batch_token_usage": {
    "total_input_tokens": 14200,
    "total_output_tokens": 3800,
    "total_cost_usd": 0.0092,
    "calls_made": 6
  }
}
```

#### 9.2.9 Evaluation History

```
GET /v1/evaluations?student_id=STU-042&exam_id=EXAM-PHY-2026-03
```

#### 9.2.10 Token Usage — Current Budget Status

```
GET /v1/usage/current
```

Response:
```json
{
  "daily": {
    "total_tokens": 142800,
    "limit": 500000,
    "remaining": 357200,
    "utilization_pct": 28.6,
    "resets_at": "2026-03-25T00:00:00Z"
  },
  "weekly": {
    "total_tokens": 823000,
    "limit": null,
    "remaining": null,
    "utilization_pct": null,
    "resets_at": null
  },
  "monthly": {
    "total_tokens": 2450000,
    "limit": 10000000,
    "remaining": 7550000,
    "utilization_pct": 24.5,
    "resets_at": "2026-04-01T00:00:00Z"
  },
  "estimated_monthly_cost_usd": 87.50
}
```

#### 9.2.11 Token Usage — Historical

```
GET /v1/usage/history?period=daily&from=2026-03-01&to=2026-03-24
```

Response:
```json
{
  "period": "daily",
  "entries": [
    {
      "period_start": "2026-03-01",
      "total_tokens": 156000,
      "total_cost_usd": 4.23,
      "call_count": 2400,
      "breakdown_by_model": {
        "claude-haiku-4-5-20251001": { "tokens": 140000, "cost_usd": 0.70 },
        "claude-sonnet-4-6": { "tokens": 16000, "cost_usd": 3.53 }
      },
      "breakdown_by_caller": {
        "eval_core": { "tokens": 150000, "calls": 2350 },
        "cache_warmup": { "tokens": 6000, "calls": 50 }
      }
    }
  ]
}
```

#### 9.2.12 Token Usage — Update Config

```
PUT /v1/usage/config
```

Request:
```json
{
  "gate": {
    "max_input_tokens": 4000,
    "max_output_tokens": 2000
  },
  "budget": {
    "daily_token_limit": 500000,
    "weekly_token_limit": null,
    "monthly_token_limit": 10000000
  }
}
```

`null` = unlimited for that dimension.

---

## 10. Evaluation Engine

### 10.1 Subject-Specific Eval Templates

| Template ID | Subjects | Step Marks |
|-------------|----------|-----------|
| `stepwise_numerical` | Physics, Chemistry, Math, Accountancy numericals | Yes |
| `essay_rubric` | English, Hindi essays, History/Geo long answers | No (rubric categories) |
| `factual_recall` | Biology, History, Geography short answers | No |
| `keyword_coverage` | Science definitions, bio processes | No |
| `ledger_tabular` | Accountancy ledger/journal entries | Yes |
| `proof_derivation` | Math proofs, Physics derivations | Yes |

### 10.2 Content-Type Aware Eval Routing

```python
def evaluate(response_id: str, question_id: str) -> EvalResult:
    response = db.get_detected_response(response_id)
    question = db.get_question(question_id)
    classification = response.content_classification

    blocking = [f for f in response.flags if f.severity == "blocking"]
    if blocking:
        raise EvalBlockedError(blocking)

    if classification.content_type == ContentType.MIXED:
        scoreable_max = int(question.max_marks * (1 - question.diagram_weight))
    else:
        scoreable_max = question.max_marks

    solution = solution_cache.get(question_id)

    if solution:
        # Cache hit — compare only, through gate
        gate_response = await gate.call(
            model_id="claude-haiku-4-5-20251001",
            prompt=render_compare_template(response, solution, question, scoreable_max),
            caller_id="eval_core"
        )
    else:
        # Cache miss — generate + evaluate, through gate
        model_id = router.route(question)
        gate_response = await gate.call(
            model_id=model_id,
            prompt=render_generate_template(response, question, scoreable_max),
            caller_id="eval_core"
        )
        # Store generated solution in cache
        solution_cache.put(question_id, gate_response.parsed.reference_solution, source="llm", model=model_id)

    result = parse_eval_result(gate_response)
    result.token_usage = gate_response.usage  # attach token metadata

    return result
```

### 10.3 Complexity Router

| Complexity | Model | Trigger |
|-----------|-------|---------|
| Cache-hit (any) | Haiku | Always — compare only |
| L1 cache-miss | Haiku | Factual, single-step |
| L2 cache-miss | Sonnet | Multi-step, short-answer |
| L3 cache-miss | Opus/Sonnet | Essay, open-ended |

### 10.4 OCR Error Tolerance

Prompt instructs LLM to tolerate: `l`↔`1`, missing superscripts, `rn`→`m`, Devanagari matra errors.

---

## 11. Rubric Schemas — Per Subject Type

### 11.1 Stepwise Numerical

```json
{
  "question_id": "PHY-MECH-042",
  "question_type": "stepwise_numerical",
  "max_marks": 5,
  "diagram_weight": 0.0,
  "rubric": {
    "steps": [
      { "step_id": "S1", "description": "Identify given values", "marks": 1 },
      { "step_id": "S2", "description": "State relevant formula", "marks": 1 },
      { "step_id": "S3", "description": "Correct substitution", "marks": 1 },
      { "step_id": "S4", "description": "Computation", "marks": 1 },
      { "step_id": "S5", "description": "Final answer with unit", "marks": 1 }
    ],
    "final_answer": { "value": 9.8, "unit": "m/s²", "tolerance_percent": 2 },
    "carry_forward_marks": true
  }
}
```

### 11.2 Essay

```json
{
  "question_id": "ENG-ESSAY-015",
  "question_type": "essay",
  "max_marks": 10,
  "diagram_weight": 0.0,
  "rubric": {
    "categories": [
      { "category": "content_relevance", "marks": 3 },
      { "category": "organization", "marks": 2 },
      { "category": "language_quality", "marks": 3 },
      { "category": "grammar_spelling", "marks": 2 }
    ],
    "required_length_words": { "min": 150, "max": 300 }
  }
}
```

### 11.3 Biology with Diagram

```json
{
  "question_id": "BIO-CELL-012",
  "max_marks": 5,
  "expects_diagram": true,
  "diagram_weight": 0.40,
  "rubric": {
    "text_rubric": {
      "required_facts": ["double membrane", "cristae", "matrix", "ATP synthesis"]
    },
    "diagram_rubric": {
      "required_labels": ["outer membrane", "inner membrane", "cristae", "matrix"]
    }
  }
}
```

Without diagram eval: `scoreable_max = 5 × 0.60 = 3`. With future diagram eval: full 5.

### 11.4 Accountancy Ledger

```json
{
  "question_id": "ACC-JNL-003",
  "question_type": "ledger_tabular",
  "max_marks": 6,
  "rubric": {
    "entries": [
      { "entry_id": "E1", "debit_account": "Purchases A/c", "credit_account": "Cash A/c", "amount": 50000, "narration_required": true, "marks": 2 },
      { "entry_id": "E2", "debit_account": "Cash A/c", "credit_account": "Sales A/c", "amount": 75000, "narration_required": true, "marks": 2 }
    ],
    "balancing_marks": 2,
    "tolerance_amount": 0
  }
}
```

---

## 12. Database Schema

```sql
-- Question metadata
CREATE TABLE questions (
    question_id     VARCHAR(64) PRIMARY KEY,
    exam_id         VARCHAR(64),
    subject         VARCHAR(32) NOT NULL,
    grade           INTEGER,
    question_text   TEXT NOT NULL,
    question_type   VARCHAR(32) NOT NULL,
    complexity      VARCHAR(4) NOT NULL DEFAULT 'L1',
    eval_template   VARCHAR(64) NOT NULL,
    max_marks       INTEGER NOT NULL,
    rubric          JSONB NOT NULL,
    has_diagram     BOOLEAN DEFAULT FALSE,
    expects_diagram BOOLEAN DEFAULT FALSE,
    diagram_weight  FLOAT DEFAULT 0.0,
    expected_word_range JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE question_solutions (
    question_id     VARCHAR(64) REFERENCES questions(question_id),
    version         INTEGER NOT NULL DEFAULT 1,
    reference_solution  TEXT NOT NULL,
    solution_source VARCHAR(16) NOT NULL,
    model_used      VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (question_id, version)
);

-- Submissions
CREATE TABLE submissions (
    submission_id   VARCHAR(64) PRIMARY KEY,
    student_id      VARCHAR(64) NOT NULL,
    exam_id         VARCHAR(64) NOT NULL,
    source          VARCHAR(16) NOT NULL,
    page_count      INTEGER NOT NULL,
    submitted_at    TIMESTAMPTZ DEFAULT NOW(),
    segmentation_status VARCHAR(16) DEFAULT 'pending'
);

CREATE TABLE submission_pages (
    submission_id   VARCHAR(64) REFERENCES submissions(submission_id),
    page_num        INTEGER NOT NULL,
    raw_asset_ref   TEXT NOT NULL,
    ocr_result      JSONB,
    boundaries      JSONB,
    PRIMARY KEY (submission_id, page_num)
);

-- Detected responses
CREATE TABLE detected_responses (
    response_id     VARCHAR(64) PRIMARY KEY,
    submission_id   VARCHAR(64) REFERENCES submissions(submission_id),
    question_id     VARCHAR(64),
    detected_text   TEXT NOT NULL,
    source_pages    INTEGER[] NOT NULL,
    content_type    VARCHAR(32) NOT NULL,
    content_classification JSONB,
    eval_status     VARCHAR(32) NOT NULL DEFAULT 'pending',
    segmentation_confidence FLOAT NOT NULL,
    segmentation_method     VARCHAR(32) NOT NULL,
    ocr_confidence  FLOAT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Unified flags
CREATE TABLE response_flags (
    flag_id         VARCHAR(64) PRIMARY KEY,
    response_id     VARCHAR(64) REFERENCES detected_responses(response_id),
    submission_id   VARCHAR(64) REFERENCES submissions(submission_id),
    source          VARCHAR(32) NOT NULL,
    flag_type       VARCHAR(64) NOT NULL,
    severity        VARCHAR(16) NOT NULL,
    reason          TEXT NOT NULL,
    suggested_action VARCHAR(64),
    metadata        JSONB,
    resolved        BOOLEAN DEFAULT FALSE,
    resolved_by     VARCHAR(64),
    resolved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_flags_response ON response_flags(response_id);
CREATE INDEX idx_flags_submission ON response_flags(submission_id);
CREATE INDEX idx_flags_severity ON response_flags(severity);
CREATE INDEX idx_flags_type ON response_flags(flag_type);

-- Evaluations
CREATE TABLE evaluations (
    evaluation_id   VARCHAR(64) PRIMARY KEY,
    response_id     VARCHAR(64) REFERENCES detected_responses(response_id),
    question_id     VARCHAR(64) REFERENCES questions(question_id),
    student_id      VARCHAR(64) NOT NULL,
    eval_path       VARCHAR(16) NOT NULL,
    model_used      VARCHAR(64) NOT NULL,
    content_type    VARCHAR(32) NOT NULL,
    total_score     INTEGER NOT NULL,
    max_score       INTEGER NOT NULL,
    scoreable_max   INTEGER NOT NULL,
    step_marks      JSONB,
    overall_feedback TEXT,
    ocr_concerns    JSONB,
    token_usage     JSONB,
    raw_llm_response JSONB,
    evaluated_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(response_id)
);

CREATE INDEX idx_eval_student ON evaluations(student_id);
CREATE INDEX idx_eval_question ON evaluations(question_id);
CREATE INDEX idx_eval_path ON evaluations(eval_path);

-- Token usage log (append-only, one row per LLM call)
CREATE TABLE token_usage_log (
    id              BIGSERIAL PRIMARY KEY,
    model           VARCHAR(64) NOT NULL,
    caller          VARCHAR(64) NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cache_read_tokens    INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    total_tokens    INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    estimated_cost_usd  NUMERIC(10, 6),
    called_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tul_called_at ON token_usage_log(called_at);
CREATE INDEX idx_tul_caller ON token_usage_log(caller);
CREATE INDEX idx_tul_model ON token_usage_log(model);

-- Token usage rollups (materialized by cron job)
CREATE TABLE token_usage_rollup (
    period_type     VARCHAR(8) NOT NULL,
    period_start    DATE NOT NULL,
    period_end      DATE NOT NULL,
    total_tokens    BIGINT NOT NULL,
    total_input     BIGINT NOT NULL,
    total_output    BIGINT NOT NULL,
    total_cost_usd  NUMERIC(12, 6),
    call_count      INTEGER NOT NULL,
    breakdown_by_model   JSONB,
    breakdown_by_caller  JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (period_type, period_start)
);

-- Gate configuration (single row, runtime-updatable)
CREATE TABLE gate_config (
    id              INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    max_input_tokens    INTEGER,
    max_output_tokens   INTEGER,
    daily_token_limit   BIGINT,
    weekly_token_limit  BIGINT,
    monthly_token_limit BIGINT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO gate_config (max_input_tokens, max_output_tokens, daily_token_limit, weekly_token_limit, monthly_token_limit)
VALUES (NULL, NULL, NULL, NULL, NULL);  -- all unlimited by default
```

---

## 13. Cost Model

Assumptions: 500 students × 8 subjects × 20 questions/subject/month = 80,000 evaluations/month.

| Phase | Cache-Hit Rate | Haiku Calls | Sonnet Calls | Monthly Cost |
|-------|---------------|-------------|-------------|-------------|
| Month 1 | 20% (cold) | 16,000 | 64,000 | ~$656 |
| Month 2 | 70% | 68,000 | 12,000 | ~$188 |
| Month 3+ | 90%+ | 76,000 | 4,000 | ~$116 |
| Steady state (warm) | 98% | 79,200 | 800 | ~$87 |

Cost reduction levers: teacher-uploaded solutions (100% cache hit), prompt caching (~50% reduction), batch API (50% off), shared question banks across schools.

LLM Gate budget config allows setting monthly cap to prevent runaway costs during cold-cache period.

---

## 14. Implementation Phases

### Phase 0 — Foundation + LLM Gate (Week 1–2)

| Task | Output |
|------|--------|
| Database schema migration (all tables: questions, solutions, submissions, responses, flags, evaluations, token_usage_log, token_usage_rollup, gate_config) | SQL migrations |
| Question metadata + solution CRUD API | Endpoints |
| Prompt templates (3: stepwise, essay, factual) | Jinja2 files |
| **LLM Gate with budget checking** | `gate.py`, `budget.py` |
| **Token logger** | `token_logger.py` |
| **LLM Client wrapper (LiteLLM) — called only by gate** | `llm_client.py` |
| **Usage API endpoints** | `routes/usage.py` |
| **Token rollup cron script** | `scripts/token_rollup.py` |
| **Gate config defaults (all unlimited)** | `config.py`, DB seed |

**Why gate in Phase 0:** If the client is built without the gate first, every service that calls it directly becomes refactor debt. Gate-first means every service is gate-aware from day one.

### Phase 1 — Eval Core (Week 3–4)

| Task | Output |
|------|--------|
| Cache-hit eval endpoint (calls gate, not client directly) | `POST /v1/evaluate` |
| Cache-miss flow (calls gate with routed model) | generate + store + evaluate |
| Batch cache warm-up script (calls gate, usage logged) | CLI tool |
| Eval result storage + retrieval API (includes token_usage) | `GET /v1/evaluations` |

**Milestone:** Evaluate manually-provided text. Token usage tracked from first call.

### Phase 2 — Segmentation + Content Classification (Week 5–7)

| Task | Output |
|------|--------|
| Boundary detector (pen + camera) | `boundary_detector.py` |
| Q marker regex parser | `marker_parser.py` |
| Segmenter (state machine + cross-page) | `segmenter.py` |
| Content classifier (CV heuristic v1) | `cv_heuristic.py` |
| Clubbed response detector | `clubbed_detector.py` |
| Unified flag system | `flags.py` |
| `POST /v1/submissions` endpoint | Full pipeline |
| `GET /v1/submissions/{id}/responses` | With flags, content_type, eval_status |
| `GET /v1/submissions/{id}/flagged` | Filtered view |

**Milestone:** Accept raw pages → segmented responses with flags.

### Phase 3 — Integration + Resolution (Week 8–9)

| Task | Output |
|------|--------|
| Wire segmentation → eval (auto-eval ready responses) | Integration |
| Batch evaluate (budget-aware — partial results on exhaustion) | `POST /v1/evaluate/batch` |
| Flag resolution API | `PATCH /v1/submissions/{id}/responses/{id}` |
| Manual score API | Same PATCH endpoint |
| Teacher review queue | `GET /v1/submissions/{id}/flagged` + filters |

### Phase 4 — Subject Expansion (Week 10–12)

| Task | Output |
|------|--------|
| Ledger/table eval template | `ledger_tabular.jinja2` |
| Proof/derivation template | `proof_derivation.jinja2` |
| Hindi Devanagari prompts | Hindi-specific templates |
| PaddleOCR Devanagari benchmarking | Accuracy report |

### Phase 5 — Diagram Eval Prep (Week 13+)

| Task | Output |
|------|--------|
| Vision LLM classifier (implements same interface) | `vision_llm.py` |
| Diagram eval prompt template | `diagram_eval.jinja2` |
| Swap classifier in config (no code change elsewhere) | Config update |
| Benchmark diagram eval accuracy | Report |

### Phase 6 — Optimization (Week 15+)

| Task | Output |
|------|--------|
| Prompt caching integration | Cost reduction |
| Batch API for non-urgent evals | 50% cost cut |
| Analytics pipeline (stroke data) | Separate service |
| Token usage dashboard (reads rollup data) | Monitoring view |
| Budget alerting (80% threshold warning flag) | Gate enhancement |

---

## 15. Folder Structure

```
eval-engine/
├── docker-compose.yml
├── api/
│   ├── main.py
│   ├── routes/
│   │   ├── submissions.py          # POST submissions, GET responses, GET flagged, PATCH resolve
│   │   ├── evaluate.py             # POST evaluate, POST batch, GET evaluations
│   │   ├── solutions.py            # GET/PUT solutions, POST questions
│   │   └── usage.py                # GET /v1/usage/current, history. PUT /v1/usage/config
│   └── models/
│       ├── submission.py           # Pydantic: Submission, ResponseSegment, Flag
│       ├── evaluation.py           # Pydantic: EvalResult, StepMark, TokenUsage
│       ├── solution.py             # Pydantic: Solution, Rubric
│       ├── flags.py                # Pydantic: Flag, FlagType enum, Severity enum
│       └── usage.py                # Pydantic: UsageCurrent, UsageHistory, GateConfigUpdate
├── services/
│   ├── ocr/
│   │   ├── interface.py
│   │   ├── paddle_ocr.py
│   │   └── onnx_hwr.py
│   ├── segmentation/
│   │   ├── boundary_detector.py
│   │   ├── marker_parser.py
│   │   ├── segmenter.py
│   │   └── confidence.py
│   ├── content_classifier/
│   │   ├── interface.py            # ContentClassifierInterface ABC
│   │   ├── cv_heuristic.py         # Current: OpenCV-based
│   │   └── vision_llm.py           # Future: Vision model-based (stub)
│   ├── clubbed_detector/
│   │   └── detector.py             # H1-H4 heuristics
│   ├── flags/
│   │   ├── registry.py             # All flag types, severities
│   │   └── resolver.py             # Flag resolution logic
│   ├── llm_gate/                   # THE SINGLE DOOR
│   │   ├── gate.py                 # LLMGate class — all LLM calls go through here
│   │   ├── budget.py               # Budget checking, CurrentUsage queries
│   │   ├── token_logger.py         # Append to token_usage_log
│   │   └── config.py               # GateConfig, BudgetConfig, runtime updates
│   ├── eval/
│   │   ├── eval_core.py            # Calls gate.call(), never llm_client directly
│   │   └── llm_router.py           # Complexity → model tier selection
│   └── solutions/
│       ├── cache.py
│       └── warmup.py               # Calls gate.call() for batch generation
├── prompts/
│   ├── eval/
│   │   ├── stepwise_numerical.jinja2
│   │   ├── essay_rubric.jinja2
│   │   ├── factual_recall.jinja2
│   │   ├── keyword_coverage.jinja2
│   │   ├── ledger_tabular.jinja2
│   │   ├── proof_derivation.jinja2
│   │   └── diagram_eval.jinja2     # Future: stub
│   └── generate/
│       ├── solution_stepwise.jinja2
│       ├── solution_essay.jinja2
│       └── solution_factual.jinja2
├── db/
│   └── migrations/
│       ├── 001_initial.sql         # questions, solutions, submissions, pages, responses
│       ├── 002_content_flags.sql   # response_flags, content classification columns
│       └── 003_token_usage.sql     # token_usage_log, token_usage_rollup, gate_config
├── tests/
│   ├── test_segmentation.py
│   ├── test_marker_parser.py
│   ├── test_content_classifier.py
│   ├── test_clubbed_detector.py
│   ├── test_eval_core.py
│   ├── test_llm_gate.py           # Gate budget checks, limit enforcement
│   ├── test_token_rollup.py       # Rollup + cleanup logic
│   └── fixtures/
│       ├── sample_pages/
│       ├── sample_responses/
│       └── sample_diagrams/
├── scripts/
│   ├── cache_warmup.py
│   └── token_rollup.py            # Cron: daily/weekly/monthly rollup + cleanup
└── cron/
    └── crontab                     # Schedule for token_rollup.py
```

---

## 16. Risk Register

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|-----------|
| OCR too low for Hindi handwriting | Med | High | Benchmark PaddleOCR Devanagari. Fallback: Google Vision. |
| LLM inconsistent scores | Med | High | temperature=0. Cache results. Dual-eval sample, flag divergence >1 mark. |
| Students ignore boundary/marker rules | High | Med | Confidence gating. Teacher review. Iterative instruction improvement. |
| Diagram detection false positives | Med | Med | Conservative thresholds. Teacher override via PATCH. |
| Diagram detection false negatives | Med | Med | `expects_diagram` cross-check. |
| Clubbed responses undetected | Low | High | H3 (missing question) catches most. H4 as P2. |
| Cost escalation with low cache-hit | Low | Med | Gate budget limits. Mandatory cache warm-up. Teacher solution upload. |
| Accountancy ledger OCR fails | Med | Med | Table detection routes to specialized template. |
| Content classifier misclassifies tables as diagrams | Low | Med | Table detection runs first, takes priority. |
| Budget exhaustion during batch exam grading | Med | Med | Gate returns partial results. Alert at 80% utilization. Admin can increase limit via API. |
| Token rollup cron failure | Low | Low | Raw logs retained for 7 days regardless. Rollup can rerun. Idempotent inserts. |
| Gate adds latency to LLM calls | Low | Low | Budget check is single indexed query (~1ms). Negligible vs LLM latency (~2-15s). |
