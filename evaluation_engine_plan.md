# Evaluation Engine — System Design Document

> **Paginated Correctness Recognition (PCR) Engine**
> Standalone, plug-and-play evaluation system for student answer correctness.
> Backend-first. MongoDB. Clean modular code.

---

## 1. What This System Does

Takes a **question + student answer** (text, handwritten images, or both) and returns a **structured evaluation** — is it correct, what score, what feedback, what went wrong, and what the correct solution is.

```
INPUT  →  [ Evaluation Engine ]  →  OUTPUT
```

- **INPUT:** Question text, options, correct answer, student answer (typed/handwritten images), question figures
- **OUTPUT:** is_correct, score (0-1), extracted_answer, feedback, reasoning, correct_solution, token_usage

The engine **never touches any database** for question data. The caller loads the question, packs it into `EvalRequest`, and gets back `EvalResult`. This is what makes it plug-and-play.

---

## 2. Public API — How Any Product Uses This

```python
from evaluation_engine import evaluate, EvalRequest, EvalResult, EngineConfig

# Configure once
config = EngineConfig(
    openai_api_key="sk-...",
    openai_model="gpt-4o",
    mongodb_uri="mongodb://...",   # OPTIONAL — for token tracking/budget
)

# Evaluate any student answer
result = await evaluate(
    EvalRequest(
        question_text="What is Newton's second law?",
        correct_answer="F = ma",
        student_canvas_pages=["data:image/png;base64,..."],
    ),
    config=config,
)

print(result.is_correct)           # True
print(result.score)                # 0.85
print(result.feedback)             # "Good understanding of force and acceleration..."
print(result.token_usage.total_tokens)  # 1559
```

That's it. Three imports, one config, one function call.

---

## 3. Input Model — `EvalRequest`

Everything the engine needs arrives in this single object. No database IDs, no internal refs — just data.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question_text` | `str` | Yes | The question being evaluated |
| `question_type` | `QuestionType` | No | `"mcq"` / `"essay"` / `"numerical"` / `"subjective"` — auto-detected if omitted |
| `options` | `List[OptionItem]` | No | MCQ options: `[{label: "A", content: "7 days"}, ...]` |
| `correct_answer` | `str` | No | Admin-provided answer. If omitted, LLM solves the question itself |
| `question_images` | `List[str]` | No | Question figure images (base64 data URLs) |
| `student_answer_text` | `str` | No | Typed answer text |
| `student_canvas_pages` | `List[str]` | No | Handwritten answer pages (base64 data URLs) |
| `student_uploaded_images` | `List[str]` | No | Uploaded answer images (base64 data URLs) |
| `student_document_text` | `str` | No | Extracted text from uploaded PDFs/DOCX |
| `question_id` | `str` | No | For logging/tracking only |
| `student_id` | `str` | No | For logging/tracking only |
| `language` | `Language` | No | `"english"` / `"hindi"` — auto-detected if omitted |

**OptionItem:**
| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | "A", "B", "C", "D" |
| `content` | `str` | The text content of the option |
| `is_image` | `bool` | Whether this option is an image |
| `image_base64` | `str` | Base64 data URL if option is image |

---

## 4. Output Model — `EvalResult`

Everything the caller needs to display results, track history, and debug issues.

| Field | Type | Description |
|-------|------|-------------|
| **Core Evaluation** | | |
| `is_correct` | `bool` | Final correctness judgment |
| `score` | `float` | 0.0 – 1.0 |
| **Student Answer** | | |
| `extracted_answer` | `str` | What the student wrote (from OCR/extraction) |
| `answer_source` | `AnswerSource` | How the answer was obtained (typed, vision, etc.) |
| `extraction_confidence` | `float` | 0.0 – 1.0 confidence of extraction |
| **Feedback** | | |
| `feedback` | `str` | Detailed evaluation feedback for the student |
| `reasoning` | `str` | Evaluation reasoning |
| `work_shown` | `str` | Student's visible work/steps |
| `what_went_wrong` | `str` | Explanation of the student's mistake |
| `correct_solution` | `str` | Step-by-step correct solution |
| **Correct Answer** | | |
| `correct_answer_display` | `str` | Human-readable correct answer (e.g. "A (7 days)") |
| `correct_answer_source` | `str` | `"admin_provided"` / `"llm_solved"` / `"unknown"` |
| `solved_answer` | `str` | LLM's own solution when no admin answer was provided |
| **Metadata** | | |
| `question_type_detected` | `QuestionType` | What the engine detected |
| `language_detected` | `Language` | What language was detected |
| `token_usage` | `TokenUsage` | Aggregate token counts + estimated cost across all LLM calls |
| `extraction_result` | `ExtractionResult` | Detailed Stage 2A output |
| `had_override` | `bool` | True if post-validation overrode LLM judgment |

**TokenUsage:**
| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Total prompt tokens across all calls |
| `output_tokens` | `int` | Total completion tokens across all calls |
| `total_tokens` | `int` | Sum |
| `model` | `str` | Model used |
| `estimated_cost_usd` | `float` | Estimated cost |

---

## 5. Pipeline Architecture — 5 Stages

```
EvalRequest
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 0 — Input Processing            (pure, no LLM) │
│  • Auto-detect question type (MCQ/essay/numerical)    │
│  • Resolve correct answer ("A" → "7 days")            │
│  • Detect language (Hindi/English)                     │
│  • Format options text for prompts                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1 — Image Enhancement           (pure, no LLM) │
│  • Upscale student images to 1500px                    │
│  • Contrast 1.5x, sharpen, brightness 1.05x           │
│  • Adaptive stroke thickening                          │
│  • Pillow optional — raw images if unavailable         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2A — Unbiased Extraction        (LLM via Gate)  │
│  • Send student images WITHOUT correct answer          │
│  • Extract: final_answer, transcription, confidence    │
│  • "Strict OCR extractor, never hallucinate"           │
│  • Skipped if no student images                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2B — Full Evaluation            (LLM via Gate)  │
│  • Full prompt: question + options + correct answer    │
│    + extracted student answer + language rules          │
│  • Different prompts: has_answer / essay / solve-self  │
│  • All images: student + question figures              │
│  • Parse structured JSON response                      │
│  • Retry with simpler prompt if JSON fails             │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3 — Post-LLM Validation         (pure, no LLM) │
│  • MCQ: letter match OR value match override           │
│  • Numeric: semantic equivalence (9 = nine = 9.0)      │
│  • Essay: score clamping 0.0 – 1.0                     │
│  • Override LLM if contradiction detected              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
                EvalResult
```

Stages communicate via internal `PipelineContext` (never exposed to caller).

---

## 6. LLM Gate — Single Door for All LLM Calls

Every LLM call in Stage 2A and 2B goes through the Gate. No stage calls the LLM provider directly.

```
Pipeline Stage
    │
    ▼
LLMGate.call_with_images()
    │
    ├── 1. Estimate input tokens
    ├── 2. Check per-call input limit ──→ raise TokenLimitExceeded
    ├── 3. Check budget headroom ───────→ raise BudgetExhausted
    ├── 4. Cap output tokens
    ├── 5. Forward to LLM Provider
    │       ├── Connection pooling (httpx)
    │       ├── Concurrency semaphore
    │       └── 3x retry, exponential backoff
    ├── 6. Log token usage → MongoDB (optional)
    └── 7. Return LLMResponse
```

### Gate Config

```python
GateConfig:
    max_input_tokens: int | None      # Per-call input limit. None = unlimited
    max_output_tokens: int | None     # Per-call output limit. None = unlimited

BudgetConfig:
    daily_token_limit: int | None     # None = unlimited
    weekly_token_limit: int | None
    monthly_token_limit: int | None
```

### Token Usage Log (MongoDB)

```javascript
// Collection: token_usage_log
{
    "model": "gpt-4o",
    "caller": "stage_2a_extract",       // which pipeline stage
    "input_tokens": 1247,
    "output_tokens": 312,
    "total_tokens": 1559,
    "estimated_cost_usd": 0.00078,
    "called_at": ISODate("2026-03-24T10:30:00Z"),
    "metadata": {
        "question_id": "PHY-042",
        "student_id": "stu-001"
    }
}

// Indexes:
//   called_at: 1  (TTL: 30 days auto-delete)
//   caller: 1
//   model: 1
```

### Budget Check Query

Single aggregation, three window sums, index-backed:

```javascript
db.token_usage_log.aggregate([
    { $match: { called_at: { $gte: month_start } } },
    { $group: {
        _id: null,
        daily:   { $sum: { $cond: [{ $gte: ["$called_at", today_start] }, "$total_tokens", 0] } },
        weekly:  { $sum: { $cond: [{ $gte: ["$called_at", week_start] }, "$total_tokens", 0] } },
        monthly: { $sum: "$total_tokens" }
    }}
])
```

### MongoDB is Optional

| MongoDB configured | Token logging | Budget enforcement | Engine works? |
|---|---|---|---|
| Yes | Writes to `token_usage_log` | Checks limits before each call | Yes |
| No | No-op (tokens still in `EvalResult`) | No-op (all calls pass) | Yes |

---

## 7. LLM Provider Abstraction

```
BaseLLMProvider (abstract)
    │
    ├── call(messages, max_tokens, temperature) → LLMResponse
    └── call_with_images(images, prompt, max_tokens, temperature, system_prompt) → LLMResponse
          │
          ▼
OpenAIProvider (implemented)
    • AsyncOpenAI client
    • httpx connection pooling (20 keepalive, 100 max)
    • asyncio.Semaphore(200) for concurrency
    • 3x retry with exponential backoff
    • Handles max_tokens vs max_completion_tokens per model
```

Adding a new provider (Anthropic, Gemini) = implement one class with two methods, plug into Gate.

---

## 8. Folder Structure

```
stoody-backend/evaluation_engine/
│
├── __init__.py                        # Public: evaluate(), EvalRequest, EvalResult, EngineConfig
├── config.py                          # EngineConfig, GateConfig, BudgetConfig
│
├── models/
│   ├── __init__.py
│   ├── enums.py                       # QuestionType, Language, AnswerSource
│   ├── request.py                     # EvalRequest, OptionItem
│   └── response.py                    # EvalResult, TokenUsage, ExtractionResult
│
├── pipeline/
│   ├── __init__.py
│   ├── context.py                     # PipelineContext (internal state between stages)
│   ├── orchestrator.py                # EvalPipeline.run() — wires stages + manages lifecycle
│   ├── stage_0_input.py               # Input normalization, type detection, answer resolution
│   ├── stage_1_enhance.py             # Image enhancement (Pillow, self-contained)
│   ├── stage_2a_extract.py            # Unbiased answer extraction (LLM, no correct answer)
│   ├── stage_2b_evaluate.py           # Full LLM evaluation, JSON parse, retry
│   └── stage_3_validate.py            # Post-validation, MCQ/numeric overrides
│
├── llm/
│   ├── __init__.py
│   ├── base_provider.py               # Abstract BaseLLMProvider, LLMResponse
│   ├── openai_provider.py             # OpenAI implementation
│   ├── gate.py                        # LLMGate: budget → limits → call → log
│   ├── token_logger.py                # MongoDB token_usage_log writer
│   └── budget.py                      # BudgetTracker: daily/weekly/monthly queries
│
├── prompts/
│   ├── __init__.py
│   ├── templates.py                   # All prompt constants (system prompts, rules, schemas)
│   └── builder.py                     # PromptBuilder: assembles prompts per question type
│
└── utils/
    ├── __init__.py
    ├── json_parser.py                 # robust_json_parse()
    ├── equivalence.py                 # answers_are_equivalent(), parse_number()
    ├── language.py                    # detect_language(), get_language_instruction()
    └── text.py                        # truncate_for_prompt(), coerce_float()
```

**25 files. ~2,250 lines. Zero external Stoody imports.**

---

## 9. Dependency Map

```
External packages (all in requirements.txt already):
    pydantic       — models
    openai         — LLM calls (AsyncOpenAI)
    httpx          — connection pooling for OpenAI
    Pillow         — image enhancement (OPTIONAL, graceful fallback)
    motor          — async MongoDB (OPTIONAL, for token logging/budget)

Internal import hierarchy (no circular deps):
    models/        ← depends on nothing
    utils/         ← depends on models/ (enums only)
    prompts/       ← depends on models/ (enums only)
    llm/           ← depends on models/, config
    pipeline/      ← depends on all above
    __init__.py    ← depends on pipeline/, models/, config
```

---

## 10. Integration Example — Stoody Backend

When the engine is ready, the existing 900-line `/evaluate` endpoint in `practice_async.py` shrinks to ~40 lines:

```python
from evaluation_engine import evaluate, EvalRequest, OptionItem, EngineConfig

@router.post("/evaluate")
async def evaluate_submission(request: Request, payload: EvaluateRequest, ...):
    # 1. Load question from MongoDB (caller's job)
    question_doc = await _load_question_doc(db, payload.questionId, is_b2c)
    question_images = await _figure_images_base64(question_doc, db, is_b2c)

    # 2. Build EvalRequest (pack all data into the engine's input model)
    eval_request = EvalRequest(
        question_text=question_doc.get("text", ""),
        options=[
            OptionItem(label=chr(65+i), content=str(o))
            for i, o in enumerate(question_doc.get("options", []))
        ],
        correct_answer=question_doc.get("correctAnswer"),
        question_images=question_images,
        student_answer_text=payload.answerText,
        student_canvas_pages=payload.canvasPages,
        question_id=payload.questionId,
    )

    # 3. Evaluate (one line)
    result = await evaluate(eval_request, engine_config)

    # 4. Save to DB (caller's job)
    await save_practice_attempt(db, result, payload, current_user)

    # 5. Return
    return {"success": True, "evaluation": result.model_dump()}
```

---

## 11. Other Products — Plug-and-Play Examples

### Example: Homework Checker App
```python
from evaluation_engine import evaluate, EvalRequest, EngineConfig

config = EngineConfig(openai_api_key=os.environ["OPENAI_KEY"])

for question in homework_questions:
    result = await evaluate(EvalRequest(
        question_text=question["text"],
        correct_answer=question["answer"],
        student_answer_text=student_answers[question["id"]],
    ))
    report.append({"q": question["id"], "correct": result.is_correct, "score": result.score})
```

### Example: Exam Grading Service
```python
from evaluation_engine import evaluate, EvalRequest, EngineConfig

config = EngineConfig(
    openai_api_key="sk-...",
    mongodb_uri="mongodb://...",                  # track costs
    budget=BudgetConfig(daily_token_limit=500000),  # cap spending
)

for student_paper in exam_papers:
    for page_image in student_paper["pages"]:
        result = await evaluate(EvalRequest(
            question_text=question_bank[q_id]["text"],
            correct_answer=question_bank[q_id]["answer"],
            student_canvas_pages=[page_image],
        ), config=config)
```

### Example: API Microservice
```python
from fastapi import FastAPI
from evaluation_engine import evaluate, EvalRequest, EngineConfig

app = FastAPI()
config = EngineConfig(openai_api_key="sk-...")

@app.post("/evaluate")
async def eval_endpoint(req: EvalRequest):
    result = await evaluate(req, config)
    return result.model_dump()
```

---

## 12. Implementation Phases

### Phase 1 — Foundation (Models + Utils)
No LLM, no MongoDB. Pure Python + Pydantic.

| File | What |
|------|------|
| `config.py` | EngineConfig, GateConfig, BudgetConfig |
| `models/enums.py` | QuestionType, Language, AnswerSource |
| `models/request.py` | EvalRequest, OptionItem |
| `models/response.py` | EvalResult, TokenUsage, ExtractionResult |
| `utils/text.py` | truncate_for_prompt, coerce_float |
| `utils/language.py` | detect_language, get_language_instruction |
| `utils/equivalence.py` | answers_are_equivalent, parse_number, resolve_correct_answer |
| `utils/json_parser.py` | robust_json_parse |

### Phase 2 — LLM Layer (Provider + Gate + Budget)

| File | What |
|------|------|
| `llm/base_provider.py` | Abstract BaseLLMProvider, LLMResponse |
| `llm/openai_provider.py` | OpenAI: pooling, semaphore, retry, vision |
| `llm/token_logger.py` | MongoDB writer (optional, graceful no-op) |
| `llm/budget.py` | BudgetTracker: daily/weekly/monthly aggregation |
| `llm/gate.py` | LLMGate: budget → limits → call → log |
| `prompts/templates.py` | All prompt constants |

### Phase 3 — Pipeline (All Stages + Orchestrator)

| File | What |
|------|------|
| `prompts/builder.py` | PromptBuilder: assemble prompts per question type |
| `pipeline/context.py` | PipelineContext dataclass |
| `pipeline/stage_0_input.py` | Type detection, answer resolution, language |
| `pipeline/stage_1_enhance.py` | Image enhancement (Pillow, self-contained) |
| `pipeline/stage_2a_extract.py` | Unbiased extraction via Gate |
| `pipeline/stage_2b_evaluate.py` | Full evaluation via Gate, JSON parse, retry |
| `pipeline/stage_3_validate.py` | MCQ/numeric/essay post-validation |
| `pipeline/orchestrator.py` | EvalPipeline.run(): wire stages |

### Phase 4 — Public API

| File | What |
|------|------|
| `__init__.py` | evaluate() function, exports |
| All `__init__.py` | Sub-package re-exports |

---

## 13. Error Handling

| Error | When | What Happens |
|-------|------|--------------|
| `TokenLimitExceeded` | Input prompt exceeds `GateConfig.max_input_tokens` | Raised before LLM call. Caller catches. |
| `BudgetExhausted` | Daily/weekly/monthly token limit hit | Raised before LLM call. Includes `period`, `used`, `limit`, `resets_at`. |
| LLM call failure | API error after 3 retries | `EvalResult` returned with `is_correct=False`, `feedback="Evaluation failed"` |
| JSON parse failure | LLM returns unparseable response | Retry with simpler prompt. If still fails, use raw response as feedback. |
| No student submission | Neither text nor images provided | `EvalResult` with `score=0`, `feedback="No answer provided"` |
| Pillow unavailable | Image enhancement import fails | Falls back to raw images. Pipeline continues. |
| MongoDB unavailable | Token logger / budget tracker can't connect | Graceful no-op. Pipeline continues without logging/budget. |

---

## 14. Question Type Handling

| Type | Detection | Correct Answer | Scoring | Post-Validation |
|------|-----------|---------------|---------|-----------------|
| **MCQ** | Has options AND not marked "subjective" | Admin provides letter (A/B/C/D) | Binary: 0 or 1 | Letter match OR value match override |
| **Essay** | Keywords: explain, describe, discuss, compare, etc. | Optional (LLM evaluates content quality) | Continuous: 0.0 – 1.0 | Score clamping |
| **Numerical** | Default for non-MCQ, non-essay | Admin provides value | Binary: 0 or 1 | Semantic equivalence (9 = nine = 9.0, unit stripping) |
| **Subjective** | Explicitly marked or default fallback | Optional | Continuous: 0.0 – 1.0 | Semantic equivalence safety net |

---

## 15. What This Engine Does NOT Do

These are the **caller's** responsibility:

- Load questions from database
- Load question images from database/S3
- Authenticate users
- Track practice sessions
- Save evaluation results to database
- Extract text from uploaded PDFs/DOCX (pass as `student_document_text`)
- Manage rate limiting on API endpoints
- Handle multi-tenancy / data isolation

The engine is pure evaluation logic. Data in, evaluation out.
