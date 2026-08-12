# LLM Gate Spec

**Status:** ACTIVE  
**Authority:** Shared LLM gate contract for DCR and PCR.

---

## 1. Summary

The LLM gate is the single allowed door for all LLM-mediated work in ExamPen.

It serves:

- DCR Vision OCR recognition (always, via `dcr_ai`)
- future DCR Devanagari support
- PCR evaluation
- PCR cache warmup
- PCR clubbed-response topic-discontinuity checks
- PCR practice evaluation

No engine, endpoint, or helper module may call an LLM provider directly outside this contract.

---

## 2. Responsibilities

The gate owns:

- allowed caller identities
- per-call input and output limits
- daily, weekly, and monthly budget enforcement
- append-only token logging
- token usage rollups
- usage APIs under `/v1/evalpen/usage/*`

The gate does not own:

- conducted-exam artifact capture
- DCR or PCR evaluation semantics
- practice persistence

---

## 3. Architecture

```text
All LLM callers
  │
  ├── dcr_ai
  ├── dcr_devanagari
  ├── pcr_eval_core
  ├── pcr_objective_extraction
  ├── pcr_cache_warmup
  ├── pcr_clubbed_h4
  └── pcr_practice
  │
  ▼
┌─────────────────────────────────────────────────┐
│                    LLM Gate                     │
│                                                 │
│ 1. Validate caller_id                           │
│ 2. Estimate prompt size                         │
│ 3. Enforce per-call limits                      │
│ 4. Enforce daily / weekly / monthly budgets     │
│ 5. Call provider                                │
│ 6. Append token log                             │
│ 7. Return content + usage metadata              │
└─────────────────────────────────────────────────┘
```

---

## 4. Contract

### 4.1 Call Shape

```text
call(model_id, prompt, caller_id, **kwargs) -> GateResponse
```

Required inputs:

- `model_id`
- `prompt`
- `caller_id`

Optional inputs:

- `max_output_tokens`
- `metadata`
- `temperature` where a caller-specific override is explicitly allowed

### 4.2 Response Shape

```text
GateResponse:
  content: str
  usage:
    model: str
    caller: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    timestamp: datetime
```

---

## 5. Allowed Callers

Only registered callers may invoke the gate. Any unregistered `caller_id` is rejected.

| caller_id | Pipeline | Purpose |
|---|---|---|
| `pcr_eval_core` | PCR | Evaluate student response on cache hit or miss |
| `pcr_objective_extraction` | PCR | Transcribe pure Objective PCR answer sheets before deterministic scoring |
| `pcr_cache_warmup` | PCR | Generate or refresh reference solutions |
| `pcr_clubbed_h4` | PCR | Topic discontinuity check for clubbed detection |
| `pcr_practice` | PCR | Stateless live practice evaluation |
| `dcr_ai` | DCR | Vision OCR recognition of rendered stroke images |
| `dcr_devanagari` | DCR | Future LLM-assisted Devanagari recognition |
| `credits_quality_judge` | Credits | Decide whether a rendered canonical stroke page or uploaded notebook page contains useful, legible writing rather than blank, random, or scribbled content |

New callers must be added here before implementation.

---

## 6. Configuration

```text
GateConfig:
  max_input_tokens: int | null
  max_output_tokens: int | null

BudgetConfig:
  daily_token_limit: int | null
  weekly_token_limit: int | null
  monthly_token_limit: int | null
```

Rules:

- `null` means unlimited for that dimension
- per-call limits are enforced before provider invocation
- budget limits are enforced against rollup-aware current usage

---

## 7. Storage Model

MongoDB only. Collections live in the tenant/admin DB.

### 7.1 `llm_gate_config`

Single document per tenant.

Fields:

- `_id = "gate_config"`
- `max_input_tokens`
- `max_output_tokens`
- `daily_token_limit`
- `weekly_token_limit`
- `monthly_token_limit`
- `updated_at`

### 7.2 `llm_token_usage_log`

Append-only, one document per gate call.

Fields:

- `model`
- `caller`
- `input_tokens`
- `output_tokens`
- `cache_read_tokens`
- `cache_creation_tokens`
- `total_tokens`
- `estimated_cost_usd`
- `called_at`

Indexes:

- `{ called_at: 1 }` with TTL retention of 7 days
- `{ caller: 1 }`
- `{ model: 1 }`

### 7.3 `llm_token_usage_rollup`

Fields:

- `period_type` = `daily | weekly | monthly`
- `period_start`
- `period_end`
- `total_tokens`
- `total_input`
- `total_output`
- `total_cost_usd`
- `call_count`
- `breakdown_by_model`
- `breakdown_by_caller`

Indexes:

- `{ period_type: 1, period_start: 1 }` unique compound

---

## 8. Usage Lifecycle

```text
llm_token_usage_log
  │
  ├─ raw rows kept for 7 days
  ├─ midnight -> daily rollup
  ├─ Monday  -> weekly rollup
  └─ month   -> monthly rollup
```

Retention intent:

- raw log: 7 days
- daily rollups: 1 month
- weekly rollups: 3 months
- monthly rollups: 3 months

---

## 9. Errors

### 9.1 Budget Exhausted

```json
{
  "error": "budget_exhausted",
  "period": "daily",
  "used_tokens": 498500,
  "limit_tokens": 500000,
  "resets_at": "2026-03-25T00:00:00Z"
}
```

### 9.2 Per-Call Limit Exceeded

```json
{
  "error": "token_limit_exceeded",
  "limit_type": "per_call_input",
  "estimated_tokens": 6200,
  "allowed_tokens": 4000
}
```

Batch evaluation may return partial completion plus refusal metadata if budget runs out mid-batch.

---

## 10. API Surface

All gate usage APIs are namespaced under `/v1/evalpen/usage/`.

- `GET /v1/evalpen/usage/current`
- `GET /v1/evalpen/usage/history`
- `PUT /v1/evalpen/usage/config`

See `api/eval-usage.openapi.yaml` for wire format.

---

## 11. Hard Rules

1. MongoDB only.
2. Shared across DCR and PCR.
3. Practice persistence remains external even though practice calls may log tokens here.
4. Gate logging is not a license to create new practice artifact storage.
5. Caller IDs in implementations must match the allow-list in this document exactly.

---

## 12. Provider Configuration (Model Agnostic)

The gate is model-agnostic. The provider and model are selected via environment variables:

```text
AI_PROVIDER — selects the active provider (openai, mistral, anthropic, gemini)
             If not set, the gate auto-detects from model_id prefix.

Per-provider env vars:
  openai:    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
  mistral:   MISTRAL_API_KEY, MISTRAL_OCR_MODEL
  anthropic: ANTHROPIC_API_KEY
  gemini:    GOOGLE_GEMINI_API_KEY / GEMINI_API_KEY
```

The gate supports multimodal (vision) calls via the `messages` parameter. All Vision OCR for DCR and PCR goes through the gate.
