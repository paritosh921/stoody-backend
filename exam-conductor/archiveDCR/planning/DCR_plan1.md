# DCR Assessment Engine — Implementation Plan

## Context

Students write responses on pre-printed DCR (Direct Character Recognition) sheets using BLE pens. The exam-conductor hub syncs these pen strokes to the database. The DCR engine evaluates student responses by:
1. Extracting the tutor's correct answer table from an uploaded document (via existing tutor document upload flow)
2. Rendering each student's synced strokes onto a digitally-generated DCR template image
3. OCR-ing the composite to extract the student's response table
4. Comparing the two tables and applying configurable scoring rules

The tutor can also optionally attach detailed explanations per question (not used for scoring, but surfaced to students later).

---

## Architecture Overview

```
Tutor uploads filled answer key       Student strokes synced by hub
  (PDF/image via document upload)       (stored in canvas_pages / strokes)
           │                                        │
           ▼                                        ▼
    OCR → extract Q→answer table         Fetch strokes from DB
           │                                        │
           ▼                                        ▼
    Store in dcr_answer_keys          Generate DCR template SVG
                                      + overlay strokes → composite PNG
                                                    │
                                                    ▼
                                           OCR → extract Q→response table
                                                    │
                                                    ▼
                                        Dict comparison (answer key vs student)
                                                    │
                                                    ▼
                                         Apply scoring rules → store result
```

---

## New Files

| File | Purpose |
|------|---------|
| `backend/models/dcr_assessment.py` | Pydantic models for all DCR entities |
| `backend/services/dcr_assessment_service.py` | Core engine: template generation, OCR, scoring, orchestration |
| `backend/api/v1/dcr_async.py` | FastAPI router with all DCR endpoints |

## Modified Files

| File | Change |
|------|--------|
| `backend/main_async.py` | Register DCR router |
| `backend/core/database.py` | Add DCR collection indexes in `ensure_indexes_for_db()` |

---

## 1. DCR Template Generator

Since the BLE pen only captures handwriting strokes (not the pre-printed grid), we need a digital template to composite strokes onto. The system generates an SVG template matching the physical DCR sheet layout.

**Location**: Part of `dcr_assessment_service.py`

**Function**: `generate_dcr_template_svg(config) -> str`

**Config parameters** (provided by tutor when creating an answer key):
- `total_questions: int` — e.g. 50
- `columns: int` — 1 or 2 (default 2)
- `header_fields: List[str]` — e.g. ["Name", "Roll No", "Subject", "Paper Set"]
- `page_size: str` — book type code (A4, A5, LS, MS, etc.)

**Generated template SVG structure**:
```
┌──────────────────────────────┐
│  Name: _________             │
│  Roll No: _______            │
│  Subject: ________           │
│  Paper Set: ____             │
├───────┬──────┬───────┬───────┤
│  Q1   │      │  Q26  │      │
│  Q2   │      │  Q27  │      │
│  Q3   │      │  Q28  │      │
│  ...  │      │  ...  │      │
│  Q25  │      │  Q50  │      │
└───────┴──────┴───────┴───────┘
```

The SVG uses the same mm-space coordinate system as `stroke_pdf_generator.py` (`BOOK_DIMENSIONS_MM`), so student strokes overlay perfectly since they're already in mm-space.

**Alternative**: Tutor can upload a blank template image instead of using the generator. Stored in S3, referenced by the answer key config.

---

## 2. Pydantic Models (`backend/models/dcr_assessment.py`)

### DCRTemplateConfig
```python
class DCRTemplateConfig(BaseModel):
    total_questions: int
    columns: int = 2
    page_size: str = "A4"  # book type code
    header_fields: List[str] = ["Name", "Roll No", "Subject", "Paper Set"]
    custom_template_url: Optional[str] = None  # S3 URL if tutor uploaded blank template
```

### DCRAnswerKeyEntry
```python
class DCRAnswerKeyEntry(BaseModel):
    question_number: int
    correct_options: List[str]       # ["A"] or ["A","C"] for multi-correct
    marks: float = 4.0
    negative_marks: float = 1.0
    explanation: Optional[str] = None  # detailed response (shown to students, not used for scoring)
```

### DCRAnswerKey (stored in `dcr_answer_keys` collection)
```python
class DCRAnswerKey(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    subject: str
    exam_name: str = ""
    paper_set: Optional[str] = None
    total_questions: int
    template_config: DCRTemplateConfig
    entries: List[DCRAnswerKeyEntry]
    default_marks: float = 4.0
    default_negative_marks: float = 1.0
    unattempted_marks: float = 0.0
    source_document_id: Optional[str] = None   # ref to uploaded document if created via OCR
    admin_id: str
    created_at: datetime
    status: str = "active"  # active | archived
```

### DCRQuestionResult
```python
class DCRQuestionResult(BaseModel):
    question_number: int
    student_option: Optional[str]   # None = unattempted
    correct_options: List[str]
    is_correct: bool
    score_awarded: float
    ocr_confidence: float
```

### DCRStudentResult (stored in `dcr_student_results` collection)
```python
class DCRStudentResult(BaseModel):
    session_id: str
    answer_key_id: str
    student_id: str
    admin_id: str
    extracted_responses: Dict[str, Optional[str]]   # {"1": "A", "2": null, ...}
    question_results: List[DCRQuestionResult]
    total_score: float
    max_possible_score: float
    correct_count: int
    wrong_count: int
    unattempted_count: int
    ocr_confidence_avg: float
    low_confidence_questions: List[int]
    needs_manual_review: bool
    processing_status: str   # pending | processing | completed | failed | review_needed
    error_message: Optional[str] = None
    rendered_image_url: Optional[str] = None   # S3 URL of composite image
    manually_overridden_questions: List[int] = []
    processed_at: Optional[datetime] = None
```

### DCRAssessmentSession (stored in `dcr_assessment_sessions` collection)
```python
class DCRAssessmentSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer_key_id: str
    title: str
    subject: str
    admin_id: str
    student_ids: List[str]
    stroke_source: DCRStrokeSource   # how to find student DCR sheets
    status: str = "created"          # created | processing | completed | partial | failed
    total_students: int
    processed_count: int = 0
    failed_count: int = 0
    created_at: datetime
    completed_at: Optional[datetime] = None
```

### DCRStrokeSource
```python
class DCRStrokeSource(BaseModel):
    book_type: str              # e.g. "LS" (A4 portrait)
    page_numbers: List[int]     # which pages are the DCR sheet
    copy_id: Optional[str] = None
```

---

## 3. Service Layer (`backend/services/dcr_assessment_service.py`)

Module-level async functions (same pattern as `note_classification_service.py`).

### 3.1 Template Generation

**`generate_dcr_template_svg(config: DCRTemplateConfig) -> str`**
- Pure SVG generation using mm-space coordinates from `BOOK_DIMENSIONS_MM`
- Draws header fields (boxed rows at top)
- Draws Q-number + answer-cell grid below
- Returns SVG string compatible with `svg_to_png_bytes()`

### 3.2 Answer Key Creation — OCR Path

**`create_answer_key_from_document(tenant_db, admin_id, document_id, template_config, scoring_config) -> dict`**
1. Fetch document from `documents` collection (already uploaded via tutor document flow)
2. Download the file from S3 (PDF/image)
3. If PDF, convert to images (reuse `document_processor.py` patterns)
4. OCR the image with the structured extraction prompt (see 3.4)
5. Parse into `List[DCRAnswerKeyEntry]`
6. Store in `dcr_answer_keys` collection

### 3.3 Answer Key Creation — Manual Path

**`create_answer_key_manual(tenant_db, admin_id, config) -> dict`**
- Direct insert from tutor-provided JSON entries
- No OCR needed; just validate and store

### 3.4 OCR Prompt (Template-Agnostic)

Used for BOTH the tutor's uploaded answer key sheet AND the student composite images.

```
You are an OCR system reading a filled answer sheet (DCR/OMR format).

The image shows a response sheet with:
- Header area: student name, roll number, subject, paper set
- Answer grid: question numbers (Q1, Q2, ...) with handwritten option letters (A/B/C/D/E)

EXTRACT all question-answer pairs.

RULES:
1. For each question number, identify the handwritten option letter next to it
2. Read ALL questions, top to bottom, left column first if two-column layout
3. Empty/unclear cells → mark as "?"
4. Only accept single capital letters A-E as valid responses
5. Ignore header fields — only extract Q→answer pairs

RESPOND IN EXACT JSON:
{
    "total_detected": <int>,
    "responses": [
        {"q": 1, "ans": "A", "conf": 0.95},
        {"q": 2, "ans": "C", "conf": 0.88},
        ...
    ]
}

conf guidelines: 0.9+ clear, 0.7-0.9 readable, 0.5-0.7 uncertain, <0.5 very unclear
```

### 3.5 Student Processing Pipeline

**`process_student_dcr(tenant_db, session_id, answer_key, student_id, stroke_source) -> dict`**

Per-student pipeline:
1. **Fetch strokes** — reuse `_fetch_page_strokes()` pattern from `note_classification_service.py` (canvas_pages preferred, strokes fallback)
2. **Generate template SVG** — `generate_dcr_template_svg(answer_key.template_config)`
3. **Render strokes SVG** — `build_svg_from_strokes(strokes, book_type)` from `stroke_pdf_generator.py`
4. **Composite** — merge template SVG + strokes SVG into single image:
   - Both share the same mm-space viewBox (same page dimensions)
   - Template as background layer, strokes on top
   - Convert composite SVG → PNG via `svg_to_png_bytes(svg, scale=1.0)` (higher scale than classification's 0.5 for letter clarity)
5. **OCR** — send PNG to vision model with structured prompt
6. **Parse** — extract `Dict[int, Optional[str]]` responses
7. **Score** — call `score_student()` (see 3.6)
8. **Store** — upsert into `dcr_student_results`
9. **Upload** — optionally store composite PNG to S3 for review

### 3.6 Scoring Algorithm

**`score_student(responses, answer_key_entries, confidences, confidence_threshold=0.5) -> dict`**

Pure Python, synchronous, O(n) per student:
```python
for each question in answer_key_entries:
    student_opt = responses.get(question_number)
    if student_opt is None or student_opt == "?":
        → unattempted (score = unattempted_marks, default 0)
    elif student_opt.upper() in correct_options:
        → correct (score = +marks)
    else:
        → wrong (score = -negative_marks)

    if confidence < threshold:
        → flag for manual review

return {question_results, total_score, max_possible, correct/wrong/unattempted counts,
        low_confidence_questions, needs_manual_review}
```

No NumPy needed — the data is small per-student (30-200 questions). The OCR API calls are the bottleneck, not comparison.

### 3.7 Batch Orchestrator

**`run_assessment_batch(tenant_db, db_name, session_id, concurrency=5) -> None`**

Background task (fire-and-forget via `asyncio.create_task`):
```python
sem = asyncio.Semaphore(concurrency)  # match OPENAI_CONCURRENCY_LIMIT

async def process_with_semaphore(student_id):
    async with sem:
        try:
            await process_student_dcr(...)
            await update session.processed_count += 1
        except Exception:
            await mark student result as failed
            await update session.failed_count += 1

tasks = [process_with_semaphore(sid) for sid in student_ids]
await asyncio.gather(*tasks, return_exceptions=True)
await update session status → "completed" or "partial"
```

Client polls `GET /dcr/assessments/{session_id}` for progress.

---

## 4. API Endpoints (`backend/api/v1/dcr_async.py`)

Router: `APIRouter(prefix="/dcr", tags=["DCR Assessment"])`

### Answer Key Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/answer-keys` | Create answer key manually (JSON entries) |
| `POST` | `/answer-keys/from-document` | Create from uploaded document (OCR path) |
| `GET` | `/answer-keys` | List answer keys for admin |
| `GET` | `/answer-keys/{key_id}` | Get specific answer key |
| `PUT` | `/answer-keys/{key_id}` | Update answer key |
| `DELETE` | `/answer-keys/{key_id}` | Soft-delete (archive) |

### Assessment Sessions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/assessments` | Start assessment (kicks off background batch) |
| `GET` | `/assessments` | List sessions |
| `GET` | `/assessments/{session_id}` | Session status + summary |
| `GET` | `/assessments/{session_id}/results` | All student results |
| `GET` | `/assessments/{session_id}/results/{student_id}` | One student's detailed result |

### Manual Review

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/results/{result_id}/override` | Override specific question responses |
| `POST` | `/results/{result_id}/re-ocr` | Re-run OCR for one student |

### Analytics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/assessments/{session_id}/analytics` | Class-wide stats (mean, median, per-Q accuracy, distribution) |

### Auth & Dependencies
- `get_current_user`, `get_database` from `api.v1.auth_async`
- `get_admin_id_from_user` from `api.v1.questions_async`
- Admin/tutor role required for all endpoints

---

## 5. Database Collections & Indexes

All in per-tenant DB (`skb_<institution_id>`).

### `dcr_answer_keys`
```
Indexes:
  { admin_id: 1, status: 1 }          — list active keys
  { id: 1 }  unique                    — lookup by uuid
```

### `dcr_assessment_sessions`
```
Indexes:
  { admin_id: 1, status: 1, created_at: -1 }  — list sessions
  { id: 1 }  unique
```

### `dcr_student_results`
```
Indexes:
  { session_id: 1, student_id: 1 }  unique  — one result per student per session
  { answer_key_id: 1, student_id: 1 }       — lookup by key+student
  { admin_id: 1, session_id: 1 }            — admin queries
  { processing_status: 1 }                  — worker polling
```

Add to `ensure_indexes_for_db()` in `core/database.py` using existing `_ensure_index_with_spec_check` pattern.

---

## 6. Key Reuse Points

| What | From | How |
|------|------|-----|
| Stroke fetching | `note_classification_service._fetch_page_strokes()` | Refactor into shared util or import directly |
| SVG rendering | `stroke_pdf_generator.build_svg_from_strokes()` | Import directly |
| PNG conversion | `stroke_pdf_generator.svg_to_png_bytes()` | Import directly, scale=1.0 for DCR |
| S3 upload | `utils/s3_storage.upload_file()` | For composite images |
| Vision OCR | `services/async_openai_service.analyze_image_async()` | With DCR-specific prompt |
| Mistral OCR fallback | `note_classification_service._mistral_ocr_image()` | Pattern reuse |
| Document access | `documents` collection via existing PDF upload | For tutor answer key sheet |
| Auth dependencies | `api/v1/auth_async.py` | `get_current_user`, `get_database` |
| Admin ID resolution | `api/v1/questions_async.get_admin_id_from_user()` | Import directly |
| Book dimensions | `stroke_pdf_generator.BOOK_DIMENSIONS_MM` | Template generation |

---

## 7. Router Registration (`main_async.py`)

```python
try:
    from api.v1.dcr_async import router as dcr_router
    _dcr_available = True
except Exception as e:
    dcr_router = None
    _dcr_available = False
    logging.warning(f"DCR Assessment routes disabled: {str(e)}")

# In router block:
if _dcr_available:
    app.include_router(dcr_router, prefix=f"{API_V1_PREFIX}/dcr", tags=["DCR Assessment"])
```

---

## 8. Implementation Sequence

### Phase 1: Models + Template Generator
1. Create `models/dcr_assessment.py` with all Pydantic models
2. Implement `generate_dcr_template_svg()` in the service
3. Add database indexes to `core/database.py`

### Phase 2: Answer Key Pipeline
1. Implement manual answer key creation (direct JSON → DB)
2. Implement OCR-based answer key creation (from uploaded document)
3. Answer key CRUD endpoints

### Phase 3: Student Processing Engine
1. Implement stroke fetch + template composite
2. Implement OCR extraction for student sheets
3. Implement scoring algorithm
4. Implement batch orchestrator with semaphore concurrency

### Phase 4: API + Integration
1. Assessment session endpoints (start, status, results)
2. Manual override endpoint
3. Analytics endpoint
4. Register router in `main_async.py`

---

## 9. Verification Plan

1. **Unit test scoring**: Create known answer key + student responses → verify scores
2. **Template generation**: Generate SVG → render to PNG → visual inspection
3. **Composite test**: Fetch real strokes from DB → overlay on template → verify alignment
4. **OCR accuracy**: Run OCR on composite images → compare extracted answers with known correct ones
5. **End-to-end**: Create answer key → start assessment → verify results via GET endpoints
6. **Concurrency**: Test with 10+ students to verify semaphore + progress tracking
