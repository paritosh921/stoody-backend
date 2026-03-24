# DCR Assessment Engine Plan

## Summary
Build DCR as a new backend assessment pipeline that starts after hub sync but does not use `canvas_pages`. Instead, create a DCR-specific sheet-response store where synced strokes are attached to a registered blank DCR template and persisted as the canonical student response sheet under the tutor's conducted test context, keyed by assessment + set variant + student identity fields extracted from the sheet.

The blank DCR template becomes the ground-truth canvas. Synced strokes are overlaid onto that template to form a renderable response sheet. OCR and evaluation operate on this stored sheet representation, not on copy/page-based note infrastructure.

## Key Changes
### 1. Ownership model
Anchor DCR to the existing Test Series / conducted-test flow, but introduce a separate DCR response-sheet domain:
- `dcr_templates`: blank template + answer-key template registration per Test Series and set variant
- `dcr_sheet_responses`: canonical stored student DCR sheets with template reference, raw synced strokes, rendered overlay asset, OCR header fields, and identity match state
- `dcr_evaluations`: per-student evaluated result with answer table, score, per-question verdicts, confidence, and review state
- `dcr_jobs`: async extraction/evaluation/review jobs

Do not store DCR sheets in `canvas_pages`, because DCR responses do not naturally carry `copy_id` / `page_number`.

### 2. Template registration and answer-key flow
For each set variant:
1. Tutor links DCR to an existing Test Series.
2. Tutor uploads:
   - the blank DCR template sheet
   - the annotated DCR answer-key sheet in the same layout
3. Backend registers one layout for that variant:
   - header boxes: name, roll number, subject, paper set
   - answer-table geometry: question-number column and response-cell column
4. Backend extracts the tutor answer-key table from the annotated sheet and maps it to the existing Test Series questions.
5. Existing question metadata remains the source for:
   - `correct_answer`
   - `points`
   - `penalty`
   - optional detailed explanations/solutions shown later to students

### 3. Student response-sheet ingestion after sync
Once hub sync has completed for a DCR submission:
1. Backend receives or resolves the synced raw stroke payload associated with the DCR conducted test event.
2. Backend creates a `dcr_sheet_response` record, not a canvas page.
3. That record stores:
   - `document_id` / conducted test reference
   - `set_variant`
   - raw strokes
   - template reference
   - rendered overlay sheet asset
   - extracted header fields
   - matched student id if resolved
   - ingest confidence / mismatch flags

The stored response sheet is the canonical artifact for all downstream work.

### 4. Rendering model
Use the blank template as background and render synced strokes on top of it:
- produce a normalized SVG/PNG/PDF representation per student sheet
- preserve raw strokes separately for re-rendering if template calibration changes
- store both:
   - immutable raw stroke payload
   - generated overlay render artifact

This gives one stable visual sheet for OCR, review, and later display in tutor/student portals.

### 5. Extraction and evaluation flow
Async batch pipeline:
1. Load registered template for the set variant.
2. Render each stored student sheet overlay.
3. Crop header fields and answer cells using template coordinates.
4. OCR:
   - roll number
   - student name
   - subject
   - paper set
   - each answer cell
5. Normalize into:
   - `student_identity`
   - `student_answers_by_qno`
6. Match student primarily by roll number, with name and set validation.
7. Persist the extracted answer table on the `dcr_sheet_response`.
8. Compare against the registered answer key using normalized question-number maps.
9. Apply question-wise `points` and `penalty` from existing question/test metadata.
10. Store finalized `dcr_evaluation` and optionally mirror summary fields into conducted-test attempt views.

### 6. Fast comparison strategy
Keep comparison deterministic and cheap:
- precompute `answer_key_by_qno`
- precompute `scoring_rule_by_qno`
- normalize each student response into `dict[qno] -> option`
- evaluate with a tight in-memory loop and bulk database writes

The expensive step is OCR/extraction. Scoring itself stays O(question count) per sheet and should scale well to hundreds of sheets.

### 7. Manual review flow
Low-confidence or mismatched sheets go to a segregated review queue:
- unreadable header identity
- roll number not matched
- set variant mismatch
- ambiguous answer-cell OCR
- missing answer rows or duplicate extracted rows

Teacher review flow:
- open grouped low-confidence sheets
- inspect rendered overlay against extracted fields
- edit identity and/or answer table
- submit bulk finalization

Reviewed sheets then produce normal finalized `dcr_evaluation` records.

### 8. APIs and interfaces
Add APIs for:
- create/list DCR assessments under an existing Test Series
- upload blank template and answer-key template for a set variant
- preview extracted answer key
- ingest/list stored DCR sheet responses for a conducted test
- trigger async extraction/evaluation jobs
- get batch progress and summary
- fetch one student's rendered DCR sheet + extracted answers + evaluated result
- list manual-review queue
- bulk submit reviewed corrections

Result payload should expose:
- student identity fields
- rendered sheet asset reference
- extracted answer table
- question-wise evaluation array
- total score
- confidence / review state
- optional detailed explanation payload from existing question data

## Test Plan
- Blank template registration correctly defines header and answer-cell geometry.
- Tutor answer-key sheet extraction produces the right question-to-answer mapping for one set variant.
- Synced DCR strokes create a `dcr_sheet_response` without requiring `copy_id` or `page_number`.
- Rendering overlays strokes correctly on the blank template and is stable across re-reads.
- OCR extracts roll number, name, set, and answer cells from rendered student sheets.
- Student matching by roll number works and name/set mismatches move the sheet to review.
- Correct / incorrect / unattempted scoring respects existing `points` and `penalty`.
- Batch evaluation of hundreds of stored DCR sheets remains deterministic regardless of worker chunking.
- Bulk manual correction finalizes queued sheets and updates summaries correctly.

## Assumptions
- Hub sync/upstream stroke capture remains out of scope; DCR starts after sync.
- DCR responses are stored in a new sheet-response store, not `canvas_pages`.
- One template layout is registered per set variant in v1.
- The tutor's conducted-test context is the parent container for student DCR submissions/results.
- Roll number is the primary matching key; name and set are validation signals.
- Detailed explanations shown to students come from existing question/test metadata, not DCR OCR.
