# Chapter 03: Scoring Pipeline

## Status
- **Phase:** W6 — Documentation
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6.A6.1)
- **Build status:** DRAFT

## Overview

The scoring pipeline transforms processed stroke data into AI-recognized text, evaluates answers against rubrics, and produces event-sourced scores. It spans three services (`svc-doc-assembly`, `svc-ai-pipeline`, `svc-score-engine`) and includes teacher review, override audit trails, and an objection lifecycle.

## Architecture Context

The scoring pipeline consumes `stroke.processed` events from the stroke pipeline (Chapter 02) and produces scores visible through BFF services to teacher and student UIs.

```
stroke.processed                page.ready                ai.result              score.updated
      |                              |                         |                       |
      v                              v                         v                       v
+------------+    +----------+    +----------+    +----------+    +----------+    +----------+
| svc-doc-   |--->| MinIO    |--->| svc-ai-  |--->| svc-     |--->| svc-     |--->| svc-     |
| assembly   |    | (S3)     |    | pipeline |    | score-   |    | analytics|    | teacher/ |
| (render +  |    | (page    |    | (HWR,    |    | engine   |    | (percen- |    | student  |
|  miss det) |    |  images) |    |  steps)  |    | (event   |    |  tiles)  |    | BFFs     |
+------------+    +----------+    +----------+    | sourced) |    +----------+    +----------+
                                                  +----------+
```

## Stage 1: svc-doc-assembly (Page Rendering + Miss Detection)

**Input:** `stroke.processed` events from NATS; processed strokes from TimescaleDB.

**Processing:**
1. Read normalized strokes for a student's exam from TimescaleDB.
2. Render strokes to SVG page images per question region.
3. Write page images to MinIO (S3). S3 write first, then PostgreSQL metadata second.
4. Detect miss indicators: `miss_no_strokes` (empty region), `miss_sync_failure` (incomplete sync metadata).
5. Publish `page.ready` event on NATS.

**Output:** Page images in MinIO, miss indicator `auto_state` in PostgreSQL, `page.ready` event.

**Error handling:**
- If PG metadata write fails after S3 write: orphaned S3 object is acceptable (garbage collected). Reverse order would create dangling references.
- Miss indicator has two columns: `auto_state` (computed, never manually edited) and `override_state` (teacher-set). Display logic: show `override_state` if non-NULL, else `auto_state`.

**Event schema:** `contracts/events/page.ready.schema.json`

**Source files:** `services/svc-doc-assembly/src/domain/`, `services/svc-doc-assembly/src/storage/`

## Stage 2: svc-ai-pipeline (HWR, Step Detection, Diagram Classification)

**Input:** `page.ready` events from NATS; page images from MinIO.

**Processing:**
1. Load page image from MinIO.
2. **HWR/OCR:** ONNX Runtime inference. Produces recognized text per question with confidence score per character.
3. **Step detection:** Segment multi-step math answers (e.g., 4-step answer split into 4 steps).
4. **Diagram classification:** Classify regions as text vs. diagram.
5. Store AI results in PostgreSQL with `model_version`.
6. Publish `ai.result` event on NATS.

**Output:** AI recognition results in PostgreSQL, `ai.result` event.

**Error handling:**
- Low confidence (< 0.85 per character): flagged for teacher review with amber highlight (mitigation A4.6).
- If > 30% of answer below threshold: entire answer flagged.
- Model versioning: re-running AI with new model creates new version, does not overwrite old.
- Fallback: if no stroke data, process copy image from `svc-copy-upload` instead.

**Event schema:** `contracts/events/ai.result.schema.json`
- Key fields: `exam_id`, `student_id`, `model_version`, `question_results[]` with `recognized_text`, `confidence`, `step_breakdown`

**Source files:** `services/svc-ai-pipeline/src/domain/`, `services/svc-ai-pipeline/src/adapters/`

## Stage 3: svc-score-engine (Event-Sourced Scoring)

**Input:** `ai.result` events from NATS.

**Processing:**
1. Receive AI results per student per exam.
2. Evaluate each question against rubric: per-step mark allocation.
3. Append `ai_draft` score event to event store (PostgreSQL).
4. Update materialized view for current score (same transaction).
5. Publish `score.updated` event on NATS.

**Output:** Score events in PostgreSQL event store, materialized score view, `score.updated` event.

**Event schema:** `contracts/events/score.updated.schema.json`
- Key fields: `exam_id`, `student_id`, `lifecycle_state`, `total_score`, `reason`

**API contract:** `api/score-engine.openapi.yaml`
- `GET /api/v1/scores/{exam_id}/students/{student_id}` — current score projection
- `PATCH /api/v1/scores/{exam_id}/students/{student_id}/questions/{question_id}` — teacher override
- `GET /api/v1/scores/{exam_id}/students/{student_id}/history` — audit trail
- `POST /api/v1/scores/{exam_id}/finalize` — finalize scores
- `POST /api/v1/scores/{exam_id}/publish` — publish to students

**Source files:** `services/svc-score-engine/src/domain/score_fsm.py`, `services/svc-score-engine/src/domain/rubric_eval.py`

## Score Lifecycle FSM

```
                 AI pipeline
                  produces
                   result
                     |
                     v
+----------+    +-----------+    +----------+    +-----------+    +--------+
| ai_draft |    | teacher_  |    | finalized|    | published |    | locked |
|          |--->| reviewed  |--->|          |--->|           |--->|        |
+----------+    +-----------+    +----------+    +-----------+    +--------+
                     ^                                |
                     |                                v
                +----------+                  +------------------+
                | override |                  | objection_window |
                | applied  |                  +------------------+
                +----------+
```

**States:**
| State | Meaning | Allowed Transitions |
|---|---|---|
| `ai_draft` | AI-generated score, awaiting teacher review | -> `teacher_reviewed` |
| `teacher_reviewed` | Teacher has reviewed (confirmed or overridden) | -> `finalized` |
| `finalized` | All scores for exam reviewed and locked for publication | -> `published` |
| `published` | Scores visible to students/parents, objection window opens | -> `objection_window` -> `locked` |
| `objection_window` | Students may file objections | -> `locked` (after window closes) |
| `locked` | Final, immutable | Terminal state |

**Invalid transitions rejected:** e.g., `ai_draft` -> `locked` returns error.

## Rubric Evaluation with Step Marking

Rubric defines per-question mark allocation with optional step breakdown:

```
Question 1 (Total: 4 marks)
  Step 1: Formula identification    -> 2 marks
  Step 2: Substitution              -> 1 mark
  Step 3: Final answer              -> 1 mark
```

AI step detection segments the student's answer. Each step is matched to the rubric step. The score engine computes: `sum(step_scores) = total_question_score`.

Teacher can override at step level or question level. Both produce audit events.

## Override Audit Trail

Every score modification is an **append-only event** in the event store:

```json
{
  "event_type": "override_applied",
  "teacher_id": "T-42",
  "question_id": "Q1",
  "old_value": 3,
  "new_value": 4,
  "reason": "Step 2 partial credit: student showed correct substitution approach",
  "timestamp": "2026-03-20T10:30:00Z"
}
```

- No UPDATE, no DELETE on score events.
- Materialized view updated atomically within the same PostgreSQL transaction.
- Teacher BFF sends override request; `svc-score-engine` validates RBAC and appends.
- Full history available via `GET /api/v1/scores/{exam_id}/students/{student_id}/history`.

## Rubric Version Control

- Every rubric edit creates a new version (mitigation A5.5).
- Scores record `rubric_version` used at time of scoring.
- If rubric updated mid-scoring: "re-score affected papers with new rubric" is an explicit teacher action, not automatic.

## Downstream Consumers

| Consumer | What It Reads | Trigger |
|---|---|---|
| `svc-analytics` | `score.updated` events | Recomputes percentiles, leaderboards |
| `svc-review` | Score context for objection review | On objection filed |
| `svc-plagiarism` | AI recognized text | After `ai.result` events |
| `svc-teacher-bff` | Materialized score views | Teacher dashboard queries |
| `svc-student-bff` | Materialized score views | Student portal queries |
| `svc-notify` | `score.updated` (published) | Triggers student notifications |
| Stoody webhook | `score.published` | `POST /api/webhooks/exampen/scores` |

## Testing

- **Unit:** U-SCR-01 (FSM valid transitions), U-SCR-02 (FSM invalid transitions rejected), U-SCR-03 (rubric step marking), U-SCR-04 (override with audit trail), U-DOC-01 (stroke-to-SVG), U-DOC-02 (miss indicator: no strokes), U-DOC-03 (miss indicator: sync failure), U-AI-01 (HWR English), U-AI-02 (HWR Devanagari), U-AI-03 (step detection), U-AI-04 (diagram classifier)
- **Integration:** I-SCR-01 (AI result -> score event), I-SCR-02 (override REST -> event appended), I-SCR-03 (NATS event published after commit), I-DOC-01 (stroke -> MinIO page image), I-DOC-02 (miss indicator stored), I-AI-01 (page image -> HWR output), I-AI-02 (model version recorded)
- **E2E:** E2E-02 (page assembly -> AI recognition), E2E-03 (AI result -> score generation), E2E-04 (score override -> analytics update), E2E-07 (copy image -> OCR -> score), E2E-09 (miss indicator propagation)

## Failure Modes & Mitigations

| ID | Failure | Mitigation |
|---|---|---|
| A4.6 | AI misrecognition | Confidence threshold flagging, teacher review required |
| A5.5 | Rubric change after partial scoring | Rubric versioning, explicit re-score action |
| PL5 | Plagiarism false positive | High threshold, question-type adjustment, teacher review mandatory |
| Q1 | Question miss ambiguity | Three distinct indicator states, teacher override workflow |

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Initial draft: doc-assembly, AI pipeline, score engine, FSM, rubric eval, audit trail | Claude Agent (W6.A6.1) |
