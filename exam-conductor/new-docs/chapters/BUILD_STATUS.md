# Build Status

Last updated: 2026-05-09

> **This document is a progress tracker.** It reports which implementation tasks are complete, in progress, or pending. It may be stale relative to actual code state. It must **not** be used as authority for architecture decisions, API behavior, storage contracts, schema shapes, or lifecycle rules. When this document conflicts with a root architecture spec (`architecture/*.md`), an integration spec (`integration/*.md`), an OpenAPI file (`api/*.openapi.yaml`), or an event schema (`contracts/events/*.schema.json`), the other document is correct and this one must be updated.

This file is the execution tracker for getting the active ExamPen codebase working against the promoted root specs.

Active authority:

- `architecture/DUAL_MODE_ARCHITECTURE.md`
- `architecture/PCR_EVAL_ENGINE_SPEC.md`
- `architecture/LLM_GATE_SPEC.md`
- `architecture/TAMPER_PROOF_SPEC.md`

---

## Current Code Reality

- `backend/exam-conductor/dcr/` has full implementation: models, repository, recognizer, matcher, service.
- `backend/exam-conductor/pcr/` has full implementation: domain modules, storage repos, services (OCR, submission, eval core, solution cache).
- 18 ExamPen routers mounted in `backend/main_async.py`; `_evalpen_available = True` verified on import.
- All LLM-mediated work (DCR Vision OCR, PCR evaluation, practice eval) routes through the shared gate.
- Conducted-exam question-paper upload already exists in the tutor/backend path and should be integrated, not rebuilt.
- **Frontend teacher UI** (`frontend/src/components/exam-pen/`) is substantially built: 5-tab model (Exams, Workspace, Results, Recheck, Conversations), IDE-style workspace with StudentExplorerPane, QuestionPaperPane, StudentCopyPane, QuestionInspectorPane, CollectionMonitor, ExamResults with publish, RecheckTab with request management, ConversationTab, WorkspaceSetupPanel, ExamPenReadinessIndicator mounted in DocumentDetailPanel.
- **Frontend student portal** (`ExamPenStudent.tsx`) is substantially built: published exam list, per-question score breakdown with reference answers, recheck request dialog, student-side conversation threads.
- **Super-admin ExamPen page** (`ExamPenManagementPage.tsx`) is partially built: tenant feature gate toggle, hub fleet listing with provision code generation, partial usage analytics (tokens today from aggregate API).
- **Mobile app** (`stoody-multi-pen/mobile-app/`) has hub list, exam selection, session dashboard, camera fallback upload with offline retry queue.
- **Hub authority alignment** now targets the converged `stoody-multi-pen/edge_hub/` runtime for new edge work. `stoody-multi-pen/HUB-exam-conductor/` remains a reference/decomposition donor for ExamPen service behavior.
- **Shared status module** (`frontend/src/utils/examPenStatus.ts`) adopted across all status-displaying components.

---

## Modular Delivery Map

```text
                  EXAMPEN IMPLEMENTATION WORK

   Shared ingest substrate ─────┬───────────────┬───────────────┐
                                │               │               │
                                ▼               ▼               ▼
                           DCR engine      PCR engine     Backend wiring
                                │               │               │
                                └───────┬───────┘               │
                                        ▼                       │
                                  Shared LLM gate               │
                                        │                       │
                                        └──────────┬────────────┘
                                                   ▼
                                          Tests + ops + rollout
```

Principle: hub and conducted-exam ingest are independent from evaluation. DCR, PCR, and the gate must remain separately assignable.

---

## Workstream Summary

| Workstream | Status | Why It Exists | Can Run In Parallel With |
|---|---|---|---|
| Shared ingest substrate | COMPLETE | Hub and canonical conducted-exam artifact storage | DCR, PCR, gate |
| DCR engine | COMPLETE (recognizer stub) | Conducted structured exam evaluation | PCR, gate |
| PCR engine | COMPLETE (OCR stub) | Conducted subjective evaluation + stateless practice eval | DCR, gate |
| LLM gate / container | COMPLETE | Shared LLM control plane and token accounting | DCR, PCR |
| Backend integration | COMPLETE | Router mount, feature gating, auth, tenant wiring | gate, engine internals |
| Review / reporting integration | COMPLETE | Tutor/admin operational use of DCR/PCR outputs | DCR, PCR |
| Tests / ops / rollout | COMPLETE (unit tests) | Validation and safe deployment | all after first working slices |

---

## Swarm Task Packs

Use these tasks for spawned agents. Each task has a bounded write set. An agent should not edit files outside its listed ownership unless the task is reassigned.

### Wave 0 — Start Immediately In Parallel

#### SWM-001 — LLM Gate Core

- Status: COMPLETE
- Objective: Create the shared gate module with caller validation, provider adapter, Mongo-backed budget/log storage, and no direct provider leakage.
- Files to create:
  - `backend/exam-conductor/llm_gate/__init__.py`
  - `backend/exam-conductor/llm_gate/models.py`
  - `backend/exam-conductor/llm_gate/repository.py`
  - `backend/exam-conductor/llm_gate/budget.py`
  - `backend/exam-conductor/llm_gate/provider.py`
  - `backend/exam-conductor/llm_gate/gate.py`
- Files to modify:
  - `backend/exam-conductor/__init__.py`
- Depends on: active root specs only
- Review when done: caller IDs match the spec exactly, Mongo-only storage, no FastAPI route work in this task.

#### SWM-002 — Shared Ingest Core

- Status: COMPLETE
- Objective: Create the conducted-exam artifact substrate used by both DCR and PCR, including provenance, hashing, immutability, and evaluator-ready references.
- Files to create:
  - `backend/exam-conductor/ingest/__init__.py`
  - `backend/exam-conductor/ingest/models.py`
  - `backend/exam-conductor/ingest/hashing.py`
  - `backend/exam-conductor/ingest/repository.py`
  - `backend/exam-conductor/ingest/service.py`
- Depends on: active root specs only
- Review when done: artifacts are admin/tenant scoped, immutable after write, and practice persistence is untouched.

#### SWM-003 — DCR Core

- Status: COMPLETE
- Objective: Create the active DCR engine with Vision OCR recognition and scoring over canonical conducted-exam artifacts.
- Files to create:
  - `backend/exam-conductor/dcr/__init__.py`
  - `backend/exam-conductor/dcr/models.py`
  - `backend/exam-conductor/dcr/repository.py`
  - `backend/exam-conductor/dcr/recognizer.py`
  - `backend/exam-conductor/dcr/matcher.py`
  - `backend/exam-conductor/dcr/service.py`
- Depends on: active root specs only
- Review when done: Vision OCR goes through the shared gate, and storage matches DCR collections in the active architecture spec.

#### SWM-004 — PCR Domain

- Status: COMPLETE
- Objective: Build the pure PCR domain logic for segmentation, classification, and flagging.
- Files to create:
  - `backend/exam-conductor/pcr/domain/__init__.py`
  - `backend/exam-conductor/pcr/domain/response_models.py`
  - `backend/exam-conductor/pcr/domain/boundary_detector.py`
  - `backend/exam-conductor/pcr/domain/marker_parser.py`
  - `backend/exam-conductor/pcr/domain/segmenter.py`
  - `backend/exam-conductor/pcr/domain/content_classifier.py`
  - `backend/exam-conductor/pcr/domain/clubbed_detector.py`
  - `backend/exam-conductor/pcr/domain/flag_registry.py`
- Depends on: active root specs only
- Review when done: regex, thresholds, cross-page stitching, and flag types match the PCR spec.

#### SWM-005 — PCR Storage

- Status: COMPLETE
- Objective: Create the PCR Mongo repository layer for submissions, detected responses, evaluations, questions, and solutions.
- Files to create:
  - `backend/exam-conductor/pcr/storage/__init__.py`
  - `backend/exam-conductor/pcr/storage/submission_repo.py`
  - `backend/exam-conductor/pcr/storage/response_repo.py`
  - `backend/exam-conductor/pcr/storage/evaluation_repo.py`
  - `backend/exam-conductor/pcr/storage/question_repo.py`
  - `backend/exam-conductor/pcr/storage/solution_repo.py`
- Depends on: SWM-002
- Review when done: collections and indexes match the PCR spec, and no practice collection is added.

### Wave 1 — Start After Wave 0 Interfaces Exist

#### SWM-006 — PCR Eval Core

- Status: COMPLETE
- Objective: Implement PCR orchestration from `PageOCR` to evaluation, including solution cache and gate-backed evaluation.
- Files to create:
  - `backend/exam-conductor/pcr/services/__init__.py`
  - `backend/exam-conductor/pcr/services/ocr_service.py`
  - `backend/exam-conductor/pcr/services/submission_service.py`
  - `backend/exam-conductor/pcr/services/solution_cache.py`
  - `backend/exam-conductor/pcr/services/eval_core.py`
- Depends on: SWM-001, SWM-004, SWM-005
- Review when done: all LLM-mediated work goes through the gate and blocking flags halt auto-eval.

#### SWM-007 — LLM Gate Usage API

- Status: COMPLETE
- Objective: Expose the usage/config surface for the shared gate.
- Files to create:
  - `backend/api/v1/evalpen_usage_async.py`
- Depends on: SWM-001
- Review when done: route shapes match `api/eval-usage.openapi.yaml` and no unrelated backend files are touched.

#### SWM-008 — Backend Wiring

- Status: COMPLETE
- Objective: Mount ExamPen route groups and feature gates into the main backend runtime.
- Files to modify:
  - `backend/main_async.py`
  - `backend/core/tenant_features.py`
- Depends on: SWM-001, SWM-003, SWM-006, SWM-007, SWM-009, SWM-010
- Review when done: only router mount and feature-flag wiring happen here; no engine logic lives in these files.

#### SWM-009 — PCR API Surface

- Status: COMPLETE
- Objective: Expose conducted-exam PCR endpoints for submissions, evaluation, solutions, and practice.
- Files to create:
  - `backend/api/v1/evalpen_submissions_async.py`
  - `backend/api/v1/evalpen_evaluate_async.py`
  - `backend/api/v1/evalpen_solutions_async.py`
  - `backend/api/v1/evalpen_practice_async.py`
- Depends on: SWM-005, SWM-006
- Review when done: routes match `api/eval-submissions.openapi.yaml`, `api/eval-evaluate.openapi.yaml`, `api/eval-solutions.openapi.yaml`, and `api/eval-practice.openapi.yaml`.

#### SWM-010 — DCR API Surface

- Status: COMPLETE
- Objective: Expose conducted-exam DCR execution and retrieval endpoints.
- Files to create:
  - `backend/api/v1/evalpen_dcr_async.py`
- Depends on: SWM-003, SWM-002
- Review when done: DCR uses canonical artifact references only and does not create practice behavior.

### Wave 2 — Bridge Existing Backend Surfaces

#### SWM-011 — Practice and OCR Gate Bridge

- Status: COMPLETE
- Objective: Remove direct provider usage from existing backend practice/OCR paths and route them through the shared gate without changing persistence semantics.
- Files to modify:
  - `backend/api/v1/practice_async.py`
  - `backend/core/ocr_service.py`
- Depends on: SWM-001, SWM-009
- Review when done: persistence behavior is unchanged, but provider access is exclusively through the gate.

#### SWM-012 — Hub and Conducted Ingest Bridge

- Status: COMPLETE
- Objective: Connect existing hub/copy surfaces to the shared ingest substrate for conducted exams.
- Files to modify:
  - `backend/api/v1/hub.py`
  - `backend/api/v1/copies_async.py`
  - `backend/api/v1/copy_sets_async.py`
- Depends on: SWM-002
- Review when done: conducted-exam artifacts flow into the ingest substrate with proper provenance and idempotency.

#### SWM-013 — Tutor and Question Metadata Integration

- Status: COMPLETE
- Objective: Attach existing tutor-side exam/question data to DCR/PCR execution without rebuilding question-paper upload.
- Files to modify:
  - `backend/api/v1/tutor_async.py`
- Files to create:
  - `backend/exam-conductor/pcr/metadata_adapter.py`
  - `backend/exam-conductor/dcr/metadata_adapter.py`
- Depends on: SWM-003, SWM-005
- Review when done: existing tutor/backend question-paper path is reused and exam metadata resolves `exam_type`, questions, and answer-key inputs.

#### SWM-014 — Review and Publication Surface

- Status: COMPLETE
- Objective: Add teacher-facing review, flagged-response handling, and score publication hooks for DCR/PCR outputs.
- Files to create:
  - `backend/api/v1/evalpen_review_async.py`
  - `backend/api/v1/evalpen_flagged_async.py`
- Files to modify:
  - `backend/api/v1/tutor_async.py`
- Depends on: SWM-009, SWM-010, SWM-013
- Review when done: blocked PCR responses can be reviewed, manual actions are audited, and DCR/PCR outputs are visible to authorized tutor/admin readers.

### Wave 3 — Validation

#### SWM-015 — ExamPen Test Harness

- Status: COMPLETE
- Objective: Add the first dedicated ExamPen test suite covering gate, ingest, DCR, PCR, and practice boundary rules.
- Files to create:
  - `backend/tests/exam_conductor/__init__.py`
  - `backend/tests/exam_conductor/test_gate.py`
  - `backend/tests/exam_conductor/test_ingest.py`
  - `backend/tests/exam_conductor/test_dcr.py`
  - `backend/tests/exam_conductor/test_pcr.py`
  - `backend/tests/exam_conductor/test_practice_boundary.py`
- Depends on: SWM-001, SWM-002, SWM-003, SWM-006, SWM-011
- Review when done: test IDs in `governance/TEST_SUITE_SPEC.md` are covered by named tests.

---

## Swarm Rules

1. One spawned agent claims one `SWM-*` task.
2. The agent should edit only the listed files.
3. If a task needs a new file outside its write set, that is a task-boundary problem and should be reviewed before proceeding.
4. `backend/main_async.py`, `backend/core/tenant_features.py`, `backend/api/v1/practice_async.py`, `backend/core/ocr_service.py`, and `backend/api/v1/tutor_async.py` are conflict-prone files and must each have a single owning task.
5. A task is reviewable only when its claimed file set matches this board.

---

## Recommended Spawn Order

1. Spawn in parallel: `SWM-001`, `SWM-002`, `SWM-003`, `SWM-004`
2. Then spawn in parallel: `SWM-005`, `SWM-006`, `SWM-007`, `SWM-009`, `SWM-010`
3. Then spawn in parallel: `SWM-011`, `SWM-012`, `SWM-013`
4. Then: `SWM-008`
5. Then: `SWM-014`
6. Finally: `SWM-015`

---

## Legacy Backlog Mapping

- `ING-*` is now covered by `SWM-002` and `SWM-012`
- `GATE-*` is now covered by `SWM-001`, `SWM-007`, `SWM-011`
- `DCR-*` is now covered by `SWM-003`, `SWM-010`, `SWM-013`
- `PCR-*` is now covered by `SWM-004`, `SWM-005`, `SWM-006`, `SWM-009`
- `BE-*` and `OPS-*` are now covered by `SWM-008`, `SWM-012`, `SWM-013`, `SWM-014`
- `QA-*` is now covered by `SWM-015`

---

## Ground Rules

1. Build against active `new-docs` docs only.
2. Treat DCR, PCR, ingest, and gate as separate modules.
3. Do not redesign practice-mode persistence.
4. Do not bypass the shared LLM gate.
5. Do not accept client-submitted answer text as authoritative for conducted exams.

---

## Deployment Checklist

- [x] sys.path setup in main_async.py for exam-conductor imports
- [x] ensure_indexes() called during app lifespan startup
- [x] LLM Vision OCR via shared gate (no ONNX/PaddleOCR dependencies)
- [x] exampen feature flag in tenant_features.py (MAX tier, default OFF)
- [x] 18 ExamPen routers mounted in main_async.py with graceful degradation
- [x] All 16 router modules pass py_compile and direct import smoke (Step 23)
- [x] All 23 exam-conductor sub-packages import cleanly (Step 23)
- [x] ExamPen import block in main_async.py:303-351 succeeds in isolation (Step 23)
- [ ] Enable exampen feature for test tenant
- [ ] Run pytest backend/tests/exam_conductor/ -v
- [x] Verify backend starts — `import main_async` succeeds, `_evalpen_available = True`, all 18 ExamPen routers + all other routes load (Step 24)
- [ ] Test endpoint access with feature flag ON vs OFF

## Known Gaps (non-blocking)

| Component | Status | Impact |
|---|---|---|
| DCR/PCR Vision OCR | Working (LLM Vision through gate) | Recognition quality depends on gate provider and model configuration |
| Practice image OCR | Working (LLM Vision camera adapter) | Camera-based practice eval functional via gate |
| Token rollup cron | Celery beat schedule in `celery_app.py` runs daily at 01:00 UTC | Auto-scheduled via `celery -A celery_app worker -B` |
| exampen_question_regions | Not populated — requires exam paper template/layout system for bbox data | DCR uses whole-page fallback from answer keys (`evalpen_dcr_async.py:274-294`). Population TODO at `tutor_async.py:2936`. No code gap — waiting on frontend template editor. |
| `pandas` / venv deps | Resolved (Step 24) — `pip install -r requirements.txt` installed all missing deps including pandas, pyotp, etc. | `main_async.py` now loads fully with `_evalpen_available = True`. |
| Backend recheck/conversation routers | Frontend defines `RecheckStatus` (with `submission_id`) and `ConversationThread` types; backend endpoints not yet mounted | Frontend recheck/conversation UI will be non-functional until backend routers are implemented. |
| Plagiarism API | Spec exists in `api/plagiarism.openapi.yaml` and chapter 13 | Not mounted; not implemented |
| Analytics API | Spec exists in `api/analytics.openapi.yaml` and chapter 15 | Not mounted; not implemented. Super-admin shows token usage only. |
| Hub production packaging | Runtime authority now targets converged `edge_hub`; `HUB-exam-conductor` remains reference implementation | Need migration plan and validation for independent ExamPen mode services inside `edge_hub` |
| Mobile camera fallback student ID | Manual entry required | Roster-backed student selector not yet implemented |
| Super-admin exam/submission counts | Frontend shows placeholder dashes | Needs new backend endpoint for per-tenant ExamPen stats |
| Hub decommission endpoint | Spec exists in `SUPERADMIN_SPEC.md` §5.2 | Not implemented |
| Hub detail endpoint | Spec exists in `SUPERADMIN_SPEC.md` §5.2 | Not implemented |

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-05-09 | Aligned status tracker with hub authority change: new edge work targets converged `edge_hub` runtime; `HUB-exam-conductor` is reference/decomposition donor. | Codex |
| 2026-05-02 | Reconciled with current frontend/mobile/super-admin implementation. Expanded Current Code Reality with frontend teacher workspace, student portal, super-admin, mobile status. Expanded Known Gaps with recheck/conversation backend parity, plagiarism, analytics, hub productionization, mobile student ID, super-admin endpoints. | Claude |
| 2026-03-25 | All SWM-001 through SWM-015 marked COMPLETE. Added deployment checklist and known stubs section. Infrastructure fixes applied (sys.path, indexes, requirements). | Claude |
| 2026-03-24 | Replaced the completed documentation-migration tracker with the active implementation backlog, using modular workstreams for ingest, DCR, PCR, gate, backend integration, and validation. | Codex |
