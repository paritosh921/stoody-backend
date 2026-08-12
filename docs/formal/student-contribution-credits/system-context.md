# Student contribution credits — system context

Status: Phase 0 evidence packet  
Date: 2026-08-12

Context kind: existing-repository
Context status: confirmed-as-built
Baseline preserved: yes

Evidence: The original Phase 0 sections below are preserved unchanged as the pre-implementation evidence baseline.

## Current owners and paths

| Concern | Current owner and evidence | Constraint classification |
|---|---|---|
| Cloud tenant, auth, student identity, durable records | Backend `core/tenant.py`, `core/database.py`, `core/user_identity.py`, auth dependencies | `X-OWN-001`, `BE-TEN-001..003`, `BE-AUTH-*` |
| Canonical mobile pen stroke shape | `stoody-multi-pen/mobile-app/src/utils/penCanvas.ts`; owner-produced BLE contract from edge hub | `X-STROKE-002`, `MP-STROKE-001..004` |
| Mobile at-least-once page delivery | `mobile-app/src/services/strokeSyncOutbox.ts`, `src/hooks/useStrokeSync.ts`, `src/services/studentStrokeService.ts` | `MP-MOB-001`, `MP-MOB-004` |
| Backend canvas persistence | `backend/api/v1/strokes_async.py` `_merge_stroke_docs`, `_build_merged_page_doc`, PUT `/pages`, POST `/pages/batch` | `X-OFFLINE-003`, backend Mongo atomicity cautions |
| Student answer-copy submission | `backend/api/v1/evalpen_student_submission_async.py` POST `/student/exams/{exam_id}/answer-copy` | `BE-UPLOAD-001..004`, `BE-PCR-*` |
| Private evidence read | `backend/utils/s3_storage.py` `download_private_object` with bucket/prefix and byte bounds | `BE-UPLOAD-003..004` |
| Durable background execution | `backend/celery_app.py`; PCR durable job and reconciler patterns | `BE-PCR-*`, `BE-SEP-005` |
| LLM authority | `backend/exam-conductor/llm_gate/*`; `new-docs/architecture/LLM_GATE_SPEC.md` | registered-caller and audit-log rules |
| Tutor scope | `backend/utils/tutor_scoping.py` `get_tutor_scoped_students` | backend authorization and assignment boundary |
| Mobile profile/dashboard | `mobile-app/src/screens/student/StudentProfileScreen.tsx`, `StudentProfileStack.tsx`, `src/screens/home/DashboardScreen.tsx` | UI projection; `MP-MOB-*` preserved |
| Web student/tutor dashboards | `frontend/src/pages/Dashboard.tsx`, `src/components/tutor/TeacherOverview.tsx`, shared API client | `FE-AUTH-001`, `FE-API-001..002`, `FE-EXAM-001` |

## Current upload boundaries

### Stoody BLE page

1. Mobile converts canonical samples into a `CompletedStroke` carrying millimetre coordinates, timestamps, pressure, source mode, and processing version.
2. The durable tenant/student/copy-scoped outbox batches a page and retries at least once until the backend acknowledges.
3. Backend validates canonical fields when `processingVersion` is present, merges by stroke ID, and stores the canvas page.
4. Credit enqueue may run only after the page write succeeds. It receives server-resolved tenant/student/page identity and the stored page version; it cannot use client credit claims.
5. A periodic backend reconciliation scan repairs a crash between steps 3 and 4.

### Student notebook/answer-copy image

1. Authenticated student selects an eligible exam and captures/selects page images in the mobile app.
2. Existing student answer-copy endpoint scans and stores private immutable page evidence, reserves the student submission, canonicalizes ingest, and queues PCR.
3. Credit enqueue may run only after canonical ingest succeeds. PCR enqueue failure does not invalidate the durable source or the credit job.
4. The credits worker reads bounded private image bytes through the existing private S3 helper and never changes ingest/PCR state.

## Proposed credit-owned state

All new collections are tenant-database collections and must be registered/classified before use.

- `student_credit_policies`: one current policy document per tenant, containing versioned thresholds, caps, tier thresholds, enabled state, and earning cutoff.
- `student_credit_jobs`: durable recoverable job state keyed uniquely by source type, source ID, and source version. States: `pending`, `processing`, `retry`, `completed`, `failed`; processing claims have lease token and expiry.
- `student_credit_judgments`: immutable terminal or latest-visible judgment keyed uniquely by tenant student and source version, with pinned policy version, metrics, decision, reasons, and computed target.
- `student_credit_ledger`: append-only awarded delta keyed uniquely by judgment/source version. Positive earned awards only in this release; future debit types are schema-reserved but not writable.

Balances, statistics, tiers, recent activity, and leaderboard are Mongo aggregations over judgments/ledger. No independently mutable `balance` field is authoritative.

## Competing actors and interleavings

- API writer persists the primary source and best-effort enqueues.
- Reconciler discovers eligible durable sources without jobs.
- Multiple Celery workers may claim, time out, retry, and redispatch the same job.
- Admin may change policy while an older job is pending; each job pins its creation policy version.
- A student may add strokes while a previous cumulative page version is being judged.
- Readers may request summary/leaderboard while a judgment or ledger insert is in flight.

## Failure boundaries

- Source write fails: no job and no credits.
- Source succeeds, enqueue fails/crashes: source stays successful; reconciler repairs.
- Worker cannot read source/private evidence: retry; never reject quality.
- Deterministic quality is below a hard gate: terminal rejection, zero award, source unchanged.
- LLM gate unavailable, budget-exhausted, malformed, or incomplete: retry/fail visibly; never quality-reject.
- Worker crashes after judgment insert but before ledger/job completion: unique keys plus retry finish without duplicate award.
- Worker lease expires: another worker may reclaim; stale lease holder cannot commit completion.
- Policy changes: pending jobs keep pinned policy; completed entries remain immutable.

## UI design system decision

Use the existing neutral card surfaces and selectable Stoody primary theme. Tier color is a contained accent, never a new page-wide theme:

| Tier | Threshold | Symbol | Accent |
|---|---:|---|---|
| Seed | 0 | sprouting dot/leaf geometry | fresh green |
| Scribe | 100 | original pen-nib path | blue |
| Pathfinder | 300 | compass path | violet |
| Beacon | 700 | lantern/radiating mark | amber |
| Luminary | 1500 | asymmetric learning-star mark | rose/gold |

Use Lucide primitives or original SVG composition, not third-party education-platform artwork. Gold is reserved for tier/top-rank accents. Mobile profile placement is below identity and before settings. Full detail uses one clear hierarchy: total and tier, progress, accepted/rejected/pending, contribution split, recent decisions, then leaderboard. Web uses compact dashboard cards that open a detail dialog/drawer rather than expanding the dashboard permanently.

## Separate validation

TLC abstracts cryptography, pixel mathematics, semantic model accuracy, Mongo/S3/Redis/Celery durability, actual scheduler timing, React layout pixels, camera permission behavior, and deployment. Those require focused tests, builds, controlled judge fixtures, service integration, and later live validation.

## Confirmed as-built context

Evidence: `backend/services/student_credits.py` is the sole credits-domain transition owner. It implements tenant policy snapshots, durable source-version jobs, deterministic metrics, the registered semantic judge, lease/retry recovery, immutable judgments, append-only ledger awards, cumulative-page deltas, and UTC student-day cap locking.

Evidence: `backend/api/v1/strokes_async.py` and `backend/api/v1/evalpen_student_submission_async.py` invoke best-effort credit enqueue only after their existing source writes complete. Their helpers catch credit failures, so the canvas and ExamPen source pipelines retain their pre-implementation success semantics.

Evidence: `backend/core/tenant.py` classifies all five credit collections as tenant scoped; `backend/core/database.py` installs policy, source-version, judgment, ledger, due-job, and lock indexes; `backend/celery_app.py` owns background processing and periodic reconciliation.

Evidence: `backend/api/v1/credits_async.py` owns authenticated policy, summary, and leaderboard interfaces. Student reads resolve the authenticated student record, tutor reads reuse `get_tutor_scoped_students`, and student leaderboard peer identifiers are privacy reduced.

Evidence: `backend/exam-conductor/llm_gate/models.py` and `backend/exam-conductor/new-docs/architecture/LLM_GATE_SPEC.md` register `credits_quality_judge`; the service uses the shared strict Responses-schema path rather than a direct provider client.

Evidence: `stoody-multi-pen/mobile-app/src/services/creditsService.ts`, `studentAnswerCopyService.ts`, the credits screens/cards, and the three navigation stacks are projection/upload adapters only. They do not calculate or persist authoritative balances. The existing user-owned `.serena/project.yml` edit is outside this baseline.

Evidence: `frontend/src/services/creditsService.ts`, `src/components/credits/CreditsProjectionCard.tsx`, `src/pages/Dashboard.tsx`, and `src/components/tutor/TeacherOverview.tsx` provide student and tutor read projections through the existing shared API client.

Final state ownership: backend tenant MongoDB owns policies, jobs, judgments, ledger, and locks; existing canvas/ExamPen owners retain primary source state; Celery owns asynchronous execution; mobile and web remain non-authoritative projections.

Final concurrency and recovery: unique source-version keys and leases fence duplicate workers; group locks serialize cumulative page awards; UTC student-day locks serialize cap calculation across distinct uploads; the periodic reconciler repairs missing jobs and expired leases.

Final compatibility boundary: a credit rejection means reward ineligibility only. Credits never reject, delete, roll back, or mutate canvas, answer-copy ingest, PCR, publication, or practice state.

Residual runtime boundary: local validation did not exercise deployed Mongo/S3/Redis/Celery, live LLM judgments, a physical camera/device, or production deployment. Android assembly remains blocked before source compilation by repository-wide React Native native-module variant resolution.
