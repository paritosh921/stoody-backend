# Student contribution credits — traceability

| Requirement / constraint | Formal behavior or property | Implementation evidence required |
|---|---|---|
| `CR-001..006`, `CR-008` | `SourceLoop` source completion, decision branches | Source adapters, deterministic metric fixtures, shared-gate semantic judge tests |
| `CR-007`, `CR-020`, `SAFE-006` | `PolicyLoop`, `jobPolicy`, `CommittedJudgmentIsPinned` | Admin-only policy API, validation, immutable version/cutoff tests |
| `CR-009`, `LIVE-002..003` | processing retry/failure and lease epoch branches | CAS lease claim/reclaim, next-attempt, retry exhaustion tests |
| `CR-010`, `SAFE-002` | `PersistJudgment`, `PersistLedger`, unique source state, totals invariants | Mongo unique indexes and duplicate/concurrent worker tests |
| `CR-011` | `pageAward`, positive delta, `PageAwardsMatchLedger` | cumulative page versions in both processing orders |
| `CR-012`, `SAFE-003`, `X-AUTH-005`, `BE-TEN-001..003` | `StudentOf`, per-student totals, source ownership | tenant DB resolution and server-resolved student identity tests |
| `CR-013`, `SAFE-005` | ledger plus `TotalsMatchLedger`, rejected zero award | append-only ledger and aggregation tests |
| `CR-007`, `SAFE-003` | `DailyCap`, `DailyCapRespected`, capped `PersistLedger` | durable student-day award lock and concurrent distinct-source cap test |
| `CR-014..018` | projection-only boundary; not pixel-modeled | mobile/web component, navigation, privacy, responsive build tests |
| `CR-019` | no debit transition exists | API negative test/no redemption route; schema-only future entry type |
| `SAFE-001` | `NoCreditBeforeSource`, `JobRequiresSource` | enqueue occurs only after successful write; failure injection test |
| `SAFE-004`, `PipelineIsIndependent` | source/normal state changes only at upload | existing stroke/ingest/PCR regression tests |
| `SAFE-007` | bounded students; authorization abstracted | tenant leaderboard and tutor scoped-student tests |
| `LIVE-001` | source complete with `jobState = none`, later `SourceLoop` enqueue | reconciliation test for crash gap |
| `LIVE-004` | `PersistJudgment`, `PersistLedger`, `FinishJob`, temporal terminal property | job completion then summary/leaderboard integration test |
| `X-STROKE-002`, `MP-STROKE-*` | source geometry is environmental input | canonical-only filter and BLE conformance runner |
| `X-OFFLINE-003`, `MP-MOB-004` | source completion precedes observer; no outbox transition changed | mobile stroke-sync focused tests |
| `BE-UPLOAD-*`, `BE-PCR-*` | photo source completes before credit observer | private S3 bounded read and answer-copy/PCR regression tests |
| shared LLM gate | accept/reject/transient nondeterminism | registered caller spec/model edit and gate contract tests |
| `BE-SEP-005`, `MP-SEP-002..004` | outside TLC | builds, service integration/deploy, device/hardware validation report |

## Confirmed as-built evidence

Code evidence: `services/student_credits.py` owns policy, job, judgment, ledger, lock, metric, semantic-judge, reconciliation, summary, and leaderboard behavior; source adapters are in `api/v1/strokes_async.py` and `api/v1/evalpen_student_submission_async.py`; authenticated APIs are in `api/v1/credits_async.py`; background execution is in `celery_app.py`.

Test evidence: `tests/test_student_credits.py`, `tests/test_canvas_stroke_canonical_contract.py`, `tests/test_database_indexes.py`, `tests/exam_conductor/test_student_answer_copy_submission.py`, and `tests/exam_conductor/test_private_s3_student_copy_storage.py` passed together after the final concurrency fix (34 tests).

Validation evidence: TLC safety and liveness passed after adding `DailyCapRespected`; mobile TypeScript and 16 focused Jest assertions passed; the frontend production build passed; backend route registration and Python compilation passed; `git diff --check` passed in all touched repositories.

| Final mapping | Code evidence | Test/validation evidence | Status |
|---|---|---|---|
| `REQ-CR-001..013`, `REQ-CR-020`, all `REQ-SAFE-*`, all `REQ-LIVE-*` | backend credit domain, source adapters, tenant schema/indexes, Celery, shared LLM caller, authenticated API | bounded TLC plus focused backend suites and route/static checks | delivered locally |
| `REQ-CR-014..015`, `REQ-CR-017..018` | mobile services, profile/detail/upload/leaderboard screens and navigation | Jest, TypeScript, feature design-token scan | delivered locally |
| `REQ-CR-016..018` | frontend service, projection card, student and tutor dashboard integration | Vite production build | delivered locally |
| `REQ-CR-019` | no debit/redemption transition or endpoint | source/route inspection | preserved non-goal |
| Android native packaging | unchanged repository build system | blocked before source compilation by native-module variant resolution | residual gap accepted |
| Live services, model calibration, deployment, device/hardware | outside local implementation evidence | not exercised | residual gap accepted |
