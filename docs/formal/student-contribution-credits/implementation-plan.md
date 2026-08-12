# Student contribution credits — implementation plan

Formal gate: passed on 2026-08-12. This plan is derived from the checked packet.

Canonical plan scope: `REQ-CR-001` through `REQ-CR-020`, with the `REQ-SAFE-*` and `REQ-LIVE-*` aliases defined in `requirements.md`.

| Order | Requirement/context ID | PlusCal label/TLA+ property | Code owner/interface | Atomicity/persistence rule | Implementation task | Validation | Depends on |
|---:|---|---|---|---|---|---|---|
| 1 | `CR-007`, `BE-TEN-003` | `PolicyLoop`, `jobPolicy` | backend credits domain + tenant DB | policy updates create a new immutable version; jobs pin it | Add policy models/defaults, tenant collection classification/indexes, admin GET/PUT API | policy validation/auth/version tests | formal gate |
| 2 | `CR-012..013`, `SAFE-002..003` | `StudentOf`, ledger/totals invariants | backend credits repository | unique judgment and ledger keys; append-only awards; no mutable balance truth | Add collection indexes, identity resolver, aggregate summary/tier/leaderboard queries | duplicate key, tenant isolation, aggregation tests | 1 |
| 3 | `CR-003..004`, `CR-011` | accepted/rejected decision, cumulative delta | backend stroke judge | metrics from stored canonical geometry; target delta committed once | Implement canonical-only geometry metrics, anti-repeat/scribble gates, PIL rendering, semantic result contract | stroke fixtures, ordering, client-count distrust tests | 1,2 |
| 4 | `CR-005..006` | accepted/rejected decision | backend image judge/private S3 | bounded read-only evidence; no ingest mutation | Implement CV metrics and semantic image quality contract | sharp/blur/dark/blank fixture tests and read failure retry | 1,2 |
| 5 | shared gate / ExamPen authority | transient decision branch | `exam-conductor/llm_gate` | all semantic calls use registered audited caller | Document/register `credits_quality_judge`; add strict JSON multimodal call | gate allow-list and malformed/budget failure tests | 3,4 |
| 6 | `LIVE-001..004`, `SAFE-001..005` | `SourceLoop`, `PersistJudgment`, `PersistLedger`, `FinishJob` | backend durable worker/Celery | best-effort post-success enqueue; reconciler repairs; CAS lease; retry separate from reject | Implement job repository, source adapters, worker task, reconciler, unique judgment/ledger keys, source-group CAS lock, and UTC student-day cap lock | crash-gap, lease reclaim, stale/duplicate dispatch, and concurrent daily-cap tests | 2..5 |
| 7 | `CR-001`, `SAFE-004`, `MP-MOB-004` | source completion before job | backend `strokes_async.py` | enqueue after successful page persistence only | Emit server-resolved page-version credit source without changing response or merge semantics | existing stroke tests + enqueue failure non-regression | 6 |
| 8 | `CR-005`, `SAFE-004`, `BE-UPLOAD-*` | source completion before job | backend student answer-copy route | enqueue after canonical ingest; never roll back source/PCR | Record mobile/web channel and enqueue photo source best-effort | answer-copy/PCR regression and enqueue failure test | 6 |
| 9 | `CR-013..017`, `SAFE-007` | projections of ledger | backend `/credits` API | read-only aggregation; student privacy and tutor scoping | Add summary, recent activity, student leaderboard, tutor leaderboard endpoints | role/scope/privacy/status tests | 2,6 |
| 10 | `CR-005`, `CR-014` | source adapter projection | mobile student home/API | existing private upload remains source authority | Add exam option fetch and multi-image capture/selection upload screen using Vision Camera/document picker, then show accepted processing ack | service/component/type tests + Android compile | 8 |
| 11 | `CR-014`, `CR-018` | read-only projection | mobile profile stack | UI never computes or mutates balance | Add compact profile card and full credits screen with original tier badge, stats, split, recent, leaderboard | loading/empty/error/privacy/navigation tests | 9 |
| 12 | `CR-015`, `CR-017` | scoped projection | tutor mobile dashboard | use tutor-scoped backend response | Add compact leaderboard card and full route | tutor navigation and scope response tests | 9,11 |
| 13 | `CR-016..018`, `FE-API-*` | read-only projection | web shared API/dashboard components | shared API meanings; no client authority | Add shared credits panel/dialog to student Dashboard and TeacherOverview with responsive original tier visuals | component tests, typecheck/build | 9 |
| 14 | all preserved constraints | all safety properties | cross-repo | no source pipeline change outside observer hook | Run backend focused/regression tests, mobile Jest/typecheck/Android build, frontend tests/build, BLE conformance | recorded command evidence | 1..13 |

## State ownership mapping

- `currentPolicy`: backend tenant `student_credit_policies` repository; only admin policy API writes.
- `sourceState` / `normalPipeline`: existing canvas and ExamPen ingest owners; credits only read.
- `jobState`, `jobPolicy`, `attempts`, `leaseEpoch`: backend `student_credit_jobs` repository and CAS worker operations.
- `decision`, `targetCredits`: credits quality service output persisted with job/judgment.
- `judgment`, `judgmentPolicy`: backend `student_credit_judgments` unique terminal record.
- `ledgerCommitted`, `ledger`: backend `student_credit_ledger` unique append-only record.
- `pageAward`, `studentTotal`: query projections over ledger, not independently writable state.

## Change control

If implementation reveals a new source writer, a need to alter source ingest state, non-atomic judgment/award behavior that cannot be recovered idempotently, or a new role/tenant boundary, stop and revise the formal packet before continuing that task.
