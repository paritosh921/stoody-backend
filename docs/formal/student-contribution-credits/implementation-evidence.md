# Student contribution credits — implementation evidence

Implementation status: complete
Code evidence: Backend credit ownership is implemented in `services/student_credits.py`, exposed by `api/v1/credits_async.py`, and connected after durable source writes by `api/v1/strokes_async.py` and `api/v1/evalpen_student_submission_async.py`.
Configuration/schema evidence: `core/tenant.py` classifies the tenant collections; `core/database.py` creates their indexes; `celery_app.py` schedules durable processing and reconciliation; `exam-conductor/llm_gate/models.py` registers the semantic judge caller.
Validation evidence: Revised TLC safety and liveness runs exited 0; 34 focused backend tests, 16 mobile tests, mobile TypeScript, backend compilation/route registration, web production build, and repository diff checks passed.
Separately validated obligations: Deterministic quality fixtures, strict semantic schema wiring, authorization, privacy, idempotency, retry/reconciliation, daily-cap concurrency, API contracts, mobile services/navigation, and web projection compilation passed locally. Deployed Mongo/S3/Redis/Celery, live LLM quality calibration, physical camera/device behavior, production deployment, and Android native assembly remain unverified.
Deviations from checked packet: resolved
Residual gaps: Android assembly is blocked before source compilation by repository-wide native-module variant resolution; no live external services, device, hardware, deployment, commit, or push validation was performed.

## Requirement evidence

| Scope | As-built code evidence | Test or validation evidence |
|---|---|---|
| `REQ-CR-001..004`, `REQ-CR-008`, `REQ-SAFE-008` | `services/student_credits.py` canonical stroke filtering, geometry metrics, hard gates, rendering, and strict semantic judgment; `api/v1/strokes_async.py` post-write adapter | `tests/test_student_credits.py`; `tests/test_canvas_stroke_canonical_contract.py` |
| `REQ-CR-005..006` | `services/student_credits.py` bounded private image read and image metrics; `api/v1/evalpen_student_submission_async.py` post-ingest adapter; mobile answer-copy screen/service | image metric, S3-prefix, all-page, schema, upload-service, and answer-copy ingest tests |
| `REQ-CR-007`, `REQ-CR-020`, `REQ-SAFE-006` | versioned tenant policy repository/API, pinned snapshots, earning cutoff, source/submission caps, and durable UTC student-day lock | policy-schema, cutoff/idempotency, disabled-reconciler, and concurrent daily-cap tests; `DailyCapRespected` TLC invariant |
| `REQ-CR-009..011`, `REQ-LIVE-001..004` | durable jobs, lease claims, bounded retries, unique judgments/ledger, cumulative group locks, and missing-source reconciliation in `services/student_credits.py`; Celery tasks in `celery_app.py` | credit job, duplicate ledger, positive-delta, stale recovery, and reconciliation tests; TLC safety/liveness runs |
| `REQ-CR-012..013`, `REQ-SAFE-002..003` | tenant-scoped collections and server-resolved student record identity; append-only ledger aggregation | database-index, ledger idempotency, cap concurrency, summary, and route registration checks |
| `REQ-CR-014..015`, `REQ-CR-017..018` | mobile profile card, student detail screen, tutor leaderboard route/card, privacy-aware API types, original constellation badges | 16 focused Jest assertions, TypeScript pass, and no hardcoded hex in feature-added lines |
| `REQ-CR-016..018` | frontend shared credits service/card integrated into student `Dashboard` and tutor `TeacherOverview` | Vite production build passed with 3,489 modules transformed |
| `REQ-CR-019` | no redemption/debit API or client operation exists; earned ledger writes use the `earned_award` entry type | route/source inspection; no redemption behavior in the delivered scope |
| `REQ-SAFE-001`, `REQ-SAFE-004` | credit hooks run after source persistence and catch observer failures; credits service has no canvas/PCR/publication/practice writers | canonical canvas, answer-copy, private-S3, and upload-security regression tests |
| `REQ-SAFE-005`, `REQ-SAFE-007` | quality reject writes zero award; retry/failure remains distinct; tutor endpoints reuse existing scope resolver; student peers are privacy reduced | rejection/idempotency and leaderboard tie tests; API/source authorization review |

## Validation layers

- Bounded model: safety run generated 13,387,737 states and 3,730,552 distinct states; liveness run generated 67,845 states and 26,792 distinct states; both found no error.
- Backend behavior: 34 focused tests passed after the final concurrency change. A broader run passed 71 tests and exposed two unrelated OCR default assertions (`gpt-4o` expected while the checkout resolves `gpt-5.6-terra`).
- Static/build: Python compilation, backend route registration, mobile TypeScript, mobile focused Jest, web production build, and `git diff --check` passed.
- Android: Gradle failed before application compilation because all autolinked React Native modules exposed no matching Android variants. The feature source was not reached by this gate.
- Runtime/deployment: no local servers, remote deployment, production services, physical device, camera, BLE hardware, commit, or push were exercised.

## Resolved deviation

Controller review found that initial code serialized cumulative awards only by source group, allowing distinct concurrent source groups to race the daily-cap read. Before confirmation, implementation gained a durable UTC student-day lock, the PlusCal model gained a capped `PersistLedger` transition and `DailyCapRespected`, TLC was rerun successfully, and a concurrent regression test was added. No unresolved behavioral deviation remains within the confirmed local scope.
