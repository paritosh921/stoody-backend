# Student contribution credits — boundary decisions

Status: Phase 0 decision ledger  
Date: 2026-08-12

Post-implementation review: complete

## Conflict resolution

The phrase “bad quality submissions ... rejected” conflicts with the requirement that credits run only after upload completion and not disturb the normal pipeline. Resolution: rejection is a **credit eligibility judgment only**. The source upload remains durable and continues through its existing academic pipeline.

## Classified constraints

| ID / concern | Class | Decision |
|---|---|---|
| `X-OWN-001` durable cloud owner | `PRESERVED` | Backend owns credit records. Mobile/web are read-only projections except normal source-upload APIs. |
| `X-STROKE-002`, `MP-STROKE-001..004` canonical geometry | `PRESERVED` | Credits consume canonical output and do not add another stroke repair/canonicalization authority. |
| `X-OFFLINE-003`, `MP-MOB-004` at-least-once mobile sync | `PRESERVED` | Existing outbox and merge flow remain unchanged; credits begin after backend success. |
| `X-INGEST-004`, `BE-UPLOAD-*` private/canonical ingest | `PRESERVED` | Student photo reward observes canonical student answer-copy ingest. Tutor camera fallback is ineligible. |
| `X-AUTH-005`, `BE-TEN-001..003`, `FE-AUTH-001` | `MODELED` | Tenant/student identity accompanies every job/judgment/award; cross-tenant transitions are forbidden. |
| New collections under `BE-TEN-003` | `MODELED` | Explicitly tenant scoped and registered before application code uses them. |
| Mongo duplicate/CAS behavior | `MODELED` | Unique source-version keys, lease-token CAS, immutable terminal decision, unique ledger insert. Concrete index behavior is separately tested. |
| Shared LLM gate | `PRESERVED` | Add and document one registered `credits_quality_judge` caller before first call. No direct provider SDK usage. |
| PCR/DCR/publication/practice ownership | `PRESERVED` | Credits cannot write these collections or alter their status. |
| Policy update during pending job | `MODELED` | Job pins `policy_version`; later updates affect later jobs only. |
| Crash after source success before enqueue | `MODELED` | Reconciler can create the missing job. |
| Duplicate dispatch/expired lease/stale worker | `MODELED` | At-most-one terminal judgment and award; stale lease cannot finish another worker's claim. |
| Cumulative page later receives more strokes | `MODELED` | Compute positive delta above awards already earned for the page; unchanged replay yields zero. |
| AI/storage/network transient failure | `MODELED` | Retry/failure state is distinct from quality rejection. |
| LLM semantic accuracy | `SEPARATELY_VALIDATED` | Calibrated fixtures and production monitoring; TLC treats semantic outcome nondeterministically. |
| Image and stroke metric math | `SEPARATELY_VALIDATED` | Deterministic unit tests with fixtures; TLC uses bounded quality/quantity values. |
| S3/Mongo/Redis/Celery durability and deployment | `SEPARATELY_VALIDATED` | Integration/deploy gates; no local service startup unless requested. |
| Mobile camera/Android permission and pixel layout | `SEPARATELY_VALIDATED` | Type/Jest/build locally, physical-device acceptance later. |
| Historical backfill | `IRRELEVANT` | No automatic backfill in v1; earning cutoff defaults to policy creation. |
| Redemption/goodies | `IRRELEVANT` | Explicit future scope. No debit endpoint or balance spending now. |

## Confirmed as-built disposition

All `MODELED` and `PRESERVED` boundaries above remain aligned with the delivered code. The implementation review found one material concurrency gap in initial code: distinct source groups could race the daily-cap read. It was resolved before confirmation by adding a durable UTC student-day award lock, extending the bounded model with `DailyCapRespected`, rerunning TLC, and adding a concurrent cap regression test.

The `SEPARATELY_VALIDATED` obligations remain accurately bounded: deterministic metrics, schemas, authorization, idempotency, concurrency, routes, TypeScript, Jest, and the web production build were locally exercised; deployed services, live semantic quality, physical camera/device behavior, and Android native assembly were not established.

Material unresolved unknowns: none
Conflict status: resolved

## Modeled state boundary

The formal model includes two students, durable source completion, optional missed enqueue, reconciliation, job claim/lease expiry, retry, policy pinning, accept/reject decisions, cumulative target credits, atomic judgment/ledger effects, duplicate dispatch, and projection totals.

It does not model coordinate arrays, image pixels, prompt text, provider behavior, HTTP rendering, physical storage, or UI layout. Those are inputs or separate validation boundaries.

## Material unknowns

None blocking formalization. Product choices resolved for v1:

- only students earn; tutors/admins view;
- only mobile canonical BLE writing and authenticated student answer-copy images earn;
- no historical backfill and no automatic rejudgment after policy edits;
- peer labels are privacy-reduced for student viewers;
- leaderboard ties sort by credits, then accepted count, then earliest achievement;
- infrastructure failure is never presented as rejection;
- reward caps are configurable anti-farming controls;
- redemption is not implemented.
