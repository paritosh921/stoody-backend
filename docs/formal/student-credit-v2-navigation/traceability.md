# Student credit V2 and mobile navigation traceability

Date: 2026-08-12  
Status: confirmed-as-built traceability

Stable mapping groups: `REQ-V2-AWARD`, `REQ-V2-POLICY`, `REQ-V2-IDENTITY`, `REQ-V2-NAVIGATION`, `REQ-V2-SAFETY`, and `REQ-V2-LIVENESS`. Detailed `V2-*` identifiers remain the row-level requirements below.

| Requirement or constraint | Formal behavior/property | Code owner/interface | Verification class |
|---|---|---|---|
| `V2-CR-001`, `V2-CR-002` | `StrokeAward`, `PhotoAward`, `AwardFormulaBoundaries`, `V2AwardCapRespected` | `backend/services/student_credits.py` judge functions | Backend unit tests with boundary measurements/pages |
| `V2-CR-003`, `V2-SAFE-004` | `DayCap`, `ActivationGuardPreserved`; award step caps under shared daily state | Backend ledger commit/student-day lock | Concurrent daily-cap tests plus two-source TLC safety |
| `V2-CR-004`, `V2-CR-011` | `TierOf`, `TierBoundaries`; tier is backend-derived | Backend summary/leaderboard and mobile/web services | Backend boundary tests; mobile/web normalization tests |
| `V2-CR-005` | `balance`/`ledger` are unchanged by `Activate`; `LedgerMatchesBalance` | Existing append-only ledger; activation endpoint | Test pre/post balance equality and changed tier projection |
| `V2-CR-006`, `V2-SAFE-005`, `MP-CREDIT-004` | atomic `Enqueue`, immutable `jobPolicy`, `PolicySnapshotPinned` | Enqueue policy snapshot and worker judge | Snapshot race/pinning tests; source jobs retain V1 after policy changes |
| `V2-CR-007`, `V2-SAFE-006` | `Activate` guard; recurring `DayBoundary`; `ActivationEventuallySucceeds` | Admin V2 activation plus policy-transition lock | Idempotency, open-job conflict, daily-over-cap conflict, successful activation tests |
| `V2-CR-008` | static valid formula/menu/tier invariants represent accepted preset shape | Policy validator and Pydantic/API error mapping | Invalid tier/threshold/cap relationship tests |
| `V2-CR-009`, `V2-LIVE-001`, `V2-LIVE-002` | `RecoverCompletion`, `TerminalMissingCompletionIsVisible`, `AllJobsEventuallyTerminal` | Reconciler persisted lookup count/status/reason | Bounded missing-source and recoverable-source tests |
| `V2-CR-010`, `V2-SAFE-003`, `MP-CREDIT-003` | one source actor/job/ledger slot per immutable source; no second award transition | Photo descriptor; existing unique indexes/idempotent ledger | Immutable re-upload/source-version and duplicate replay tests |
| `V2-CR-012`, `V2-SAFE-008` | `surface="avatar"`; `AvatarIsIdentityOnly` | `StoodyHeader`; identity overlay component | Render/actions/accessibility/dismiss tests |
| `V2-CR-013` | single-valued `surface` and UI transitions | Header modal state | Tests opening one closes the other and backdrop/back dismiss |
| `V2-CR-014`, `V2-CR-015` | `StudentMenu`, `TutorMenu`, `MenuShapeCorrect` | Student/teacher hamburger components and navigation types/stacks | Exact order and nested-route Jest tests |
| `V2-CR-016` | Credits route commit; Profile is distinct route | Student profile stack/screen; tutor Home stack | Student Profile has no credit fetch/card; route tests |
| `V2-CR-017`, `V2-SAFE-007` | Rewards route only; no backend write variable in UI process | Shared Rewards screen | Component/navigation tests and source search for mutation API |
| `V2-CR-018` | Existing dashboard projection outside modeled canonical navigation | Tutor dashboard credit card | Preserved by diff/build verification |
| `V2-LIVE-003` | `activationRequested ~> activePolicy="v2"` | Explicit admin operation after drain/day guard | API success test and TLC liveness |
| `V2-LIVE-004` | `RequestedNavigationCommits` | Hamburger navigation callbacks/stacks | Student/tutor route tests |
| `V2-SAFE-001`, `MP-CREDIT-002` | Source completion precedes actor lifecycle; credit variables cannot alter source state | Existing best-effort enqueue call sites | Preserve source handlers; existing/new failure-isolation tests |
| `V2-SAFE-002`, `MP-CREDIT-001`, `MP-MOB-001` | only backend actor writes ledger/policy/tier | Backend services; clients projection only | Client source review and tests; no client mutation endpoints |
| `MP-SEP-002` | Outside TLC | `mobile-app` Jest/typecheck/Android Gradle build | Separate build/test evidence |

## State ownership

| Modeled state | Implementation owner |
|---|---|
| `activePolicy`, activation guard | Tenant `student_credit_policies` document via backend service only |
| `jobState`, `jobPolicy`, completion lookups | Tenant `student_credit_jobs` via backend enqueue/reconciler/worker only |
| `decision` | Tenant `student_credit_judgments` via backend worker only |
| `ledger`, `balance`, `dailyAward` | Append-only tenant ledger plus aggregation under backend locks |
| tier ladder/placement | Backend policy and summary/leaderboard projection |
| `surface`, `requestedRoute`, `route` | Mobile header/navigation state; no credit authority |

## As-built code evidence

| Stable requirement | Delivered code evidence |
|---|---|
| `REQ-V2-AWARD` | `services/student_credits.py`: `DEFAULT_POLICY`, `V2_AWARD_POLICY`, stroke/photo judge award formulas, `tier_for_credits` |
| `REQ-V2-POLICY` | `services/student_credits.py`: `_validate_policy_semantics`, `_validate_tier_input_order`, `get_credit_policy`, `initialize_credit_policy`, `activate_v2_credit_policy`, `enqueue_credit_job`, `reconcile_credit_jobs`, `get_student_credit_summary`; `api/v1/credits_async.py`: `/policy/activate-v2` |
| `REQ-V2-IDENTITY` | `mobile-app/src/components/layout/ProfileMenu.tsx`; `StoodyHeader.tsx` |
| `REQ-V2-NAVIGATION` | `StudentHamburgerMenu.tsx`, `teacherMoreMenu.ts`, `TeacherHamburgerMenu.tsx`, `StudentProfileStack.tsx`, `MoreStack.tsx`, `RewardsPlaceholderScreen.tsx`, `StudentProfileScreen.tsx` |
| `REQ-V2-SAFETY` | Backend transition/group/day locks and unique judgment/ledger indexes; projection-only `mobile-app/src/services/creditsService.ts` and `frontend/src/services/creditsService.ts` |
| `REQ-V2-LIVENESS` | Bounded missing-completion reconciliation, worker retry states and menu navigation callbacks/stacks |

## Test and validation evidence

| Stable requirement | Test or validation evidence |
|---|---|
| `REQ-V2-AWARD`, `REQ-V2-POLICY`, `REQ-V2-SAFETY`, `REQ-V2-LIVENESS` | `tests/test_student_credits.py`: 34 passing tests covering arithmetic boundaries, tier boundaries, activation guards/idempotency, enqueue exclusion, policy semantics, preserved ledger, immutable photos, projection-only reads and bounded recovery |
| `REQ-V2-IDENTITY`, `REQ-V2-NAVIGATION` | Mobile full Jest run: 45 suites/244 tests; post-device-review route/fallback suite: 4 suites/15 tests; TypeScript check passed. Profile and Credits target distinct nested routes, and the display-only five-tier fallback cannot identify Seed as highest. |
| All modeled groups | Fresh `Model.cfg`: 14,449 generated/4,104 distinct states; fresh `ModelSafety.cfg`: 670,913 generated/161,424 distinct states; no TLC errors |
| Separately validated build obligations | Frontend Vite production build passed; repository Android release builder and four stroke-render preflight tests passed; final APK hash and successful USB install/launch are recorded in `implementation-evidence.md` |

## Preserved constraints

- Upload/PCR/canvas handlers remain primary and return independently of credit observer failures.
- Existing V1 unique indexes, judgment keys, source group locks, and student-day locks remain in force.
- Existing student/tutor authorization and privacy-scoped leaderboard behavior remains unchanged.
- Existing mobile `No data` defensive communication remains unchanged.
- Existing tutor dashboard credits projection remains present.
- Unrelated `.serena/project.yml` is excluded from all edits and staging.
