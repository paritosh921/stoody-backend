# Student credit V2 and mobile navigation implementation evidence

Date: 2026-08-12  
Implementation status: complete
Code evidence: backend policy, activation, enqueue, reconciliation and summary symbols plus mobile/web projection and navigation paths listed below.
Validation evidence: fresh TLC checks, 34 backend tests, 244 mobile tests, TypeScript, frontend build and final Android release build listed below.
Deviations from checked packet: resolved

## Code evidence

- Credit economics and tier authority: `services/student_credits.py` `DEFAULT_POLICY`, `V2_AWARD_POLICY`, `_validate_policy_semantics`, `_validate_tier_input_order`, `tier_for_credits`, and `get_student_credit_summary` implement the 250 mm/credit, 5-credit pen-page cap, 1-credit photo-page rate, 10-credit photo-submission cap, 100-credit daily cap, backend tier ladder, policy version and full-ladder projection.
- Explicit policy lifecycle: `services/student_credits.py` `get_credit_policy`, `initialize_credit_policy`, `update_credit_policy`, `activate_v2_credit_policy`, `_acquire_named_lock`, and `enqueue_credit_job` keep reads projection-only, initialize at write boundaries, serialize enqueue snapshots against activation, preserve stored ledger history and refuse unsafe activation.
- Recovery and idempotency: `services/student_credits.py` `_acquire_missing_completion_lookup_slot`, `_finalize_missing_completion_failure`, `reconcile_credit_jobs`, existing unique indexes, group/day locks and ledger commit logic keep retries bounded and awards idempotent without changing the primary upload records.
- Administrative interface: `api/v1/credits_async.py` `POST /policy/activate-v2` and the policy update handler expose typed 409 conflict and 422 semantic-validation outcomes to admins.
- Backend-driven clients: `mobile-app/src/services/creditsService.ts`, `mobile-app/src/types/credits.ts`, `mobile-app/src/screens/student/StudentCreditsScreen.tsx`, and `frontend/src/services/creditsService.ts` normalize backend policy versions, ladder definitions and placement without a competing local threshold ladder.
- Identity-only avatar surface: `mobile-app/src/components/layout/ProfileMenu.tsx` and `StoodyHeader.tsx` implement the centered read-only identity overlay and mutual exclusion with the hamburger surface.
- Canonical navigation: `StudentHamburgerMenu.tsx`, `teacherMoreMenu.ts`, `TeacherHamburgerMenu.tsx`, `StudentProfileStack.tsx`, `MoreStack.tsx`, and navigation types place Learning Credits immediately above Rewards for students and tutors. Student Profile no longer owns the credits card.
- Future redemption boundary: `mobile-app/src/screens/RewardsPlaceholderScreen.tsx` is a shared read-only placeholder with no credit mutation or redemption API.
- Policy/runbook evidence: `stoody-multi-pen/docs/STUDENT_CREDIT_AWARD_POLICY.md` and `stoody-multi-pen/docs/formal/constraint-index.md` record the V2 formulas, ownership, activation guards, migration rule and remaining rollout work.

## Configuration and schema evidence

- Existing MongoDB collections remain the persistence owners: `student_credit_policies`, `student_credit_jobs`, `student_credit_judgments`, `student_credit_ledger`, and `student_credit_locks`.
- No destructive schema migration is introduced. Existing tenant policies remain stored until explicit V2 activation; new policy initialization uses the V2 default.
- Queued jobs retain their `policy_version` and complete `policy_snapshot`; activation never rewrites queued jobs, judgments or ledger entries.
- Photograph evidence remains immutable at source version `1`; corrected uploads require a new submission identity.

## Validation evidence

- Formal liveness: `tla_tool.py check Model.tla --config Model.cfg` passed with 14,449 generated states, 4,104 distinct states, depth 20 and zero states left on queue.
- Formal two-source safety: `tla_tool.py check Model.tla --config ModelSafety.cfg` passed with 670,913 generated states, 161,424 distinct states, depth 30 and zero states left on queue.
- Backend behavior: `backend/venv/Scripts/python.exe -m pytest tests/test_student_credits.py -q` passed 34 tests. Coverage includes reward/tier boundaries, semantic policy validation, activation guards/idempotency, enqueue exclusion, balance preservation, immutable photos, summary ladder authority, projection read-only behavior and bounded missing-completion failure.
- Mobile behavior: full `npm test` passed 45 suites and 244 tests. After the final controller corrections, focused header/menu tests passed 2 suites/3 tests and `npm run typecheck` passed.
- Web projection/build: `frontend npm run build` passed; only pre-existing browser-data, mixed-import and chunk-size warnings were reported.
- Android package: the final `build-install-android.bat --install` run passed its four stroke-render preflight tests, Gradle release assembly and ADB replacement install. The final APK is `mobile-app/apk-output/StoodyAndroid-release.apk` (50,659,836 bytes; SHA-256 `BC5CB4510FB9CA90C1574E85E9BDA92B0AC5F0E3D146CEF9E95C0B9FDDAA813C`). Package `com.stoodyapp` version `1.0.3` (`versionCode 4`) was installed and launched on device `KJ95NJIFY9XGRKIZ`.
- Diff hygiene: `git diff --check` passed in backend, frontend and stoody-multi-pen. The unrelated pre-existing `stoody-multi-pen/.serena/project.yml` change was not edited or included in this scope.

## Separately validated obligations

- Upload-pipeline isolation is preserved structurally: credit enqueue remains a best-effort observer after primary writes and source handlers were not rewritten by this change.
- Existing daily/group/ledger concurrency controls remain in place in addition to the new policy-transition lock.
- The direct long-path `npm run android:debug` path failed during native-module variant resolution; the repository-supported short-path release builder completed successfully twice, including once after all final source corrections.
- Spark implementation agents produced the bounded backend/mobile changes. The controller re-inspected and corrected them. GLM 5.2 was invoked twice through OpenCode for advisory review but timed out without output, so no GLM finding is treated as validation evidence.

## Deviation resolution detail

Resolved. Controller review found that unordered tier input was being sorted rather than rejected and that terminal missing-completion reconciliation needed a status/count predicate to avoid racing a claimed worker job. Both were corrected and covered by tests; the checked ownership, activation and recovery model did not require revision.

## Residual gaps

- No production tenant policy was activated and no production ledger/job/balance audit was run. Existing tenants therefore continue using their stored policy until an admin explicitly activates V2.
- No staging calibration was performed with representative genuine handwriting, sparse pages, random scribbles, blurred/glare/skewed photos or semantic-judge outages.
- The final APK was installed and launched on one physical Android device, and the user confirmed the corrected Profile/Credits behavior. Wider device coverage remains unverified.
- Rewards is intentionally a non-functional placeholder; redemption ownership, catalog, inventory, transaction and reversal semantics remain future work.
- Deployment, service restart behavior and live MongoDB/Celery concurrency were not exercised in this local verification.
