# Student credit V2 and mobile navigation implementation plan

Date: 2026-08-12  
Status: implemented, validated and user-confirmed

Stable requirement groups used by this plan: `REQ-V2-AWARD`, `REQ-V2-POLICY`, `REQ-V2-IDENTITY`, `REQ-V2-NAVIGATION`, `REQ-V2-SAFETY`, and `REQ-V2-LIVENESS`. The table retains the detailed `V2-*` and constraint identifiers for precise traceability.

| Order | Requirement/context ID | PlusCal label/TLA+ property | Code owner/interface | Atomicity/persistence rule | Implementation task | Validation | Depends on |
|---:|---|---|---|---|---|---|---|
| 1 | `V2-CR-001..004`, `V2-CR-008` | formula/tier invariants | `backend/services/student_credits.py`, `api/v1/credits_async.py` | one normalized, semantically valid policy document | Add V2 preset/defaults, tier thresholds, formula values, cross-field/tier validator, typed API errors | policy/arithmetic/tier boundary tests | formal gate |
| 2 | `V2-CR-006..007`, `V2-SAFE-005..006` | `Enqueue`, `Activate`, pinning/activation properties | backend policy/job collections and lock collection | enqueue snapshot+insert and activation serialize on tenant policy-transition lock; activation preserves ledger | Add lock helper, guarded/idempotent admin `/credits/policy/activate-v2`, open-job and per-student UTC-day checks | race-shape, conflict, idempotency, balance-preservation tests | 1 |
| 3 | `V2-CR-009..010`, `V2-LIVE-001..002` | `RecoverCompletion`, terminal/liveness properties | credit reconciler and photo descriptor | persisted lookup count reaches terminal once; immutable photo identity stays version 1 | Bound missing completion lookups with stable failure reason; codify/test immutable photo rejudgment semantics | reconciler recovery/terminal and duplicate photo tests | 1 |
| 4 | `V2-CR-004..005`, `V2-CR-011` | `TierOf`, `TierBoundaries`, preserved balance | backend summary response | tier ladder/placement produced from one policy read; ledger untouched | Include full `tiers` and policy version in summary; verify new ladder reprojects existing totals | summary/tier/balance tests | 1 |
| 5 | `MP-CREDIT-001`, backend-driven ladder gate | tier properties | `frontend/src/services/creditsService.ts` and credits components/tests | web renders backend resolved placement; no local thresholds | Extend full-ladder projection types/normalization as needed; keep no-data behavior | frontend unit tests/typecheck/build | 4 |
| 6 | `V2-CR-012..013`, `V2-SAFE-008` | UI surface/identity invariant | `StoodyHeader.tsx`, replace `ProfileMenu.tsx` with identity overlay | one modal surface at a time; avatar overlay has no action callbacks | Build centered responsive identity-only overlay; remove header profile/logout wiring; mutually close overlay/drawer | component/header tests and accessibility assertions | formal gate |
| 7 | `V2-CR-014..018`, `V2-LIVE-004`, `V2-SAFE-007` | menu shape and navigation commit | hamburger components, navigation types/stacks, profile/rewards screens | menu press dismisses before route; Rewards has no credit write | Add adjacent Credits/Rewards menu entries for both roles, routes and shared placeholder; remove student Profile credit fetch/card; preserve tutor dashboard card | menu ordering/routes, Profile absence, Rewards no-write tests | 6 |
| 8 | `V2-CR-011`, `MP-CREDIT-005` | backend tier authority | mobile credits types/service/screens/tests | client uses returned tier/ladder and never derives placement from total | Add `tiers` contract, remove `DEFAULT_TIERS`, `tierForTotal`, `tierProgressForTotal`; neutral missing-payload display only | service/screen component tests | 4, 7 |
| 9 | docs and rollout contract | activation/model decisions | policy doc, constraint index, formal packet, API docs if present | proposed document becomes implemented-but-activation-explicit; no false deployed claim | Update policy/readiness text and cross-links; retain activation runbook and operational limitations | Markdown link/diff checks | 1..8 |
| 10 | all | all invariants/properties | backend, frontend, mobile | no unverified owner or mutation path | Controller reviews all delegated diffs; run GLM 5.2 narrow reviews; fix only verified findings | diff review, tests, builds, fresh TLC | 1..9 |

## Delegation and control

- Use `codex exec` with model `gpt-5.3-codex-spark` for bounded implementation tasks split by backend and mobile/web ownership. Each prompt names permitted files, formal packet, required tests, and the unrelated-file exclusion.
- Use `opencode run --model zai-coding-plan/glm-5.2` for compact post-diff reviews and noisy failure triage. GLM output is advisory; the controller verifies every finding against source/tests.
- The controller (this agent) owns architecture decisions, formal change control, diff inspection, test execution, integration corrections, and completion claims.

## Plan gate checklist

- Every V2 requirement and selected constraint has a plan row or explicit preserved obligation.
- Each modeled state has one implementation owner.
- Atomic policy activation/enqueue and daily-cap behavior match the checked model.
- Environmental outcomes (missing source, worker retry, day rollover, API failure) have deterministic tests.
- The two TLC counterexample classes have implementation requirements/tests.
- No material unresolved decision remains. Production activation itself is operational work and is not silently inferred from a code deploy.
