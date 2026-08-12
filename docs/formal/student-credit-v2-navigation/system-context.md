# Student credit V2 and mobile account navigation system context

Status: pre-implementation discovery snapshot  
Date: 2026-08-12
Context kind: existing-repository
Context status: confirmed-as-built
Baseline preserved: yes

Evidence: repository paths, symbols and observed pre-implementation behaviors below were inspected from the synchronized nested repositories on 2026-08-12.  
Assumption: production tenant balances, queues and deployment state are outside this local checkout and require a separate authorized operational audit.

## Repository and ownership map

| Concern | Current owner and evidence | V2 change |
|---|---|---|
| Policy, jobs, judgment, ledger, tier placement | `backend/services/student_credits.py`; `backend/api/v1/credits_async.py` | V2 defaults, semantic validation, guarded activation, full tier projection, bounded missing-source recovery |
| Stroke source | `backend/api/v1/strokes_async.py`; canonical `canvas_pages` versions | Observer only; formula changes, source pipeline unchanged |
| Photo source | `backend/api/v1/evalpen_student_submission_async.py`; immutable `submission_id`; `evalpen_answer_pages` | Observer only; 1/page and cap 10; version `1` remains valid for immutable evidence |
| Mobile credits projection | `stoody-multi-pen/mobile-app/src/services/creditsService.ts`; credits screens/components | Remove local tier authority; render backend ladder/placement |
| Web credits projection | `frontend/src/services/creditsService.ts`; `CreditsProjectionCard.tsx` | Preserve backend placement and full ladder contract; no local thresholds |
| Header/account UI | `StoodyHeader.tsx`; `ProfileMenu.tsx` | Replace action dropdown with centered identity-only overlay |
| Mobile navigation | `StudentHamburgerMenu.tsx`; `TeacherHamburgerMenu.tsx`; native stacks/types | Add canonical Credits and Rewards routes; remove profile credits card |

## Current-state facts re-verified

- Backend V1 defaults award 4 credits per 250 mm, cap strokes at 40/page, photos at 2/page and 20/submission, with a daily cap of 200.
- Backend jobs already store `policy_version` and a full `policy_snapshot` at enqueue.
- Judgment and ledger uniqueness are enforced per `(source_type, source_id, source_version)`/`judgment_key`; source and student-day locks guard cumulative and daily-cap calculations.
- Existing `get_credit_policy` creates a policy document on first access. Existing stored policy documents will not acquire changed defaults merely from a code deploy.
- Existing policy request validation checks scalar ranges but not cross-field or tier-ladder semantics.
- Reconciliation currently skips a due job forever when source completion cannot be recovered.
- Photo submissions reject a second upload after canonical immutable evidence exists; the existing `source_version = "1"` therefore identifies one immutable submission.
- Backend summary currently returns resolved tier placement but not the full ladder. Mobile has a hard-coded ladder and fallback tier calculation; this is the concrete authority drift.
- The top-right mobile avatar currently opens `ProfileMenu`, which contains Profile and Sign out actions.
- Student hamburger already contains Profile and Sign out; teacher hamburger includes the canonical More catalogue (including Profile) and Sign out.
- Student Profile currently fetches and renders `CreditsSummaryCard`; tutor dashboard also presents an informational credits projection.
- Student profile stack is registered as a hidden tab and already hosts `StudentCredits`; tutor credits is in `HomeStack`.

## State model

Backend abstract state:

- `activePolicy`: `Legacy` or `V2`.
- `jobs[j]`: absent, pending, processing, retry, completed, or failed, plus captured policy and bounded completion-lookup count.
- `judged[j]`, `ledger[j]`, `award[j]`: immutable/idempotent result projection.
- `dailyAward`: sum of modeled positive entries for the student/day.
- `balance`: append-only sum; activation never changes it.
- `tier`: derived only from `balance` and `activePolicy` ladder.

Mobile abstract state:

- `surface`: none, avatar overlay, or hamburger drawer.
- `route`: home/profile/credits/rewards/other.
- `role`: student or tutor.
- Avatar open and hamburger open are never simultaneously true.
- Rewards navigation changes only route/surface; it cannot change backend balance.

## Important transitions

1. A successful source upload may enqueue an idempotent credit job with the active policy snapshot; enqueue failure is swallowed/repaired later.
2. A worker claims a due job, judges it with its captured policy, then commits one judgment and one non-negative ledger delta under locks and the captured daily cap.
3. Missing source-completion evidence increments a bounded reconciliation counter; at the bound, the job becomes terminal failed with zero award.
4. Admin V2 activation checks semantic validity, that no non-terminal job exists, and that no student's current UTC-day awards already exceed the V2 cap, then atomically changes the current policy. Existing balance and historical records remain untouched.
5. Summary and leaderboard derive tier placement from the current backend ladder. Clients render those values and do not infer thresholds.
6. Avatar tap opens an identity-only centered overlay. Hamburger tap first closes the avatar overlay and opens the drawer. A menu destination closes the drawer before navigation.

## Failure and concurrency boundaries

- Policy activation racing with enqueue: the activation operation and enqueue snapshot/insert share a tenant policy-transition lock. Activation also refuses while open jobs exist or the current UTC day is already above the V2 cap; tests cover the transition contract.
- Duplicate delivery/worker replay: unique indexes and idempotent read-before-insert preserve one judgment/ledger entry.
- Concurrent daily awards: the student-day lock preserves the V2 cap.
- Old and new jobs: guarded activation avoids a queue containing both versions at the transition; historical terminal jobs may retain either version.
- Missing source evidence: bounded reconciliation terminates; a new immutable upload uses a new source identity rather than mutating the failed one.
- Credits endpoint unavailable: clients show `No data`; navigation and source pipelines remain usable.
- UI modal overlap: opening one account surface closes the other; route actions are absent from the avatar overlay.

## Selected constraint-index obligations

- `MP-MOB-001`: backend remains mobile authority.
- `MP-CREDIT-001`: backend is the only writable credit/tier owner.
- `MP-CREDIT-002`: post-upload non-blocking observer.
- `MP-CREDIT-003`: append-only idempotent awards under locks.
- `MP-CREDIT-004`: captured policy snapshots are immutable.
- `MP-CREDIT-005`: eliminate the mobile tier drift surface.
- `MP-SEP-002`: focused mobile tests and Android build are separate real-code evidence.

## Confirmed as-built context

Evidence: repository inspection, executable tests/builds, fresh TLC results and explicit user confirmation recorded in `implementation-evidence.md` and `confirmation-record.md` on 2026-08-12.

The preceding sections remain the preserved pre-implementation discovery baseline. The delivered system now has these confirmed owners and boundaries:

- Policy, award formulas, tier definitions, activation, judgment and ledger ownership remain solely in `services/student_credits.py`; `api/v1/credits_async.py` exposes the admin-only activation boundary and read-only role projections.
- New tenant policy initialization uses V2 economics. Existing stored tenant policies remain unchanged until guarded `POST /api/v1/credits/policy/activate-v2` activation; activation preserves historical ledger entries and rejects open jobs or a current-day balance already above the V2 cap.
- Enqueue and activation serialize through the tenant policy-transition lock. Jobs persist their complete policy snapshot/version, and worker processing continues under existing group, student-day and uniqueness controls.
- Missing source-completion evidence uses a persisted bounded counter and reaches a stable terminal failure without overriding a concurrently claimed worker job. Photograph submissions remain immutable version-1 sources identified by submission ID.
- Summary and leaderboard policy reads no longer initialize durable policy documents. Backend summaries return the policy version and full tier ladder; mobile and web remain projection-only consumers.
- `StoodyHeader` owns mutually exclusive avatar/hamburger surfaces. `ProfileMenu` is an identity-only centered overlay; student and tutor hamburger catalogs own Credits and Rewards navigation. `RewardsPlaceholderScreen` has no credit mutation path.
- Persistence remains in the existing tenant MongoDB policy/job/judgment/ledger/lock collections. No destructive migration or historical balance rewrite is introduced.
- Compatibility limits remain explicit: existing tenant activation is operational, production calibration is separate, and Rewards redemption is future work. The corrected release APK was installed and launched on one USB-connected Android device; authenticated screen behavior was then user-verified as correct, but broader hardware/device coverage remains separate.

## Separate validation boundary

TLC explores the abstract policy/job/ledger/navigation state machine. It does not prove MongoDB atomicity, Celery delivery, React Native layout pixels, API authorization, image-quality correctness, or production data state. Those require focused tests, builds, and deployment/operational evidence.
