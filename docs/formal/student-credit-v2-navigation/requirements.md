# Student credit V2 and mobile account navigation requirements

Status: pre-implementation formal-design input  
Date: 2026-08-12

## Stable requirement aliases

| Stable ID | Detailed requirements |
|---|---|
| `REQ-V2-AWARD` | `V2-CR-001` through `V2-CR-004` |
| `REQ-V2-POLICY` | `V2-CR-005` through `V2-CR-011` |
| `REQ-V2-IDENTITY` | `V2-CR-012` and `V2-CR-013` |
| `REQ-V2-NAVIGATION` | `V2-CR-014` through `V2-CR-018` |
| `REQ-V2-SAFETY` | `V2-SAFE-001` through `V2-SAFE-008` |
| `REQ-V2-LIVENESS` | `V2-LIVE-001` through `V2-LIVE-004` |

## Scope

This packet supersedes only the award values, tier thresholds, and mobile entry-point requirements in the confirmed V1 `student-contribution-credits` baseline. V1 ownership, append-only ledger, post-upload observer behavior, privacy, and source-pipeline isolation remain binding.

## Functional requirements

- `V2-CR-001` Pen writing awards use `ceil(eligible_path_length_mm / 250) * 1`, capped at 5 credits per accepted page. Rejected pages award zero.
- `V2-CR-002` Notebook photographs award 1 credit for each accepted page, capped at 10 credits per immutable submission. The existing all-pages-must-pass submission judgment remains unchanged unless separately requested.
- `V2-CR-003` A student can receive at most 100 positive earned credits per UTC ledger day across all sources.
- `V2-CR-004` The backend-owned tier ladder is Seed 0, Scribe 100, Pathfinder 500, Beacon 1,500, Luminary 4,000.
- `V2-CR-005` Existing ledger entries are preserved without clawback or rewriting. Current tier is re-derived from the active ladder, so V2 activation may visibly demote a student's badge while preserving the exact balance.
- `V2-CR-006` Each job retains the complete policy snapshot and version captured at enqueue. Activating V2 must not reinterpret pending, retrying, processing, completed, or failed V1 jobs.
- `V2-CR-007` Existing-tenant V2 activation is an explicit admin operation. It is idempotent and must refuse while any non-terminal credit job exists or any student has already received more than 100 positive credits in the current UTC day. New tenants start with the V2 defaults.
- `V2-CR-008` Policy updates reject semantically invalid combinations: empty or duplicate tier identifiers/names, non-increasing thresholds, first threshold other than zero, coverage/density minima above maxima, per-unit awards above their source caps, or malformed V2 preset data.
- `V2-CR-009` A job whose source completion time cannot be recovered becomes terminal after a bounded number of reconciliation lookups; it must not remain pending forever.
- `V2-CR-010` Photo submissions are immutable award sources identified by submission ID and source version `1`. Re-upload creates a new submission ID. Re-judging the same evidence is idempotent and cannot create a second ledger award.
- `V2-CR-011` Summary responses include the complete backend tier ladder used for placement. Mobile and web must not calculate tier placement from local threshold tables.
- `V2-CR-012` Tapping the top-right avatar opens a centered, modal, picture-in-picture-style identity card containing brief identity information only. It contains no profile, sign-out, credits, rewards, or navigation action.
- `V2-CR-013` The avatar overlay and hamburger drawer are mutually exclusive and dismissible by backdrop or platform back action.
- `V2-CR-014` Student hamburger navigation contains Profile, Learning Credits, Rewards, and Sign out. Rewards appears immediately below Learning Credits. Existing entitled destinations remain available.
- `V2-CR-015` Tutor hamburger navigation contains Learning Credits and Rewards immediately adjacent, with Rewards below Learning Credits. Existing More destinations and Sign out remain available.
- `V2-CR-016` Learning Credits opens the existing role-appropriate credits screen. Student Profile no longer fetches or renders a credits card.
- `V2-CR-017` Rewards opens a native read-only placeholder page consistent with the existing Stoody mobile theme. It clearly states that redemption will be available later and exposes no spend, reserve, transfer, or redemption operation.
- `V2-CR-018` Existing informational tutor dashboard credit projection may remain; the hamburger becomes the canonical mobile navigation entry point.

## Safety requirements

- `V2-SAFE-001` Credit failures never fail, roll back, delay, or rewrite the source stroke/photo upload or assessment pipeline.
- `V2-SAFE-002` The backend is the only writable owner of policy, judgment, ledger, balance, and tier placement.
- `V2-SAFE-003` A source version has at most one judgment and at most one ledger entry; replays and concurrent workers do not duplicate awards.
- `V2-SAFE-004` Daily positive awards never exceed 100 under V2, including concurrent source completions.
- `V2-SAFE-005` A V1-snapshot job always uses V1 values and a V2-snapshot job always uses V2 values.
- `V2-SAFE-006` V2 activation cannot occur while a non-terminal job exists or the current day's positive awards already violate the V2 cap.
- `V2-SAFE-007` The Rewards page performs no credit mutation and cannot reduce a balance.
- `V2-SAFE-008` The avatar identity overlay exposes no privileged account action.

## Liveness and recovery requirements

- `V2-LIVE-001` Under eventual worker availability and recoverable source evidence, every queued job eventually becomes completed or terminal failed.
- `V2-LIVE-002` Under repeated missing completion evidence, a job eventually becomes terminal failed after the configured bounded lookup count.
- `V2-LIVE-003` After old jobs drain, an admin V2 activation attempt can succeed.
- `V2-LIVE-004` Every Credits or Rewards menu selection closes the drawer and reaches the selected screen.

## Non-goals

- Redeeming, reserving, transferring, expiring, purchasing, or manually debiting credits.
- Rewriting historical ledger balances to match V2 formulas.
- Replacing the current deterministic and semantic handwriting/image judge.
- Changing the source upload, PCR, canvas ownership, or BLE stroke pipeline.

## Acceptance summary

The implementation is acceptable only when TLC checks the modeled safety/liveness properties, backend tests cover V2 arithmetic/validation/activation/recovery, mobile tests cover overlay and menu routes/order, web/mobile consume backend tier placement, and affected builds/type checks pass.
