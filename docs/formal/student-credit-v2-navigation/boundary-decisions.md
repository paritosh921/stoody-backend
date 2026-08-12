# Student credit V2 and mobile account navigation boundary decisions

Status: approved-by-request assumptions for formalization  
Date: 2026-08-12
Material unresolved unknowns: none
Conflict status: resolved
Post-implementation review: complete

Boundary classifications: policy/award/concurrency/navigation transitions are `MODELED`; repository ownership and upload isolation are `PRESERVED`; production deployment, calibration and physical-device behavior are `SEPARATELY_VALIDATED`; production tenant state is `ASSUMED` absent until audited; redemption internals are `IRRELEVANT` to this placeholder scope.

## Decision table

| ID | Decision | Rationale and consequence |
|---|---|---|
| `BD-V2-001` | V2 is a named backend preset and the default for newly created tenant policies. Existing tenants use an explicit idempotent admin activation operation. | Changing Python defaults alone cannot update stored policies; hidden read-side migration is unsafe. |
| `BD-V2-002` | Activation and enqueue share a tenant policy-transition lock. Activation refuses with conflict while any pending, processing, or retry credit job exists or any student is already above 100 awarded credits in the current UTC day. No force flag is provided. | Prevents a policy race and avoids making the new daily-cap invariant false immediately at activation. Operations drain/recover the queue or wait for the UTC day boundary, then retry. |
| `BD-V2-003` | Historical balances are preserved exactly; tier is re-derived from V2 thresholds. Visual demotion is accepted and no compensating credit is issued. | The user explicitly wants stricter tiers; ledger rewriting would violate append-only ownership and auditability. |
| `BD-V2-004` | The 250 mm formula uses ceiling, matching current behavior, but reduces the multiplier to one and cap to five. | Any non-zero accepted contribution interval earns its first unit; hard/semantic quality gates still reject random or insufficient writing. |
| `BD-V2-005` | Photo acceptance semantics remain all-pages-must-pass for a submission. If accepted, award is `min(page_count, 10)`. | V2 changes generosity, not judgment semantics. Partial-page credit would be a separate product change. |
| `BD-V2-006` | Photo source version remains `1` because a canonical submission is immutable. Re-upload uses a new submission ID. Same-evidence re-judgment cannot change or duplicate its ledger award. | Explicitly resolves photograph version/re-judgment without inventing mutable evidence versions. |
| `BD-V2-007` | Missing completion evidence uses a bounded reconciliation lookup counter and then terminal failure with a stable reason. | Eliminates infinite pending jobs while retaining normal worker retries for recoverable judging errors. |
| `BD-V2-008` | Full tier ladder and resolved placement are returned by backend summaries. For backward compatibility with an older deployed summary contract, mobile may render the complete V2 ladder as a display-only fallback, but it never derives the current tier from the total. Backend ladder and placement always replace the fallback when present. | Preserves all five tier affordances during staggered rollout without reintroducing client-side placement authority or falsely marking Seed as the highest tier. |
| `BD-V2-009` | The existing `ProfileMenu` concept becomes a centered identity-only `UserSummaryOverlay`; all header-level actions are removed. | Directly matches the requested picture-in-picture interaction and prevents duplicate action menus. |
| `BD-V2-010` | Hamburger order is Profile, Learning Credits, Rewards, then remaining account/action destinations as appropriate; Rewards is immediately below Credits. | Makes the drawer canonical while preserving existing entitled destinations. Teacher placement is inserted after My Profile before the remaining More catalogue. |
| `BD-V2-011` | One reusable Rewards screen serves student and tutor stacks and contains only explanatory placeholder content. | Avoids duplicated design and preserves the no-redemption safety boundary. |
| `BD-V2-012` | Remove only the student Profile credits card/fetch. Keep the tutor dashboard's informational credit projection. | The request specifically relocates the Profile entry point; the dashboard projection is informational and remains useful. |

## Conflict ledger

- Confirmed V1 `CR-014` required a student Profile credits card. `V2-CR-016` explicitly supersedes it: the card/fetch is removed and hamburger navigation owns the entry.
- Confirmed V1 `CR-019` prohibited redemption. The new Rewards page does not conflict because it is a read-only placeholder with no debit or redemption operation.
- The earlier V1 tier thresholds are superseded by `V2-CR-004`; historical balances remain governed by the append-only V1 safety baseline.
- The former mobile `DEFAULT_TIERS` placement calculator is removed. The remaining complete V2 catalog is display-only compatibility data and never calculates placement from total credits.

## Operational activation contract

1. Deploy backend and clients supporting full backend tier payloads and V2 semantics.
2. Inspect/repair pending, retry, and processing jobs until the open count is zero; verify no student is above 100 positive credits for the current UTC day.
3. Invoke the admin V2 activation operation per tenant.
4. Record the returned policy version and observe award/job metrics.
5. Existing balances stay fixed; only tier projection and future jobs use V2.

No production tenant activation or production-balance inspection is claimed by this local implementation unless separately executed with authorized deployment credentials.

## Confirmed boundary review

- All `MODELED` policy, concurrency, recovery and navigation decisions are represented by the delivered code and tests.
- `PRESERVED` upload isolation, append-only ledger ownership and privacy-scoped leaderboard behavior remain intact.
- `SEPARATELY_VALIDATED` local builds and tests are complete; deployment, production tenant audit, calibration and device runtime remain residual operational work accepted at confirmation.
- No material unchecked deviation remains. The two controller findings—tier input ordering and the missing-completion worker race—were resolved before confirmation.
- Post-confirmation device review found two mobile compatibility defects: a missing backend ladder collapsed the display to Seed/highest, and Profile restored the stack's last Credits route. The client now retains all five display tiers for an older API, never reports highest without a confirmed final tier, and explicitly targets `Profile` versus `StudentCredits`; focused tests, typecheck, release build, USB install and launch passed before push.
