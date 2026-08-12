# Student contribution credits — requirements

Status: Phase 0 input for `tla-plus-implementation`  
Date: 2026-08-12

Canonical requirement aliases: `REQ-CR-001` through `REQ-CR-020`, `REQ-SAFE-001` through `REQ-SAFE-008`, and `REQ-LIVE-001` through `REQ-LIVE-004` refer to the stable IDs below.

## Product outcome

Reward authenticated students for useful learning evidence that has already been durably accepted by Stoody's existing upload pipelines. Canonical Stoody BLE writing earns more than notebook photographs. The reward system is an asynchronous observer: it must never block, rewrite, delete, accept, reject, or otherwise change the source upload or its academic evaluation.

## Functional requirements

- `CR-001` A student can earn credits from canonical Stoody BLE strokes uploaded through the existing mobile page-sync path.
- `CR-002` Only canonical pen strokes (`processingVersion = ble-canonical-v1`, source mode `live` or `offlineReplay`) are eligible. Touch strokes and client-reported aggregate counts are not credit authority.
- `CR-003` Stroke quantity is computed server-side from validated geometry, including unique stroke count, point count, physical path length, duration, coverage, repetition, and shape variation.
- `CR-004` Stroke quality is judged by deterministic geometry checks followed by a semantic handwriting/scribble check through the shared ExamPen LLM gate. Random lines, repeated traces, empty pages, and obvious scribbles must not earn credits.
- `CR-005` A student can earn lower credits for authenticated student answer-copy/notebook images submitted from the mobile app through the existing private answer-copy ingest path.
- `CR-006` Image quality is computed server-side from resolution, blur, exposure, contrast, page coverage, skew/perspective, glare/clipping, and ink/text density, followed by a semantic written-page/legibility check through the shared LLM gate.
- `CR-007` Per-tenant administrators can adjust enabled state, acceptance thresholds, quantity conversion, per-submission caps, daily caps, and the earning-start cutoff. Every saved policy has an immutable version identifier. Concurrent awards for one student are serialized for the UTC earning day so the configured daily cap cannot be exceeded.
- `CR-008` A quality result above all configured acceptance gates is credit-accepted; a result below a gate is credit-rejected with machine-readable reason codes. Credit rejection does not reject the source upload.
- `CR-009` A transient worker, storage, AI-gate, budget, or network failure remains retryable/failed and is not reported as a quality rejection.
- `CR-010` Every source version has at most one terminal judgment and at most one positive ledger delta, despite duplicate dispatch, response loss, concurrent workers, or restart.
- `CR-011` A cumulative canvas page can earn only the positive difference between its newly computed target and earlier awards for that same page. Re-uploading unchanged strokes cannot re-award them.
- `CR-012` Credit identity is the durable tenant-scoped student record identity, with business student ID and username retained as lookup/display attributes; it is not a client-supplied name alone.
- `CR-013` The durable append-only ledger is credit truth. Balance, tier, statistics, and leaderboard are projections derived from ledger/judgment records.
- `CR-014` The mobile student profile shows a compact credits card below identity and before settings. Tapping it opens credits total, tier progress, accepted/rejected/pending statistics, source split, recent judgments, and leaderboard.
- `CR-015` The tutor mobile dashboard shows a compact contribution leaderboard entry and can open the full leaderboard.
- `CR-016` The student web dashboard shows total/tier/progress and opens the full credits view. The tutor web dashboard shows the scoped contribution leaderboard and opens the same detail view.
- `CR-017` Students see tenant peers using privacy-reduced labels (first name plus last initial) and their own full label. Tutors see only students in their existing authorized scope and may see full names.
- `CR-018` Tiers and icons use an original Stoody learning-constellation system, not copied Khan Academy artwork or naming: Seed, Scribe, Pathfinder, Beacon, and Luminary.
- `CR-019` The current release does not redeem or debit credits. Ledger schema must permit a separately authorized future redemption design without changing earned-award history.
- `CR-020` Existing historical uploads are not automatically backfilled. Only sources completed on or after the tenant policy's `earning_started_at` are eligible.

## Safety requirements

- `SAFE-001` No credit can exist before the corresponding primary source upload is durable and successful.
- `SAFE-002` Duplicate jobs, concurrent claims, crashes, and retries cannot double-award a source version.
- `SAFE-003` A student cannot receive another student's credits and no tenant can read or write another tenant's credit data.
- `SAFE-004` Credits processing cannot mutate canonical canvas, ingest, PCR, publication, or practice records.
- `SAFE-005` Rejected quality yields zero ledger delta; infrastructure failure is never converted into rejection.
- `SAFE-006` Policy changes do not silently rewrite completed judgments or balances.
- `SAFE-007` Leaderboards are tenant-scoped and tutor views retain existing assignment/class scoping.
- `SAFE-008` The worker never trusts client-provided stroke counts, quality scores, student identifiers, or award amounts.

## Liveness and recovery requirements

- `LIVE-001` If the API process crashes after the primary write but before job creation, a reconciler eventually creates the missing job.
- `LIVE-002` A claimed job whose worker dies becomes reclaimable after its lease expires.
- `LIVE-003` Transient failures retry with bounded attempts and explicit next-attempt time; exhaustion is visible as failed, not silently lost.
- `LIVE-004` A successfully committed terminal judgment eventually yields a completed job and appears in summary/leaderboard projections.

## Compatibility and non-goals

- Preserve the current mobile stroke outbox, backend canvas merge contract, student answer-copy ingest, private S3 evidence, PCR queueing, publication, and practice persistence.
- Preserve existing endpoint status meanings and authentication headers.
- Do not reward tutor camera-fallback uploads because the uploader supplies another student's identifier.
- Do not reward touch drawing or web-canvas uploads in this release.
- Do not implement goodies, redemption, transfers, purchases, or negative balances.
- Do not reject, hide, or delete source learning evidence because it was credit-ineligible.

## Acceptance evidence

- TLC checks the bounded asynchronous workflow and listed safety/liveness properties.
- Backend tests cover metrics, policy validation, tenant/student isolation, job recovery, lease reclaim, duplicate dispatch, atomic award, and scoped leaderboard.
- Mobile tests/typecheck cover navigation, profile placement, student photo upload request, credits states, and tutor scope projection.
- Frontend tests/build cover student/tutor dashboard placement and response-state rendering.
- Existing focused stroke/ExamPen tests and the cross-repo stroke conformance runner remain green.
- Live AI, S3, Mongo/Redis/Celery, deployment, and device/camera quality remain separately validated.
