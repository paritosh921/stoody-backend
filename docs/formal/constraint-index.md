# Backend constraint catalog for formal refactors

Snapshot: `main` at `7805428efa54f4b4ce06df8f19016be750f0a212` on 2026-08-11
Repository root: `D:\SOFTWARE_Projects_LP\skiller-bot\backend`
Status: evidence catalog; re-verify selected entries before use

## Scope and authority

This catalog covers the FastAPI backend's tenant, auth, persistence, upload, ExamPen, and hub-control boundaries. `AGENTS.md` applies. It records current integration constraints, not desired behavior. Existing behavior that conflicts with a future request must be resolved explicitly rather than silently preserved.

Database/driver atomicity, crypto, filesystem durability, scanners, cloud services, Celery, deployment, and performance are not proved by TLC; model their observable outcomes and validate the mechanisms separately.

## Constraint index

### Tenant and data ownership

| ID | Constraint | Evidence anchors | Validation anchor / model cue |
|---|---|---|---|
| `BE-TEN-001` | Tenant-scoped operations require a request/task-local `TenantContext` carrying `admin_id` and normally `db_name`; absence must fail rather than fall back to another tenant database. | `core/tenant.py` `TenantContext.require`; `core/database.py`; route `_get_tenant_db` helpers; `docs/TENANT_ISOLATION.md` | Put tenant/admin/database identity in `TypeOK`. Safety: no transition reads or writes another tenant. |
| `BE-TEN-002` | Tenant-scoped collections must receive the current `admin_id` constraint or use the resolved tenant database. Explicit cross-tenant/master operations require a named bypass/authority path. | `core/tenant.py` `TenantAwareDB`, scoped/global collection sets, `_TenantBypassContext`; auth dependencies | Model normal and privileged actors separately; forbid accidental bypass actions. |
| `BE-TEN-003` | A new or renamed collection cannot remain an unclassified “unknown collection.” Current `TenantAwareDB` warns and proceeds without a tenant filter for unknown names, so collection registration is a security-critical refactor obligation. | `TenantAwareDB._get_tenant_filter` | For any task adding a collection, classify ownership before formalization and add tenant-isolation tests. |
| `BE-TEN-004` | Tutor visibility, admin ownership, hub assignment, student/pen binding, and tenant identity are distinct checks; satisfying one does not imply the others. | `api/v1/exam_orch_async.py`; `hub_ops_async.py`; `stroke_ingest_async.py`; tutor-scope tests | Model each authorization guard separately and include mismatched combinations. |

### Authentication and sessions

| ID | Constraint | Evidence anchors | Validation anchor / model cue |
|---|---|---|---|
| `BE-AUTH-001` | Protected routes derive identity and authority from verified backend tokens/cookies plus current database state, never client role claims alone. | `core/auth.py` `AuthManager`; route dependencies in `api/v1/auth_async.py`, `auth_cookie.py`, and `totp_2fa.py` | Model credential validity as guarded input; cryptographic verification is separate validation. |
| `BE-AUTH-002` | Admin/tutor password login is a two-stage 2FA state machine where required; no long-lived authenticated session is created before required OTP completion. | `api/v1/totp_2fa.py`; legacy entry points in `auth_async.py` and `auth_cookie.py`; `tests/test_legacy_2fa_enforcement.py` | Model password accepted, setup required, OTP pending, OTP failure/expiry, and session issued. |
| `BE-AUTH-003` | Global session invalidation stores `min_token_issued_at`; verification rejects tokens older than the cutoff. In-memory caching means propagation is bounded rather than instantaneous. | `core/auth.py` `verify_token_and_get_user`; invalidation endpoint in `auth_async.py`; frontend `../frontend/docs/FORCE_LOGOUT_ON_DEPLOY.md` | Model worker cache as stale/fresh and state the propagation bound as an assumption or liveness property. |
| `BE-AUTH-004` | Mobile session duration is selected only for requests identified by `X-App-Source: stoody-mobile`; web/non-mobile session policy and 2FA behavior remain separate. | `core/mobile_auth.py`; auth route call sites; `tests/test_mobile_auth_session_ttl.py` | Model client class explicitly; prevent mobile duration from leaking to untagged clients. |
| `BE-AUTH-005` | Hub tokens are scoped, bound to hub identity, tenant context, assignment/manifest rules, and expiry. Local-access issuance does not grant arbitrary backend authority. | `api/v1/hub_ops_async.py` token creators and `_require_hub_scope`/`_require_hub_id_match`; hub auth tests | Model scope membership and token expiry; validate signatures/secrets separately. |

### Persistence, concurrency, and idempotency

| ID | Constraint | Evidence anchors | Validation anchor / model cue |
|---|---|---|---|
| `BE-DB-001` | Every state transition must name its MongoDB atomic boundary. Multi-document workflows are not assumed atomic merely because they execute in one coroutine. | Route/service update sequences; `core/database.py` | Model interleaving between reads and writes and crashes between durable effects. |
| `BE-DB-002` | Idempotency identities must be stable and enforced with a unique index or conditional update where concurrent duplicates are possible. | Exam `session_request_key` unique sparse index; ingest `dedup_hash` index; duplicate-key handling | Model two equal concurrent requests and response loss followed by retry. |
| `BE-DB-003` | Retryable background/external work records explicit pending/success/failure state and does not equate request acceptance with completion. | Exam processing jobs; hub commands/acks; OCR/task services | Split enqueue/accept, execution, durable result, and acknowledgement labels. |
| `BE-RISK-001` | Current exam lifecycle transition reads the state and later updates by `exam_id`; the update predicate does not include the previously read lifecycle state. Concurrent transitions are therefore a material race to examine in any lifecycle refactor. | `api/v1/exam_orch_async.py` `transition_lifecycle` | Mark `UNKNOWN` until task discovery decides the desired concurrency contract; model competing transitions and implement CAS/transaction only after a checked design. |
| `BE-RISK-002` | Recheck score resolution spans publication-snapshot amendment and recheck-request finalization in separate MongoDB operations. A crash or conditional-update loss between them is not covered by a cross-document transaction. | `api/v1/evalpen_recheck_async.py` resolve path; `api/v1/evalpen_review_async.py` published amendment path | Mark the desired recovery contract `UNKNOWN` for any recheck refactor; model failure after amendment and before request finalization. |
| `BE-RISK-003` | `data/private_uploads` currently contains git-tracked clean, quarantine, and rejected upload artifacts. That repository state conflicts with the intended private runtime-storage boundary and must never be treated as a fixture or acceptable persistence mechanism. | `git ls-files data/private_uploads`; `docs/UPLOAD_SECURITY.md`; `core/upload_security/storage.py` | Security remediation and history/retention decisions require a separate authorized task. Future refactors must preserve privacy regardless of current tracked artifacts. |

### Exam lifecycle and ingest

| ID | Constraint | Evidence anchors | Validation anchor / model cue |
|---|---|---|---|
| `BE-EXAM-001` | Current lifecycle order is `draft -> armed -> in_progress -> collection_closed -> uploading -> ready_for_eval`; skipping and reverse transitions are rejected. | `exam_orch_async.py` `LIFECYCLE_STATES`, `LIFECYCLE_TRANSITIONS`, `transition_lifecycle` | Model every state and invalid action. Treat terminal quiescence explicitly if deadlock checking is enabled. |
| `BE-EXAM-002` | Arming requires preflight success; entering `ready_for_eval` requires upload/processing blockers to be absent. | `_build_preflight`, `_ready_for_eval_issues`, `transition_lifecycle`; focused exam tests | Model preflight and processing outcomes nondeterministically; safety: guards cannot be bypassed. |
| `BE-EXAM-003` | Finalized exam documents are hard-locked against question/metadata mutation, and finalization is the sole authority for syncing ExamPen question/answer metadata. | `api/v1/pdf_async.py`; `questions_async.py`; backend `AGENTS.md`; finalization tests | Model draft/finalized and one sync effect. Include duplicate finalize and mutation-after-finalize. |
| `BE-INGEST-001` | Stroke chunks are accepted only for authorized tenant/hub actors, a valid canonical exam type, a matching body exam type, a valid lifecycle window, and a matching pen/student binding. | `stroke_ingest_async.upload_stroke_chunk` and `_resolve_exam_context_for_ingest` | Model every guard independently, including mismatched combinations. |
| `BE-INGEST-002` | Chunk payload hash is recomputed server-side. Dedup identity combines exam, normalized pen MAC, chunk index, and payload hash; a duplicate race returns the existing semantic acknowledgement. | `_compute_payload_hash`, `_compute_dedup_hash`, unique index, duplicate-key handler | Safety: same identity produces at most one logical artifact. Liveness: retry can learn acceptance. |
| `BE-INGEST-003` | Per-pen finalization validates completeness/checksum before canonical ingest and must be retry-safe. Chunk acceptance is not finalization. | `stroke_ingest_async.finalize_pen_upload`; ingest repository/service; edge upload worker tests | Model missing/duplicate chunks, lost response, checksum mismatch, and repeated complete. |
| `BE-EXAM-004` | DCR and PCR share ingest/orchestration but retain distinct evaluation ownership. A refactor may not merge their internal state or write through the wrong engine merely because both consume the same artifact. | `exam-conductor/new-docs/governance/STATE_OWNERSHIP_MAP.md`; `exam-conductor/dcr`; `exam-conductor/pcr` | Model shared routing plus distinct engine actions; preserve state ownership map. |

### PCR grading, publication, analytics, and recheck

| ID | Constraint | Evidence anchors | Validation anchor / model cue |
|---|---|---|---|
| `BE-PCR-001` | PCR grading is pinned to a frozen grading contract and paper/evidence version. A contract migration is audited, uses optimistic predicates, and must not race an active queued/processing owner or silently reinterpret completed evidence. | `services/pcr_grading_contract_migration.py`; `exam-conductor/pcr/services/full_document_grading.py`; PCR architecture specs and focused tests | Model old/new contract, active lease/job, migration claim, supersession, requeue, crash, retry, and concurrent migration. |
| `BE-PCR-002` | The visual evidence graph fixes page/region evidence ownership for each question. Missing or ambiguous ownership remains explicit and can require review; downstream grading must not silently borrow evidence from another question. | `exam-conductor/pcr/services/visual_evidence_graph.py`; `objective_answer_sheet.py`; `full_document_grading.py`; evidence-graph tests | Model evidence present, missing, ambiguous, stale-version, and cross-question candidates. Validate image geometry separately. |
| `BE-PCR-003` | Objective marks are assigned deterministically from the frozen answer key and scoring policy. An LLM may transcribe through the approved gate caller but may not assign objective marks, guess an unrecognizable non-empty response, or mutate the frozen key. | `services/objective_scoring_service.py`; `exam-conductor/pcr/services/objective_answer_sheet.py`; `exam-conductor/new-docs/architecture/LLM_GATE_SPEC.md`; objective-scoring tests | Model attempted/not-attempted/unrecognizable/correct/incorrect and penalty rules; validate OCR and prompt behavior separately. |
| `BE-PUB-001` | Student-facing scores and analytics derive from an integrity-validated publication snapshot, not mutable OCR or evaluation working records. An explicit audited amendment replaces the published snapshot using an optimistic snapshot predicate. | `services/exampen_submission_readiness.py`; `services/exam_analytics.py`; `api/v1/evalpen_review_async.py`; analytics/publication tests | Model publish, stale amendment, successful amendment, invalid snapshot, and concurrent reader. |
| `BE-RECHECK-001` | At most one active recheck exists per submission/question through sparse unique `active_key`. Claim is conditional and actor-scoped; resolution requires the claimant and a resolution lock, preserves maximum score, records audit/conversation state, and removes the active key only at terminal resolution. | `api/v1/evalpen_recheck_async.py`; recheck indexes; recheck/student-BFF tests | Model duplicate create, competing claims, duplicate resolve, stale lock, score/no-score terminal states, and lost responses. Include `BE-RISK-002`. |
| `BE-CONTENT-001` | Institution content-category IDs are stable, normalized identifiers. Existing IDs cannot be removed or changed; retirement uses archive state, names/IDs remain unique, and selection is tenant-owned. | `core/content_categories.py`; settings/content routes; `tests/test_content_categories.py` | Model create, rename display label, archive, existing reference, duplicate ID/name, and cross-tenant selection. |

### Hub control and manifests

| ID | Constraint | Evidence anchors | Validation anchor / model cue |
|---|---|---|---|
| `BE-HUB-001` | Provisioning, registration, heartbeat, assignment, tutor manifest, command polling/ack, data upload, and local-access issuance are different authorities and transitions. | `api/v1/hub_ops_async.py` route handlers and scope helpers | Model hub lifecycle and command lifecycle separately; do not use one “authenticated” boolean. |
| `BE-HUB-002` | Hub device credentials are stored/compared as hashes, can be rotated/revoked, and manifest refresh has explicit requested/ack/status/error projection. | `hub_ops_async.py` credential/manifest helpers and handlers | Model old/new/revoked credentials and hub pull races; validate hash/signature functions separately. |
| `BE-HUB-003` | Command creation is not command completion. Polling and acknowledgement must preserve command identity and tolerate retry/duplicate delivery. | `_create_hub_command`, `poll_hub_commands`, `ack_hub_command` | Model at-least-once delivery and lost ack; require idempotent command effects where applicable. |

### Upload security

| ID | Constraint | Evidence anchors | Validation anchor / model cue |
|---|---|---|---|
| `BE-UPLOAD-001` | Every user-controlled upload route declares a policy and route mapping. Request-size enforcement is an outer bound, not a substitute for file policy. | `core/upload_security/policies.py`, `routes.py`, `coverage.py`; `middleware/request_size_limit.py`; `docs/UPLOAD_SECURITY.md` | Treat policy selection as a guard; run route-coverage tests separately. |
| `BE-UPLOAD-002` | Binary upload order is bounded read, type/signature validation, quarantine, malware scan, parser guards, private release, then verdict persistence. Production scanner failure is fail-closed when configured. | `core/upload_security/service.py`, `validation.py`, `scanner.py`, `storage.py`, `verdicts.py` | Model scan clean/infected/unavailable and crash boundaries. Filesystem/scanner behavior is separate validation. |
| `BE-UPLOAD-003` | Quarantine, clean, and rejected data remain private from nginx/static serving; cleanup retention is operational state with a deploy gate and metrics. | `docs/UPLOAD_SECURITY.md`; cleanup and deploy-validation scripts | Validate permissions, mounts, services, health, and metrics on deployment. |
| `BE-UPLOAD-004` | A clean authoring PDF is promoted to immutable content-addressed private S3 only after SHA-256 verification. The verdict is updated to the durable URI before local staging deletion; deletion failure is a cleanup warning, not a rollback of durable success. | `core/upload_security/durable_authoring.py`; `tests/test_durable_authoring_storage.py` | Model hash mismatch, S3 failure, verdict-update failure, crash before/after verdict update, and cleanup failure. Validate S3 privacy and lifecycle separately. |

### Separate-validation obligations

| ID | Obligation | Concrete validation |
|---|---|---|
| `BE-SEP-001` | Python behavior | Run focused `pytest` suites derived from selected guards/actions, then broader regressions proportional to scope. |
| `BE-SEP-002` | Syntax/static integration | Run `python -m compileall` or targeted `py_compile`; run configured lint/type checks where applicable. |
| `BE-SEP-003` | MongoDB indexes and atomicity | Inspect effective indexes and exercise concurrent duplicate/CAS cases against a representative database or faithful test double. |
| `BE-SEP-004` | Upload filesystem/scanner | Run upload security tests and `scripts/validate_upload_security_deploy.py` in the target environment. |
| `BE-SEP-005` | Deployment and external services | Verify CI/CD, health, MongoDB/Redis/Celery/S3/AI dependencies, and runtime configuration separately. Do not start local servers unless requested. |

## Known discovery hazards

- `AGENTS.md` contains architecture notes as well as current constraints; verify code and tests before treating prose as current.
- Route-local `_get_tenant_db` helpers are numerous. Trace the selected endpoint end to end instead of assuming one helper implementation covers all routes.
- Motor/Mongo operations in a coroutine are not a transaction by default.
- Many external tasks are accepted asynchronously. A `2xx` often means accepted/enqueued, not completed.
- Material lifecycle, ownership, or collection-name uncertainty is `UNKNOWN`, never silently out of scope.
- The ExamPen governance/spec hierarchy is authoritative for architecture boundaries; implementation drift is a defect to resolve, not a reason to weaken those boundaries.

## Suggested focused evidence searches

```powershell
rg -n "TenantContext|TenantAwareDB|_get_tenant_db" core api tests
rg -n "LIFECYCLE_TRANSITIONS|transition_lifecycle|session_request_key" api tests
rg -n "dedup_hash|finalize_pen_upload|payload_hash" api exam-conductor tests
rg -n "secure_upload|UploadFile|UPLOAD_POLICY" api core middleware tests
```
