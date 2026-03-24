# Chapter 05: Authentication and Authorization

## Status
- **Phase:** W6 — Documentation
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6.A6.1)
- **Build status:** DRAFT

## Overview

ExamPen does not issue its own user session tokens. Stoody is the identity provider and JWT issuer. `svc-auth` validates Stoody-issued JWTs via JWKS, normalizes claims, maps Stoody roles to ExamPen-specific roles, and enforces tenant isolation via PostgreSQL Row-Level Security (RLS). Parent-child scoping ensures parents see only their linked children's data.

## Architecture Context

```
+----------+     JWT     +-----------+     Normalized     +------------------+
|  Stoody  |------------>| svc-auth  |     Claims         | All ExamPen      |
| (Issuer) |   Bearer    | (Validate |-------------------->| Services         |
|          |   Token     |  + Map)   |                    | (Read claims     |
+----+-----+             +-----+-----+                    |  for RBAC + RLS) |
     |                         |                           +------------------+
     | JWKS                    | Stoody Profile API
     v                         v
 /.well-known/           /api/users/{id}
 jwks.json               /api/parents/{id}/children
```

## Stoody JWT -> ExamPen Role Mapping

### Authentication Flow

1. User logs into Stoody. Stoody issues a JWT with standard claims (`sub`, `iss`, `aud`, `exp`, `iat`, `jti`) plus Stoody-specific claims (`role`, `tenant_id`).
2. Stoody frontend includes JWT as `Authorization: Bearer <token>` on all ExamPen BFF API calls.
3. `svc-auth` receives the token and:
   a. Fetches Stoody's JWKS from `/.well-known/jwks.json` (cached, refreshed on `kid` mismatch).
   b. Validates signature, expiry, and issuer.
   c. Checks ExamPen-side revocation state.
   d. Enriches claims with Stoody profile data via `GET /api/users/{user_id}`.
   e. Maps Stoody role to ExamPen roles.
   f. Returns `NormalizedClaims` to the calling service.

### JWKS Cache Behavior

| Scenario | Behavior |
|---|---|
| Normal operation | Use cached JWKS keyset (TTL configurable, default 24h) |
| `kid` mismatch | Fetch new keyset from Stoody. If fetch succeeds, update cache. |
| JWKS fetch fails + cache valid | Use cached keyset, log warning |
| JWKS fetch fails + cache expired | Reject all tokens with HTTP 503 |
| Stoody key rotation | Multiple concurrent keysets supported in cache |

### Role Mapping Table

Stoody provides a base role. `svc-auth` maps it to one or more ExamPen roles:

| Stoody Role | ExamPen Roles | Notes |
|---|---|---|
| `super_admin` | `super_admin` | Full system access |
| `principal` | `principal` | Institute-wide exam access |
| `hod` | `hod` | Department-scoped access |
| `tutor` | `tutor`, `invigilator`*, `evaluator`*, `reviewer`* | *Assigned per-exam by exam creator |
| `student` | `student` | Own scores and objections only |
| `parent` | `parent` | Linked children's scores only |
| (unknown) | `no_exampen_access` | New Stoody roles default to no access |

Exam-specific roles (`invigilator`, `evaluator`, `reviewer`) are assigned per-exam through `svc-exam-orch`. A tutor may be an evaluator for one exam and an invigilator for another.

### Normalized Claims Schema

From `api/auth.openapi.yaml` — `NormalizedClaims`:

```
{
  user_id:          string        # Stoody user ID
  tenant_id:        string        # Multi-tenant isolation key
  stoody_role:      enum          # Original Stoody role
  exampen_roles:    string[]      # Mapped ExamPen roles
  token_source:     "stoody_jwt"  # Always this value
  token_status:     enum          # "valid" | "revoked"
  subject_ids:      string[]      # Subjects the user teaches (tutors)
  class_ids:        string[]      # Classes the user is associated with
  child_student_ids: string[]     # Children (parents only)
  profile:          Profile       # display_name, email, phone, institute_name
}
```

## RBAC Matrix: 7 Roles x All Actions

| Action | Super Admin | Principal | HOD | Tutor (Evaluator) | Invigilator | Student | Parent |
|---|---|---|---|---|---|---|---|
| Create exam | Y | Y | Y | Y (own subjects) | - | - | - |
| Define rubric | Y | Y | Y | Y (own exams) | - | - | - |
| Assign invigilators | Y | Y | Y | - | - | - | - |
| Start/stop exam (hub) | - | - | - | - | Y (assigned) | - | - |
| View all scores | Y | Y | Y (own dept) | Y (own exams) | - | - | - |
| Edit scores | - | - | Y | Y (assigned evaluator) | - | - | - |
| Finalize scores | Y | Y | Y | Y (own exams) | - | - | - |
| Publish scores | Y | Y | Y | Y (own exams) | - | - | - |
| Review objections | - | - | Y | Y (assigned) | - | - | - |
| Escalate objection | Y | Y | Y | - | - | - | - |
| View own scores | - | - | - | - | - | Y | Y (child) |
| File objection | - | - | - | - | - | Y | - |
| Chat (tutor side) | - | - | - | Y (own students) | - | - | - |
| Chat (student side) | - | - | - | - | - | Y | - |
| View leaderboard | Y | Y | Y | Y | - | Y (own pos) | Y (child) |
| Export data | Y | Y | Y (own dept) | Y (own exams) | - | - | - |
| Plagiarism review | Y | Y | Y | Y (own exams) | - | - | - |
| Register pens | - | - | - | - | Y | - | - |
| Upload copy images | - | - | - | - | Y | - | - |
| Monitor sync | Y (via web) | - | - | - | Y (BLE) | - | - |

## RLS Tenant Isolation

### Mechanism

1. Every PostgreSQL table includes a `tenant_id` column.
2. Application middleware sets `SET app.current_tenant = '{tenant_id}'` per request (extracted from normalized JWT claims).
3. RLS policies enforce: `tenant_id = current_setting('app.current_tenant')`.
4. Cross-tenant query returns empty result set, not an error (mitigation A8.1).

### Enforcement Rules

- Every new migration MUST include RLS policy or an explicit exemption comment.
- CI check validates RLS presence on all new tables.
- Integration test I-AUTH-03: tenant A cannot read tenant B data.
- BFF services have zero write access to any database (ownership violation rule).

### RLS Policy Example

```sql
ALTER TABLE scores ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON scores
  USING (tenant_id = current_setting('app.current_tenant')::uuid);

CREATE POLICY tenant_insert ON scores
  FOR INSERT
  WITH CHECK (tenant_id = current_setting('app.current_tenant')::uuid);
```

## Parent-Child Scoping

Parents see only their linked children's data. The scope chain:

1. Parent authenticates with Stoody JWT (role = `parent`).
2. `svc-auth` calls Stoody: `GET /api/parents/{user_id}/children` -> returns `[student_id_1, student_id_2, ...]`.
3. `child_student_ids` populated in `NormalizedClaims`.
4. BFF services filter all queries: `WHERE student_id IN (child_student_ids)`.
5. Integration test I-BFF-S03: parent JWT -> child score view allowed only for linked children.

## Revocation

`svc-auth` manages ExamPen-side token revocation:

- `POST /api/v1/auth/revocations` — revoke a token JTI with mandatory reason.
- `GET /api/v1/auth/revocations/{jti}` — check revocation status.
- Revocation state stored in PostgreSQL, checked during introspection.
- Stoody remains the source of truth for primary identity. ExamPen revocation is supplementary.

## Hub Authentication

Hub invigilator authentication uses a separate mechanism:

1. Backend `svc-auth` generates rotating 24-hour auth codes.
2. Codes pushed to hub during provisioning and daily sync.
3. Hub caches codes in `invig_codes` SQLite table with `valid_from` / `valid_until`.
4. Invigilator writes 12-byte code to BLE Auth characteristic (`6f5f2001-...`).
5. Hub validates against cached codes. Expired codes rejected even from cache.
6. Multiple failed auth attempts -> 5-minute lockout (mitigation S3).

## Interfaces

- **API contract:** `api/auth.openapi.yaml`
  - `POST /api/v1/auth/introspect` — validate Stoody JWT, return normalized claims
  - `GET /api/v1/auth/me` — current bearer's normalized claims
  - `POST /api/v1/auth/revocations` — revoke a token
  - `GET /api/v1/auth/revocations/{jti}` — check revocation

- **Security scheme:** `StoodyBearer` — HTTP Bearer with JWT format

- **Stoody endpoints consumed:**
  - `GET /.well-known/jwks.json` — signing keys
  - `GET /api/users/{user_id}` — profile enrichment
  - `GET /api/parents/{user_id}/children` — parent scope resolution

## Configuration

| Variable | Description |
|---|---|
| `STOODY_JWKS_URL` | Stoody JWKS endpoint URL |
| `STOODY_API_URL` | Stoody REST API base URL |
| `JWKS_CACHE_TTL_HOURS` | JWKS cache TTL (default: 24) |
| `DATABASE_URL` | PostgreSQL connection string (svc-auth schema) |
| `REVOCATION_CHECK_ENABLED` | Enable/disable revocation checks (default: true) |

## Testing

- **Unit:** U-AUTH-01 (JWT validation + claim normalization), U-AUTH-02 (RBAC role hierarchy), U-AUTH-03 (tenant isolation RLS), U-AUTH-04 (JWKS cache expiry + refresh), U-AUTH-05 (parent-child scope resolution)
- **Integration:** I-AUTH-01 (Stoody JWT introspection via REST), I-AUTH-02 (JWKS fetch + token validation with mock Stoody), I-AUTH-03 (multi-tenant RLS: tenant A cannot read tenant B)
- **BFF RBAC:** I-BFF-T02 (student JWT -> 403 on teacher endpoints), I-BFF-S03 (parent JWT -> child scope enforcement)

## Failure Modes & Mitigations

| ID | Failure | Mitigation |
|---|---|---|
| A8.1 | Multi-tenant data leak | RLS on every table, CI check for new migrations |
| A8.2 | DPDPA violation (children's data) | Data minimization, parent consent via Stoody, auto-delete after retention period |
| S3 | BLE MITM on invigilator channel | Rotating auth codes, BLE 4.2 LESC, app-level challenge-response planned for V2 |

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Initial draft: auth flow, role mapping, RBAC matrix, RLS, parent scoping, hub auth | Claude Agent (W6.A6.1) |
