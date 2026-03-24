# svc-auth

ExamPen authentication and authorization service. Validates Stoody-issued JWTs via JWKS, normalizes claims to the ExamPen role model, enriches profiles from the Stoody user API, and manages ExamPen-side token revocations.

## Ownership Declaration

- **Writes:** Normalized auth claims, ExamPen role mappings, revocation state
- **Reads from:** Stoody JWKS endpoint (JWT signing keys), Stoody user API (profile enrichment), Stoody parent API (child-student scope)
- **Never writes to:** Exam sessions, scores, strokes, objections, chat, analytics, or any other service's database
- **Transactional boundaries:** Revocation insert/delete is single-row atomic in PostgreSQL. Role mapping upsert is single-row atomic.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/introspect` | Validate a Stoody JWT, return normalized claims |
| GET | `/api/v1/auth/me` | Return claims for current bearer token |
| POST | `/api/v1/auth/revocations` | Revoke a token JTI (principal+) |
| GET | `/api/v1/auth/revocations/{jti}` | Check revocation status (principal+) |
| DELETE | `/api/v1/auth/revocations/{jti}` | Un-revoke a token JTI (principal+) |

## Running Locally

```bash
# Requires: PostgreSQL, stoody-mock running on port 9100
cd services/svc-auth
pip install -e ".[dev]"
uvicorn src.main:app --port 8001 --reload
```

## Running Tests

```bash
# Unit tests (domain logic — no I/O, no DB)
pytest tests/test_role_mapper.py tests/test_claims.py -v

# Integration tests (mocked DB and Stoody)
pytest tests/test_routes_introspect.py tests/test_routes_revocation.py -v
```

## Dependencies

- `libs/exampen-common-py` (JWT validation, DB pool, RLS, logging)
- PostgreSQL (revocations + role_mappings tables)
- Stoody platform (JWKS endpoint, user API) — or `stoody-mock` for development

## Depended On By

- All ExamPen backend services (via `/introspect` for JWT validation)
- `svc-teacher-bff`, `svc-student-bff` (for RBAC gating)
- `svc-exam-orch` (for identity resolution)
