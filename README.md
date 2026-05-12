# Stoody Backend

FastAPI backend for the Stoody learning platform. It owns tenant-aware APIs, authentication, content ingestion, canvas persistence, test/evaluator data, admin/tutor/student workflows, file storage, and platform integration endpoints.

This root README is the backend entry point. Detailed runbooks and architecture notes live in [docs/](docs/).

## Runtime

- Python 3.11 is the recommended local runtime.
- Main entry point: `main_async.py`
- API base path: `/api/v1`
- Default local port: `5001`
- Primary database: MongoDB Atlas or a configured MongoDB instance
- Optional services: Redis, S3-compatible storage, OAuth providers, OCR/AI providers

## Quick Start

```powershell
cd backend
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main_async.py
```

Health and API checks:

```powershell
curl http://localhost:5001/health
curl http://localhost:5001/api/v1/health
```

## Configuration

Create `backend/.env` for local development. The exact production values are deployment-specific, but the common categories are:

```env
HOST=0.0.0.0
PORT=5001
MONGODB_URI=mongodb+srv://...
MONGODB_DB_NAME=skillbot_db
JWT_SECRET_KEY=...
FRONTEND_URL=http://localhost:8080
GOOGLE_CLIENT_ID=...
OPENAI_API_KEY=...
MISTRAL_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...
S3_BUCKET_NAME=...
```

## Current Architecture

- `main_async.py` boots the FastAPI application and registers versioned routers.
- `api/v1/` contains the active route modules.
- `services/` contains business logic and integrations.
- `models/` contains data contracts and persistence models.
- `core/` contains shared configuration, auth, security, tenant, and database helpers.
- `scripts/` contains operational utilities. See [scripts/README.md](scripts/README.md).
- `exam-conductor/` contains the ExamPen/DCR/PCR architecture and implementation docs.

## Auth, Tenant, And Storage Rules

- JWT-bearing requests are tenant-aware where tenant isolation is required.
- Tenant identity must be resolved through the established backend auth and tenant helpers, not from ad hoc request fields.
- Canvas and document persistence use the backend storage abstractions rather than direct file writes in API handlers.
- S3 object storage is documented in [docs/S3_STORAGE_MIGRATION.md](docs/S3_STORAGE_MIGRATION.md).
- Tenant isolation is documented in [docs/TENANT_ISOLATION.md](docs/TENANT_ISOLATION.md).

## Documentation Index

- [docs/README.md](docs/README.md) - curated backend documentation index
- [docs/QUICK_START.md](docs/QUICK_START.md) - local startup path
- [docs/BACKEND_SETUP.md](docs/BACKEND_SETUP.md) - environment and setup details
- [docs/TENANT_ISOLATION.md](docs/TENANT_ISOLATION.md) - tenant isolation rules
- [docs/DATABASE_manage_STRICT_UNIFIED.md](docs/DATABASE_manage_STRICT_UNIFIED.md) - strict database/auth/tenant operational runbook
- [docs/S3_STORAGE_MIGRATION.md](docs/S3_STORAGE_MIGRATION.md) - object storage migration notes
- [docs/B2C_USER_SUPPORT.md](docs/B2C_USER_SUPPORT.md) - B2C user support workflows
- [docs/CURRENT_BACKEND_NOTES.md](docs/CURRENT_BACKEND_NOTES.md) - consolidation notes and removed-doc rationale

## Verification

```powershell
cd backend
python -m compileall .
python -m pytest
```

Run narrower tests when working on a focused area. Keep implementation claims tied to current source files and current tests, not historical plan markdown.
