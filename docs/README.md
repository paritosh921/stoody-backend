# Backend Documentation

This folder contains the backend documents that are still useful for current development. Old implementation plans, duplicate script indexes, and stale endpoint snapshots have been removed or folded into the references below.

## Start Here

- [Backend README](../README.md) - current backend overview and entry point.
- [QUICK_START.md](QUICK_START.md) - shortest local startup path.
- [BACKEND_SETUP.md](BACKEND_SETUP.md) - environment setup and configuration details.

## Architecture And Operations

- [TENANT_ISOLATION.md](TENANT_ISOLATION.md) - tenant isolation rules, enforcement points, and backend behavior.
- [DATABASE_manage_STRICT_UNIFIED.md](DATABASE_manage_STRICT_UNIFIED.md) - strict database/auth/tenant runbook consolidated from the older database docs.
- [S3_STORAGE_MIGRATION.md](S3_STORAGE_MIGRATION.md) - object storage migration and runtime storage notes.
- [B2C_USER_SUPPORT.md](B2C_USER_SUPPORT.md) - B2C account support and operations.
- [CURRENT_BACKEND_NOTES.md](CURRENT_BACKEND_NOTES.md) - what was consolidated or removed during the docs cleanup.

## Scripts

The maintained script reference is now [../scripts/README.md](../scripts/README.md). The previous `SCRIPTS_INDEX.md` and `SCRIPTS_SUMMARY.md` files duplicated that index and were removed.

## Quick Commands

```powershell
cd backend
python main_async.py
```

```powershell
cd backend
python -m compileall .
python -m pytest
```
