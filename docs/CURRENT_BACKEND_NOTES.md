# Current Backend Notes

This file records the backend documentation cleanup decisions so future developers know where the useful information went.

## Canonical Backend Entry Points

- Root overview: [../README.md](../README.md)
- Detailed docs index: [README.md](README.md)
- Active backend application entry point: `backend/main_async.py`
- Active API package: `backend/api/v1/`
- Maintained script index: `backend/scripts/README.md`

## Consolidated Documents

- `DATABASE_manage_STRICT_UNIFIED.md` was moved from the backend root into `backend/docs/` because it is an operational runbook, not a root entry point.
- `DATABASE_manage_STRICT_claude.md` was removed because its relevant operational guidance had already been represented by the unified strict database runbook.
- `SCRIPTS_INDEX.md` and `SCRIPTS_SUMMARY.md` were removed because they duplicated the maintained `backend/scripts/README.md`.
- `MULTI_TENANCY_ARCHITECTURE.md` was removed as a separate stale analysis document. Current tenant behavior should be documented in `TENANT_ISOLATION.md`.

## Removed Historical Plans

- `ENDPOINT_DOCUMENTATION.md` described an older Flask-style API surface and was not safe as a current endpoint authority.
- `smartboard_implementation_plan.md` was a historical migration plan, not a current implementation guide.

## Current Documentation Policy

Keep the backend root limited to `README.md`. Put durable setup, architecture, and operations material in `backend/docs/`. Put script usage in `backend/scripts/README.md`. Delete plan/checklist files once the implementation has landed unless they still contain non-duplicated operational knowledge.
