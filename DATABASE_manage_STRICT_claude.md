# DATABASE_manage_STRICT_claude.md

## Backend Database Cleanup & Deployment Plan

> **Status:** Pre-deployment reference document
> **EC2 Current:** Commit `0ead8e01` (old, stable)
> **HEAD:** Commit `2908d41` (9 commits ahead, contains all fixes + feature v2)

---

## Table of Contents

1. [Situation Summary](#1-situation-summary)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [What Is NOT Broken](#3-what-is-not-broken)
4. [Architecture Reference](#4-architecture-reference)
5. [The 9 Commits at HEAD](#5-the-9-commits-at-head)
6. [Phase 1: Fix Health Check](#phase-1-fix-health-check)
7. [Phase 2: Migration Script Spec](#phase-2-migration-script-spec)
8. [Phase 3: Deployment Sequence](#phase-3-deployment-sequence)
9. [Verification Checklist](#verification-checklist)
10. [Rollback Plan](#rollback-plan)
11. [Files Modified Summary](#files-modified-summary)

---

## 1. Situation Summary

EC2 is running an OLD commit (`0ead8e01`) because the 9 newer commits (up to `2908d41`) caused **502 errors** when deployed. The newer commits contain:

- Auth flow corrections (commits 1-5)
- Feature v2 tier system with super-admin controls (commits 6, 8)
- Database fallback removal + cleanup (commit 7)
- `admin_id` orphan migration script (commit 9)

The backend uses a **per-tenant-database architecture** (`skb_{institution_id}`), but a fallback in `core/database.py:_get_context_db()` was silently routing operations to a default database (`skillbot_db_fallback`) when `TenantContext` was not set. This created ghost databases (`CIEL-1001`, `skillbot_db`, `skillbot_db_fallback`, `default`) and caused data to land in the wrong place.

There are **1-3 legacy tenants** that need migration (e.g., CIEL with db `skb_indl-ciel-1001`).

---

## 2. Root Cause Analysis

### Root Cause 1: Health Check Crash (502s)

`main_async.py:1030` calls `mongo_count("questions")` which goes through `_get_context_db()` -> returns `None` (no tenant context during health checks) -> generates noisy warnings every ALB health check cycle. This caused the 502 errors on deployment.

### Root Cause 2: Legacy Data Invisible

`TenantAwareDB` (`core/tenant.py`) filters all **13 tenant-scoped collections** by `admin_id`, but old documents lack this field entirely. Queries return empty results, making existing data invisible.

**The 13 filtered collections:**
`students`, `documents`, `tutors`, `questions`, `question_attempts`, `student_activity_log`, `chat_sessions`, `student_test_attempts`, `assignments`, `meetings`, `notifications`, `class_schedules`, `smartboard_sessions`

### Root Cause 3: Legacy Tenants Missing v2 Fields

`skb_master.tenants` docs for old tenants lack `enabled_features_v2`, `subscription_tier`, `max_students`, etc. Feature middleware may malfunction when encountering these incomplete records.

### Root Cause 4: Fallback Database Creation

The old code writes to `skillbot_db_fallback` when context is not available. MongoDB auto-creates databases on first write. Returning `None` instead of a fallback database prevents this unwanted database creation.

### Root Cause 5: Legacy Tenants Not Linked to Super-Admin

Old tenants lack `assigned_superadmin_id`, breaking super-admin management workflows.

### Key Insight: Login vs. Data Discrepancy

- **Login** uses `_resolve_tenant_for_auth()` which finds tenant in `skb_master`, gets admin from tenant DB, creates JWT with `admin_id` + `db_name`
- **Data queries** use `TenantAwareDB` which adds `{"admin_id": admin_oid}` filter to ALL tenant-scoped collections
- If documents do not have matching `admin_id`, queries return empty
- **This is NOT a code bug -- it is a data migration issue**

---

## 3. What Is NOT Broken

- The code at HEAD is **architecturally correct**
- Login works (tenant resolution, JWT creation, middleware context setting)
- `_get_context_db()` returning `None` is **correct behavior** (stops ghost database creation)
- All `mongo_*` wrappers handle `None` gracefully (return `None`/`[]`/`0`/`False`)
- Feature v2 code is complete and functional
- The hundreds of database calls throughout admin and PDF modules work correctly for authenticated requests (middleware sets context from JWT)

---

## 4. Architecture Reference

### Files That Use `db.mongo_*` Wrappers (via `_get_context_db()`)

| File | Approximate Calls |
|------|-------------------|
| `api/v1/admin_async.py` | ~90 |
| `api/v1/pdf_async.py` | ~80 |
| `api/v1/mcq_async.py` | ~40 |
| `api/v1/practice_async.py` | ~30 |
| `api/v1/tutor_async.py` | ~20 |
| `api/v1/chat_async.py`, `student_async.py`, `settings_async.py`, etc. | Various |

All work correctly when `TenantMiddleware` sets context from JWT (authenticated requests).

### Key Source References

- `core/tenant.py:34-48` -- `TENANT_SCOPED_COLLECTIONS` (definitive list of 13 collections)
- `core/tenant_features.py` -- `LEGACY_DEFAULT_TENANT_FEATURES`, `build_enabled_features_v2()`
- `scripts/admin/fix_admin_id_orphans.py` -- existing per-tenant `admin_id` fixer
- `scripts/migrations/assign_superadmin_owners.py` -- existing super-admin assignment script

---

## 5. The 9 Commits at HEAD

| # | Commit | Files | Description |
|---|--------|-------|-------------|
| 1 | `f34db18` | `auth_async.py`, `superadmin_async.py` | Auth flow corrections |
| 2 | `d02fe21` | `auth_async.py` | Connection fixes |
| 3 | `3770728` | `admin_async.py`, `auth_async.py`, `superadmin_async.py` | UI behavior |
| 4 | `d26e0f4` | `auth_async.py`, `superadmin_async.py` | Behavior fixes |
| 5 | `a90774b` | `auth_async.py`, `superadmin_async.py` | Feature flags |
| 6 | `162569d` | NEW `tenant_features.py` (v1), middleware, auth | Feature enforcement |
| 7 | `d1e0c36` | `database.py` cleanup | Fallback removal, rename `mongo_db`, `ensure_indexes` refactor, removed debug endpoints |
| 8 | `0984580` | `tenant_features.py` v2 (618 lines), auth, superadmin | Feature v2 expansion + super-admin endpoints |
| 9 | `2908d41` | `fix_admin_id_orphans.py` | Migration script for orphaned `admin_id` records |

**Diffstat (13 files changed, 1920 insertions, 420 deletions):**

```
api/v1/admin_async.py                  |  26 +-
api/v1/auth_async.py                   | 794 ++++++++++++++++++++++++---------
api/v1/auth_cookie.py                  |  43 +-
api/v1/superadmin_async.py             | 302 ++++++++++---
api/v1/totp_2fa.py                     |  54 ++-
core/auth.py                           |  40 +-
core/database.py                       |  71 +--
core/tenant_features.py                | 548 +++++++++++++++++++++++
middleware/tenant_middleware.py         |  51 ++-
scripts/admin/fix_admin_id_orphans.py  | 174 ++++++++
scripts/admin/init_admin_direct.py     |  81 ++--
scripts/admin/setup_demo_admin.py      |  52 ++-
scripts/admin/update_admin_password.py | 104 +++--
```

---

## Phase 1: Fix Health Check

**Status:** Already done locally.

`main_async.py:1030` changed from:

```python
questions_count = await app.state.db.mongo_count("questions")
```

to:

```python
if app.state.db and app.state.db._legacy_default_db is not None:
    questions_count = await app.state.db._legacy_default_db["questions"].count_documents({})
```

This bypasses `_get_context_db()` entirely for the health check endpoint, using the legacy default database directly. No tenant context is needed for health checks.

**No further code changes needed.**

---

## Phase 2: Migration Script Spec

**File to create:** `scripts/migrations/migrate_legacy_tenants.py`

This is the only new file needed. It combines three concerns into one script.

### A) Stamp `admin_id` on Orphan Documents

For each active tenant in `skb_master`:

1. Get the tenant DB by `db_name`
2. Find the admin (first `master_admin` or any admin) in `tenant_db["admins"]`
3. For each of the 13 collections from `TENANT_SCOPED_COLLECTIONS`:
   - Find docs where `admin_id` is missing or null
   - Find docs where `admin_id` is string (should be ObjectId)
   - Update with correct admin ObjectId

### B) Backfill Master Tenant Fields

For each tenant in `skb_master` missing v2 fields:

- Set `enabled_features_v2` using defaults (standard tier)
- Set `subscription_tier` = `"standard"` if missing
- Set `max_students` = `100` if missing
- Set `max_tutors` = `10` if missing
- Preserve existing `enabled_features` (legacy v1)

### C) Assign Super-Admin Ownership

Delegate to existing `assign_superadmin_owners.py` (already handles this correctly). The new script just reminds the operator to run it.

### Script Interface

```bash
# Dry run (all tenants)
python scripts/migrations/migrate_legacy_tenants.py --all --dry-run

# Apply (all tenants)
python scripts/migrations/migrate_legacy_tenants.py --all

# Single tenant
python scripts/migrations/migrate_legacy_tenants.py --tenant-id CIEL-1001 --dry-run
```

---

## Phase 3: Deployment Sequence

### Step 1: MongoDB Backup (on EC2)

```bash
mongodump --uri="$MONGODB_URI" --out=/tmp/backup-$(date +%Y%m%d)
```

### Step 2: Pull New Code (don't restart yet)

```bash
cd ~/backend && git fetch origin && git checkout <new-commit>
```

### Step 3: Run Migration (while old code still serves traffic)

```bash
# Dry run first
python scripts/migrations/migrate_legacy_tenants.py --all --dry-run

# Apply
python scripts/migrations/migrate_legacy_tenants.py --all

# Assign super-admin ownership
python scripts/migrations/assign_superadmin_owners.py --include-non-pending

# Verify zero orphans for known tenant
python scripts/admin/fix_admin_id_orphans.py \
  --db-name skb_indl-ciel-1001 \
  --admin-email cielknowledge@gmail.com \
  --dry-run
```

### Step 4: Restart Backend with New Code

```bash
pip install -r requirements.txt
sudo systemctl restart stoody-backend
```

### Step 5: Smoke Test

```bash
# Health check
curl -s http://127.0.0.1:5001/health | python3 -m json.tool

# Admin login
curl -s -X POST http://127.0.0.1:5001/api/v1/auth/admin/login \
  -H "Content-Type: application/json" \
  -d '{"email":"cielknowledge@gmail.com","password":"...","tenant_id":"CIEL-1001"}'

# Data retrieval with token
curl -s http://127.0.0.1:5001/api/v1/admin/students \
  -H "Authorization: Bearer <token>"

# Super-admin tenant list
curl -s http://127.0.0.1:5001/api/v1/superadmin/tenants \
  -H "Authorization: Bearer <superadmin-token>"
```

---

## Verification Checklist

- [ ] Health check returns 200 with no warning stack traces in logs
- [ ] No `"No tenant context set"` warnings during normal operation
- [ ] No `"DEPRECATED: get_mongo_db()"` warnings
- [ ] Admin login returns JWT with `db_name` and `admin_id`
- [ ] Dashboard shows student/document counts (not zeros)
- [ ] Document upload works
- [ ] Super-admin can see all tenants (including legacy ones)
- [ ] No new documents appear in `skillbot_db_fallback`
- [ ] Feature enforcement works (disabled features return 403)
- [ ] MongoDB Compass shows no new ghost databases created

---

## Rollback Plan

```bash
git checkout 0ead8e01
sudo systemctl restart stoody-backend
```

Migration data is **backward-compatible** -- old code ignores new fields (`admin_id` stamps, `enabled_features_v2`, `subscription_tier`, etc.).

---

## Files Modified Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/main_async.py` | **Already modified** | Health check uses `_legacy_default_db` directly |
| `backend/scripts/migrations/migrate_legacy_tenants.py` | **To create** | Comprehensive migration: `admin_id` stamps + master field backfill |
| `backend/core/database.py` | **Already modified** | Fallback removal, `mongo_db` -> `_legacy_default_db`, `ensure_indexes` refactor |
| `backend/api/v1/auth_async.py` | **Already modified** | `/user` endpoint fix, logout fallback cleanup, debug endpoints removed |
| `backend/api/v1/admin_async.py` | **Already modified** | Availability checks fixed |
| `backend/middleware/tenant_middleware.py` | **Already modified** | Warning log for cookie-only path |
| `backend/scripts/admin/init_admin_direct.py` | **Already modified** | `--db-name` param, hardcoded creds removed |
| `backend/scripts/admin/setup_demo_admin.py` | **Already modified** | `--db-name` param, hardcoded creds removed |
| `backend/scripts/admin/update_admin_password.py` | **Already modified** | `--db-name` param, hardcoded creds removed |
| `backend/scripts/admin/fix_admin_id_orphans.py` | **Already modified** | Per-tenant `admin_id` fixer |

---

## Original 5-Phase Implementation Plan (Reference)

The code changes were guided by a 5-phase plan that has been **fully implemented**:

1. **Phase 1: Core Database Fix** (`core/database.py`) -- Eliminate fallback, rename `mongo_db` to `_legacy_default_db`, rework `ensure_indexes`, deprecate `get_mongo_db()` and `get_mongo_collection()`
2. **Phase 2: Auth Endpoint Fixes** (`api/v1/auth_async.py`) -- Fix `/user` endpoint, fix logout fallback, remove debug endpoints
3. **Phase 3: Admin Endpoint Fixes** (`api/v1/admin_async.py`) -- Fix availability checks to use `db.mongo_client` instead of `db.get_mongo_db()`
4. **Phase 4: Middleware Warning** (`middleware/tenant_middleware.py`) -- Add warning log for cookie-only path without `db_name`
5. **Phase 5: Legacy Scripts** (`scripts/admin/`) -- Add `--db-name` parameter to 3 scripts, remove hardcoded credentials
