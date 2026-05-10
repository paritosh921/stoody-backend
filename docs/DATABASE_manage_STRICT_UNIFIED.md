# DATABASE_manage_STRICT_UNIFIED.md

## Purpose

Unified strict-mode backend stabilization and deployment plan, combining:

- `DATABASE_manage_STRICT_codex.md` (strict auth/tenant acceptance criteria)
- retired `DATABASE_manage_STRICT_claude.md` notes (operational execution, commands, rollback)

This version incorporates feedback to add an explicit EC2 source/runtime parity phase and avoids over-claiming a single 502 root cause.

---

## Decision

Use **Strict Tenant Model** only:

1. No fallback tenant DB reads/writes for tenant-scoped routes.
2. Tenant context (`db_name`) is mandatory for authenticated tenant operations.
3. Super-admin remains the approval/activation authority.

---

## Root Cause Position (Corrected)

1. The health-check context path is a confirmed fix item.
2. Health-check warning/error path is **not** treated as the sole proven 502 cause.
3. There is a likely deployment/runtime mismatch risk (wrong checkout, wrong venv, wrong `ExecStart`, or stale process target), which must be verified first.

---

## Phase 0: EC2 Runtime and Source Parity (Mandatory Gate)

Run on EC2 before any migration/restart:

```bash
cd ~/backend
git log -1 --oneline
git rev-parse --abbrev-ref HEAD
sudo systemctl cat stoody-backend
sudo systemctl status stoody-backend --no-pager
which python
python -V
```

Verify:

1. Deployed commit is exactly the intended commit.
2. `ExecStart` points to the correct backend path and virtualenv.
3. Gunicorn app target matches intended module (`main_async:app` or configured equivalent).
4. Service is not running an old checkout or alternate venv.

Do not proceed until all 4 are confirmed.

---

## Phase 1: Backup + Pre-Deploy Safety

```bash
mongodump --uri="$MONGODB_URI" --out=/tmp/backup-$(date +%Y%m%d-%H%M%S)
cd ~/backend && git fetch origin
git checkout <target-commit>
```

If dependencies changed:

```bash
pip install -r requirements.txt
```

---

## Phase 2: Data Migration (Before Restart)

### 2.1 Required tenant integrity checks in `skb_master.tenants`

Each tenant must have:

1. `tenant_id`
2. `institution_id`
3. `db_name`
4. `status`
5. `assigned_superadmin_id`

Rows missing required linkage must be corrected or quarantined.

### 2.2 Migrate legacy tenant-scoped documents

Use migration tooling to ensure `admin_id` linkage is present and typed correctly (`ObjectId`) across tenant-scoped collections.

Recommended execution:

```bash
# Dry-run comprehensive migration
python scripts/migrations/migrate_legacy_tenants.py --all --dry-run

# Apply migration
python scripts/migrations/migrate_legacy_tenants.py --all

# Super-admin ownership assignment for legacy tenants
python scripts/migrations/assign_superadmin_owners.py --include-non-pending
```

Optional targeted verification:

```bash
python scripts/admin/fix_admin_id_orphans.py \
  --db-name <tenant_db_name> \
  --admin-email <tenant_admin_email> \
  --dry-run
```

### 2.3 Backfill tenant feature metadata

For legacy tenants missing v2 fields, backfill:

1. `enabled_features_v2`
2. `subscription_tier`
3. `max_students`
4. `max_tutors`

Keep legacy feature fields as-is for compatibility, but v2 must become canonical.

---

## Phase 3: Deploy and Restart

```bash
cd ~/backend
sudo systemctl restart stoody-backend
sudo journalctl -u stoody-backend --no-pager -n 200
```

Immediate smoke checks:

```bash
curl -s http://127.0.0.1:5001/health | python3 -m json.tool
curl -s -I http://127.0.0.1:5001/health
```

---

## Phase 4: Strict Auth and Feature-Gate Acceptance Tests

### A. Core strict auth tests

1. Active admin login succeeds and returns token containing non-null `tenant_id`, `institution_id`, `db_name`.
2. Pending/unapproved tenant login fails with `403` (tenant inactive).
3. Missing `db_name` in authenticated tenant request fails closed (`401`/explicit tenant DB missing).
4. Tenant row exists but tenant DB missing returns deterministic failure (`503` or configured error), never fallback data.
5. `/auth/user` returns clear tenant-state/auth-state errors; no silent default DB behavior.

### B. Feature enforcement tests

1. Disable feature in super-admin for tenant.
2. Re-login as tenant admin.
3. Route mapped to disabled feature returns `403`.
4. Re-enable and re-login restores access.
5. Confirm enforcement for admin-facing features that cascade to tutor/student visibility/usage.

### C. Health and observability tests

1. `/health` works without tenant context.
2. No recurring `"No tenant context set"` warnings for normal authenticated flows.
3. No new writes to fallback/ghost DBs.

---

## Phase 5: Rollback (Fast Path)

If deployment fails:

```bash
cd ~/backend
git checkout 0ead8e01
sudo systemctl restart stoody-backend
```

Then re-check:

```bash
curl -s http://127.0.0.1:5001/health
sudo journalctl -u stoody-backend --no-pager -n 100
```

---

## Operational Rules Going Forward

1. Never reintroduce fallback DB behavior for tenant routes.
2. Do not issue tenant JWT/session states with empty `db_name`.
3. Treat manual Compass edits as unsafe until validated by migration scripts.
4. Keep migration scripts idempotent and dry-run capable.
5. Every production deploy must pass Phase 0 parity checks before restart.

---

## Deliverables

1. `scripts/migrations/migrate_legacy_tenants.py` — **CREATED** (admin_id orphan fix + feature v2 backfill + integrity validation + quarantine).
2. Migration execution logs (dry-run + apply) — run on EC2 before deploy.
3. Acceptance test evidence for strict auth and feature gates.
4. Final deployed commit hash + systemd `ExecStart` evidence.

## Phase 0 Evidence Template (fill on EC2)

```
# Run these on EC2, paste output here before proceeding:
git log -1 --oneline              → ___
git rev-parse --abbrev-ref HEAD   → ___
sudo systemctl cat stoody-backend → ExecStart=___
which python                      → ___
python -V                         → ___

# All 4 must match expectations. Do not proceed until confirmed.
```

## Phase 4 Evidence Template (fill after deploy)

```
# Health check
curl -s http://127.0.0.1:5001/health | python3 -m json.tool
# Expected: {"healthy": true, "mongodb": {"connected": true, ...}}

# Admin login (active tenant)
curl -s -X POST http://127.0.0.1:5001/api/v1/auth/admin/login \
  -H "Content-Type: application/json" \
  -d '{"email":"...","password":"...","tenant_id":"..."}'
# Expected: 200 with token containing db_name, tenant_id, admin_id

# Pending tenant login (should fail)
# Expected: 403 "Tenant is not active"

# Missing db_name request (should fail)
# Expected: 401 "Tenant database missing"

# Feature-gated route with disabled feature
# Expected: 403 "Feature ... is disabled by super admin"

# Re-enable feature, re-login, retry
# Expected: 200
```

## Code Changes Implemented

| File | Change |
|------|--------|
| `core/database.py` | Removed `_legacy_default_db`, `get_mongo_db()` raises RuntimeError, `get_mongo_collection()` raises RuntimeError, cleaned health_check reset |
| `middleware/tenant_middleware.py` | Strict mode: exact-match exempt paths (no prefix leak), all auth requests without `db_name` rejected (401), static resource bypass |
| `main_async.py` | Health check uses `master_db.tenants.count_documents()` instead of legacy DB question count |
| `api/v1/student_bulk_upload.py` | Fixed broken `db.mongo_db["students"]` → `tenant_db["students"]` |
| `api/v1/admin_async.py` | Removed `skillbot_db` references in comments |
| `api/v1/pdf_async.py` | Removed `skillbot_db` reference in comment |
| `update_reattempts.py` | Rewritten to require `--db-name` flag (was hardcoded to `skillbot_db`) |
| `scripts/migrations/migrate_legacy_tenants.py` | **NEW**: migration with orphan fix + feature backfill + integrity quarantine |
| `CLAUDE.md` | Updated DB architecture docs, marked auth_bypass as resolved |
| `api/v1/__pycache__/auth_bypass.cpython-*.pyc` | Deleted |
