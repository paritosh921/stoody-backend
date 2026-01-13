# Multi-Tenant Data Isolation Guide

## Overview

This document explains the tenant isolation architecture that ensures data from one admin/institution cannot be accessed by another.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Request                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TenantMiddleware (sets TenantContext)              │
│                                                                 │
│  - Extracts admin_id from JWT token                             │
│  - Sets TenantContext for the request                           │
│  - Clears context after request completes                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Route Handler                                │
│                                                                 │
│  Uses: tenant_db = Depends(get_tenant_db)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TenantAwareDB (automatic filtering)                │
│                                                                 │
│  - Automatically adds admin_id to queries                       │
│  - Validates tenant context exists                              │
│  - Prevents cross-tenant data access                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DatabaseManager                              │
│                                                                 │
│  - Raw MongoDB operations                                       │
│  - Used only by TenantAwareDB                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. TenantContext (`core/tenant.py`)

Async-safe context variable that holds tenant information:

```python
from core.tenant import TenantContext

# Set at request start
TenantContext.set(admin_id="abc123", user_type="admin")

# Get current tenant
admin_id = TenantContext.get_admin_id()
admin_oid = TenantContext.get_admin_oid()  # As ObjectId

# Clear at request end
TenantContext.clear()
```

### 2. TenantAwareDB (`core/tenant.py`)

Database wrapper that automatically enforces tenant isolation:

```python
from core.tenant import TenantAwareDB

tenant_db = TenantAwareDB(db_manager)

# This automatically adds {"admin_id": current_admin_id}
students = await tenant_db.find("students", {"grade": "10"})

# Insert automatically sets admin_id
await tenant_db.insert_one("students", {"name": "John"})

# Global collections are NOT filtered
admins = await tenant_db.find("admins", {"email": "foo@bar.com"})
```

### 3. get_tenant_db Dependency (`middleware/tenant_middleware.py`)

FastAPI dependency for route handlers:

```python
from middleware.tenant_middleware import get_tenant_db

@router.get("/students")
async def get_students(tenant_db: TenantAwareDB = Depends(get_tenant_db)):
    # No need to add admin_id - it's automatic!
    return await tenant_db.find("students", {})
```

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| TenantContext class | **DONE** | `core/tenant.py` |
| TenantAwareDB wrapper | **DONE** | `core/tenant.py` |
| TenantMiddleware | **DONE** | `middleware/tenant_middleware.py` |
| Middleware registration | **DONE** | Registered in `main_async.py` |
| Strokes isolation | **DONE** | User lookup approach in `strokes_async.py` |
| SmartBoard isolation | **DONE** | admin_id in all inserts/queries in `smartboard_async.py` |
| Data migration script | **DONE** | `scripts/migrations/backfill_admin_id.py` |
| Tenant isolation tests | **DONE** | `tests/test_tenant_isolation.py` |
| Route migration | Partial | Critical routes fixed, gradual migration ongoing |

## Collection Classification

### Tenant-Scoped Collections (automatic admin_id filter)
- `students`
- `documents`
- `tutors`
- `questions`
- `question_attempts`
- `student_activity_log`
- `chat_sessions`
- `student_test_attempts`
- `assignments`
- `meetings`
- `notifications`
- `class_schedules`
- `smartboard_sessions`

### Special Handling Collections
- `strokes` - Written by BLE agent without admin_id. Uses **user lookup approach**: query valid user_ids for tenant first, then filter strokes by those users.

### Global Collections (no filter)
- `admins`
- `system_settings`
- `audit_logs`

## Migration Guide

### Before (fragile - easy to forget admin_id):

```python
@router.get("/students")
async def get_students(
    current_user: Dict = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    admin_id = current_user.get("admin_id") or current_user.get("user_id")

    # BUG RISK: Easy to forget admin_id filter!
    students = await db.mongo_find("students", {"grade": "10"})  # WRONG!

    # Correct but verbose:
    students = await db.mongo_find("students", {
        "grade": "10",
        "admin_id": ObjectId(admin_id)  # Must remember this!
    })

    return students
```

### After (robust - automatic filtering):

```python
from middleware.tenant_middleware import get_tenant_db

@router.get("/students")
async def get_students(tenant_db: TenantAwareDB = Depends(get_tenant_db)):
    # admin_id filter is AUTOMATIC - impossible to forget!
    students = await tenant_db.find("students", {"grade": "10"})
    return students
```

## Security Features

### 1. Automatic Filter Injection
All queries to tenant-scoped collections automatically include `admin_id`.

### 2. Tenant Context Validation
Operations on tenant-scoped collections fail if context isn't set:
```
TenantContextError: Cannot query tenant-scoped collection 'students' without tenant context
```

### 3. Cross-Tenant Access Prevention
If a query includes a different `admin_id`, it raises:
```
TenantIsolationError: Tenant isolation violation: filter admin_id does not match tenant context
```

### 4. Audit Logging
All tenant bypass operations are logged as warnings.

## Bypassing Tenant Filter (Admin Operations Only)

For system-wide operations (use with extreme caution):

```python
async def admin_system_report(tenant_db: TenantAwareDB):
    # This bypasses tenant filter - logs a warning
    with tenant_db.bypass_tenant_filter():
        all_students = await tenant_db.find("students", {})
    return all_students
```

## Testing Tenant Isolation

```python
import pytest
from core.tenant import TenantContext, TenantAwareDB, TenantContextError

async def test_tenant_isolation():
    # Set context for admin A
    TenantContext.set(admin_id="admin_a_id", user_type="admin")

    # Query only returns admin A's students
    students = await tenant_db.find("students", {})
    assert all(str(s["admin_id"]) == "admin_a_id" for s in students)

async def test_missing_context_raises():
    TenantContext.clear()

    with pytest.raises(TenantContextError):
        await tenant_db.find("students", {})

async def test_cross_tenant_access_blocked():
    TenantContext.set(admin_id="admin_a_id", user_type="admin")

    with pytest.raises(TenantIsolationError):
        # Trying to access admin B's data - blocked!
        await tenant_db.find("students", {"admin_id": "admin_b_id"})
```

## Strokes Isolation (User Lookup Approach)

The `strokes` collection is written by the Stoody BLE agent without admin_id. Instead of modifying the BLE agent, we use query-time user lookup:

```python
# In strokes_async.py

# 1. Get valid user_ids for this tenant
admin_id = current_user.get("admin_id")
tenant_students = await db.mongo_find(
    "students",
    {"admin_id": ObjectId(admin_id)},
    projection={"_id": 1}
)
valid_user_ids = [str(s["_id"]) for s in tenant_students]

# 2. Filter strokes to only those users
query = {"user_id": {"$in": valid_user_ids}}
strokes = await db.mongo_find("strokes", query)
```

**Trade-off:** Adds one extra query per request, but avoids BLE agent changes.

## Deployment Checklist

### Phase 1: Core Infrastructure
1. [x] Register `TenantMiddleware` in `main_async.py` (after subdomain middleware)
2. [x] Add `smartboard_sessions` to `TENANT_SCOPED_COLLECTIONS` in `core/tenant.py`

### Phase 2: Fix Data Isolation Gaps
3. [x] Update `strokes_async.py` to use user lookup approach
4. [x] Update `smartboard_async.py` to include admin_id in:
   - Session creation (insert)
   - Session queries (find)
   - Question attempt creation (insert)
   - Question attempt queries (find)

### Phase 3: Migrate Routes
5. [ ] Migrate admin routes to use `get_tenant_db` dependency (gradual, ongoing)
6. [x] Fix `PUT /students/{id}` missing admin_id validation

### Phase 4: Data Migration
7. [ ] Run backfill script for existing smartboard_sessions (run: `python scripts/migrations/backfill_admin_id.py`)
8. [ ] Run backfill script for existing question_attempts (same script handles both)

### Phase 5: Testing
9. [x] Add tenant isolation unit tests (`tests/test_tenant_isolation.py`)
10. [ ] Test cross-tenant access prevention manually
11. [ ] Enable audit logging for production
