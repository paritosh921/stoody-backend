# Multi-Tenancy Architecture Analysis

## Executive Summary

This document analyzes different approaches to multi-tenant data isolation for the SkillBot platform, comparing their trade-offs and providing recommendations for implementation.

---

## Current State: Query-Level Filtering

### How It Works
Every database query manually includes `admin_id` filter:

```python
students = await db.mongo_find("students", {
    "grade": "10",
    "admin_id": ObjectId(admin_id)  # Must remember to add this!
})
```

### Problems Identified
1. **Fragile** - Developers must remember to add `admin_id` to every query
2. **Error-prone** - Missing filter = cross-tenant data leakage (security vulnerability)
3. **Hard to audit** - No centralized enforcement, bugs scattered across codebase
4. **Recently discovered bugs** - Multiple endpoints were missing `admin_id` filters

### Files with Data Isolation Bugs (Fixed)
- `admin_async.py:1207-1228` - class-section monitoring stats
- `admin_async.py:1451-1472` - student progress endpoint
- `admin_async.py:1606-1619` - activity feed endpoint
- `admin_async.py:1895-1910` - test attempts endpoint
- `tutor_async.py:485-507` - tutor students endpoint

---

## Multi-Tenancy Approaches Comparison

| Approach | Data Isolation | Complexity | Performance | Cost | Compliance |
|----------|---------------|------------|-------------|------|------------|
| Separate DB Instances | Strongest | Very High | Best | Highest | Easiest |
| Database Per Tenant | Strong | Medium-High | Good | Same* | Easy |
| **Tenant-Aware Data Layer** | Good | Medium | Good | Same | Medium |
| Query-Level Filters (Current) | Weak | Low | Good | Same | Hard |

*MongoDB pricing is based on storage and operations, NOT number of databases.

---

## Option 1: Separate MongoDB Instances

### Architecture
```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  MongoDB Instance 1 │  │  MongoDB Instance 2 │  │  MongoDB Instance 3 │
│  (DPS Delhi)        │  │  (KV School)        │  │  (Lincoln High)     │
│                     │  │                     │  │                     │
│  URI: mongodb://    │  │  URI: mongodb://    │  │  URI: mongodb://    │
│  dps.cluster.net/   │  │  kvs.cluster.net/   │  │  lincoln.cluster.net│
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

### Pros
- Strongest isolation (physical separation)
- Independent scaling per tenant
- Easy compliance (GDPR, data residency)
- No risk of cross-tenant bugs

### Cons
- Highest operational complexity
- Highest cost (separate clusters)
- Complex connection management
- Difficult cross-tenant analytics

### When to Use
- Enterprise customers requiring dedicated infrastructure
- Strict data residency requirements (data must stay in specific region)
- Customers willing to pay premium for isolation

---

## Option 2: Database Per Tenant (Recommended for SkillBot)

### Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                    MongoDB Cluster (Single URI)                     │
│  mongodb+srv://user:pass@cluster.mongodb.net/                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ skb_master   │  │ skb_in_dps01 │  │ skb_in_kvs01 │   ...        │
│  │ (Super Admin)│  │ (DPS Delhi)  │  │ (KV School)  │              │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤              │
│  │ - tenants    │  │ - students   │  │ - students   │              │
│  │ - billing    │  │ - documents  │  │ - documents  │              │
│  │ - analytics  │  │ - tutors     │  │ - tutors     │              │
│  │ - audit_logs │  │ - questions  │  │ - questions  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### How MongoDB URI Works
```python
# Single URI in environment (no database specified)
MONGODB_URI = "mongodb+srv://user:pass@cluster.mongodb.net/"

# Connection at application startup
client = AsyncIOMotorClient(MONGODB_URI)

# Database selection at runtime based on authenticated user
master_db = client["skb_master"]                    # Super-admin
tenant_db = client[f"skb_{tenant.database_name}"]   # e.g., "skb_in_dps01"
```

### Super-Admin Data Access
```python
# Master database tracks all tenants
tenants_collection = master_db["tenants"]

# Tenant registry document
{
    "_id": ObjectId("..."),
    "tenant_id": "skb_in_dps01",
    "admin_id": "abc123",
    "database_name": "skb_in_dps01",
    "institution": "Delhi Public School, R.K. Puram",
    "region": "in",
    "institution_code": "dps",
    "instance": "001",
    "subdomain": "dpsrkp",
    "created_at": ISODate("2024-01-15T10:00:00Z"),
    "status": "active",
    "plan": "enterprise",
    "storage_quota_gb": 50,
    "student_limit": 5000
}

# Super-admin can:
# 1. Query master DB for tenant list
all_tenants = await master_db["tenants"].find({}).to_list()

# 2. Connect to specific tenant DB when needed
for tenant in tenants:
    tenant_db = client[tenant["database_name"]]
    student_count = await tenant_db["students"].count_documents({})

# 3. Run aggregation across all tenant DBs for analytics
```

### Proposed Naming Convention
```
skb_{region}_{institution_code}_{instance}

Format:
- skb_        : Prefix (SkillBot)
- {region}    : 2-letter country/region code (in, us, uk, ae, sg)
- {inst_code} : 3-6 character institution identifier
- {instance}  : 3-digit instance number (001-999)

Examples:
- skb_in_dps_001      → India, Delhi Public School, instance 1
- skb_in_kvs_001      → India, Kendriya Vidyalaya, instance 1
- skb_in_dav_001      → India, DAV School, instance 1
- skb_us_lincoln_001  → USA, Lincoln High, instance 1
- skb_ae_gems_001     → UAE, GEMS School, instance 1
- skb_master          → Super-admin master database (special)
```

### Tenant Registry Schema
```javascript
// Collection: skb_master.tenants
{
    "_id": ObjectId("..."),

    // Identification
    "tenant_id": "skb_in_dps_001",
    "database_name": "skb_in_dps_001",
    "admin_id": ObjectId("..."),  // Primary admin user

    // Institution Details
    "institution_name": "Delhi Public School, R.K. Puram",
    "institution_code": "dps",
    "region": "in",
    "instance": "001",

    // Access
    "subdomain": "dpsrkp",
    "custom_domain": "learn.dpsrkp.edu.in",  // Optional

    // Status
    "status": "active",  // active, suspended, trial, cancelled
    "created_at": ISODate("2024-01-15T10:00:00Z"),
    "activated_at": ISODate("2024-01-16T00:00:00Z"),
    "trial_ends_at": ISODate("2024-02-15T00:00:00Z"),

    // Plan & Limits
    "plan": "enterprise",  // free, starter, professional, enterprise
    "student_limit": 5000,
    "tutor_limit": 100,
    "storage_quota_gb": 50,
    "features": ["smartboard", "ai_tutor", "analytics", "api_access"],

    // Usage Tracking
    "current_students": 3450,
    "current_tutors": 45,
    "storage_used_gb": 12.5,

    // Billing (if applicable)
    "billing_email": "accounts@dpsrkp.edu.in",
    "billing_cycle": "annual",
    "next_billing_date": ISODate("2025-01-15T00:00:00Z"),

    // Metadata
    "created_by": ObjectId("..."),  // Super-admin who created
    "notes": "Premium customer since 2024"
}
```

### Pros
- Strong isolation (separate databases)
- Same MongoDB cluster = same cost
- Easy backup/restore per tenant (`mongodump --db skb_in_dps_001`)
- Easy tenant deletion (`DROP DATABASE skb_in_dps_001`)
- Smaller indexes per database = faster queries
- Impossible to accidentally query wrong tenant
- Clear audit trail

### Cons
- More complex connection management
- Need tenant routing logic
- Cross-tenant queries require iteration
- Initial migration effort

### Cost Analysis
MongoDB Atlas pricing is based on:
- Cluster tier (RAM, CPU, storage)
- Storage used (GB)
- Data transfer
- Operations (reads/writes)

**NOT** on number of databases. Therefore:
- 1 database with 100GB = 100 databases with 1GB each (same cost)
- Database-per-tenant has NO additional cost

---

## Option 3: Tenant-Aware Data Layer (Scaffolded)

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Request                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TenantMiddleware (sets TenantContext)              │
│  - Extracts admin_id from JWT token                             │
│  - Sets async-safe context variable                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Route Handler                                │
│  Uses: tenant_db = Depends(get_tenant_db)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TenantAwareDB (automatic filtering)                │
│  - Automatically injects admin_id into queries                  │
│  - Validates tenant context exists                              │
│  - Blocks cross-tenant access attempts                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DatabaseManager                              │
│  - Raw MongoDB operations                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Files
- `core/tenant.py` - TenantContext and TenantAwareDB classes
- `middleware/tenant_middleware.py` - FastAPI integration

### Usage Example
```python
# BEFORE (fragile):
@router.get("/students")
async def get_students(
    current_user: Dict = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    admin_id = current_user.get("admin_id")
    # BUG RISK: Easy to forget admin_id!
    students = await db.mongo_find("students", {"grade": "10"})

# AFTER (robust):
from middleware.tenant_middleware import get_tenant_db

@router.get("/students")
async def get_students(tenant_db: TenantAwareDB = Depends(get_tenant_db)):
    # admin_id filter is AUTOMATIC
    students = await tenant_db.find("students", {"grade": "10"})
```

### Pros
- Prevents "forgot admin_id" bugs
- Gradual migration (works alongside existing code)
- Centralized enforcement
- Audit logging for bypass attempts

### Cons
- Still single database (logical separation only)
- admin_id in every document (storage overhead)
- Complex queries with $and conditions
- Tenant deletion requires scanning all collections

---

## Recommendation

### Short-Term (Immediate)
Complete **Tenant-Aware Data Layer** (Option 3):
- Scaffolded in `core/tenant.py` and `middleware/tenant_middleware.py`
- Requires: middleware registration in `main_async.py` + route migration
- Gradual migration possible, no breaking changes

### Medium-Term (3-6 months)
Migrate to **Database Per Tenant** (Option 2):
- Stronger isolation
- Better performance (smaller indexes)
- Easier compliance
- Same cost as current approach

### Long-Term (Enterprise customers)
Offer **Separate DB Instances** (Option 1) as premium tier:
- For customers with strict compliance requirements
- Higher price point to cover operational costs

---

## Migration Path: Current → Database Per Tenant

### Phase 1: Preparation
1. Implement tenant registry in master database
2. Create database provisioning scripts
3. Update connection logic to support multi-database

### Phase 2: New Tenants
1. All new registrations get separate database
2. Existing tenants remain in shared database
3. Test thoroughly with new tenants

### Phase 3: Migration
1. Create migration script to copy tenant data to new database
2. Migrate tenants in batches (start with small/test tenants)
3. Update tenant registry with new database name
4. Verify data integrity
5. Remove old data from shared database

### Phase 4: Cleanup
1. Remove admin_id from all queries (no longer needed)
2. Remove TenantAwareDB layer (database isolation handles it)
3. Archive shared database
4. Update documentation

---

## Security Considerations

### Current Risks
1. Cross-tenant data leakage via missing filters
2. No audit trail for data access
3. Difficult compliance reporting

### With Database Per Tenant
1. Physical database separation = impossible cross-tenant access
2. Database-level access logs
3. Easy per-tenant audit reports
4. Simple GDPR compliance (delete database = delete all tenant data)

---

## Current Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| TenantContext class | Scaffolded | `core/tenant.py` |
| TenantAwareDB wrapper | Scaffolded | `core/tenant.py` |
| TenantMiddleware | Scaffolded | `middleware/tenant_middleware.py` |
| Middleware registration | **NOT DONE** | `main_async.py` |
| Route migration | **NOT DONE** | Routes use manual admin_id |
| SmartBoard isolation | **NOT DONE** | Missing admin_id in inserts/queries |
| Strokes isolation | **NOT DONE** | Will use user lookup approach |

**Verdict:** The tenant-aware layer is scaffolded but not wired into the running backend.

---

## Appendix: Collection Classification

### Tenant-Scoped Collections
These contain tenant-specific data and require isolation:
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
These require custom isolation logic:
- `strokes` - Written by BLE agent without admin_id. Uses **user lookup approach**: query valid user_ids for tenant first, then filter strokes by those users. No BLE agent changes required.

### Global Collections (Master Database Only)
These contain system-wide data:
- `tenants` - Tenant registry
- `admins` - Admin accounts (with tenant reference)
- `system_settings` - Global configuration
- `audit_logs` - System-wide audit trail
- `billing` - Subscription/payment data
- `feature_flags` - Global feature toggles

---

## References

- [MongoDB Multi-Tenancy Patterns](https://www.mongodb.com/docs/manual/tutorial/model-data-for-multi-tenancy/)
- [SaaS Multi-Tenant Architecture](https://docs.aws.amazon.com/wellarchitected/latest/saas-lens/multi-tenant-architecture.html)
- [GDPR Data Isolation Requirements](https://gdpr.eu/data-protection/)
