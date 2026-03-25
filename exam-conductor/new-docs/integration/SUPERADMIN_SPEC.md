# SUPERADMIN_SPEC.md
# ExamPen — Super-Admin Control Surface

**Status:** ACTIVE
**Authority:** Platform-level ExamPen administration via the super-admin desktop app.

Reference: `architecture/LLM_GATE_SPEC.md`, `integration/HUB_DEPLOYMENT_SPEC.md`, `governance/STATE_OWNERSHIP_MAP.md`

---

## 1. Summary

The super-admin app (`super-admin/`) is the platform management console for Stoody. It manages tenants, feature flags, and cross-tenant operations. This spec defines the ExamPen-specific surfaces that the super-admin needs.

The super-admin does NOT interact with individual exams, students, or evaluation results. Those are tenant-scoped and belong to the tutor/admin role within the frontend.

---

## 2. Codebase Location

| Component | Location |
|-----------|----------|
| Super-admin desktop app | `super-admin/` (Electron + React) |
| Backend super-admin API | `backend/api/v1/superadmin_async.py` |
| Feature flag catalog | `backend/core/tenant_features.py` |
| Gate config/usage API | `backend/api/v1/evalpen_usage_async.py` |

---

## 3. ExamPen Feature Flag

The `exampen` feature flag controls access to all ExamPen endpoints.

| Property | Value |
|----------|-------|
| Key | `exampen` |
| Tier | MAX |
| Default | OFF (must be explicitly enabled per tenant) |
| Audience | `tenant_admin`, `tutor` |
| Path prefix | `/api/v1/evalpen` |
| Billing code | `MAX_EXAMPEN` |

**Super-admin responsibilities:**
- Enable/disable `exampen` per tenant via the existing feature flag UI
- View which tenants have ExamPen enabled
- Bulk-enable for pilot tenants

---

## 4. LLM Gate Administration

Super-admin needs platform-wide visibility into LLM token consumption across tenants.

### 4.1 Per-Tenant Gate Config

Existing endpoint (admin-scoped, per-tenant):
- `PUT /api/v1/evalpen/usage/config` — Set token limits

**Super-admin addition needed:**
- `GET /api/v1/superadmin/evalpen/gate/tenants` — List all tenants with ExamPen enabled and their gate config
- `PUT /api/v1/superadmin/evalpen/gate/tenants/{tenant_id}/config` — Override gate config for a specific tenant
- `GET /api/v1/superadmin/evalpen/gate/usage/aggregate` — Cross-tenant usage summary

### 4.2 Platform Budget

Super-admin may set platform-wide LLM budget caps that override per-tenant limits:

| Config | Scope | Purpose |
|--------|-------|---------|
| `platform_daily_token_limit` | All tenants combined | Prevent runaway costs |
| `platform_monthly_token_limit` | All tenants combined | Monthly billing cap |
| `per_tenant_default_daily_limit` | Default for new tenants | Baseline per-tenant budget |

---

## 5. Hub Provisioning

When a school receives an ExamPen hub (Raspberry Pi), the super-admin provisions it:

### 5.1 Provisioning Flow

1. Super-admin creates a hub provisioning code in the desktop app
2. Code is printed/shared with the school's IT admin
3. School admin enters the code on the hub's TUI Setup screen
4. Hub calls `POST /api/v1/hubs/provision {hub_code}`
5. Backend validates code, assigns `hub_id`, returns config
6. Super-admin sees hub status change to "provisioned"

### 5.2 Hub Management Endpoints (super-admin scoped)

- `POST /api/v1/superadmin/evalpen/hubs/provision-code` — Generate provisioning code for a tenant
- `GET /api/v1/superadmin/evalpen/hubs` — List all provisioned hubs across tenants
- `GET /api/v1/superadmin/evalpen/hubs/{hub_id}` — Hub details (last seen, firmware, pen count)
- `DELETE /api/v1/superadmin/evalpen/hubs/{hub_id}` — Decommission a hub

### 5.3 Hub Data in skb_master

Hub provisioning records live in `skb_master` (not per-tenant DB):

Collection: `exampen_hubs`
```
{
  hub_id: string,
  institution_id: string,
  hub_code: string (hashed),
  provisioned_at: datetime,
  last_seen_at: datetime,
  firmware_version: string,
  status: "provisioned" | "active" | "decommissioned",
  config: { backend_url, wifi_mode, uplink_mode }
}
```

---

## 6. Usage Analytics Dashboard

Super-admin desktop app should display:

- **Token usage by tenant** — Daily/weekly/monthly breakdown
- **Top callers** — Which engines (DCR, PCR, practice) consume the most tokens
- **Budget utilization** — % of limit used per tenant
- **Hub fleet status** — Online/offline hubs, last sync times
- **Exam throughput** — Submissions/evaluations per day across platform

Data sources:
- `llm_token_usage_rollup` (per-tenant DB) — Token rollups
- `evalpen_submissions` (per-tenant DB) — Submission counts
- `exampen_hubs` (skb_master) — Hub status

---

## 7. Ownership Boundaries

| What | Owner |
|------|-------|
| Feature flag enable/disable | Super-admin |
| Platform budget caps | Super-admin |
| Hub provisioning codes | Super-admin |
| Hub fleet monitoring | Super-admin |
| Per-tenant gate config | Tenant admin (or super-admin override) |
| Exam creation, evaluation, review | Tenant tutor/admin (NOT super-admin) |
| Student data, scores, feedback | Tenant-scoped (super-admin has no direct access) |

---

## 8. Hard Rules

1. Super-admin does NOT access individual student exam data.
2. Super-admin does NOT trigger evaluations or override scores.
3. Hub provisioning codes are single-use and expire after 72 hours.
4. Platform budget caps cannot be bypassed by per-tenant config.
5. All super-admin actions on ExamPen resources are logged in `skb_master.superadmin_audit_log`.
