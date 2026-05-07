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

When a school receives an ExamPen hub (Raspberry Pi), the provisioning involves two parties: the super-admin creates the code, and the hub consumes it.

### 5.1 Two-Party Provisioning Flow

**Party 1 — Super-admin creates the provisioning code:**

1. Super-admin opens the desktop app and navigates to ExamPen Hub management
2. Super-admin generates a hub provisioning code via `POST /api/v1/superadmin/evalpen/hubs/provision-code` (scoped to a tenant)
3. Code is printed/shared with the school's IT admin (codes are single-use, expire after 72 hours)

**Party 2 — Admin provisions the hub on-site:**

4. School admin powers on the hub, enters WiFi credentials on TUI Setup Screen
5. Admin authenticates to the Stoody backend (admin JWT)
6. Admin enters the provisioning code on the hub TUI
7. TUI calls `POST /api/v1/hubs/provision {hub_code}` with the admin's Bearer token
8. Backend validates the code, assigns a `hub_id`, and returns:
   - `hub_id` — system-assigned hub identifier
   - `institute_id` — tenant identifier
   - `hub_token` — long-lived JWT (365 days) with `user_type: "hub"` for subsequent hub API calls
   - `invig_codes` — pre-generated invigilator auth codes for local caching
   - `pen_inventory` — known pens for this institute (may be empty array)
   - `backend_url` — absolute URL for hub-to-backend communication
   - `provisioned_at` — ISO 8601 timestamp
9. Hub stores config locally, caches invig codes and pen inventory in SQLite
10. Super-admin sees hub status change to "provisioned" in desktop app

**Key distinction:** Super-admin generates the code; admin on-site consumes it at `POST /api/v1/hubs/provision`. The hub itself does not call super-admin endpoints.

### 5.2 Super-Admin Hub Management Endpoints

These endpoints are super-admin scoped (cross-tenant visibility):

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/superadmin/evalpen/hubs/provision-code` | POST | Generate a provisioning code for a tenant |
| `/api/v1/superadmin/evalpen/hubs` | GET | List all provisioned hubs across tenants |
| `/api/v1/superadmin/evalpen/hubs/{hub_id}` | GET | Hub details (last seen, firmware, pen count) |
| `/api/v1/superadmin/evalpen/hubs/{hub_id}` | DELETE | Decommission a hub |

### 5.3 Hub-Facing Provisioning Endpoint

The hub consumes its provisioning code through the tenant-scoped admin API (NOT a super-admin endpoint):

| Endpoint | Method | Caller | Purpose |
|---|---|---|---|
| `/api/v1/hubs/provision` | POST | Admin (`admin` or `b2c_admin` role) | Consume provisioning code, receive `hub_id` + `hub_token` + config |

Full contract: `integration/HUB_DEPLOYMENT_SPEC.md` §7.

### 5.4 Hub Data in skb_master

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

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-05-02 | Added §9 implementation status. Marked implemented surfaces. Added pending items for hub detail, decommission, richer analytics, platform caps. | Claude |
| 2026-04-09 | Resolved provisioning contract authority: split into two-party flow (super-admin creates code, admin consumes it at `POST /api/v1/hubs/provision`). Added §5.3 with explicit hub-facing endpoint. Aligned response fields with HUB_DEPLOYMENT_SPEC §7. | Claude |

---

## 9. Implementation Status (2026-05-02)

The following super-admin ExamPen surfaces are implemented in `super-admin/src/`:

### Implemented

| Surface | Component / API | Status |
|---|---|---|
| Feature gate (exampen enable/disable per tenant) | `ExamPenManagementPage.tsx` → `updateTenantFeaturesV2()` | **Built** |
| Hub fleet listing | `listProvisionedHubs()` → `GET /superadmin/evalpen/hubs` | **Built** |
| Hub provision code generation | `generateHubProvisionCode()` → `POST /superadmin/evalpen/hubs/provision-code` | **Built** |
| Per-hub display in UI | Hub list with status badges, last-seen dates | **Built** |
| Provision code copy-to-clipboard | Button in `ExamPenManagementPage.tsx` | **Built** |
| Token usage (today) | `getEvalPenUsageAggregate()` → `GET /superadmin/evalpen/gate/usage/aggregate` | **Built** |
| Gate tenant listing | `GET /superadmin/evalpen/gate/tenants` | Backend exists, not yet wired in UI |
| Gate tenant config override | `PUT /superadmin/evalpen/gate/tenants/{id}/config` | Backend exists, not yet wired in UI |

### Pending (spec defined but not implemented)

| Item | Spec Reference | Status |
|---|---|---|
| Per-hub detail endpoint | §5.2 `GET /superadmin/evalpen/hubs/{hub_id}` | **Not implemented** |
| Hub decommission endpoint | §5.2 `DELETE /superadmin/evalpen/hubs/{hub_id}` | **Not implemented** |
| Per-tenant exam/submission count analytics | §6 | **Not implemented** — requires new backend endpoint |
| Platform-wide budget cap administration | §4.2 | **Not implemented** |
| Top callers (DCR/PCR/practice token breakdown) | §6 | **Not implemented** |
| Exam throughput across platform | §6 | **Not implemented** |
