# ExamPen DCR — Integration Notes

Exact changes needed to wire ExamPen DCR into the main Stoody backend.
**Do NOT apply these changes during the DCR module build phase** — they
are deferred to the integration sprint.

---

## 1. `config_async.py` additions

Add these near the other env-var reads (around the `Settings` class or
at module level alongside existing config constants):

```python
# ExamPen DCR (NATS event bus)
NATS_URL: str = os.getenv("NATS_URL", "nats://localhost:4222")
EXAMPEN_ENABLED: bool = os.getenv("EXAMPEN_ENABLED", "false").lower() == "true"
```

---

## 2. `core/tenant_features.py` additions

### 2a. Add to FEATURE_CATALOG (after the `stoody_pen_capture` entry):

```python
{
    "key": "exampen_dcr",
    "label": "ExamPen DCR",
    "description": "Digital Copy Review — pen-based exam capture, AI scoring, and objection workflow",
    "category": FEATURE_CATEGORY_MAX,
    "audience": ["tenant_admin", "tutor", "student"],
    "status": STATUS_ACTIVE,
    "default_enabled": False,
    "billing_code": "MAX_EXAMPEN_DCR",
},
```

### 2b. Add to FEATURE_PATH_PREFIXES:

```python
"exampen_dcr": (
    "/api/v1/exampen",
),
```

### 2c. Add legacy mapping (optional, for v1 compat):

```python
# In LEGACY_TO_V2_MAP — only if a legacy key is needed
"exampen": ("exampen_dcr",),

# In V2_TO_PRIMARY_LEGACY_MAP
"exampen_dcr": "exampen",
```

---

## 3. `main_async.py` additions

Inside the `lifespan` async context manager, add a NATS initialization
block guarded by `EXAMPEN_ENABLED`. This ensures NATS is only started
when the feature is explicitly enabled.

```python
# --- At top of file, with other imports ---
from config_async import NATS_URL, EXAMPEN_ENABLED

# --- Inside the lifespan function, after MongoDB/Redis init ---
nats_client = None
exampen_consumer_tasks = []

if EXAMPEN_ENABLED:
    try:
        from exampen.dcr.core.nats_client import NatsClient
        from exampen.dcr.events.stream_setup import ensure_exampen_stream
        from exampen.dcr.events.consumers import start_all_consumers

        nats_client = NatsClient(url=NATS_URL)
        await nats_client.connect()
        await ensure_exampen_stream(nats_client)
        exampen_consumer_tasks = await start_all_consumers(
            nats_client, db_manager
        )
        logger.info("ExamPen DCR event bus initialized")
    except Exception:
        logger.exception("Failed to initialize ExamPen DCR — continuing without it")
        nats_client = None

yield  # <-- existing yield

# --- In the shutdown section (after yield) ---
if nats_client is not None:
    await nats_client.close()
    logger.info("ExamPen NATS connection closed")
```

### 3b. Mount ExamPen API routers (after existing router includes):

```python
if EXAMPEN_ENABLED:
    try:
        from exampen.dcr.api import exam_router, score_router, objection_router
        app.include_router(exam_router, prefix="/api/v1/exampen", tags=["exampen"])
        app.include_router(score_router, prefix="/api/v1/exampen", tags=["exampen"])
        app.include_router(objection_router, prefix="/api/v1/exampen", tags=["exampen"])
        logger.info("ExamPen API routers mounted at /api/v1/exampen")
    except Exception:
        logger.exception("Failed to mount ExamPen routers")
```

---

## 4. `requirements.txt` additions

```
nats-py>=2.6.0,<3
```

---

## 5. Environment variables (`.env` or deployment config)

```bash
# Enable ExamPen DCR module
EXAMPEN_ENABLED=true

# NATS server URL (default: localhost for dev)
NATS_URL=nats://localhost:4222
```

---

## 6. Infrastructure prerequisites

- **NATS Server** with JetStream enabled (`nats-server -js`)
- The `EXAMPEN` stream is auto-created by `stream_setup.ensure_exampen_stream()`
- File storage, 7-day retention, captures all `EXAMPEN.>` subjects
- For production: use NATS credentials file and TLS

---

## 7. Feature gating flow

The three-layer gating model applies:

1. **Backend path prefix**: `/api/v1/exampen` is gated by `exampen_dcr` in
   `FEATURE_PATH_PREFIXES` — middleware blocks requests for tenants without
   the feature enabled.

2. **Frontend route guard**: Add `"/exampen"` to `FEATURE_GATED_STUDENT_PATHS`
   in `ProtectedRoute.tsx` (or equivalent admin/tutor guard).

3. **Frontend UI conditional**: Use `isTenantFeatureEnabled(tenantFeatures, 'exampen_dcr')`
   to show/hide ExamPen navigation items and dashboard widgets.

---

## 8. Testing checklist

- [ ] NATS not running: backend starts normally, consumers are skipped, log warnings
- [ ] EXAMPEN_ENABLED=false: no NATS connection attempted, no routers mounted
- [ ] EXAMPEN_ENABLED=true + NATS running: stream created, all 7 consumers registered
- [ ] Tenant without `exampen_dcr` feature: API returns 403 on `/api/v1/exampen/*`
- [ ] Tenant with `exampen_dcr` feature: full pipeline end-to-end
- [ ] Queue group dedup: two workers only process each message once
