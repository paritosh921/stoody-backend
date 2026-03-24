# exampen-common-py

Shared Python utilities for all ExamPen backend services. Pure infrastructure — no business logic.

## Ownership Declaration

- **Writes:** Nothing — this is a library, not a service
- **Reads from:** Stoody JWKS endpoint, PostgreSQL, NATS
- **Never writes to:** Any service database directly
- **Consumers:** All `svc-*` backend services

## Modules

| Module | Purpose |
|--------|---------|
| `auth` | Stoody JWT validation via JWKS, claim normalization, FastAPI dependency |
| `nats_client` | NATS JetStream connection factory, JSON publish/subscribe |
| `db` | PostgreSQL async pool, RLS tenant injection, health check |
| `logging` | JSON structured logging, request/correlation ID propagation |

## Installation

```bash
pip install -e libs/exampen-common-py
# or with dev dependencies
pip install -e "libs/exampen-common-py[dev]"
```

## Usage Examples

### Auth — JWT Validation

```python
from fastapi import FastAPI, Depends
from exampen_common.auth import get_current_user, ExamPenUser, JWKSManager

app = FastAPI()

# Option 1: Use the built-in FastAPI dependency
@app.get("/me")
async def whoami(user: ExamPenUser = Depends(get_current_user)):
    return {"user_id": user.user_id, "roles": user.exampen_roles}

# Option 2: Validate a token directly
async def check_token(raw_token: str):
    from exampen_common.auth import validate_token
    user = await validate_token(raw_token)
    print(user.tenant_id, user.stoody_role)

# Option 3: Custom JWKS manager (e.g. different URL or TTL)
mgr = JWKSManager(jwks_url="https://stoody.example.com/.well-known/jwks.json", ttl_seconds=3600)
await mgr.warmup()  # pre-fetch at startup
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `STOODY_JWKS_URL` | `http://localhost:8000/.well-known/jwks.json` | Stoody JWKS endpoint |
| `JWKS_TTL_SECONDS` | `86400` | Cache TTL in seconds |
| `JWT_AUDIENCE` | *(none)* | Expected `aud` claim |
| `JWT_ISSUER` | *(none)* | Expected `iss` claim |

### NATS — Publish & Subscribe

```python
from exampen_common.nats_client import create_nats_client

# Connect
client = await create_nats_client()

# Publish a JSON message
await client.publish("score.updated", {
    "exam_id": "e-123",
    "student_id": "s-456",
    "total": 87,
})

# Subscribe with consumer group (load-balanced)
async def handle_stroke(payload: dict):
    print("Received:", payload["exam_id"])

await client.subscribe(
    "stroke.raw",
    handle_stroke,
    durable="stroke-proc",
    queue="proc-workers",
)

# Cleanup
await client.close()
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `NATS_CREDS` | *(none)* | Path to NATS credentials file |
| `NATS_RECONNECT_DELAY` | `2` | Seconds between reconnect attempts |
| `NATS_MAX_RECONNECT` | `-1` | Max reconnect attempts (-1 = infinite) |

### DB — PostgreSQL with RLS

```python
from exampen_common.db import create_pool, session_factory, rls_session, get_health
from sqlalchemy import text

# Create pool at startup
engine = create_pool()
sf = session_factory(engine)

# Per-request: session with RLS tenant injection
async for session in rls_session(sf, tenant_id="tenant-abc"):
    result = await session.execute(text("SELECT * FROM exams"))
    exams = result.fetchall()

# Health check (e.g. for /healthz endpoint)
status = await get_health(engine)
# {"status": "healthy", "result": 1}
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://exampen:exampen@localhost:5432/exampen` | Connection string |
| `DB_POOL_SIZE` | `5` | Connection pool size |
| `DB_MAX_OVERFLOW` | `10` | Max overflow connections |
| `DB_POOL_TIMEOUT` | `30` | Pool checkout timeout (seconds) |

### Logging — Structured JSON

```python
from fastapi import FastAPI
from exampen_common.logging import configure_logging, get_logger, RequestIdMiddleware

# Configure at startup
configure_logging(level="DEBUG", service_name="svc-score-engine")

app = FastAPI()
app.add_middleware(RequestIdMiddleware)

log = get_logger(__name__)
log.info("Score computed", extra={"exam_id": "e-123", "score": 87})
# Output: {"asctime": "...", "level": "INFO", "service": "svc-score-engine",
#          "request_id": "abc123", "message": "Score computed",
#          "exam_id": "e-123", "score": 87}
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Root log level |
| `SERVICE_NAME` | `exampen` | Service identifier in log records |

## Running Tests

```bash
cd libs/exampen-common-py
pip install -e ".[dev]"
pytest tests/ -v
```

## Dependencies

- Python >= 3.12
- aiohttp (JWKS fetch)
- asyncpg + SQLAlchemy async (PostgreSQL)
- nats-py (NATS JetStream)
- PyJWT with cryptography (RS256 validation)
- python-json-logger (structured output)
- FastAPI / Starlette (dependency injection, middleware)
