# svc-stroke-ingest

Chunk-oriented stroke ingestion service for ExamPen. Accepts uploads from
hub-uplink, validates CRC-32 integrity, enforces idempotency, publishes
`stroke.raw` events to NATS JetStream, and tracks per-pen upload progress
for reconciliation after hub reconnects.

## Ownership Declaration

- **Writes:** NATS JetStream `stroke.raw` events (the primary write), Redis
  idempotency keys, PostgreSQL `upload_progress` table (best-effort tracking)
- **Reads from:** Redis (idempotency check), PostgreSQL (upload status)
- **Never writes to:** TimescaleDB strokes (owned by `svc-stroke-proc`),
  exam session state (owned by `svc-exam-orch`), any other service's schema
- **Transactional boundaries:** Validate -> NATS JetStream acknowledged
  publish. If publish fails, return HTTP 503; hub retries.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/strokes/ingest` | Ingest one chunk from hub |
| GET | `/api/v1/exams/{exam_id}/upload-status` | Per-pen upload reconciliation |

Contract: `new-docs/api/stroke-ingest.openapi.yaml`

## Run Locally

```bash
# From repo root with infra running (NATS, Redis, PostgreSQL)
cd services/svc-stroke-ingest
pip install -e ".[dev]" -e ../../libs/exampen-common-py
uvicorn src.main:app --reload --port 8002
```

## Run Tests

```bash
# Unit tests (domain only, no infra needed)
pytest tests/ -m unit

# Integration tests (uses mocked infra via test fixtures)
pytest tests/ -m integration

# All tests
pytest tests/
```

## Dependencies

- `libs/exampen-common-py` (auth, NATS, DB, logging)
- NATS JetStream (event publish)
- Redis (idempotency keys)
- PostgreSQL (upload progress tracking)

## Depended On By

- `hub-uplink` (HTTP client)
- `svc-stroke-proc` (NATS consumer of `stroke.raw`)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for idempotency |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `AUTH_SERVICE_URL` | `http://localhost:9100/...` | Stoody JWKS endpoint |
| `IDEMPOTENCY_TTL_SECONDS` | `604800` (7 days) | Key expiry |
| `STROKE_RAW_SUBJECT` | `stroke.raw` | NATS subject for events |
| `RATE_LIMIT_PER_HUB` | `200` | Max requests per hub per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window |
