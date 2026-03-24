# svc-stroke-proc

Stroke processing service for ExamPen. Subscribes to `stroke.raw` events
from NATS JetStream, deduplicates by idempotency key, normalizes pen
coordinates to mm, assigns strokes to question regions, commits to
TimescaleDB in an atomic transaction, and publishes `stroke.processed`
events.

## Ownership Declaration

- **Writes:** TimescaleDB `processed_strokes` hypertable (the primary
  durable write), NATS JetStream `stroke.processed` events (published
  AFTER DB commit)
- **Reads from:** NATS JetStream `stroke.raw` events (from
  `svc-stroke-ingest`), TimescaleDB (dedup check)
- **Never writes to:** Redis (owned by `svc-stroke-ingest`), exam session
  state (owned by `svc-exam-orch`), score tables, any other service's schema
- **Transactional boundaries:** Dedup -> normalize -> TimescaleDB commit
  (single PostgreSQL transaction per pen per chunk). NATS publish after
  commit. If NATS fails, data is safe in TimescaleDB.

## Processing Pipeline

```
stroke.raw event
  -> idempotency check (chunk_exists)
  -> decode payload (JSON or binary 14-byte frames)
  -> normalize coordinates (pen units -> mm, Y-invert, clamp)
  -> compute bounding boxes
  -> assign to question regions
  -> atomic DB commit (SELECT FOR UPDATE dedup)
  -> publish stroke.processed event
```

## Run Locally

```bash
# From repo root with infra running (NATS, TimescaleDB)
cd services/svc-stroke-proc
pip install -e ".[dev]" -e ../../libs/exampen-common-py
uvicorn src.main:app --reload --port 8003
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

- `libs/exampen-common-py` (NATS client, logging)
- NATS JetStream (event subscribe + publish)
- TimescaleDB / PostgreSQL (processed stroke storage)

## Depended On By

- `svc-doc-assembly` (reads processed strokes from TimescaleDB)
- `svc-doc-assembly` (NATS consumer of `stroke.processed`)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `NATS_CREDS` | `None` | Optional NATS credentials file |
| `DATABASE_URL` | `postgresql+asyncpg://...` | TimescaleDB connection |
| `STROKE_RAW_SUBJECT` | `stroke.raw` | NATS subject to subscribe |
| `STROKE_PROCESSED_SUBJECT` | `stroke.processed` | NATS subject to publish |
| `CONSUMER_DURABLE_NAME` | `svc-stroke-proc` | JetStream consumer name |
