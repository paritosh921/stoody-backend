# svc-doc-assembly

Stroke-to-page rendering, miss indicator auto-detection, and page image assembly. Consumes `stroke.processed` events from NATS, renders canonical strokes to SVG, detects per-question miss indicators, uploads page images to S3, writes metadata to PostgreSQL, and publishes `page.ready` events.

## Ownership Declaration

- **Writes:** Page images (S3), assembled page metadata (PostgreSQL `assembled_pages`), miss indicator `auto_state`
- **Reads from:** `svc-stroke-proc` (via NATS `stroke.processed` events), TimescaleDB (stroke data via `normalized_stroke_uri`)
- **Never writes to:** Scores, objections, chat, auth, exam lifecycle, stroke tables
- **Transactional boundaries:** S3 write first, PG metadata second. If PG fails after S3, orphaned S3 object is acceptable (garbage collected). Reverse order would create dangling reference.

## Run Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Run unit tests (domain only, zero I/O)
pytest tests/ -m unit

# Run service
uvicorn src.main:app --port 8000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://exampen:exampen@localhost:5432/exampen` | PostgreSQL connection |
| `NATS_URL` | `nats://localhost:4222` | NATS server |
| `MINIO_URL` | `localhost:9000` | MinIO/S3 endpoint |
| `MINIO_BUCKET` | `exampen-pages` | S3 bucket for page images |
| `MINIO_ACCESS_KEY` | `minioadmin` | S3 access key |
| `MINIO_SECRET_KEY` | `minioadmin` | S3 secret key |
| `MINIO_SECURE` | `false` | Use HTTPS for S3 |
| `SERVICE_PORT` | `8000` | HTTP port |

## Dependencies

- `svc-stroke-proc` (upstream: produces `stroke.processed` events)
- PostgreSQL (metadata storage)
- MinIO/S3 (page image storage)
- NATS JetStream (event bus)

## Depends On This

- `svc-ai-pipeline` (reads page images from S3 via `page.ready` events)
- `svc-teacher-bff` (reads assembled page metadata for display)
- `svc-student-bff` (reads assembled page metadata for display)
