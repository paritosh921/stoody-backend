# svc-copy-upload

Fallback photo-based answer capture service. Accepts photographed answer pages when stroke-based capture is unavailable, stores them in S3/MinIO, records metadata in PostgreSQL, and publishes `copy.ready` events to NATS for downstream processing.

## Ownership Declaration

- **Writes:** `copy_images` table (PostgreSQL), copy image objects (S3/MinIO)
- **Reads from:** `svc-auth` (JWT validation via exampen_common)
- **Never writes to:** scores, strokes, page images (stroke-derived), exam sessions
- **Transactional boundaries:** Multipart upload -> S3 write -> PG metadata write. Same order as `svc-doc-assembly` page images. If PG fails after S3, orphaned S3 object is acceptable.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/exams/{exam_id}/copies/upload` | Upload one photographed answer page |
| GET | `/api/v1/exams/{exam_id}/copies/{student_id}` | List uploaded copy pages for a student |
| GET | `/api/v1/exams/{exam_id}/copies/{student_id}/{page_number}` | Get specific copy image (presigned URL) |

## Run Locally

```bash
# From repo root
docker compose up postgres minio nats -d
cd services/svc-copy-upload
pip install -e ".[dev]"
python -m src.main
```

## Run Tests

```bash
cd services/svc-copy-upload
pytest tests/ -m unit        # Domain logic only
pytest tests/ -m integration # Mocked S3/PG/NATS
pytest tests/                # All tests
```

## Dependencies

- `libs/exampen-common-py` (auth, db, nats, logging)
- PostgreSQL (metadata storage)
- MinIO/S3 (image storage)
- NATS JetStream (event publishing)

## Depends On This

- `svc-ai-pipeline` (reads copy images when no stroke data available)
- `svc-teacher-bff` (reads for display)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://exampen:exampen@localhost:5432/exampen` | PostgreSQL connection |
| `NATS_URL` | `nats://localhost:4222` | NATS server |
| `MINIO_URL` | `http://localhost:9000` | MinIO/S3 endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | S3 access key |
| `MINIO_SECRET_KEY` | `minioadmin` | S3 secret key |
| `MINIO_BUCKET` | `exampen-copies` | S3 bucket name |
| `PRESIGNED_URL_EXPIRY` | `3600` | Presigned URL TTL (seconds) |
