# svc-ai-pipeline

AI recognition pipeline for ExamPen. Consumes `page.ready` events from NATS,
runs handwriting recognition (HWR), step detection, and content classification
on page images, then stores results in PostgreSQL and publishes `ai.result`
events.

## Ownership Declaration

- **Writes:** AI recognition results (per-question, per-student, per-model-version)
- **Reads from:** `svc-doc-assembly` (page images in S3), `svc-copy-upload` (copy images in S3)
- **Never writes to:** scores, objections, strokes, exam lifecycle state
- **Transactional boundaries:** AI results written to PostgreSQL per-question per-student. Published to NATS after PG commit. Model version stored with every result. Re-running AI with new model creates new version row; old versions are never overwritten.

## Running Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the service
uvicorn src.main:app --reload

# Run unit tests
pytest tests/ -m unit

# Run all tests
pytest tests/
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://localhost:5432/exampen_ai` | PostgreSQL connection string |
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `MINIO_URL` | `http://localhost:9000` | MinIO/S3 endpoint |
| `MODEL_DIR` | `/models` | Directory containing ONNX model files |
| `CONFIDENCE_THRESHOLD` | `0.85` | Per-character confidence threshold for flagging |

## Dependencies

- `svc-doc-assembly` — provides page images in S3 (consumed via `page.ready` events)
- `svc-copy-upload` — provides copy images in S3 (fallback source)
- PostgreSQL — result storage
- NATS JetStream — event consumption and publication
- MinIO/S3 — page image retrieval
- ONNX Runtime — model inference

## Depended On By

- `svc-score-engine` — reads AI results to create score drafts
- `svc-plagiarism` — reads recognized text for similarity analysis
- `svc-teacher-bff` — reads AI results for teacher display
