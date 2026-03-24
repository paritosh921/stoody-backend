# ExamPen Pipeline Tests (L5 - E2E)

End-to-end tests that verify multi-service pipeline correctness per
`TEST_SUITE_SPEC.md` section 2.3 (test IDs E2E-01 through E2E-13).

## Test Matrix

| File | Test ID | What It Verifies |
|------|---------|------------------|
| `test_e2e_01_stroke_pipeline.py` | E2E-01 | Stroke ingestion -> processing -> TimescaleDB storage, dedup |
| `test_e2e_02_page_assembly.py` | E2E-02 | Page assembly -> page image in S3, page.ready event |
| `test_e2e_03_ai_scoring.py` | E2E-03 | AI result -> score generation in ai_draft state |
| `test_e2e_04_score_override.py` | E2E-04 | Score override via REST -> analytics percentile recalculation |
| `test_e2e_05_objection.py` | E2E-05 | Objection lifecycle: file -> resolve (approve) -> re-score -> notification |
| `test_e2e_06_plagiarism.py` | E2E-06 | Plagiarism detection: known pairs flagged, false positives below threshold |
| `test_e2e_07_copy_fallback.py` | E2E-07 | Copy image upload -> OCR -> score (fallback path) |
| `test_e2e_08_full_simulation.py` | E2E-08 | Full 40-student x 10-question exam simulation |
| `test_e2e_09_miss_indicators.py` | E2E-09 | Miss indicator propagation through pipeline |
| `test_e2e_10_teacher_bff.py` | E2E-10 | Teacher BFF score aggregation and dashboard data delivery |
| `test_e2e_11_student_objection.py` | E2E-11 | Student BFF objection lifecycle end-to-end |
| `test_e2e_12_webhook.py` | E2E-12 | Stoody webhook delivery on score publication |
| `test_e2e_13_full_integration.py` | E2E-13 | Full happy-path integration smoke: create exam -> strokes -> AI -> score -> review -> publish -> objection -> resolve -> analytics -> webhook |

## Prerequisites

### Required Infrastructure

Full Docker Compose stack must be running:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Services required:
- PostgreSQL 16 + TimescaleDB
- NATS JetStream
- MinIO (S3-compatible)
- All pipeline services: svc-stroke-ingest, svc-stroke-proc, svc-doc-assembly,
  svc-ai-pipeline, svc-score-engine, svc-review, svc-analytics, svc-plagiarism,
  svc-copy-upload, svc-notify
- BFF services (for E2E-10, E2E-11, E2E-13): svc-teacher-bff, svc-student-bff
- Stoody mock (for E2E-12, E2E-13): test-suite/stoody-mock/

### Python Dependencies

```bash
pip install pytest pytest-asyncio nats-py asyncpg minio aiohttp
```

## Running Tests

### All E2E tests

```bash
pytest test-suite/pipeline-tests/ -m e2e -v
```

### Single test file

```bash
pytest test-suite/pipeline-tests/test_e2e_01_stroke_pipeline.py -v
```

### Skip slow tests (E2E-08 full simulation)

```bash
pytest test-suite/pipeline-tests/ -m e2e -v --ignore=test-suite/pipeline-tests/test_e2e_08_full_simulation.py
```

## Configuration

All infrastructure endpoints are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXAMPEN_NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `EXAMPEN_PG_DSN` | `postgresql+asyncpg://exampen:exampen@localhost:5432/exampen` | PostgreSQL DSN |
| `EXAMPEN_MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint |
| `EXAMPEN_MINIO_ACCESS_KEY` | `exampen` | MinIO access key |
| `EXAMPEN_MINIO_SECRET_KEY` | `exampen123` | MinIO secret key |
| `EXAMPEN_MINIO_BUCKET` | `exampen-pages` | MinIO bucket name |
| `EXAMPEN_SCORE_ENGINE_URL` | `http://localhost:8003` | Score engine REST URL |
| `EXAMPEN_REVIEW_URL` | `http://localhost:8005` | Review service REST URL |
| `EXAMPEN_COPY_UPLOAD_URL` | `http://localhost:8006` | Copy upload service REST URL |
| `EXAMPEN_TEACHER_BFF_URL` | `http://localhost:8010` | Teacher BFF REST URL |
| `EXAMPEN_STUDENT_BFF_URL` | `http://localhost:8011` | Student BFF REST URL |
| `EXAMPEN_STOODY_WEBHOOK_URL` | `http://localhost:9090` | Stoody mock webhook receiver |
| `EXAMPEN_EVENT_TIMEOUT` | `30` | Default NATS event wait timeout (seconds) |

## Architecture

Tests use NATS events as the coordination mechanism:

1. **Publish** an upstream event (e.g., `stroke.raw`)
2. **Subscribe** to the expected downstream event (e.g., `stroke.processed`)
3. **Assert** the downstream event arrived with correct data
4. Optionally **verify** side effects in PostgreSQL, MinIO, or REST APIs

All event subscriptions are set up BEFORE publishing to avoid race conditions.
Timeout handling prevents tests from hanging indefinitely.

## Fixture Data

Tests use factories from `conftest.py` and static fixtures from
`test-suite/fixtures/`:

- `fixtures/students.json` - 40 student records
- `fixtures/exams/` - 3 exam definitions with rubrics
- `fixtures/ai_results.json` - AI recognition results
- `fixtures/scores.json` - Score records in various lifecycle states
- `fixtures/objections.json` - 5 objection records
