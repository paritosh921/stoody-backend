# svc-score-engine

Event-sourced scoring service for ExamPen. Consumes AI recognition results, evaluates them against versioned rubrics, stores every score mutation as an immutable event, and exposes HTTP endpoints for teacher overrides and exam-level workflow (finalize, publish, lock).

## Ownership Declaration

- **Writes:** `score_events` (append-only), `score_materialized` (projection), `rubrics` (versioned)
- **Reads from:** `svc-ai-pipeline` (via NATS `ai.result` events), own PostgreSQL tables
- **Never writes to:** any other service's database, objection state, plagiarism flags, analytics
- **Transactional boundaries:** event append + materialized view update in same PG transaction; NATS publish AFTER commit

## Run Locally

```bash
# Install dependencies
pip install -e ".[dev]"

# Run service
uvicorn src.main:app --reload

# Run unit tests (domain-only, no I/O)
pytest tests/ -m unit

# Run all tests
pytest tests/
```

## Run via Docker

```bash
docker build -t svc-score-engine .
docker run -p 8000:8000 \
  -e SCORE_ENGINE_DATABASE_URL=postgresql+asyncpg://score:score@host:5432/exampen_scores \
  -e SCORE_ENGINE_NATS_URL=nats://host:4222 \
  svc-score-engine
```

## Depends On

- PostgreSQL (score_events, score_materialized, rubrics tables)
- NATS JetStream (consumes `ai.result`, publishes `score.updated`)
- `svc-ai-pipeline` (produces AI results)

## Depended On By

- `svc-review` (reads scores for objection context)
- `svc-analytics` (consumes `score.updated` events)
- `svc-teacher-bff` / `svc-student-bff` (read aggregation)
