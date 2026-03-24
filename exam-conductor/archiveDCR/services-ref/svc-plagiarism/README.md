# svc-plagiarism

Plagiarism detection service for ExamPen. Consumes `plagiarism.check` events
from NATS (triggered when all AI results are ready for an exam), computes
TF-IDF text similarity + structural (Levenshtein) similarity + temporal
correlation + seating proximity for every student pair per question, generates
flags for pairs exceeding the review threshold, and publishes
`plagiarism.result` events.

Teacher verdict persistence (confirmed/dismissed with mandatory reason) is
also owned by this service. Plagiarism detection NEVER auto-penalizes --
teacher review is always required.

## Ownership Declaration

- **Writes:** Plagiarism flags (composite scores, evidence), teacher verdicts
- **Reads from:** `svc-ai-pipeline` (AI-recognized text via shared DB read), exam session metadata (temporal/proximity data)
- **Never writes to:** scores, objections, strokes, exam lifecycle state, AI results
- **Transactional boundaries:** Flags written in bulk per exam after all AI results ready. Teacher verdicts update the same service-owned row. NATS event published after PG commit.

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
| `DATABASE_URL` | `postgresql://localhost:5432/exampen_plagiarism` | PostgreSQL connection string |
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `REVIEW_THRESHOLD` | `0.75` | Composite score threshold for review_recommended |
| `STRONG_THRESHOLD` | `0.90` | Composite score threshold for strong_match |
| `MOCK_MODE` | `false` | Enable mock mode (no real DB/NATS connections) |

## Dependencies

- `svc-ai-pipeline` -- provides AI-recognized text (read-only)
- PostgreSQL -- flag and verdict storage
- NATS JetStream -- event consumption and publication

## Depended On By

- `svc-teacher-bff` -- reads flags and verdicts for teacher display
- `svc-score-engine` -- reads flags as context (does not modify)
