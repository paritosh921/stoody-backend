# svc-analytics

Percentiles, leaderboards, and class statistics for ExamPen exams. Subscribes to `score.updated` NATS events from svc-score-engine and recomputes materialized analytics views. svc-analytics is the ONLY writer of percentile data (per STATE_OWNERSHIP_MAP.md).

## Ownership Declaration

- **Writes:** Exam percentiles, leaderboard cache, score cache (event-driven), question response cache
- **Reads from:** `svc-score-engine` (via `score.updated` NATS events), `svc-auth` (JWT validation via shared `exampen-common` dependency)
- **Never writes to:** Scores, exam sessions, strokes, objections, chat, or any other service's database
- **Transactional boundaries:** Percentile recomputation is DELETE + INSERT in a single transaction (idempotent). Leaderboard recomputation follows the same pattern. Score cache upserts are single-row UPSERT operations.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/analytics/exams/{exam_id}/leaderboard` | Leaderboard rows for an exam |
| GET | `/api/v1/analytics/exams/{exam_id}/class-stats` | Class-level statistics |
| GET | `/api/v1/analytics/students/{student_id}/performance` | Cross-exam student trend |
| GET | `/api/v1/analytics/exams/{exam_id}/student/{student_id}` | Per-student exam performance |
| GET | `/api/v1/analytics/exams/{exam_id}/questions` | Question-wise difficulty analysis |

## Running Locally

```bash
# Requires: PostgreSQL with svc-analytics schema applied, NATS JetStream
cd services/svc-analytics
pip install -e ".[dev]"
uvicorn src.main:app --port 8009 --reload
```

## Running Tests

```bash
# Unit tests (domain logic -- no I/O, no DB)
pytest tests/test_percentile.py tests/test_leaderboard.py tests/test_class_stats.py -v
```

## Dependencies

- `libs/exampen-common-py` (JWT validation, DB pool, RLS, logging)
- PostgreSQL (exam_percentiles, leaderboard_cache, exam_score_cache, question_response_cache tables)
- NATS JetStream (score.updated event subscription)
- `svc-score-engine` (event source for score changes)

## Depended On By

- `svc-teacher-bff` (read-only aggregation of analytics)
- `svc-student-bff` (read-only aggregation of analytics)
