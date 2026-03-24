# svc-student-bff

ExamPen student BFF (Backend For Frontend) -- read-only aggregator for student and parent views. Aggregates data from backing services (svc-score-engine, svc-review, svc-analytics, svc-chat) and relays mutations to the owning service.

## Ownership Declaration

- **Writes:** NONE. This service has ZERO database access.
- **Reads from:** `svc-score-engine` (scores, answer insights), `svc-review` (objection status), `svc-analytics` (performance history, trends, strengths), `svc-chat` (message threads), Stoody API (parent-child resolution)
- **Relays mutations to:** `svc-review` (file objection), `svc-chat` (send message)
- **Never writes to:** Any database. All mutations go through backing service APIs.
- **Transactional boundaries:** None. Read-only aggregator.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/student/exams/{exam_id}/score` | Score summary (total, percentage, percentile) |
| GET | `/api/v1/student/exams/{exam_id}/questions` | Question-wise breakdown |
| GET | `/api/v1/student/exams/{exam_id}/questions/{qid}/answer` | Answer image + AI analysis |
| POST | `/api/v1/student/exams/{exam_id}/objections` | File objection (relay to svc-review) |
| GET | `/api/v1/student/objections` | List own objections |
| GET | `/api/v1/student/objections/{id}` | Objection status + resolution |
| GET | `/api/v1/student/performance/history` | Score history across exams |
| GET | `/api/v1/student/performance/trends` | Trend data for charts |
| GET | `/api/v1/student/performance/strengths` | AI-generated strengths/weaknesses |
| GET | `/api/v1/student/exams/{exam_id}/chat/{teacher_id}` | Chat thread messages |
| POST | `/api/v1/student/exams/{exam_id}/chat/{teacher_id}` | Send message (relay to svc-chat) |

## RBAC

- **Student**: sees own data only. Can file objections and send chat messages.
- **Parent**: sees linked children's data only (resolved via Stoody API). Read-only -- cannot file objections or send messages.
- **Teacher/Admin**: 403 Forbidden (must use svc-teacher-bff instead).

## Running Locally

```bash
cd services/svc-student-bff
pip install -e ".[dev]"
uvicorn src.main:app --port 8010 --reload
```

## Running Tests

```bash
# All tests (mocked backing services)
pytest tests/ -v

# Score route tests
pytest tests/test_routes_scores.py -v

# Parent scope tests
pytest tests/test_parent_scope.py -v

# RBAC tests
pytest tests/test_rbac.py -v
```

## Dependencies

- `libs/exampen-common-py` (JWT validation, logging)
- Stoody platform (JWKS endpoint, parent-child API) or `stoody-mock`
- svc-score-engine, svc-review, svc-analytics, svc-chat (backing services)

## Depended On By

- Stoody student portal frontend (consumes BFF API)
- Stoody parent portal frontend (consumes BFF API)
- ExamPen mobile student app (consumes BFF API)
