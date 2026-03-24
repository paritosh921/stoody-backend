# svc-review

ExamPen objection lifecycle service. Manages the full objection cycle: filing by students, assignment to evaluators, resolution (approve/reject), and escalation to HOD or senior evaluators.

## Ownership Declaration

- **Writes:** Objection state (FSM: `filed -> assigned -> reviewing -> resolved | escalated`)
- **Reads from:** `svc-score-engine` (score context for objection detail), `svc-auth` (JWT validation)
- **Never writes to:** Scores, exam sessions, strokes, chat, analytics, or any other service's database
- **Transactional boundaries:** State transitions use `SELECT ... FOR UPDATE` for single-writer locking. Objection resolution triggers re-score command to `svc-score-engine` via NATS (PG write first, NATS publish after commit).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/objections` | File a new objection (student only, during objection window) |
| GET | `/api/v1/objections` | List objections (filter by exam_id, status) |
| GET | `/api/v1/objections/{id}` | Get objection detail with context |
| POST | `/api/v1/objections/{id}/assign` | Assign objection to an evaluator |
| POST | `/api/v1/objections/{id}/resolve` | Approve (triggers re-score via NATS) or reject (mandatory reason) |
| POST | `/api/v1/objections/{id}/escalate` | Escalate to HOD or senior evaluator |

## Running Locally

```bash
# Requires: PostgreSQL, NATS, stoody-mock running on port 9100
cd services/svc-review
pip install -e ".[dev]"
uvicorn src.main:app --port 8007 --reload
```

## Running Tests

```bash
# Unit tests (domain logic -- no I/O, no DB)
pytest tests/test_objection_fsm.py tests/test_objection_rules.py -v

# Integration tests (mocked DB, NATS, and auth)
pytest tests/test_routes.py -v
```

## Dependencies

- `libs/exampen-common-py` (JWT validation, DB pool, NATS client, logging)
- PostgreSQL (objections table)
- NATS JetStream (objection events, re-score commands)
- Stoody platform (JWKS endpoint) or `stoody-mock` for development

## Depended On By

- `svc-score-engine` (reads re-score commands from NATS)
- `svc-student-bff` (reads objection status for student portal)
- `svc-teacher-bff` (reads objection inbox for teacher dashboard)
