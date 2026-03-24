# svc-exam-orch

Exam orchestrator service for ExamPen. Manages exam lifecycle (FSM), pen-student bindings, invigilator/evaluator assignments, and scheduling. This is the authoritative owner of exam session state and server-confirmed pen bindings.

## Ownership Declaration

- **Writes:** Exam session lifecycle (FSM), pen-student bindings (authoritative), invigilator/evaluator assignments
- **Reads from:** Stoody platform (student roster, tutor list, class/subject data via REST), `svc-auth` (JWT validation via shared lib)
- **Never writes to:** Score tables, stroke tables, objection tables, chat tables, analytics tables
- **Transactional boundaries:** Exam state transitions use `SELECT ... FOR UPDATE` row-level locking. NATS `exam.lifecycle` events published AFTER PostgreSQL commit.

## Run Locally

```bash
# With Docker Compose (from repo root)
docker compose up svc-exam-orch

# Direct (requires PostgreSQL + NATS running)
cd services/svc-exam-orch
pip install -e ".[dev]"
uvicorn src.main:app --reload --port 8002
```

## Run Tests

```bash
cd services/svc-exam-orch
pytest tests/ -v

# Unit tests only (domain layer, no I/O)
pytest tests/test_exam_fsm.py tests/test_exam_models.py tests/test_binding_logic.py -v

# Integration tests (route handlers with mocked DB/NATS)
pytest tests/test_routes_exams.py tests/test_routes_bindings.py -v
```

## Depends On

- `libs/exampen-common-py` (auth, DB pool, NATS client, logging)
- Stoody platform APIs (student roster, tutor list) — works with mock in `MOCK_MODE`
- PostgreSQL (exam state, bindings, assignments)
- NATS JetStream (lifecycle event publishing)

## Depended On By

- `svc-stroke-ingest` (reads exam state as ingestion gate)
- `svc-invig-console` (reads exam state for display)
- `svc-teacher-bff` (reads exam data for teacher dashboard)
- Hub (receives commands derived from exam lifecycle)

## API Contract

`new-docs/api/exam-orch.openapi.yaml`

## Validation Levels Achieved

- L1: Docker image builds
- L3: Unit tests (domain FSM, models, binding logic)
- L4: Integration tests (routes with mocked infrastructure)
