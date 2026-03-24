# svc-chat

Append-only messaging service for teacher-student exam chat threads. Messages are immutable once written -- no UPDATE, no DELETE at any layer (application or database). This guarantees DPDPA audit safety for minors' data.

## Ownership Declaration

- **Writes:** Chat messages (append-only), read receipts (append-only)
- **Reads from:** `svc-auth` (JWT validation via shared `exampen-common` dependency)
- **Never writes to:** Exam sessions, scores, strokes, objections, analytics, or any other service's database
- **Transactional boundaries:** Single INSERT per message. Single INSERT/UPSERT per read receipt. No UPDATE or DELETE at the database level -- enforced by PostgreSQL triggers.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat/threads/{exam_id}/{other_user_id}` | Append a message to a thread |
| GET | `/api/v1/chat/threads/{exam_id}/{other_user_id}` | Get messages in a thread |
| POST | `/api/v1/chat/threads/{exam_id}/{other_user_id}/read` | Mark thread as read (appends receipt) |

## Running Locally

```bash
# Requires: PostgreSQL with svc-chat schema applied
cd services/svc-chat
pip install -e ".[dev]"
uvicorn src.main:app --port 8011 --reload
```

## Running Tests

```bash
# Unit tests (domain logic -- no I/O, no DB)
pytest tests/test_message_rules.py tests/test_thread_logic.py tests/test_append_only.py -v

# Integration tests (mocked DB and auth)
pytest tests/test_routes_messages.py -v
```

## Dependencies

- `libs/exampen-common-py` (JWT validation, DB pool, RLS, logging)
- PostgreSQL (chat_messages + read_receipts tables)
- `svc-auth` (user identity and role resolution)

## Depended On By

- `svc-teacher-bff` (read-only aggregation of chat threads)
- `svc-student-bff` (read-only aggregation of chat threads)
