# svc-invig-console

Real-time WebSocket invigilator dashboard backend for ExamPen. Subscribes to hub status events via NATS, proxies exam session data from svc-exam-orch, and pushes 1 Hz dashboard snapshots to connected invigilator clients.

## Ownership Declaration

- **Writes:** Nothing -- this service owns zero authoritative state
- **Reads from:** `svc-exam-orch` (exam session data via REST), hub status relay (NATS events)
- **Never writes to:** Exam tables, score tables, stroke tables, or any database
- **Transactional boundaries:** None -- pure read/relay service

## Run Locally

```bash
# With Docker Compose (from repo root)
docker compose up svc-invig-console

# Direct (requires NATS + svc-exam-orch running)
cd services/svc-invig-console
pip install -e ".[dev]"
uvicorn src.main:app --reload --port 8010
```

## Run Tests

```bash
cd services/svc-invig-console
pytest tests/ -v

# Unit tests only (domain layer, no I/O)
pytest tests/test_status_aggregator.py -v

# Integration tests (WebSocket with mocked infrastructure)
pytest tests/test_websocket.py -v
```

## API Contract

`new-docs/api/invig-console.openapi.yaml`

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/invigilator/sessions` | List active exam sessions |
| GET | `/api/v1/invigilator/sessions/{exam_id}` | Session detail with pen count, sync progress |
| GET | `/api/v1/invigilator/sessions/{exam_id}/sync` | Per-pen sync progress |
| GET | `/api/v1/invigilator/sessions/{exam_id}/dongles` | Dongle health and capacity |

### WebSocket

| Path | Description |
|------|-------------|
| WS `/api/v1/invigilator/ws` | Live dashboard updates (1 Hz snapshots) |

WebSocket protocol:
1. Authenticate via `?token=<jwt>` query param or first message `{"type": "auth", "token": "<jwt>"}`
2. Subscribe: `{"type": "subscribe", "exam_id": "<uuid>"}`
3. Receive: `{"event_type": "session.snapshot", "payload": {...}}` at 1 Hz

## Depends On

- `libs/exampen-common-py` (auth, NATS client, logging)
- `svc-exam-orch` (exam session data via REST)
- NATS JetStream (hub status relay events)

## Depended On By

- `frontend/invigilator-console` (WebSocket consumer)
- `mobile/exampen-mobile` (when in hub-control mode)

## Validation Levels Achieved

- L1: Docker image builds
- L3: Unit tests (domain status aggregator)
- L4: Integration tests (WebSocket with mocked infrastructure)
