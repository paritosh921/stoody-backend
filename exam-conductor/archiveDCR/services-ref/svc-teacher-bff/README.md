# svc-teacher-bff

Read-only aggregation layer for the ExamPen teacher UI. This BFF proxies data from multiple backing services into unified responses optimised for teacher-facing screens (class score overview, student drill-down, objection inbox, analytics, plagiarism review).

## Ownership Declaration

- **Writes:** NONE — this service has ZERO write access to any database
- **Reads from:** svc-score-engine, svc-analytics, svc-review, svc-plagiarism, svc-chat, svc-exam-orch, svc-doc-assembly (all via HTTP)
- **Never writes to:** Any database. All mutations are relayed to backing service APIs.
- **Transactional boundaries:** None — pure HTTP aggregation

## Running Locally

```bash
# From the service directory
pip install -e ".[dev]"
uvicorn src.main:app --reload --port 8010
```

## Running Tests

```bash
pytest tests/ -v
```

## Dependencies

| Backing Service | Purpose |
|----------------|---------|
| svc-exam-orch | Exam list, exam detail with roster |
| svc-score-engine | Scores, overrides, finalize, publish |
| svc-doc-assembly | Answer page images, miss indicators |
| svc-analytics | Leaderboard, class stats, question analysis |
| svc-review | Objection inbox, detail, resolve, escalate |
| svc-plagiarism | Plagiarism flags, teacher verdicts |
| svc-chat | Teacher-student messaging |

## What Depends on This

- `teacher-dashboard` (web frontend)
- `exampen-mobile` (teacher view mode)
- Stoody tutor portal (via API-driven native embed)

## API Contract

`new-docs/api/teacher-bff.openapi.yaml`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STOODY_JWKS_URL` | `http://localhost:9100/.well-known/jwks.json` | Stoody JWKS endpoint |
| `SCORE_ENGINE_URL` | `http://localhost:8002` | svc-score-engine base URL |
| `ANALYTICS_URL` | `http://localhost:8003` | svc-analytics base URL |
| `REVIEW_URL` | `http://localhost:8004` | svc-review base URL |
| `PLAGIARISM_URL` | `http://localhost:8005` | svc-plagiarism base URL |
| `CHAT_URL` | `http://localhost:8006` | svc-chat base URL |
| `EXAM_ORCH_URL` | `http://localhost:8007` | svc-exam-orch base URL |
| `DOC_ASSEMBLY_URL` | `http://localhost:8008` | svc-doc-assembly base URL |
| `BACKING_SERVICE_TIMEOUT` | `10` | HTTP timeout (seconds) for backing service calls |
