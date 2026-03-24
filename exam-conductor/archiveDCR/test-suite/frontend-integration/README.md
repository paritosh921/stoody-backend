# Frontend <-> BFF Integration Tests (W6.A2 / L5)

Integration tests that verify all three ExamPen web frontends against real BFF
services. Unlike unit or component tests, these tests run against a fully
operational backend stack --- no mocks.

## Scope

| Frontend | BFF | Key Flows |
|---|---|---|
| `teacher-dashboard` (port 5173) | `svc-teacher-bff` | Exam CRUD, score review, objection inbox, analytics |
| `student-portal` (port 5174) | `svc-student-bff` | Score summary, question breakdown, objection filing, chat |
| `invigilator-console` (port 5175) | `svc-invig-console` | Session list, WebSocket dashboard, pen grid |

## Approach

- **Framework**: [Playwright](https://playwright.dev/) with TypeScript.
- **Real services**: Tests assume the full Docker Compose stack is running,
  including all BFF services, backing services, NATS, PostgreSQL, and the
  Stoody mock.
- **Seeded data**: The seed script (`scripts/seed-data.sh`) must be run before
  the suite so that exams, students, scores, and objections exist.

### What the tests verify

1. **CRUD flows** -- create, read, update, delete operations through the UI
   that hit the real BFF and backing services.
2. **RBAC enforcement** -- a student token cannot access teacher routes (must
   redirect or receive 403). Tested via injected auth cookies/storage.
3. **WebSocket real-time updates** -- the invigilator console receives live
   status updates over WebSocket.
4. **Data correctness** -- rendered data matches what the BFF returns (spot
   checks via API comparison).

## Prerequisites

### 1. Infrastructure

```bash
# Start all services
docker compose -f infra/docker-compose.yml up -d

# Seed test data
./scripts/seed-data.sh --students 40 --exams 3 --questions-per-exam 10
```

### 2. Playwright install

```bash
cd test-suite/frontend-integration
npm install
npx playwright install --with-deps chromium firefox
```

### 3. Frontend dev servers (or use built assets)

```bash
# Option A: dev servers
cd frontend/teacher-dashboard && npm run dev &
cd frontend/student-portal   && npm run dev &
cd frontend/invigilator-console && npm run dev &

# Option B: build and serve via BFF (Traefik static-file routes)
# Ensure infra/traefik/ routes are configured.
```

## Running Tests

```bash
# All integration tests
npx playwright test

# Single frontend
npx playwright test tests/teacher-dashboard.spec.ts

# Headed mode for debugging
npx playwright test --headed

# Generate HTML report
npx playwright test --reporter=html
```

## Configuration

All base URLs are configurable via environment variables (see
`playwright.config.ts`). Defaults assume local dev servers.

| Variable | Default | Description |
|---|---|---|
| `TEACHER_BASE_URL` | `http://localhost:5173` | Teacher dashboard URL |
| `STUDENT_BASE_URL` | `http://localhost:5174` | Student portal URL |
| `INVIG_BASE_URL` | `http://localhost:5175` | Invigilator console URL |
| `TEACHER_BFF_URL` | `http://localhost:8010` | Teacher BFF for API calls |
| `STUDENT_BFF_URL` | `http://localhost:8011` | Student BFF for API calls |
| `INVIG_BFF_URL` | `http://localhost:8090` | Invigilator console service |

## Test IDs

Tests reference the following TEST_SUITE_SPEC identifiers:

- I-BFF-T01, I-BFF-T02 (teacher BFF aggregation, RBAC)
- I-BFF-S01, I-BFF-S02, I-BFF-S03 (student BFF score view, objection, parent)
- I-INVIG-01 (WebSocket status feed)
- E2E-10, E2E-11 (teacher BFF aggregation, student objection lifecycle)

## File Structure

```
frontend-integration/
  playwright.config.ts    -- Playwright configuration for all 3 frontends
  package.json            -- Dependencies
  README.md               -- This file
  tests/
    teacher-dashboard.spec.ts
    student-portal.spec.ts
    invigilator-console.spec.ts
```
