# ExamPen Scripts

Development and CI/CD utility scripts.

## Scripts

### seed-data.sh — Test Data Generator

Generates realistic test data for development and testing environments.

```bash
# Default: 40 students, 3 exams, 10 questions/exam
./scripts/seed-data.sh

# Custom parameters
./scripts/seed-data.sh --students 40 --exams 3 --questions-per-exam 10

# Different random seed for data variety
./scripts/seed-data.sh --seed 99

# Generate and load into PostgreSQL
./scripts/seed-data.sh --load-sql --db-url postgresql://user:pass@localhost:5432/exampen
```

**Output** (in `test-suite/fixtures/`):
- Binary stroke files in P05 14-byte frame format
- JSON exam definitions with rubrics
- JSON student/tutor/score/objection records
- Known plagiarism pairs
- BLE simulator configuration
- Idempotent PostgreSQL seed SQL

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--students N` | 40 | Number of students |
| `--exams N` | 3 | Number of exams |
| `--questions-per-exam N` | 10 | Questions per exam |
| `--seed N` | 42 | Random seed (deterministic output) |
| `--output DIR` | `test-suite/fixtures/` | Output directory |
| `--load-sql` | off | Load seed.sql into database |
| `--db-url URL` | — | PostgreSQL URL (required with --load-sql) |

### generate-mocks.sh — Mock Server Generation

Generates `infra/docker-compose.mock.yml` from all OpenAPI specs in `new-docs/api/`. Each spec gets a [Stoplight Prism](https://stoplight.io/open-source/prism) mock server on a dedicated port.

```bash
# Generate the docker-compose.mock.yml file
./scripts/generate-mocks.sh

# Start all 12 mock servers
docker compose -f infra/docker-compose.mock.yml up

# Start a single mock (e.g., auth on port 4010)
docker compose -f infra/docker-compose.mock.yml up mock-auth

# Test a mock endpoint
curl http://localhost:4010/api/v1/auth/introspect
```

**Port mapping:**

| Service | Port | OpenAPI Spec |
|---------|------|--------------|
| auth | 4010 | auth.openapi.yaml |
| exam-orch | 4011 | exam-orch.openapi.yaml |
| stroke-ingest | 4012 | stroke-ingest.openapi.yaml |
| score-engine | 4013 | score-engine.openapi.yaml |
| review | 4014 | review.openapi.yaml |
| analytics | 4015 | analytics.openapi.yaml |
| plagiarism | 4016 | plagiarism.openapi.yaml |
| chat | 4017 | chat.openapi.yaml |
| copy-upload | 4018 | copy-upload.openapi.yaml |
| teacher-bff | 4019 | teacher-bff.openapi.yaml |
| student-bff | 4020 | student-bff.openapi.yaml |
| invig-console | 4021 | invig-console.openapi.yaml |

**Requires:** Docker with Compose v2.

### validate-contracts.sh — Contract Validation

Validates all OpenAPI YAML specs and event JSON Schemas for structural correctness.

```bash
# Validate all contracts
./scripts/validate-contracts.sh
```

**Checks performed:**
- OpenAPI specs (`new-docs/api/*.openapi.yaml`): valid YAML, has `openapi`, `info`, and `paths` keys, version starts with `3.`
- Event schemas (`new-docs/contracts/events/*.schema.json`): valid JSON, has `type` or `$schema` key

**Requires:** Python 3.8+. Uses `pyyaml` if installed (`pip install pyyaml`) for full YAML parsing; falls back to regex-based structural checks otherwise. Exit code 1 on any invalid file.

**Makefile targets** (suggested):
```makefile
mock:
	./scripts/generate-mocks.sh
	docker compose -f infra/docker-compose.mock.yml up

validate-contracts:
	./scripts/validate-contracts.sh
```

### pre-commit-check.sh — Pre-Commit Validation

Enforces code quality rules before commit:

1. **File size limits** — Python (300), TypeScript (250), SQL (200), Config (150) lines
2. **Domain purity** — No I/O imports (asyncio, aiohttp, sqlalchemy, nats, httpx, requests) in `*/domain/*.py`
3. **Cross-service imports** — No service importing from another service's `src/`

```bash
# Check staged files only
./scripts/pre-commit-check.sh

# Check all tracked files
./scripts/pre-commit-check.sh --all
```

**Install as git hook:**
```bash
ln -sf ../../scripts/pre-commit-check.sh .git/hooks/pre-commit
```

**Exemptions:** Add `# EXEMPT: <reason>` (Python) or `// EXEMPT: <reason>` (TypeScript) in the first 5 lines of a file to bypass the size limit check.

## seed_data.py — Python Seed Generator (Internal)

Called by `seed-data.sh`. Not intended to be run directly, but can be:

```bash
python scripts/seed_data.py --students 40 --exams 3 --questions-per-exam 10 --seed 42 --output test-suite/fixtures
```

Requires Python 3.12+. No external dependencies (stdlib only).
