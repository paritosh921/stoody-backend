# ExamPen Load Tests & Tuning

Performance and load testing suite for ExamPen, targeting the A8.4 peak-load scenario: 10,000 students uploading 336KB of stroke data each (3.3GB burst) while teachers and students concurrently query scores.

## Quick Start

### Prerequisites

```bash
# Locust (interactive load testing)
pip install locust>=2.20

# k6 (CI load testing)
# macOS:  brew install k6
# Linux:  sudo snap install k6
# Windows: choco install k6
# Docker:  docker run --rm -i grafana/k6 run - <k6_script.js
```

### Running Locust (Interactive)

```bash
cd test-suite/load-tests

# Web UI mode — opens http://localhost:8089
locust -f locustfile.py --host http://localhost:8080

# Headless mode — for scripted runs
locust -f locustfile.py --host http://localhost:8080 \
  --headless -u 500 -r 50 --run-time 5m \
  --csv results/run

# Single scenario via tags
locust -f locustfile.py --host http://localhost:8080 \
  --headless --tags stroke -u 250 -r 25 --run-time 3m

locust -f locustfile.py --host http://localhost:8080 \
  --headless --tags teacher -u 500 -r 50 --run-time 3m

locust -f locustfile.py --host http://localhost:8080 \
  --headless --tags student -u 5000 -r 200 --run-time 5m
```

### Running k6 (CI)

```bash
cd test-suite/load-tests

# Mixed workload (default)
k6 run k6_script.js

# Single scenario
k6 run k6_script.js --env SCENARIO=stroke_burst
k6 run k6_script.js --env SCENARIO=teacher_scores
k6 run k6_script.js --env SCENARIO=student_scores

# With custom target
k6 run k6_script.js --env BASE_URL=https://staging.exampen.example.com

# Export results to JSON
k6 run k6_script.js --out json=results/k6_output.json

# Export to Prometheus (for Grafana dashboards)
k6 run k6_script.js --out experimental-prometheus-rw
```

## Scenarios

### 1. Stroke Ingestion Burst

Simulates the post-exam upload burst described in FAILURE_MITIGATION_REGISTER.md A8.4.

| Parameter | Value |
|-----------|-------|
| Target students | 10,000 |
| Data per student | 336 KB (40 chunks x 8.4 KB) |
| Total burst | 3.3 GB |
| Endpoint | `POST /api/v1/strokes/ingest` |
| Virtual users | 250 (each = 1 hub uploading for ~40 pens) |
| Expected throughput | >5,000 chunks/sec |

Each chunk contains 600 coordinate frames of 14 bytes each (P05 pen format: bookType, pageNo, X, Y, pressure, penProp, timestamp).

### 2. Teacher Score Query

Simulates 500 concurrent teachers viewing class score overviews.

| Endpoint | Weight |
|----------|--------|
| `GET /api/v1/teacher/exams/{exam_id}/scores` | 60% |
| `GET /api/v1/teacher/exams/{exam_id}/scores/{student_id}` | 20% |
| `GET /api/v1/teacher/exams` | 20% |

### 3. Student Portal

Simulates 5,000 concurrent students checking their scores.

| Endpoint | Weight |
|----------|--------|
| `GET /api/v1/student/exams/{exam_id}/scores` | 55% |
| `GET /api/v1/student/exams` | 22% |
| `GET /api/v1/student/exams/{exam_id}/answers/{question_id}` | 11% |
| `GET /api/v1/student/performance` | 11% (20% of users) |

### 4. Mixed Workload

Runs all three scenarios simultaneously with weighted user distribution:

| Scenario | Weight | Approximate % of VUs |
|----------|--------|---------------------|
| Stroke ingestion | 5 | 29% |
| Teacher scores | 2 | 12% |
| Student scores | 10 | 59% |

## Performance Budgets

Tests fail if any threshold is violated.

| Metric | Threshold |
|--------|-----------|
| Stroke ingest p95 latency | < 2 seconds |
| Stroke ingest p99 latency | < 5 seconds |
| Teacher score p95 latency | < 2 seconds |
| Teacher score p99 latency | < 3 seconds |
| Student score p95 latency | < 2 seconds |
| Student score p99 latency | < 3 seconds |
| Overall error rate | < 1% |
| Stroke ingest error rate | < 1% |

Override Locust thresholds via environment variables:

```bash
THRESHOLD_STROKE_P95=3.0 THRESHOLD_SCORE_P95=2.5 \
  locust -f locustfile.py --host http://localhost:8080 --headless ...
```

## Authentication

By default, both tools generate mock JWTs that are structurally valid but not cryptographically signed. These work when the target services are configured with `MOCK_MODE=true`.

For testing against real services, provide valid tokens:

```bash
# Locust — comma-separated token lists
export EXAMPEN_TEACHER_TOKENS="eyJ...,eyJ...,eyJ..."
export EXAMPEN_STUDENT_TOKENS="eyJ...,eyJ...,eyJ..."

# k6 — single token per role
k6 run k6_script.js \
  --env TEACHER_TOKEN=eyJ... \
  --env STUDENT_TOKEN=eyJ... \
  --env HUB_TOKEN=eyJ...
```

## Seed Fixtures

Load tests use the seed fixtures from `test-suite/fixtures/` when available. Generate them first:

```bash
python scripts/seed_data.py --students 40 --exams 3 --questions-per-exam 10
```

If fixtures are not found, the tests generate synthetic IDs automatically.

## Interpreting Results

### Locust

- **Web UI**: The Locust web UI at `http://localhost:8089` shows real-time charts for RPS, response times, and failure rates.
- **CSV output**: With `--csv results/run`, Locust writes `results/run_stats.csv`, `results/run_stats_history.csv`, and `results/run_failures.csv`.
- **Performance budget**: On exit, the test checks p95 latencies and error rates against thresholds. A non-zero exit code indicates a budget violation.

### k6

- **Console summary**: k6 prints a summary table with p50/p90/p95/p99 latencies.
- **Thresholds**: k6 exits with code 99 if any threshold is violated. CI pipelines should treat this as a test failure.
- **Custom metrics**: Look for `stroke_ingest_duration`, `teacher_score_duration`, and `student_score_duration` in the summary.
- **JSON output**: Use `--out json=results.json` and analyze with jq or import into Grafana.

### Key metrics to watch

1. **Throughput (chunks/sec)**: During stroke burst, target >5,000 chunks/sec sustained.
2. **p95 latency**: Must stay below 2s for all endpoints.
3. **Error rate**: Must stay below 1%. 409 (deduplicated) responses from stroke ingest are counted as successes.
4. **NATS consumer lag**: Monitor via `http://nats-host:8222/jsz` during the test. Lag >10s indicates under-provisioned consumers.

## Tuning Configurations

The `tuning/` directory contains recommended configurations for production deployments.

### `tuning/nats-jetstream.conf`

NATS JetStream server and stream/consumer tuning for burst load handling.

Key settings:
- 5GB memory, 10GB file storage for JetStream
- `STROKE_RAW` stream: 8GB max, 24h retention, 3-way replication
- `stroke-proc-worker` consumer: 30s ack wait, 1000 max pending, flow control enabled
- Monitoring alert thresholds for consumer lag and storage usage

### `tuning/postgresql.conf`

PostgreSQL tuning for the write-heavy exam workload on a 32GB/8-core server.

Key settings:
- 8GB shared_buffers, 24GB effective_cache_size
- WAL: 64MB buffers, 8GB max_wal_size, lz4 compression
- 15-minute checkpoint interval, 0.9 completion target
- JIT disabled (OLTP workload), SSD-optimized random_page_cost
- PgBouncer recommendations for connection pooling

### `tuning/timescaledb.conf`

TimescaleDB-specific tuning for the strokes hypertable.

Key settings:
- 1-hour chunk interval (optimal for burst writes)
- Compression: segmentby exam_id/student_id, compress after 2 hours
- Expected 10-15x compression ratio
- 2-year retention policy (DPDPA compliance)
- Continuous aggregate for stroke counts (5-minute refresh)
- Index recommendations for query patterns and RLS

### Applying tuning configs

These configs are **recommendations**, not drop-in files. To apply:

1. Start with your base PostgreSQL/TimescaleDB config (e.g., from pgtune).
2. Merge the relevant settings from `tuning/postgresql.conf` and `tuning/timescaledb.conf`.
3. For NATS, apply stream and consumer settings via `nats` CLI or your IaC tool (Terraform, Helm).
4. Run load tests to validate.
5. Iterate: adjust settings, re-run, compare.

## CI Integration

### GitHub Actions example

```yaml
load-test:
  runs-on: ubuntu-latest
  needs: [deploy-staging]
  steps:
    - uses: actions/checkout@v4

    - name: Install k6
      run: |
        sudo gpg -k
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
          --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D68
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" \
          | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update && sudo apt-get install k6

    - name: Run load tests
      run: |
        k6 run test-suite/load-tests/k6_script.js \
          --env BASE_URL=${{ secrets.STAGING_URL }} \
          --env TEACHER_TOKEN=${{ secrets.LT_TEACHER_TOKEN }} \
          --env STUDENT_TOKEN=${{ secrets.LT_STUDENT_TOKEN }} \
          --env HUB_TOKEN=${{ secrets.LT_HUB_TOKEN }} \
          --out json=load-test-results.json

    - name: Upload results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: load-test-results
        path: load-test-results.json
```

## Directory Structure

```
load-tests/
├── locustfile.py               # Locust scenarios (interactive)
├── k6_script.js                # k6 scenarios (CI)
├── tuning/
│   ├── nats-jetstream.conf     # NATS JetStream tuning
│   ├── postgresql.conf         # PostgreSQL write-heavy tuning
│   └── timescaledb.conf        # TimescaleDB hypertable tuning
└── README.md                   # This file
```
