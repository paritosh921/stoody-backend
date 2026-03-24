# ExamPen Infrastructure — Local Dev Stack

## Prerequisites

- Docker Engine 24+ with Compose v2
- 8 GB RAM minimum (PostgreSQL + TimescaleDB + NATS + MinIO + Redis + monitoring)

## Quick Start

```bash
# 1. Create .env from template
cp .env.example .env

# 2. Start infrastructure only
docker compose up -d

# 2b. Start infrastructure + monitoring
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# 3. Verify all healthy
docker compose ps
```

## Ports

| Service       | Port  | Purpose              |
|---------------|-------|----------------------|
| PostgreSQL    | 5432  | Database             |
| NATS          | 4222  | Client connections   |
| NATS Monitor  | 8222  | HTTP monitoring      |
| MinIO API     | 9000  | S3-compatible API    |
| MinIO Console | 9001  | Web UI               |
| Redis         | 6379  | Cache / rate limit   |
| Traefik       | 80    | HTTP reverse proxy   |
| Traefik       | 8080  | Dashboard            |
| Prometheus    | 9090  | Metrics              |
| Grafana       | 3000  | Dashboards           |
| Loki          | 3100  | Log aggregation      |
| Tempo         | 3200  | Distributed tracing  |

## Compose Files

| File                             | Purpose                           |
|----------------------------------|-----------------------------------|
| `docker-compose.yml`            | Core infra (PG, NATS, MinIO, Redis, Traefik) |
| `docker-compose.monitoring.yml` | Prometheus, Grafana, Loki, Tempo, Promtail |
| `docker-compose.services.yml`   | Backend services 1-13 (commented) |
| `docker-compose.bff.yml`        | BFF services 14-16 (commented)    |
| `docker-compose.test.yml`       | Ephemeral test isolation stack    |

### Running with services

```bash
# Infrastructure only
docker compose up -d

# Infrastructure + monitoring
docker compose -f docker-compose.yml \
  -f docker-compose.monitoring.yml up -d

# Infrastructure + monitoring + a specific service (uncomment it first)
docker compose -f docker-compose.yml \
  -f docker-compose.monitoring.yml \
  -f docker-compose.services.yml up -d

# All layers
docker compose -f docker-compose.yml \
  -f docker-compose.monitoring.yml \
  -f docker-compose.services.yml \
  -f docker-compose.bff.yml up -d
```

### Running tests

```bash
# Start ephemeral test stack (separate ports, no volumes)
docker compose -f docker-compose.yml -f docker-compose.test.yml up -d

# Run tests against test stack
DATABASE_HOST=localhost DATABASE_PORT=5433 \
NATS_URL=nats://localhost:4223 \
pytest services/svc-score-engine/tests/ -m integration

# Tear down
docker compose -f docker-compose.yml -f docker-compose.test.yml down
```

## Monitoring

Grafana auto-provisions three datasources (Prometheus, Loki, Tempo) and three dashboards:

- **API Performance** — request latency heatmap, req/sec, error rate, slow endpoints
- **Pipeline Health** — NATS consumer lag, stroke ingestion, page assembly, AI processing, scoring
- **Hub Fleet** — hub connectivity, dongle health, sync progress, upload status

Default login: `admin` / `admin`

### Alerts (Prometheus)

| Alert                       | Condition                   |
|-----------------------------|-----------------------------|
| HighErrorRate               | 5xx > 5% for 2m            |
| HighLatency                 | p99 > 2s for 5m            |
| NATSConsumerLag             | Significant pending lag     |
| DatabaseConnectionExhausted | Connections > 90% for 2m   |
| HubOffline                  | Health check failing > 5m  |
| StrokeIngestionStalled      | 0 throughput with pending   |
| AIProcessingSlow            | p95 > 30s for 10m          |

## Databases Created

The `init-db.sql` script creates 10 service databases with the `exampen` user:

`exampen_auth`, `exampen_exam`, `exampen_stroke` (+ TimescaleDB), `exampen_score`,
`exampen_review`, `exampen_analytics`, `exampen_plagiarism`, `exampen_chat`,
`exampen_copy`, `exampen_notify`

## MinIO Buckets

Auto-created on first start: `exampen-pages`, `exampen-copies`

## Resetting

```bash
# Stop and remove volumes (full reset)
docker compose down -v
```
