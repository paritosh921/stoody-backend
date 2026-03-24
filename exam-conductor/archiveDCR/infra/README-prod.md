# ExamPen -- Production Deployment Guide

## Prerequisites

- Docker Engine 24+ with Compose V2
- Linux host (Ubuntu 22.04 LTS recommended), 8+ CPU cores, 32 GB RAM, 500 GB SSD
- Domain name with DNS A record pointing to the server
- Port 80 and 443 open for Let's Encrypt ACME challenge and HTTPS traffic
- S3-compatible storage account for pgBackRest backups (AWS S3 or self-hosted MinIO)

## 1. Secret Management

All sensitive values are Docker secrets, never stored in environment variables or compose files.

### Create secrets directory (file-based mode)

```bash
mkdir -p infra/secrets && chmod 700 infra/secrets

# Generate strong passwords (example using openssl)
openssl rand -base64 32 > infra/secrets/postgres_password.txt
openssl rand -base64 32 > infra/secrets/redis_password.txt
openssl rand -base64 32 > infra/secrets/minio_root_password.txt
openssl rand -base64 32 > infra/secrets/nats_password.txt
openssl rand -base64 64 > infra/secrets/jwt_signing_key.txt
openssl rand -base64 32 > infra/secrets/smtp_password.txt
openssl rand -base64 32 > infra/secrets/webhook_secret.txt
openssl rand -base64 32 > infra/secrets/stoody_webhook_secret.txt
openssl rand -base64 20 > infra/secrets/s3_access_key.txt
openssl rand -base64 40 > infra/secrets/s3_secret_key.txt

# Lock permissions
chmod 400 infra/secrets/*.txt
```

### Swarm mode (alternative)

```bash
echo "$(openssl rand -base64 32)" | docker secret create postgres_password -
# Repeat for each secret. Remove file-based entries from docker-compose.secrets.yml.
```

## 2. TLS Setup

### External TLS (Let's Encrypt via Traefik)

Set your ACME email in the environment before starting:

```bash
export ACME_EMAIL=ops@yourschool.edu.in
```

Traefik auto-provisions certificates. Ensure port 80 is reachable for HTTP challenge.

### Internal TLS (service-to-service)

Generate self-signed certificates for internal services:

```bash
mkdir -p infra/certs/{nats,minio,redis}

# NATS
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout infra/certs/nats/nats.key -out infra/certs/nats/nats.crt \
  -subj "/CN=nats"

# MinIO
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout infra/certs/minio/private.key -out infra/certs/minio/public.crt \
  -subj "/CN=minio"

# Redis
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout infra/certs/redis/redis.key -out infra/certs/redis/redis.crt \
  -subj "/CN=redis"

# PostgreSQL (place in pgdata volume on first init)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout server.key -out server.crt -subj "/CN=postgres"
chmod 600 server.key
# Copy into the pgdata volume before first start.
```

## 3. Deploy

```bash
cd infra

# Start infrastructure + services
docker compose \
  -f docker-compose.prod.yml \
  -f docker-compose.prod-services.yml \
  -f docker-compose.secrets.yml \
  up -d

# Verify all healthy
docker compose \
  -f docker-compose.prod.yml \
  -f docker-compose.prod-services.yml \
  ps --format "table {{.Name}}\t{{.Status}}"
```

## 4. Backup Verification

### pgBackRest initial stanza setup

```bash
docker exec exampen-postgres-1 \
  pgbackrest --stanza=exampen stanza-create

# Run first full backup
docker exec exampen-postgres-1 \
  pgbackrest --stanza=exampen --type=full backup

# Verify
docker exec exampen-postgres-1 \
  pgbackrest --stanza=exampen info
```

### Schedule cron jobs

Install retention jobs from `backup/retention-cron.yaml` into host crontab or deploy as Kubernetes CronJobs. Each job entry includes its schedule and command.

### Monthly restore drill

```bash
# Restore to a test instance (never production)
pgbackrest --stanza=exampen --target="2026-03-01 00:00:00" \
  --type=time --target-action=promote restore
```

## 5. Monitoring Alerts to Configure

Deploy the monitoring stack (`docker-compose.monitoring.yml`) alongside production. Configure these alerts in Grafana:

| Alert | Condition | Severity |
|-------|-----------|----------|
| PostgreSQL replication lag | WAL lag > 100 MB | Critical |
| pgBackRest backup age | Last full backup > 26 hours | Critical |
| NATS consumer lag | Any consumer lag > 10 seconds | Warning |
| Service unhealthy | Any container health check failing > 2 min | Critical |
| Disk usage | Any volume > 85% | Warning |
| Memory pressure | Any container > 90% of limit | Warning |
| HTTP 5xx rate | > 1% of requests in 5 min window | Critical |
| Certificate expiry | TLS cert expires in < 14 days | Warning |
| MinIO disk usage | Bucket usage > 80% of provisioned | Warning |
| Redis evictions | Eviction rate > 100/min | Warning |

## 6. Scaling Recommendations

### Horizontal scaling (stateless services)

These services can be replicated without coordination:
- `svc-stroke-ingest` -- scale first under load (stateless ingestion)
- `svc-ai-pipeline` -- scale for AI throughput (CPU-bound)
- `svc-teacher-bff`, `svc-student-bff` -- scale for read traffic
- `svc-doc-assembly` -- scale for rendering throughput

```bash
docker compose -f docker-compose.prod.yml \
  -f docker-compose.prod-services.yml \
  up -d --scale svc-stroke-ingest=3 --scale svc-ai-pipeline=2
```

### Vertical scaling thresholds

| Service | When to scale up |
|---------|-----------------|
| PostgreSQL | Connection count > 150 or query latency p99 > 200 ms |
| NATS | Consumer lag sustained > 5 seconds |
| Redis | Memory usage > 400 MB or eviction rate rising |
| svc-ai-pipeline | Inference queue depth > 50 |

### Database scaling

- Enable PostgreSQL read replicas for BFF services (read-only aggregators)
- TimescaleDB: enable compression on chunks older than 7 days (configured in retention-cron.yaml)
- Consider moving to managed PostgreSQL (RDS/Cloud SQL) at > 5000 concurrent students

## 7. Operational Checklist

- [ ] All secrets generated and permissions locked (400)
- [ ] Internal TLS certificates generated for NATS, MinIO, Redis, PostgreSQL
- [ ] DNS A record points to server IP
- [ ] ACME_EMAIL environment variable set
- [ ] pgBackRest stanza created and first full backup verified
- [ ] Retention cron jobs installed
- [ ] Monitoring stack deployed with alerts configured
- [ ] Restore drill completed successfully
- [ ] Firewall allows only ports 80, 443 inbound
- [ ] Log rotation verified (25 MB x 5 files per container)
