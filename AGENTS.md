# Backend - Stoody AI Learning Platform

## TLA+-guided refactors

When the user invokes `tla-plus-implementation` for work in this repository, read `docs/formal/constraint-index.md` before creating the task-specific Phase 0 artifacts. Re-verify every selected evidence anchor in the current checkout, cite the relevant `BE-*` IDs in `system-context.md`/`boundary-decisions.md`, and treat material drift or uncertainty as `UNKNOWN` until resolved.

The catalog is discovery input, not a universal specification or a substitute for the skill's required `requirements.md`, `system-context.md`, `boundary-decisions.md`, PlusCal/TLA+, TLC run, and gates. Do not run the formal workflow for ordinary stateless work unless the user or skill trigger requires it.

## Overview

Async Python backend for an educational platform (Stoody) with AI-powered learning features, multi-tenancy, and real-time capabilities. Deployed on **AWS EC2** with Nginx + Supervisor.

## Tech Stack

| Category | Technology |
|----------|------------|
| Runtime | Python 3.11 (required - 3.12 has TLS issues with MongoDB Atlas) |
| Framework | FastAPI 0.104.1 (async) |
| Server | Uvicorn 0.24.0 with uvloop |
| Database | MongoDB Atlas (Motor async driver) |
| Cache | Redis (async redis-py) |
| Task Queue | Celery with Redis broker |
| Storage | AWS S3 (boto3) with local fallback |

### AI/ML Integrations
- OpenAI (GPT-4, Vision)
- Anthropic Codex
- Google Gemini
- Mistral (OCR)
- LangChain (RAG debugging)
- Sentence Transformers

## Project Structure

```
backend/
├── api/v1/                 # 36 route modules
│   ├── auth_async.py       # JWT authentication
│   ├── auth_cookie.py      # Cookie-based auth (httpOnly)
│   ├── b2c_auth_async.py   # Google OAuth
│   ├── chat_async.py       # AI chat endpoints
│   ├── questions_async.py  # Question CRUD
│   ├── practice_async.py   # Practice sessions
│   ├── mcq_async.py        # MCQ tests
│   ├── ocr.py              # OCR endpoints
│   ├── pdf_async.py        # PDF processing
│   ├── strokes_async.py    # Pen stroke data
│   ├── totp_2fa.py         # 2FA auth
│   └── ...
├── core/                   # Core services
│   ├── database.py         # MongoDB connection
│   ├── auth.py             # JWT/password auth
│   ├── cache.py            # Redis cache
│   ├── tenant.py           # Multi-tenancy
│   ├── permissions.py      # RBAC
│   └── ...
├── services/               # Business logic
│   ├── async_openai_service.py
│   ├── mistral_ocr_service.py
│   ├── document_processor.py
│   └── ...
├── models/                 # Pydantic models
├── middleware/             # Request middleware
├── utils/                  # Utilities
├── main_async.py           # FastAPI entry point
├── config_async.py         # Settings (Pydantic)
├── requirements.txt        # Dependencies
└── deploy.sh               # EC2 deployment
```

## API Endpoints

### Authentication
```
POST /api/v1/auth/2fa/login-2fa    # Admin/tutor login state machine (user_type: admin|tutor)
POST /api/v1/auth/student/login    # Student login (with tenant_id)
POST /api/v1/auth/b2c/google-callback  # Google OAuth
POST /api/v1/auth/2fa/setup/start  # 2FA setup (TOTP)
POST /api/v1/auth/2fa/setup/verify # Verify first setup OTP
POST /api/v1/auth/2fa/verify-otp   # Verify login OTP
```

### Content & Learning
```
GET/POST /api/v1/questions         # Question CRUD
POST /api/v1/questions/batch-save  # Bulk save
POST /api/v1/practice/start        # Start practice
POST /api/v1/practice/submit       # Submit answer
GET /api/v1/mcq/list               # List MCQs
POST /api/v1/chat                  # AI chat
```

### Admin & Management
```
GET /api/v1/admin/students         # List students
POST /api/v1/admin/students/bulk-upload  # Bulk import
GET /api/v1/admin/dashboard        # Analytics
POST /api/v1/admin/settings        # School config
```

### Health Checks
```
GET /health                        # Basic health
GET /api/health                    # Detailed status
GET /healthz                       # K8s probe
GET /alb-health                    # ALB probe
```

## Database Architecture

### MongoDB Databases (Strict Tenant Model)
```
skb_master/              # Tenant registry (super-admin managed)
├── tenants              # Tenant records (status, features, db_name)
├── super_admins         # Super-admin accounts
└── superadmin_messages  # Registration messaging

skb_<institution_id>/    # Per-tenant databases (e.g. skb_indl-ciel-1001)
├── admins               # School administrators
├── students             # Student accounts (admin_id scoped)
├── tutors               # Teacher accounts (admin_id scoped)
├── questions            # Question content + images (admin_id scoped)
├── documents            # PDF documents (admin_id scoped)
├── chat_sessions        # AI chat sessions (admin_id scoped)
├── assignments          # Assignments (admin_id scoped)
├── meetings             # Online class meetings (admin_id scoped)
├── notifications        # Notifications (admin_id scoped)
├── class_schedules      # Schedules (admin_id scoped)
├── smartboard_sessions  # Whiteboard sessions (admin_id scoped)
└── school_settings      # Per-tenant configuration

STOODY-b2c/              # B2C users (separate, no tenant scoping)
```

> **No fallback database.** All tenant-scoped operations require `TenantContext` with `db_name` from JWT. Requests without valid tenant context are rejected (401).

### Connection Pooling
- MongoDB: 50-500 connections
- Redis: 500 connections max

## Authentication & Security

### Auth Methods
1. **JWT** - HS256, 60-min expiry, 32+ char secret required
2. **Cookie Auth** - httpOnly, Secure, SameSite
3. **OAuth 2.0** - Google for B2C users
4. **2FA** - TOTP (Google Authenticator)

### Role-Based Access Control
| Role | Scope |
|------|-------|
| `admin` | Full school management |
| `tutor` | Assigned students, grading |
| `student` | Take tests, view content |
| `b2c_admin` | B2C tenant management |
| `superadmin` | Cross-tenant management |

### Security Features
- bcrypt password hashing (12 rounds)
- CSRF protection on state changes
- Rate limiting (600 req/min default, 120 for auth)
- Security headers (HSTS, CSP, X-Frame-Options)
- Request sanitization
- Audit logging (`logs/security.log`)

## Environment Variables

```bash
# Required
NODE_ENV=production
MONGODB_URI=mongodb+srv://...
REDIS_URL=redis://...
JWT_SECRET_KEY=<32+ chars>

# AI Services
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=...
GOOGLE_GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=...                # Note classification (free at console.groq.com)

# AWS
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=stoody-assets-prod

# OAuth
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
```

## Commands

```bash
# Development
python -m uvicorn main_async:app --reload --port 5001

# Production (via Supervisor)
sudo supervisorctl start skillbot-production
sudo supervisorctl restart skillbot-production
sudo supervisorctl status

# Logs
sudo tail -f /var/log/supervisor/skillbot-production.log
```

## Deployment Architecture (EC2)

```
CloudFront (CDN)
    ↓
ALB (SSL Termination, Health Checks)
    ↓
Nginx (Reverse Proxy, Rate Limiting, CORS)
    ↓
Supervisor (Process Manager)
    └── Uvicorn Workers (8 default)
        └── FastAPI App (Port 5001)
            ├── MongoDB Atlas
            ├── Redis
            ├── Celery
            └── External APIs
```

### Performance Tuning
- Workers: 8 (configurable)
- Max connections per worker: 2000
- Nginx proxy timeout: 300s

---

## Grafana Monitoring Strategy

### Data Sources Required
1. **Prometheus** - Application metrics (via prometheus-fastapi-instrumentator)
2. **AWS CloudWatch** - EC2/ALB/S3 metrics
3. **MongoDB Atlas** - Database metrics (native integration)
4. **Loki** - Application logs
5. **Redis Exporter** - Cache metrics

### Application Metrics (Prometheus)

Install `prometheus-fastapi-instrumentator` and expose `/metrics`:

```python
# main_async.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

| Metric | Type | Labels |
|--------|------|--------|
| `http_request_duration_seconds` | Histogram | method, endpoint, status |
| `http_requests_total` | Counter | method, endpoint, status |
| `http_requests_in_progress` | Gauge | method, endpoint |
| `http_request_size_bytes` | Histogram | method, endpoint |
| `http_response_size_bytes` | Histogram | method, endpoint |

### Custom Business Metrics

Add these custom metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Authentication
auth_attempts = Counter('auth_attempts_total', 'Auth attempts', ['method', 'success'])
active_sessions = Gauge('active_sessions', 'Active user sessions', ['user_type'])

# AI Services
ai_request_duration = Histogram('ai_request_duration_seconds', 'AI API latency', ['service'])
ai_request_total = Counter('ai_request_total', 'AI API calls', ['service', 'success'])
ai_tokens_used = Counter('ai_tokens_used_total', 'Tokens consumed', ['service', 'type'])

# Learning
practice_attempts = Counter('practice_attempts_total', 'Practice attempts', ['subject', 'difficulty'])
mcq_submissions = Counter('mcq_submissions_total', 'MCQ submissions', ['subject', 'correct'])
questions_served = Counter('questions_served_total', 'Questions served', ['subject', 'type'])

# OCR Processing
ocr_processing_duration = Histogram('ocr_processing_duration_seconds', 'OCR latency', ['provider'])
ocr_requests = Counter('ocr_requests_total', 'OCR requests', ['provider', 'success'])

# WebSocket
ws_connections = Gauge('websocket_connections', 'Active WebSocket connections', ['type'])
stroke_batches_received = Counter('stroke_batches_total', 'Stroke batches received')

# Database
db_query_duration = Histogram('db_query_duration_seconds', 'DB query latency', ['collection', 'operation'])
```

### Infrastructure Metrics (CloudWatch Agent)

Install CloudWatch Agent on EC2 for:

| Metric | Purpose | Alert Threshold |
|--------|---------|-----------------|
| `CPUUtilization` | CPU usage | > 80% |
| `MemoryUtilization` | Memory usage | > 85% |
| `DiskUtilization` | Disk usage | > 80% |
| `NetworkIn/Out` | Network traffic | Anomaly |
| `StatusCheckFailed` | Instance health | > 0 |

### MongoDB Atlas Metrics

Connect Grafana to MongoDB Atlas metrics API:

| Metric | Purpose | Alert |
|--------|---------|-------|
| `OPCOUNTER_*` | Operations/sec | Baseline |
| `CONNECTIONS` | Connection count | > 400 |
| `QUERY_TARGETING_SCANNED_OBJECTS` | Query efficiency | High ratio |
| `DOCUMENT_METRICS` | Doc operations | Anomaly |
| `CACHE_BYTES_*` | WiredTiger cache | > 90% |

### Redis Metrics (via redis_exporter)

| Metric | Purpose | Alert |
|--------|---------|-------|
| `redis_connected_clients` | Active connections | > 450 |
| `redis_memory_used_bytes` | Memory usage | > 80% |
| `redis_keyspace_hits/misses` | Cache hit ratio | < 80% |
| `redis_commands_processed_total` | Throughput | Anomaly |

### ALB Metrics (CloudWatch)

| Metric | Purpose | Alert |
|--------|---------|-------|
| `RequestCount` | Traffic volume | Anomaly |
| `TargetResponseTime` | Backend latency | p99 > 2s |
| `HTTPCode_Target_5XX` | Backend errors | > 1% |
| `HTTPCode_ELB_5XX` | ALB errors | > 0 |
| `HealthyHostCount` | Healthy targets | < expected |
| `ActiveConnectionCount` | Active conns | Capacity |

### Logging (Loki)

Ship logs via Promtail:

```yaml
# promtail-config.yaml
scrape_configs:
  - job_name: skillbot
    static_configs:
      - targets: [localhost]
        labels:
          job: skillbot-backend
          __path__: /var/log/supervisor/skillbot-production.log
    pipeline_stages:
      - regex:
          expression: '(?P<timestamp>\S+) - (?P<logger>\S+) - (?P<level>\S+) - (?P<message>.*)'
      - labels:
          level:
          logger:
```

### Recommended Dashboards

1. **API Performance**
   - Request latency heatmap (p50, p95, p99)
   - Requests/sec by endpoint
   - Error rate by endpoint
   - Slow endpoints (> 1s)

2. **Infrastructure Health**
   - EC2 CPU/Memory/Disk
   - ALB request count and latency
   - Active connections

3. **Database Performance**
   - MongoDB operations/sec
   - Query latency distribution
   - Connection pool usage
   - Slow queries

4. **Cache Performance**
   - Redis hit/miss ratio
   - Memory usage
   - Commands/sec

5. **AI Services**
   - API latency by provider (OpenAI, Codex, Gemini)
   - Token usage over time
   - Error rates

6. **Business Metrics**
   - Active users (by role)
   - Practice attempts/hour
   - MCQ completion rate
   - Popular subjects

### Alerting Rules

```yaml
# Critical
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
  for: 2m

- alert: HighLatency
  expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 2
  for: 5m

- alert: DatabaseConnectionExhausted
  expr: mongodb_connections > 450
  for: 1m

- alert: InstanceDown
  expr: up{job="skillbot-backend"} == 0
  for: 1m

# Warning
- alert: HighCPU
  expr: system_cpu_utilization > 0.8
  for: 10m

- alert: LowCacheHitRate
  expr: redis_keyspace_hits / (redis_keyspace_hits + redis_keyspace_misses) < 0.8
  for: 15m
```

### Implementation Priority

1. **Phase 1** (Week 1): Prometheus + basic HTTP metrics
2. **Phase 2** (Week 2): CloudWatch integration, EC2 metrics
3. **Phase 3** (Week 3): MongoDB Atlas + Redis metrics
4. **Phase 4** (Week 4): Custom business metrics, Loki logging

---

## Code Conventions

- **Async**: All I/O operations must be async
- **Models**: Pydantic for request/response validation
- **Errors**: Raise HTTPException with appropriate status codes
- **Logging**: Use module logger, not print()
- **Security**: Validate JWT on protected routes, use permissions decorator

## Security Bypasses & Vulnerabilities (MUST FIX before production)

### CRITICAL

1. ~~`api/v1/auth_bypass.py`~~ — **RESOLVED**: File deleted, .pyc remnants cleaned, no router registration.

2. **Hardcoded default credentials in multiple files**
   - `admin@skillbot.app` / `admin123` appears in:
     - `models/admin.py` → `create_default_admin()` (line ~134-153)
     - `scripts/admin/init_admin_direct.py` (now requires `--db-name`)
     - `scripts/admin/setup_demo_admin.py` (now requires `--db-name`)
   - **Action**: Remove hardcoded credentials; use env vars or interactive prompts for seeding

3. **Super-admin setup key in `.env` — `stoody-super-admin108`**
   - Used for first-time super-admin creation via `POST /superadmin/setup`
   - **Action**: Rotate this key on the deployed EC2 instance; use a strong random value

### HIGH

4. **`SUPERADMIN_JWT_SECRET` committed in `.env`**
   - `LoAtCioeLPxnTx7Ox-kf_iADYhH6Ju6xVfD4OGjqJ2k`
   - **Action**: Rotate this secret on EC2; ensure `.env` is in `.gitignore`

5. **No rate limiting on super-admin login endpoint**
   - `POST /superadmin/login` should have aggressive rate limiting (e.g., 5 attempts/min)

## ExamPen Offline Exam Initiation — Architectural Decisions (2026-04-02)

> Temporary section. Move relevant parts into `exam-conductor/new-docs/` once stable.

### Exam mode and finalization

- Documents gain `exam_mode` (`"dcr"` | `"pcr"` | null) set at upload time, and `exam_finalized` (bool) set by `POST /pdf/documents/{document_id}/finalize-exam`.
- **Finalize is a hard lock.** After finalization: question create/update/delete, bulk marking, document metadata edits, and recalculate-points are all rejected (403). No re-finalize.
- Finalize is the **sole sync authority** for ExamPen metadata. It calls `sync_dcr_answer_keys()` (DCR) or `sync_questions_to_exampen()` (PCR) exactly once.

### Auto-sync removed from question CRUD

- `questions_async.py` previously auto-called `sync_questions_to_exampen()` and `sync_dcr_answer_keys()` on every single-save, batch-save, and create. This was removed because it pushed metadata pre-finalize, weakening the "review then finalize" contract. The finalize endpoint now owns all exampen collection writes.

### Orphaned helpers removed

- `sync_paper_to_exampen()` in `tutor_async.py` — deleted (zero callers). Was a paper-builder integration that was never wired up.

### Dual practice evaluation surfaces

- `/api/v1/practice/evaluate` (in `practice_async.py`) — the live student practice path. Routes through the LLM gate with `caller_id="pcr_practice"`.
- `/api/v1/evalpen/practice/evaluate` (in `evalpen_practice_async.py`) — built during exam-conductor boundary work for future PCR-template-based practice. Not wired into frontend yet. Both exist intentionally; do not remove the evalpen one.

### Teacher BFF prepared exams

- `evalpen_teacher_bff_async.py` merges two sources: submission-driven exams (from `evalpen_submissions`) and finalized documents (from `documents` where `exam_finalized=true`). Prepared exams appear with `status: "prepared"` and zero submission counts.
- Tutor scoping for prepared exams matches the existing document visibility model: `teacher_ids` contains tutor ID, OR `teacher_ids` is empty/null/missing (open to all tutors).
- Question counts use live aggregation from `questions` collection, not `extracted_questions_count` (which drifts after edits).

### Frontend exam-pen flow

- Clicking a prepared exam in ExamList routes to the "Exam Setup" tab (split-pane: PDF left, questions + submission status right). Active exams route to "Review Queue".
- DocumentDetailPanel shows a "Finalize for Exam" bar when `exam_mode` is set. After finalization all edit controls are disabled (type dropdown, answer buttons, points/penalty, add/delete, bulk marking, edit dialog).

## Known Issues

- Python 3.12 incompatible with MongoDB Atlas (TLS issues)
- Some endpoints lack proper rate limiting
- Profile endpoint disabled in production (`?profile=true`)
- OCR request logging disabled (base64 spam)
