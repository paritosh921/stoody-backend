"""Environment-based configuration for svc-stroke-ingest."""

from __future__ import annotations

import os

# -- NATS --------------------------------------------------------------

NATS_URL: str = os.getenv("NATS_URL", "nats://localhost:4222")
NATS_CREDS: str | None = os.getenv("NATS_CREDS")

STROKE_RAW_SUBJECT: str = os.getenv("STROKE_RAW_SUBJECT", "EXAMPEN.stroke.raw")

# -- Auth --------------------------------------------------------------

AUTH_SERVICE_URL: str = os.getenv(
    "AUTH_SERVICE_URL",
    "http://localhost:9100/.well-known/jwks.json",
)

# -- Redis (idempotency) -----------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

IDEMPOTENCY_TTL_SECONDS: int = int(
    os.getenv("IDEMPOTENCY_TTL_SECONDS", str(7 * 24 * 3600))  # 7 days
)

# -- PostgreSQL (upload progress) ---------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen_stroke_ingest",
)

# -- Rate limiting -----------------------------------------------------

RATE_LIMIT_PER_HUB: int = int(os.getenv("RATE_LIMIT_PER_HUB", "200"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
