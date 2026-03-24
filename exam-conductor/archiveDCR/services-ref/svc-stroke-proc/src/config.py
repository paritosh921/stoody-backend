"""Environment-based configuration for svc-stroke-proc."""

from __future__ import annotations

import os

# -- NATS ------------------------------------------------------------------

NATS_URL: str = os.getenv("NATS_URL", "nats://localhost:4222")
NATS_CREDS: str | None = os.getenv("NATS_CREDS")

STROKE_RAW_SUBJECT: str = os.getenv("STROKE_RAW_SUBJECT", "EXAMPEN.stroke.raw")
STROKE_PROCESSED_SUBJECT: str = os.getenv(
    "STROKE_PROCESSED_SUBJECT", "EXAMPEN.stroke.processed"
)

# Consumer durable name for NATS JetStream subscription
CONSUMER_DURABLE_NAME: str = os.getenv(
    "CONSUMER_DURABLE_NAME", "svc-stroke-proc"
)

# -- TimescaleDB -----------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen_stroke_proc",
)

# -- Service ---------------------------------------------------------------

SERVICE_NAME: str = "svc-stroke-proc"
