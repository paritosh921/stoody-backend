"""Environment-based configuration for svc-analytics."""

from __future__ import annotations

import os


DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen_analytics",
)

AUTH_SERVICE_URL: str = os.getenv(
    "AUTH_SERVICE_URL",
    "http://localhost:8001",
)

NATS_URL: str = os.getenv(
    "NATS_URL",
    "nats://localhost:4222",
)

NATS_STREAM: str = os.getenv(
    "NATS_STREAM",
    "EXAMPEN",
)

NATS_SUBJECT: str = os.getenv(
    "NATS_SUBJECT",
    "EXAMPEN.score.updated",
)

NATS_DURABLE_NAME: str = os.getenv(
    "NATS_DURABLE_NAME",
    "svc-analytics-score-consumer",
)

DEFAULT_PASS_THRESHOLD: float = float(
    os.getenv("DEFAULT_PASS_THRESHOLD", "40.0")
)
