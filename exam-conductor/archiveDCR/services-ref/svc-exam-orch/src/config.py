"""Environment-based configuration for svc-exam-orch."""

from __future__ import annotations

import os

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen_orch",
)

NATS_URL: str = os.getenv(
    "NATS_URL",
    "nats://localhost:4222",
)

STOODY_API_URL: str = os.getenv(
    "STOODY_API_URL",
    "http://localhost:9100",
)

AUTH_SERVICE_URL: str = os.getenv(
    "AUTH_SERVICE_URL",
    "http://localhost:8001",
)

STOODY_CLIENT_TIMEOUT: int = int(os.getenv("STOODY_CLIENT_TIMEOUT", "5"))
