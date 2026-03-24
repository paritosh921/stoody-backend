"""Environment-based configuration for svc-review."""

from __future__ import annotations

import os


DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen_review",
)

STOODY_JWKS_URL: str = os.getenv(
    "STOODY_JWKS_URL",
    "http://localhost:9100/.well-known/jwks.json",
)

NATS_URL: str = os.getenv(
    "NATS_URL",
    "nats://localhost:4222",
)

SCORE_ENGINE_URL: str = os.getenv(
    "SCORE_ENGINE_URL",
    "http://localhost:8005",
)
