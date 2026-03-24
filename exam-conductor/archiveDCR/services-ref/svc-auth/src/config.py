"""Environment-based configuration for svc-auth."""

from __future__ import annotations

import os


DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen_auth",
)

STOODY_JWKS_URL: str = os.getenv(
    "STOODY_JWKS_URL",
    "http://localhost:9100/.well-known/jwks.json",
)

STOODY_API_URL: str = os.getenv(
    "STOODY_API_URL",
    "http://localhost:9100",
)

REDIS_URL: str = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

STOODY_CLIENT_TIMEOUT: int = int(os.getenv("STOODY_CLIENT_TIMEOUT", "5"))
STOODY_CLIENT_RETRIES: int = int(os.getenv("STOODY_CLIENT_RETRIES", "2"))
