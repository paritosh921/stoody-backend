"""Environment configuration for svc-copy-upload."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable service configuration loaded from environment."""

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen",
    )
    nats_url: str = os.getenv("NATS_URL", "nats://localhost:4222")
    minio_url: str = os.getenv("MINIO_URL", "http://localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "exampen-copies")
    minio_region: str = os.getenv("MINIO_REGION", "us-east-1")
    presigned_url_expiry: int = int(os.getenv("PRESIGNED_URL_EXPIRY", "3600"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8006"))


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings()
