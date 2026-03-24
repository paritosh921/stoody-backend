"""Environment configuration for svc-doc-assembly."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable service settings loaded from environment variables."""

    database_url: str
    stroke_database_url: str
    nats_url: str
    minio_url: str
    minio_bucket: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    service_port: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.environ.get(
                "DATABASE_URL",
                "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen",
            ),
            stroke_database_url=os.environ.get(
                "STROKE_DATABASE_URL",
                "postgresql://exampen:exampen@localhost:5432/exampen_stroke",
            ),
            nats_url=os.environ.get("NATS_URL", "nats://localhost:4222"),
            minio_url=os.environ.get("MINIO_URL", "localhost:9000"),
            minio_bucket=os.environ.get("MINIO_BUCKET", "exampen-pages"),
            minio_access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            service_port=int(os.environ.get("SERVICE_PORT", "8000")),
        )
