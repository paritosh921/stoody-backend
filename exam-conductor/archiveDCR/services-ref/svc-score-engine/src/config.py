"""Environment-based configuration for svc-score-engine."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Loaded from environment variables (12-factor)."""

    # -- Service identity --
    service_name: str = "svc-score-engine"
    host: str = "0.0.0.0"
    port: int = 8000

    # -- PostgreSQL --
    database_url: str = "postgresql+asyncpg://score:score@localhost:5432/exampen_scores"

    # -- NATS --
    nats_url: str = "nats://localhost:4222"
    nats_stream: str = "EXAMPEN"
    nats_consumer: str = "score-engine"

    # -- Auth (Stoody JWKS) --
    stoody_jwks_url: str = "http://localhost:8000/.well-known/jwks.json"

    # -- Feature flags --
    mock_mode: bool = False

    model_config = {"env_prefix": "SCORE_ENGINE_"}


settings = Settings()
