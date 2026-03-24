"""Environment configuration for svc-ai-pipeline."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration loaded from environment variables."""

    database_url: str = "postgresql://localhost:5432/exampen_ai"
    nats_url: str = "nats://localhost:4222"
    minio_url: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "exampen-pages"
    model_dir: str = "/models"
    confidence_threshold: float = 0.85
    mock_mode: bool = False

    model_config = {"env_prefix": ""}


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
