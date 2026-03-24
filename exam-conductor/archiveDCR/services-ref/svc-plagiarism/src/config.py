"""Environment configuration for svc-plagiarism."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration loaded from environment variables."""

    database_url: str = "postgresql://localhost:5432/exampen_plagiarism"
    nats_url: str = "nats://localhost:4222"
    review_threshold: float = 0.75
    strong_threshold: float = 0.90
    mock_mode: bool = False

    model_config = {"env_prefix": ""}


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
