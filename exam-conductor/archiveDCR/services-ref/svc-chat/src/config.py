"""Environment-based configuration for svc-chat."""

from __future__ import annotations

import os


DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen_chat",
)

AUTH_SERVICE_URL: str = os.getenv(
    "AUTH_SERVICE_URL",
    "http://localhost:8001",
)

EXAM_ORCH_URL: str = os.getenv(
    "EXAM_ORCH_URL",
    "http://localhost:8003",
)

MAX_MESSAGE_LENGTH: int = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
