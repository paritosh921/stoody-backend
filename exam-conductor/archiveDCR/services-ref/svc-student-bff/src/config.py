"""Environment-based configuration for svc-student-bff.

ZERO database URLs — this service is a read-only aggregator.
All data comes from backing service REST APIs.
"""

from __future__ import annotations

import os

STOODY_JWKS_URL: str = os.getenv(
    "STOODY_JWKS_URL",
    "http://localhost:9100/.well-known/jwks.json",
)

STOODY_API_URL: str = os.getenv(
    "STOODY_API_URL",
    "http://localhost:9100",
)

SCORE_ENGINE_URL: str = os.getenv(
    "SCORE_ENGINE_URL",
    "http://localhost:8005",
)

REVIEW_SERVICE_URL: str = os.getenv(
    "REVIEW_SERVICE_URL",
    "http://localhost:8007",
)

ANALYTICS_SERVICE_URL: str = os.getenv(
    "ANALYTICS_SERVICE_URL",
    "http://localhost:8008",
)

CHAT_SERVICE_URL: str = os.getenv(
    "CHAT_SERVICE_URL",
    "http://localhost:8009",
)

EXAM_ORCH_URL: str = os.getenv(
    "EXAM_ORCH_URL",
    "http://localhost:8002",
)

DOC_ASSEMBLY_URL: str = os.getenv(
    "DOC_ASSEMBLY_URL",
    "http://localhost:8004",
)

AI_PIPELINE_URL: str = os.getenv(
    "AI_PIPELINE_URL",
    "http://localhost:8006",
)

# HTTP client defaults
CLIENT_TIMEOUT: int = int(os.getenv("CLIENT_TIMEOUT", "5"))
CLIENT_RETRIES: int = int(os.getenv("CLIENT_RETRIES", "2"))
