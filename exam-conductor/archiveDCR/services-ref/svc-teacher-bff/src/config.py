"""Environment-based configuration for svc-teacher-bff.

All URLs point to backing services. This BFF has ZERO database access —
every piece of data comes from one of these service endpoints.
"""

from __future__ import annotations

import os

STOODY_JWKS_URL: str = os.getenv(
    "STOODY_JWKS_URL",
    "http://localhost:9100/.well-known/jwks.json",
)

SCORE_ENGINE_URL: str = os.getenv(
    "SCORE_ENGINE_URL",
    "http://localhost:8002",
)

ANALYTICS_URL: str = os.getenv(
    "ANALYTICS_URL",
    "http://localhost:8003",
)

REVIEW_URL: str = os.getenv(
    "REVIEW_URL",
    "http://localhost:8004",
)

PLAGIARISM_URL: str = os.getenv(
    "PLAGIARISM_URL",
    "http://localhost:8005",
)

CHAT_URL: str = os.getenv(
    "CHAT_URL",
    "http://localhost:8006",
)

EXAM_ORCH_URL: str = os.getenv(
    "EXAM_ORCH_URL",
    "http://localhost:8007",
)

DOC_ASSEMBLY_URL: str = os.getenv(
    "DOC_ASSEMBLY_URL",
    "http://localhost:8008",
)

# HTTP client settings
BACKING_SERVICE_TIMEOUT: int = int(os.getenv("BACKING_SERVICE_TIMEOUT", "10"))
