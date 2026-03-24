"""Environment-based configuration for svc-invig-console."""

from __future__ import annotations

import os

EXAM_ORCH_URL: str = os.getenv(
    "EXAM_ORCH_URL",
    "http://localhost:8002",
)

NATS_URL: str = os.getenv(
    "NATS_URL",
    "nats://localhost:4222",
)

AUTH_SERVICE_URL: str = os.getenv(
    "AUTH_SERVICE_URL",
    "http://localhost:8001",
)

EXAM_ORCH_TIMEOUT: int = int(os.getenv("EXAM_ORCH_TIMEOUT", "5"))

WS_PUSH_INTERVAL_SEC: float = float(os.getenv("WS_PUSH_INTERVAL_SEC", "1.0"))

# NATS subjects for hub status relay
NATS_HUB_STATUS_SUBJECT: str = os.getenv(
    "NATS_HUB_STATUS_SUBJECT",
    "EXAMPEN.hub.status.>",
)
