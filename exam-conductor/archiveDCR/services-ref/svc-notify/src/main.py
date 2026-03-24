"""svc-notify — ExamPen notification service.

FastAPI app with NATS subscriptions for score.updated, objection.*,
and exam.lifecycle events. Dispatches email, push, and SMS notifications.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.config import settings
from src.events.event_consumers import start_consumers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_nats_conn = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _nats_conn  # noqa: PLW0603
    logger.info("svc-notify starting — connecting to NATS at %s", settings.nats_url)
    _nats_conn = await start_consumers(settings.nats_url)
    yield
    if _nats_conn:
        await _nats_conn.drain()
        logger.info("NATS connection drained")


app = FastAPI(
    title="svc-notify",
    description="ExamPen notification service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "svc-notify"}
