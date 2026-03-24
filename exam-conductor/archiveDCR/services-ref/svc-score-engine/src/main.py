"""FastAPI entry point for svc-score-engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from exampen_common.auth import JWKSManager
from src.config import settings
from src.events.ai_consumer import start_ai_consumer, stop_ai_consumer
from src.events.rescore_consumer import start_rescore_consumer, stop_rescore_consumer
from src.routes.scores import router as scores_router
from src.routes.workflow import router as workflow_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm JWKS cache, start NATS consumer on startup, drain on shutdown."""
    # -- Auth: warm the JWKS keyset so first request is fast --
    jwks = JWKSManager(jwks_url=settings.stoody_jwks_url)
    try:
        await jwks.warmup()
    except Exception:
        logger.warning(
            "JWKS warmup failed — tokens will be validated on first request",
            exc_info=True,
        )
    app.state.jwks_manager = jwks

    if not settings.mock_mode:
        await start_ai_consumer()
        await start_rescore_consumer()
    yield
    if not settings.mock_mode:
        await stop_rescore_consumer()
        await stop_ai_consumer()


app = FastAPI(
    title="ExamPen Score Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(scores_router, prefix="/api/v1")
app.include_router(workflow_router, prefix="/api/v1")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
