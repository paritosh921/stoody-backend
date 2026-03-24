"""svc-teacher-bff — Read-only aggregation layer for teacher UI.

Entry point: creates the FastAPI application, warms JWKS cache,
and initialises backing-service HTTP clients.

CRITICAL: This BFF has ZERO write access to any database.
All mutations are relayed to backing service APIs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from exampen_common.auth import JWKSManager

from src.adapters.http_client import BackingClients
from src.config import STOODY_JWKS_URL
from src.routes.analytics import router as analytics_router
from src.routes.exam_mgmt import router as exam_mgmt_router
from src.routes.exams import router as exams_router
from src.routes.objections import router as objections_router
from src.routes.plagiarism import router as plagiarism_router
from src.routes.scores import router as scores_router
from src.routes.workflows import router as workflows_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm JWKS cache and create backing-service HTTP clients."""
    jwks = JWKSManager(jwks_url=STOODY_JWKS_URL)
    try:
        await jwks.warmup()
    except Exception:
        logger.warning("JWKS warmup failed — first request will trigger fetch", exc_info=True)
    app.state.jwks_manager = jwks

    clients = BackingClients()
    app.state.clients = clients
    yield
    await clients.close()


app = FastAPI(
    title="ExamPen Teacher BFF",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(exams_router, prefix="/api/v1")
app.include_router(exam_mgmt_router, prefix="/api/v1")
app.include_router(scores_router, prefix="/api/v1")
app.include_router(objections_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(plagiarism_router, prefix="/api/v1")
app.include_router(workflows_router, prefix="/api/v1")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
