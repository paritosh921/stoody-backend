"""svc-student-bff — Read-only aggregator for student and parent views.

Entry point: creates the FastAPI application, includes routers,
and manages startup/shutdown lifecycle hooks.

CRITICAL: ZERO database access.  All data comes from backing service
REST APIs (svc-score-engine, svc-review, svc-analytics, svc-chat).
All mutations are relayed to the owning service.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.auth import JWKSManager
from exampen_common.logging import RequestIdMiddleware, configure_logging

from src.adapters.analytics_client import AnalyticsClient
from src.adapters.chat_client import ChatClient
from src.adapters.review_client import ReviewClient
from src.adapters.score_client import ScoreClient
from src.adapters.stoody_client import StoodyClient
from src.config import (
    ANALYTICS_SERVICE_URL,
    CHAT_SERVICE_URL,
    REVIEW_SERVICE_URL,
    SCORE_ENGINE_URL,
    STOODY_API_URL,
    STOODY_JWKS_URL,
)
from src.routes.chat import router as chat_router
from src.routes.objections import router as objections_router
from src.routes.performance import router as performance_router
from src.routes.scores import router as scores_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: warm JWKS cache, create adapter clients.  No DB pools."""
    configure_logging()

    # Auth — validate Stoody JWTs
    jwks = JWKSManager(jwks_url=STOODY_JWKS_URL)
    await jwks.warmup()
    app.state.jwks_manager = jwks

    # Stoody platform client — parent-child resolution
    app.state.stoody_client = StoodyClient(base_url=STOODY_API_URL)

    # Backing service clients (read-only aggregation + mutation relay)
    app.state.score_client = ScoreClient(base_url=SCORE_ENGINE_URL)
    app.state.review_client = ReviewClient(base_url=REVIEW_SERVICE_URL)
    app.state.analytics_client = AnalyticsClient(
        base_url=ANALYTICS_SERVICE_URL,
    )
    app.state.chat_client = ChatClient(base_url=CHAT_SERVICE_URL)

    yield

    # No pools or connections to clean up — all HTTP clients are
    # per-request aiohttp.ClientSession instances.


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ExamPen Student BFF API",
        version="1.0.0",
        description=(
            "Read-only aggregation API for Stoody student and parent "
            "score-view surfaces.  ZERO database access."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)

    app.include_router(scores_router, prefix="/api/v1/student")
    app.include_router(objections_router, prefix="/api/v1/student")
    app.include_router(performance_router, prefix="/api/v1/student/performance")
    app.include_router(chat_router, prefix="/api/v1/student")
    return app


app = create_app()
