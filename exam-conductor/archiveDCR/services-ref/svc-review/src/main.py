"""svc-review — Objection lifecycle management.

Entry point: creates the FastAPI application, includes routers,
and manages startup/shutdown lifecycle hooks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.auth import JWKSManager
from exampen_common.db import create_pool, session_factory
from exampen_common.logging import configure_logging, RequestIdMiddleware
from exampen_common.nats_client import NatsClient

from src.adapters.score_client import ScoreEngineClient
from src.config import DATABASE_URL, NATS_URL, SCORE_ENGINE_URL, STOODY_JWKS_URL
from src.events.objection_publisher import ObjectionPublisher
from src.routes.objections import router as objections_router
from src.storage.objection_repo import ObjectionRepo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: warm JWKS cache, create DB pool, connect NATS. Shutdown: cleanup."""
    configure_logging()

    # Auth
    jwks = JWKSManager(jwks_url=STOODY_JWKS_URL)
    await jwks.warmup()
    app.state.jwks_manager = jwks

    # Database
    engine = create_pool(url=DATABASE_URL)
    sf = session_factory(engine)
    app.state.db_engine = engine
    app.state.session_factory = sf
    app.state.objection_repo = ObjectionRepo(sf)

    # NATS
    nats = NatsClient(url=NATS_URL)
    await nats.connect()
    app.state.nats_client = nats
    app.state.objection_publisher = ObjectionPublisher(nats)
    app.state.score_checker = ScoreEngineClient(base_url=SCORE_ENGINE_URL)

    yield

    # Shutdown
    await nats.close()
    await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ExamPen Review API",
        version="1.0.0",
        description="Objection intake, assignment, resolution, and escalation.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(objections_router, prefix="/api/v1/objections")
    return app


app = create_app()
