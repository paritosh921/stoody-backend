"""svc-analytics -- Percentiles, leaderboards, class statistics.

Entry point: creates the FastAPI application, includes routers,
manages startup/shutdown lifecycle hooks, and starts the NATS
score.updated event consumer.

svc-analytics is the ONLY writer of percentile data
(per STATE_OWNERSHIP_MAP.md).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.db import create_pool, session_factory
from exampen_common.logging import configure_logging, RequestIdMiddleware

from src.config import (
    DATABASE_URL,
    NATS_DURABLE_NAME,
    NATS_STREAM,
    NATS_SUBJECT,
    NATS_URL,
)
from src.events.score_consumer import ScoreConsumer
from src.routes.analytics import router as analytics_router
from src.storage.analytics_repo import AnalyticsRepo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: create DB pool + NATS consumer. Shutdown: dispose."""
    configure_logging(service_name="svc-analytics")
    engine = create_pool(url=DATABASE_URL)
    sf = session_factory(engine)

    repo = AnalyticsRepo(sf)
    app.state.db_engine = engine
    app.state.session_factory = sf
    app.state.analytics_repo = repo

    # Start NATS consumer for score.updated events
    consumer = ScoreConsumer(
        repo=repo,
        nats_url=NATS_URL,
        stream=NATS_STREAM,
        subject=NATS_SUBJECT,
        durable_name=NATS_DURABLE_NAME,
    )
    app.state.score_consumer = consumer

    try:
        await consumer.start()
    except Exception:
        # If NATS is unavailable, log warning and continue.
        # The service can still serve cached data via REST.
        import logging

        logging.getLogger(__name__).warning(
            "NATS consumer failed to start — analytics will "
            "serve stale data until NATS is available",
            exc_info=True,
        )

    yield

    await consumer.stop()
    await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ExamPen Analytics API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(
        analytics_router, prefix="/api/v1/analytics",
    )
    return app


app = create_app()
