"""svc-invig-console -- Real-time WebSocket invigilator dashboard backend.

Entry point: creates the FastAPI application, includes routers,
and manages startup/shutdown lifecycle (NATS, hub relay, exam client).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.auth import JWKSManager
from exampen_common.logging import configure_logging, RequestIdMiddleware
from exampen_common.nats_client import create_nats_client

from src.adapters.exam_client import ExamOrchClient
from src.config import EXAM_ORCH_URL, EXAM_ORCH_TIMEOUT, NATS_URL
from src.events.hub_relay import HubRelay
from src.routes.sessions import router as sessions_router
from src.routes.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: warm JWKS, connect NATS, start hub relay. Shutdown: teardown."""
    configure_logging(service_name="svc-invig-console")

    jwks = JWKSManager()
    await jwks.warmup()

    nats = await create_nats_client(url=NATS_URL)
    exam_client = ExamOrchClient(EXAM_ORCH_URL, timeout=EXAM_ORCH_TIMEOUT)
    hub_relay = HubRelay(nats)
    await hub_relay.start()

    app.state.jwks_manager = jwks
    app.state.nats_client = nats
    app.state.exam_client = exam_client
    app.state.hub_relay = hub_relay

    yield

    await hub_relay.stop()
    await nats.close()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ExamPen Invigilator Console API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(sessions_router, prefix="/api/v1/invigilator")
    app.include_router(ws_router, prefix="/api/v1/invigilator")
    return app


app = create_app()
