"""svc-exam-orch — Exam lifecycle FSM, pen binding, scheduling.

Entry point: creates the FastAPI application, includes routers,
and manages startup/shutdown lifecycle (DB pool, NATS connection).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.auth import JWKSManager
from exampen_common.db import create_pool, session_factory
from exampen_common.logging import configure_logging, RequestIdMiddleware
from exampen_common.nats_client import create_nats_client

from src.adapters.stoody_client import StoodyClient
from src.config import DATABASE_URL, NATS_URL, STOODY_API_URL, STOODY_CLIENT_TIMEOUT
from src.routes.assignments import router as assignments_router
from src.routes.bindings import router as bindings_router
from src.routes.exams import router as exams_router
from src.routes.rubrics import router as rubrics_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: warm JWKS, create DB pool, connect NATS. Shutdown: teardown."""
    configure_logging(service_name="svc-exam-orch")

    jwks = JWKSManager()
    await jwks.warmup()

    engine = create_pool(url=DATABASE_URL)
    sf = session_factory(engine)
    nats = await create_nats_client(url=NATS_URL)
    stoody = StoodyClient(STOODY_API_URL, timeout=STOODY_CLIENT_TIMEOUT)

    app.state.jwks_manager = jwks
    app.state.db_engine = engine
    app.state.session_factory = sf
    app.state.nats_client = nats
    app.state.stoody_client = stoody

    yield

    await nats.close()
    await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ExamPen Exam Orchestrator API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(exams_router, prefix="/api/v1/exams")
    app.include_router(rubrics_router, prefix="/api/v1/exams")
    app.include_router(bindings_router, prefix="/api/v1/exams/{exam_id}/bindings")
    app.include_router(assignments_router, prefix="/api/v1/exams/{exam_id}/invigilators")
    return app


app = create_app()
