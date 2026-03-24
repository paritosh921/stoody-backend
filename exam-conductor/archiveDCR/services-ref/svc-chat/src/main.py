"""svc-chat -- Append-only messaging for teacher-student exam threads.

Entry point: creates the FastAPI application, includes routers,
and manages startup/shutdown lifecycle hooks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.db import create_pool, session_factory
from exampen_common.logging import configure_logging, RequestIdMiddleware

from src.adapters.exam_enrollment import ExamEnrollmentAdapter
from src.config import DATABASE_URL, EXAM_ORCH_URL
from src.routes.messages import router as messages_router
from src.storage.message_repo import MessageRepo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: create DB pool. Shutdown: dispose pool."""
    configure_logging(service_name="svc-chat")
    engine = create_pool(url=DATABASE_URL)
    sf = session_factory(engine)

    app.state.db_engine = engine
    app.state.session_factory = sf
    app.state.message_repo = MessageRepo(sf)
    app.state.enrollment = ExamEnrollmentAdapter(EXAM_ORCH_URL)

    yield

    await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ExamPen Chat API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(messages_router, prefix="/api/v1/chat")
    return app


app = create_app()
