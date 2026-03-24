"""svc-stroke-ingest -- Chunk-oriented stroke ingestion service.

Accepts chunk uploads from hubs, validates CRC-32, checks idempotency,
publishes ``stroke.raw`` events to NATS JetStream, and tracks per-pen
upload progress for reconciliation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.logging import configure_logging, RequestIdMiddleware
from exampen_common.nats_client import create_nats_client

from src.config import DATABASE_URL, NATS_URL, NATS_CREDS, REDIS_URL
from src.adapters.nats_adapter import StrokePublisher
from src.routes.chunks import router as chunks_router
from src.routes.status import router as status_router
from src.storage.idempotency_repo import RedisIdempotencyRepo
from src.storage.upload_status_repo import UploadStatusRepo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: connect NATS, Redis, DB. Shutdown: close all."""
    configure_logging(service_name="svc-stroke-ingest")

    nats_client = await create_nats_client(url=NATS_URL, creds=NATS_CREDS)
    app.state.nats_client = nats_client
    app.state.stroke_publisher = StrokePublisher(nats_client)

    idem_repo = RedisIdempotencyRepo(REDIS_URL)
    await idem_repo.connect()
    app.state.idempotency_repo = idem_repo

    status_repo = UploadStatusRepo(DATABASE_URL)
    await status_repo.connect()
    app.state.upload_status_repo = status_repo

    yield

    await status_repo.close()
    await idem_repo.close()
    await nats_client.close()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ExamPen Stroke Ingest API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(chunks_router, prefix="/api/v1/strokes")
    app.include_router(status_router, prefix="/api/v1/exams")
    return app


app = create_app()
