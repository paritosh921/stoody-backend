"""svc-stroke-proc -- Stroke dedup, normalization, and TimescaleDB commit.

Subscribes to ``stroke.raw`` events from NATS JetStream, deduplicates,
normalizes coordinates, commits to TimescaleDB, and publishes
``stroke.processed`` events.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.logging import configure_logging, RequestIdMiddleware
from exampen_common.nats_client import create_nats_client

from src.config import DATABASE_URL, NATS_URL, NATS_CREDS, SERVICE_NAME
from src.events.raw_consumer import RawStrokeConsumer
from src.events.processed_publisher import ProcessedStrokePublisher
from src.storage.stroke_repo import StrokeRepo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: connect NATS, TimescaleDB. Shutdown: close all."""
    configure_logging(service_name=SERVICE_NAME)

    nats_client = await create_nats_client(url=NATS_URL, creds=NATS_CREDS)
    app.state.nats_client = nats_client

    publisher = ProcessedStrokePublisher(nats_client)
    app.state.publisher = publisher

    stroke_repo = StrokeRepo(DATABASE_URL)
    await stroke_repo.connect()
    app.state.stroke_repo = stroke_repo

    consumer = RawStrokeConsumer(
        nats_client=nats_client,
        stroke_repo=stroke_repo,
        publisher=publisher,
    )
    await consumer.start()
    app.state.consumer = consumer

    yield

    await consumer.stop()
    await stroke_repo.close()
    await nats_client.close()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    return FastAPI(
        title="ExamPen Stroke Processor",
        version="1.0.0",
        lifespan=lifespan,
    )


app = create_app()
