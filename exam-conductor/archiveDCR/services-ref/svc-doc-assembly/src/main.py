"""FastAPI entry point for svc-doc-assembly."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import nats
from fastapi import FastAPI

import asyncpg

from src.config import Settings
from src.adapters.s3_adapter import S3Adapter
from src.adapters.stroke_data_client import TimescaleStrokeClient
from src.events.processed_consumer import ProcessedStrokeConsumer
from src.storage.page_repo import PageRepository


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()

    nc = await nats.connect(settings.nats_url)
    js = nc.jetstream()

    s3 = S3Adapter(settings)
    await s3.ensure_bucket()

    page_repo = PageRepository(settings.database_url)
    await page_repo.connect()

    # Connect to svc-stroke-proc's TimescaleDB for real stroke data
    stroke_pool = await asyncpg.create_pool(
        settings.stroke_database_url, min_size=2, max_size=10,
    )
    stroke_source = TimescaleStrokeClient(stroke_pool)

    consumer = ProcessedStrokeConsumer(
        js=js,
        s3=s3,
        page_repo=page_repo,
        stroke_source=stroke_source,
    )
    await consumer.start()

    app.state.nc = nc
    app.state.consumer = consumer
    app.state.page_repo = page_repo
    app.state.stroke_pool = stroke_pool

    yield

    await consumer.stop()
    await page_repo.disconnect()
    await stroke_pool.close()
    await nc.close()


app = FastAPI(
    title="svc-doc-assembly",
    description="Stroke-to-page rendering and miss indicator detection",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
