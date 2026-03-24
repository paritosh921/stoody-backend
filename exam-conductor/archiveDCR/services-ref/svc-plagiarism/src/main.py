"""FastAPI application entry point for svc-plagiarism."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from fastapi import FastAPI

from .config import get_settings
from .events.check_consumer import start_consumer
from .routes.flags import router as flags_router
from .storage.flag_repo import FlagRepo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage connection pools and NATS subscriptions."""
    settings = get_settings()

    # PostgreSQL pool
    pool = await asyncpg.create_pool(dsn=settings.database_url)
    repo = FlagRepo(pool)
    app.state.flag_repo = repo
    app.state.db_pool = pool

    # NATS consumer
    nc = await start_consumer(
        nats_url=settings.nats_url,
        repo=repo,
        pool=pool,
    )
    app.state.nats_client = nc

    yield

    # Shutdown
    await nc.close()
    await pool.close()


app = FastAPI(
    title="ExamPen Plagiarism Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(flags_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
