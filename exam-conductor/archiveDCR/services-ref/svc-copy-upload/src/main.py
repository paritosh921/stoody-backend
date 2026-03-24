"""svc-copy-upload — Fallback photo-based answer capture service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI

from exampen_common.logging import RequestIdMiddleware, configure_logging
from exampen_common.nats_client import create_nats_client

from src.adapters.s3_adapter import create_s3_adapter
from src.config import load_settings
from src.routes.uploads import build_router

configure_logging(service_name="svc-copy-upload")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of external connections."""
    settings = load_settings()
    app.state.settings = settings
    app.state.nats = await create_nats_client(url=settings.nats_url)
    app.state.s3 = create_s3_adapter(settings)
    yield
    await app.state.nats.close()


app = FastAPI(
    title="ExamPen Copy Upload API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RequestIdMiddleware)
app.include_router(build_router(), prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
