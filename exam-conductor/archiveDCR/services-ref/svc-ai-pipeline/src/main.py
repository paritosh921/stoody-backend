"""FastAPI entry point for svc-ai-pipeline."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.adapters.model_adapter import ModelRegistry
from src.config import get_settings
from src.events.page_consumer import PageConsumer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start NATS subscription and model registry on startup."""
    settings = get_settings()

    registry = ModelRegistry(settings.model_dir, mock=settings.mock_mode)
    registry.load_all()
    app.state.model_registry = registry

    consumer = PageConsumer(settings, registry)
    await consumer.start()
    app.state.page_consumer = consumer

    yield

    await consumer.stop()


app = FastAPI(
    title="svc-ai-pipeline",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "svc-ai-pipeline"}


@app.get("/ready")
async def ready() -> dict:
    """Readiness probe — checks model registry is loaded."""
    registry: ModelRegistry = app.state.model_registry
    models = registry.list_models()
    return {"status": "ready", "loaded_models": models}
