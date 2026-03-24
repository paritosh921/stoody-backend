"""svc-auth — Stoody JWT validation, role mapping, and revocation.

Entry point: creates the FastAPI application, includes routers,
and manages startup/shutdown lifecycle hooks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from exampen_common.auth import JWKSManager
from exampen_common.db import create_pool, session_factory
from exampen_common.logging import configure_logging, RequestIdMiddleware

from src.config import DATABASE_URL, STOODY_JWKS_URL
from src.middleware.rls import RLSMiddleware
from src.routes.introspect import router as introspect_router
from src.routes.revocation import router as revocation_router
from src.storage.revocation_repo import RevocationRepo
from src.storage.role_mapping_repo import RoleMappingRepo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: warm JWKS cache, create DB pool. Shutdown: dispose pool."""
    configure_logging()
    jwks = JWKSManager(jwks_url=STOODY_JWKS_URL)
    await jwks.warmup()
    engine = create_pool(url=DATABASE_URL)
    sf = session_factory(engine)

    app.state.jwks_manager = jwks
    app.state.db_engine = engine
    app.state.session_factory = sf
    app.state.revocation_repo = RevocationRepo(sf)
    app.state.role_mapping_repo = RoleMappingRepo(sf)

    yield

    await engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ExamPen Auth API",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Middleware order matters: outermost first.  RequestIdMiddleware runs
    # first (sets correlation IDs), then RLSMiddleware sets tenant context.
    app.add_middleware(RLSMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(introspect_router, prefix="/api/v1/auth")
    app.include_router(revocation_router, prefix="/api/v1/auth")
    return app


app = create_app()
