"""PostgreSQL async connection pool with RLS tenant injection.

Provides:
- AsyncPG-backed connection pool (via SQLAlchemy async engine)
- RLS middleware that sets ``app.current_tenant`` per request
- Health-check query
- Environment-based configuration
"""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from exampen_common.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen",
)
_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Pool factory
# ---------------------------------------------------------------------------


def create_pool(
    url: str = _DATABASE_URL,
    pool_size: int = _POOL_SIZE,
    max_overflow: int = _MAX_OVERFLOW,
    pool_timeout: int = _POOL_TIMEOUT,
    **engine_kwargs: Any,
) -> AsyncEngine:
    """Create a SQLAlchemy async engine backed by asyncpg.

    Returns the engine so callers can create sessions via
    :func:`session_factory` or use raw connections.
    """
    engine = create_async_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        echo=False,
        **engine_kwargs,
    )
    _log.info("PostgreSQL pool created (size=%d, overflow=%d)", pool_size, max_overflow)
    return engine


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session maker bound to *engine*."""
    return async_sessionmaker(engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# RLS middleware
# ---------------------------------------------------------------------------


async def rls_middleware(
    conn: AsyncConnection,
    tenant_id: str,
) -> None:
    """Set the RLS tenant context variable on an open connection.

    Must be called at the start of every request that accesses
    tenant-scoped tables.  The PostgreSQL RLS policies reference
    ``current_setting('app.current_tenant')``.

    Parameters
    ----------
    conn:
        An active SQLAlchemy async connection.
    tenant_id:
        The tenant identifier extracted from the JWT.
    """
    # Use a parameterized SET via text() to prevent SQL injection.
    # PostgreSQL SET does not support $1 placeholders, so we use
    # set_config() which does.
    await conn.execute(
        text("SELECT set_config('app.current_tenant', :tid, true)"),
        {"tid": tenant_id},
    )
    _log.debug("RLS tenant set to %s", tenant_id)


async def rls_session(
    factory: async_sessionmaker[AsyncSession],
    tenant_id: str,
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a session with RLS tenant already set.

    Usage::

        async for session in rls_session(sf, user.tenant_id):
            result = await session.execute(...)
    """
    async with factory() as session:
        conn = await session.connection()
        await rls_middleware(conn, tenant_id)
        yield session
        await session.commit()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


async def get_health(engine: AsyncEngine) -> dict[str, Any]:
    """Run a cheap health-check query and return status metadata."""
    try:
        async with engine.connect() as conn:
            row = await conn.execute(text("SELECT 1 AS ok"))
            result = row.scalar()
        return {"status": "healthy", "result": result}
    except Exception as exc:
        _log.error("DB health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}
