"""Tests for exampen_common.db — RLS middleware, pool creation, health check.

Test IDs: U-COMMON-DB-01 through U-COMMON-DB-05
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exampen_common.db import (
    create_pool,
    get_health,
    rls_middleware,
    session_factory,
)


# ---------------------------------------------------------------------------
# U-COMMON-DB-01: RLS middleware injects tenant via set_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rls_middleware_calls_set_config():
    """Verify that rls_middleware executes set_config with the tenant id."""
    mock_conn = AsyncMock()
    await rls_middleware(mock_conn, "tenant-abc")

    mock_conn.execute.assert_awaited_once()
    call_args = mock_conn.execute.call_args
    # First positional arg is the text() clause
    sql_text = str(call_args[0][0])
    assert "set_config" in sql_text
    assert "app.current_tenant" in sql_text
    # Second positional arg is the params dict
    params = call_args[0][1]
    assert params["tid"] == "tenant-abc"


# ---------------------------------------------------------------------------
# U-COMMON-DB-02: RLS middleware prevents SQL injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rls_middleware_parameterized():
    """Ensure the tenant id goes through a parameter, not string concat."""
    mock_conn = AsyncMock()
    malicious = "'; DROP TABLE exams; --"
    await rls_middleware(mock_conn, malicious)

    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    # The malicious string is passed as a parameter, not interpolated
    assert params["tid"] == malicious


# ---------------------------------------------------------------------------
# U-COMMON-DB-03: create_pool returns engine
# ---------------------------------------------------------------------------


def test_create_pool_returns_engine():
    """create_pool should produce an AsyncEngine (mocked driver)."""
    with patch("exampen_common.db.create_async_engine") as mock_create:
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine
        engine = create_pool(url="postgresql+asyncpg://u:p@h/d")
    assert engine is mock_engine
    mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# U-COMMON-DB-04: session_factory returns maker
# ---------------------------------------------------------------------------


def test_session_factory_returns_maker():
    with patch("exampen_common.db.async_sessionmaker") as mock_maker:
        mock_engine = MagicMock()
        session_factory(mock_engine)
    mock_maker.assert_called_once_with(mock_engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# U-COMMON-DB-05: get_health returns status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_health_success():
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_conn.execute.return_value = mock_result

    # engine.connect() must return an async context manager
    cm = AsyncMock()
    cm.__aenter__.return_value = mock_conn
    cm.__aexit__.return_value = False

    mock_engine = MagicMock()
    mock_engine.connect.return_value = cm

    result = await get_health(mock_engine)
    assert result["status"] == "healthy"
    assert result["result"] == 1


@pytest.mark.asyncio
async def test_get_health_failure():
    cm = AsyncMock()
    cm.__aenter__.side_effect = ConnectionError("refused")

    mock_engine = MagicMock()
    mock_engine.connect.return_value = cm

    result = await get_health(mock_engine)
    assert result["status"] == "unhealthy"
    assert "refused" in result["error"]
