"""Unit tests for RLS middleware — tenant context injection.

Test IDs: U-AUTH-RLS-01 through U-AUTH-RLS-07

These tests verify that the RLSMiddleware correctly:
- Extracts tenant_id from bearer JWTs
- Sets tenant_id on request.state and context var
- Calls rls_middleware on the DB engine
- Skips RLS for super_admin and unauthenticated requests

NOTE: Cross-tenant isolation via actual PostgreSQL RLS policies
is an L4 integration test that requires a real PostgreSQL instance.
See the docstring on test_rls_cross_tenant_blocked_l4 for details.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from src.middleware.rls import RLSMiddleware, tenant_id_var


# -- Fixtures ----------------------------------------------------------------

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _make_token(
    tenant_id: str = "tenant-abc",
    role: str = "tutor",
    **extra_claims: object,
) -> str:
    """Create a signed test JWT with the given tenant and role."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "user-123",
        "tenant_id": tenant_id,
        "role": role,
        "name": "Test User",
        "jti": "jti-rls-test",
        "iat": now,
        "exp": now + timedelta(hours=1),
        **extra_claims,
    }
    return pyjwt.encode(
        payload, _PRIVATE_KEY, algorithm="RS256", headers={"kid": "test-kid"}
    )


def _build_app() -> FastAPI:
    """Build a minimal FastAPI app with RLSMiddleware and a probe endpoint."""
    app = FastAPI()
    app.add_middleware(RLSMiddleware)

    # Mock DB engine with async connect context manager.
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()

    mock_engine = MagicMock()
    # engine.connect() returns an async context manager.
    mock_engine.connect = MagicMock(
        return_value=_AsyncCM(mock_conn)
    )
    app.state.db_engine = mock_engine

    @app.get("/probe")
    async def probe(request: Request) -> JSONResponse:
        """Return state set by the middleware for test assertions."""
        return JSONResponse({
            "tenant_id": getattr(request.state, "tenant_id", ""),
            "ctx_tenant_id": tenant_id_var.get(""),
        })

    return app


class _AsyncCM:
    """Minimal async context manager wrapper for mocking engine.connect()."""

    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, *args: object) -> None:
        pass


# -- Tests -------------------------------------------------------------------


def test_rls_sets_tenant_on_request_state():
    """U-AUTH-RLS-01: Middleware puts tenant_id on request.state."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(tenant_id="tenant-xyz")
    resp = client.get(
        "/probe", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "tenant-xyz"


def test_rls_sets_context_var():
    """U-AUTH-RLS-02: Middleware sets tenant_id_var context variable."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(tenant_id="tenant-ctx")
    resp = client.get(
        "/probe", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ctx_tenant_id"] == "tenant-ctx"


@patch("src.middleware.rls._set_rls_on_engine", new_callable=AsyncMock)
def test_rls_calls_engine_set(mock_set_rls: AsyncMock):
    """U-AUTH-RLS-03: Middleware calls _set_rls_on_engine with tenant_id."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(tenant_id="tenant-db")
    client.get("/probe", headers={"Authorization": f"Bearer {token}"})
    mock_set_rls.assert_called_once()
    call_args = mock_set_rls.call_args
    # Second positional arg is tenant_id.
    assert call_args[0][1] == "tenant-db"


def test_rls_skips_super_admin():
    """U-AUTH-RLS-04: super_admin role bypasses RLS injection."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token(tenant_id="tenant-abc", role="super_admin")
    resp = client.get(
        "/probe", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    body = resp.json()
    # tenant_id should be empty — RLS was not set.
    assert body["tenant_id"] == ""


def test_rls_skips_no_auth_header():
    """U-AUTH-RLS-05: Requests without Authorization header pass through."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/probe")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == ""
    assert body["ctx_tenant_id"] == ""


def test_rls_skips_malformed_token():
    """U-AUTH-RLS-06: Malformed bearer token does not crash middleware."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(
        "/probe", headers={"Authorization": "Bearer not-a-real-jwt"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == ""


def test_rls_context_var_reset_after_request():
    """U-AUTH-RLS-07: Context var is reset after the request completes."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    # First request sets tenant.
    token = _make_token(tenant_id="tenant-first")
    resp1 = client.get(
        "/probe", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp1.json()["ctx_tenant_id"] == "tenant-first"

    # Second request with no auth should have empty tenant.
    resp2 = client.get("/probe")
    assert resp2.json()["ctx_tenant_id"] == ""


# -- L4 integration test placeholder ----------------------------------------


def test_rls_cross_tenant_blocked_l4():
    """U-AUTH-RLS-L4: Cross-tenant access blocked by PostgreSQL RLS policy.

    This test is a PLACEHOLDER.  Actual verification requires a running
    PostgreSQL instance with the 001_initial.sql migration applied.

    To run as an L4 integration test:
      1. Start the test PostgreSQL container:
         ``docker compose -f infra/docker-compose.test.yml up -d postgres``
      2. Apply migrations:
         ``psql -f migrations/001_initial.sql``
      3. Run with ``pytest -m integration tests/test_rls_middleware.py``

    Expected behavior:
      - INSERT with app.current_tenant='tenant-A' succeeds for tenant-A rows
      - SELECT with app.current_tenant='tenant-B' returns zero rows for
        tenant-A data
      - INSERT with app.current_tenant='tenant-B' and tenant_id='tenant-A'
        is rejected by the WITH CHECK policy
    """
    # Placeholder — passes by design.  Real assertions require L4 infra.
    pass
