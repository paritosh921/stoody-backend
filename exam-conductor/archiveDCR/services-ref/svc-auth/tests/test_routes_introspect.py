"""Integration tests for POST /introspect and GET /me routes.

Test IDs: I-AUTH-INT-01 through I-AUTH-INT-05

These tests mock the JWKS manager, Stoody client, and DB repos
so they exercise the full route → domain → response pipeline
without requiring a live database or Stoody server.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from fastapi.testclient import TestClient

from src.main import create_app


# -- Fixtures --------------------------------------------------------------

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY = _PRIVATE_KEY.public_key()


def _make_token(
    claims: dict | None = None,
    expired: bool = False,
) -> str:
    """Create a signed test JWT."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "user-123",
        "tenant_id": "tenant-abc",
        "role": "tutor",
        "name": "Alice",
        "email": "alice@test.com",
        "jti": "jti-test-001",
        "iat": now,
        "exp": now + timedelta(hours=-1 if expired else 1),
        **(claims or {}),
    }
    return pyjwt.encode(payload, _PRIVATE_KEY, algorithm="RS256", headers={"kid": "test-kid-1"})


def _mock_app() -> TestClient:
    """Build a TestClient with all async dependencies mocked."""
    app = create_app()

    # Mock JWKS manager
    jwks_mock = AsyncMock()
    jwks_mock.get_signing_key = AsyncMock(
        return_value=_PUBLIC_KEY
    )
    jwks_mock.warmup = AsyncMock()
    app.state.jwks_manager = jwks_mock

    # Mock DB engine (not used in introspect route directly)
    app.state.db_engine = MagicMock()

    # Mock revocation repo
    rev_repo = AsyncMock()
    rev_repo.is_revoked = AsyncMock(return_value={"jti": "jti-test-001", "revoked": False})
    app.state.revocation_repo = rev_repo

    # Mock role mapping repo
    role_repo = AsyncMock()
    role_repo.get_all = AsyncMock(return_value={})
    app.state.role_mapping_repo = role_repo

    # Mock session factory
    app.state.session_factory = MagicMock()

    return TestClient(app, raise_server_exceptions=False)


# -- Tests -----------------------------------------------------------------


@pytest.fixture
def client():
    return _mock_app()


def test_introspect_valid_token(client: TestClient):
    """I-AUTH-INT-01: Valid Stoody JWT returns normalized claims."""
    token = _make_token()
    with patch("src.routes.introspect.StoodyClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get_user_profile = AsyncMock(return_value={
            "name": "Alice Tutor",
            "email": "alice@school.edu",
        })
        mock_instance.get_parent_children = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        resp = client.post("/api/v1/auth/introspect", json={"token": token})

    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "user-123"
    assert body["tenant_id"] == "tenant-abc"
    assert body["stoody_role"] == "tutor"
    assert "evaluator" in body["exampen_roles"]
    assert body["token_source"] == "stoody_jwt"
    assert body["token_status"] == "valid"
    assert body["profile"]["display_name"] == "Alice Tutor"


def test_introspect_expired_token(client: TestClient):
    """I-AUTH-INT-02: Expired token returns 401."""
    token = _make_token(expired=True)
    with patch("src.routes.introspect.StoodyClient"):
        resp = client.post("/api/v1/auth/introspect", json={"token": token})
    assert resp.status_code == 401


def test_introspect_revoked_token():
    """I-AUTH-INT-03: Revoked token returns 401."""
    app = create_app()
    jwks_mock = AsyncMock()
    jwks_mock.get_signing_key = AsyncMock(return_value=_PUBLIC_KEY)
    jwks_mock.warmup = AsyncMock()
    app.state.jwks_manager = jwks_mock
    app.state.db_engine = MagicMock()
    app.state.session_factory = MagicMock()

    rev_repo = AsyncMock()
    rev_repo.is_revoked = AsyncMock(return_value={"jti": "jti-test-001", "revoked": True})
    app.state.revocation_repo = rev_repo

    role_repo = AsyncMock()
    role_repo.get_all = AsyncMock(return_value={})
    app.state.role_mapping_repo = role_repo

    client = TestClient(app, raise_server_exceptions=False)
    token = _make_token()

    with patch("src.routes.introspect.StoodyClient"):
        resp = client.post("/api/v1/auth/introspect", json={"token": token})
    assert resp.status_code == 401


def test_introspect_stoody_profile_down(client: TestClient):
    """I-AUTH-INT-04: Stoody profile down -> graceful degradation."""
    token = _make_token()
    with patch("src.routes.introspect.StoodyClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get_user_profile = AsyncMock(return_value=None)
        mock_instance.get_parent_children = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        resp = client.post("/api/v1/auth/introspect", json={"token": token})

    assert resp.status_code == 200
    body = resp.json()
    # Falls back to JWT claim name
    assert body["profile"]["display_name"] == "Alice"


def test_me_endpoint(client: TestClient):
    """I-AUTH-INT-05: GET /me returns claims from bearer header."""
    token = _make_token()
    with patch("src.routes.introspect.StoodyClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get_user_profile = AsyncMock(return_value=None)
        mock_instance.get_parent_children = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    assert resp.json()["user_id"] == "user-123"


def test_me_no_auth_header(client: TestClient):
    """I-AUTH-INT-05b: GET /me without Authorization header returns 401."""
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401
