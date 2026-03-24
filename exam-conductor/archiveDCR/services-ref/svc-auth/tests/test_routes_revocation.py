"""Integration tests for revocation routes — revoke / check / un-revoke cycle.

Test IDs: I-AUTH-REV-01 through I-AUTH-REV-05

Mocks the DB repos and auth dependency so routes can be tested
without a live database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from exampen_common.auth import ExamPenUser
from src.main import create_app


# -- Fixtures --------------------------------------------------------------

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY = _PRIVATE_KEY.public_key()


def _make_principal_token() -> str:
    """Create a JWT for a principal-level user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "admin-001",
        "tenant_id": "tenant-abc",
        "role": "admin",
        "name": "Principal Admin",
        "email": "admin@school.edu",
        "jti": "jti-admin-001",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    return pyjwt.encode(payload, _PRIVATE_KEY, algorithm="RS256", headers={"kid": "test-kid-1"})


def _make_student_token() -> str:
    """Create a JWT for a student-level user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "student-001",
        "tenant_id": "tenant-abc",
        "role": "student",
        "name": "Test Student",
        "jti": "jti-student-001",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    return pyjwt.encode(payload, _PRIVATE_KEY, algorithm="RS256", headers={"kid": "test-kid-1"})


def _build_client(*, auth_role: str = "admin") -> TestClient:
    """Build a TestClient with mocked dependencies."""
    app = create_app()

    # Mock JWKS
    jwks_mock = AsyncMock()
    jwks_mock.get_signing_key = AsyncMock(return_value=_PUBLIC_KEY)
    jwks_mock.warmup = AsyncMock()
    app.state.jwks_manager = jwks_mock
    app.state.db_engine = MagicMock()
    app.state.session_factory = MagicMock()

    # Mock revocation repo
    rev_repo = AsyncMock()
    rev_repo.revoke = AsyncMock(return_value={
        "jti": "jti-target",
        "revoked": True,
        "revoked_at": datetime.now(timezone.utc).isoformat(),
        "reason": "test revocation",
    })
    rev_repo.is_revoked = AsyncMock(return_value={"jti": "jti-target", "revoked": False})
    rev_repo.delete = AsyncMock(return_value=True)
    app.state.revocation_repo = rev_repo

    # Mock role mapping repo
    role_repo = AsyncMock()
    role_repo.get_all = AsyncMock(return_value={})
    app.state.role_mapping_repo = role_repo

    return TestClient(app, raise_server_exceptions=False)


# -- Tests -----------------------------------------------------------------


def test_revoke_accepts_and_stores():
    """I-AUTH-REV-01: POST /revocations with principal token returns 202."""
    client = _build_client()
    token = _make_principal_token()
    resp = client.post(
        "/api/v1/auth/revocations",
        json={"jti": "jti-target", "reason": "Compromised device"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["jti"] == "jti-target"
    assert body["revoked"] is True


def test_check_revocation_status():
    """I-AUTH-REV-02: GET /revocations/{jti} returns revocation status."""
    client = _build_client()
    token = _make_principal_token()
    resp = client.get(
        "/api/v1/auth/revocations/jti-target",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["jti"] == "jti-target"
    assert "revoked" in body


def test_unrevoke():
    """I-AUTH-REV-03: DELETE /revocations/{jti} un-revokes the token."""
    client = _build_client()
    token = _make_principal_token()
    resp = client.delete(
        "/api/v1/auth/revocations/jti-target",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["jti"] == "jti-target"
    assert body["revoked"] is False


def test_student_cannot_revoke():
    """I-AUTH-REV-04: Student token cannot create revocations (403)."""
    client = _build_client()
    token = _make_student_token()
    resp = client.post(
        "/api/v1/auth/revocations",
        json={"jti": "jti-target", "reason": "Should not work"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


def test_revoke_requires_reason_min_length():
    """I-AUTH-REV-05: Reason shorter than 5 chars returns 422."""
    client = _build_client()
    token = _make_principal_token()
    resp = client.post(
        "/api/v1/auth/revocations",
        json={"jti": "jti-target", "reason": "bad"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422
