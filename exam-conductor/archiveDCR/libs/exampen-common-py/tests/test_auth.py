"""Tests for exampen_common.auth — JWT validation, JWKS cache, kid mismatch.

Test IDs: U-COMMON-AUTH-01 through U-COMMON-AUTH-10
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from fastapi import HTTPException
from jwt import PyJWKSet

from exampen_common.auth import (
    ExamPenUser,
    JWKSManager,
    _normalize_claims,
    validate_token,
)

# -- Helpers — generate test RSA keys & JWKS --------------------------------


def _generate_rsa_keypair() -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def _private_pem(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _jwks_dict(pub: rsa.RSAPublicKey, kid: str = "test-kid-1") -> dict:
    """Build a minimal JWKS dict from an RSA public key."""
    from jwt.algorithms import RSAAlgorithm

    jwk = json.loads(RSAAlgorithm.to_jwk(pub))
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    return {"keys": [jwk]}


def _make_token(
    private_key: rsa.RSAPrivateKey,
    kid: str = "test-kid-1",
    claims: dict | None = None,
) -> str:
    payload = {
        "sub": "user-42",
        "tenant_id": "tenant-1",
        "role": "tutor",
        "name": "Alice",
        "email": "alice@example.com",
        "exp": int(time.time()) + 3600,
        **(claims or {}),
    }
    return pyjwt.encode(
        payload, _private_pem(private_key), algorithm="RS256", headers={"kid": kid},
    )


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture()
def rsa_keys():
    return _generate_rsa_keypair()


@pytest.fixture()
def jwks_dict(rsa_keys):
    _, pub = rsa_keys
    return _jwks_dict(pub)


@pytest.fixture()
def token(rsa_keys):
    priv, _ = rsa_keys
    return _make_token(priv)


# -- U-COMMON-AUTH-01: Normalize claims maps Stoody roles --------------------


def test_normalize_claims_tutor():
    user = _normalize_claims({"sub": "u1", "tenant_id": "t1", "role": "tutor"})
    assert user.stoody_role == "tutor"
    assert "teacher" in user.exampen_roles


def test_normalize_claims_unknown_role():
    user = _normalize_claims({"sub": "u1", "tenant_id": "t1", "role": "alien"})
    assert "no_exampen_access" in user.exampen_roles


# -- U-COMMON-AUTH-02: ExamPenUser is frozen dataclass ----------------------


def test_exampen_user_immutable():
    user = ExamPenUser(user_id="1", tenant_id="t", stoody_role="student")
    with pytest.raises(AttributeError):
        user.user_id = "changed"  # type: ignore[misc]


# -- U-COMMON-AUTH-03: Token validation with test keys ----------------------


@pytest.mark.asyncio
async def test_validate_token_success(rsa_keys, jwks_dict, token):
    async def mock_fetch(url):
        return PyJWKSet.from_dict(jwks_dict)

    mgr = JWKSManager(jwks_url="http://fake", ttl_seconds=60)
    with patch("exampen_common.auth._fetch_jwks", side_effect=mock_fetch):
        user = await validate_token(token, manager=mgr)

    assert user.user_id == "user-42"
    assert user.tenant_id == "tenant-1"
    assert user.stoody_role == "tutor"


# -- U-COMMON-AUTH-04: Expired token is rejected ----------------------------


@pytest.mark.asyncio
async def test_validate_token_expired(rsa_keys, jwks_dict):
    priv, _ = rsa_keys
    expired = _make_token(priv, claims={"exp": int(time.time()) - 100})

    async def mock_fetch(url):
        return PyJWKSet.from_dict(jwks_dict)

    mgr = JWKSManager(jwks_url="http://fake", ttl_seconds=60)
    with patch("exampen_common.auth._fetch_jwks", side_effect=mock_fetch):
        with pytest.raises(HTTPException) as exc_info:
            await validate_token(expired, manager=mgr)
        assert exc_info.value.status_code == 401


# -- U-COMMON-AUTH-05: kid mismatch triggers re-fetch -----------------------


@pytest.mark.asyncio
async def test_kid_mismatch_triggers_refetch(rsa_keys, jwks_dict):
    priv, pub = rsa_keys
    new_token = _make_token(priv, kid="new-kid")
    rotated_jwks = _jwks_dict(pub, kid="new-kid")
    call_count = 0

    async def mock_fetch(url):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return PyJWKSet.from_dict(jwks_dict)  # old keyset
        return PyJWKSet.from_dict(rotated_jwks)  # rotated

    mgr = JWKSManager(jwks_url="http://fake", ttl_seconds=60)
    with patch("exampen_common.auth._fetch_jwks", side_effect=mock_fetch):
        await mgr.warmup()  # TTL still valid after this
        user = await validate_token(new_token, manager=mgr)

    assert user.user_id == "user-42"
    assert call_count >= 2


# -- U-COMMON-AUTH-06: JWKS fetch failure with expired cache => 503 ---------


@pytest.mark.asyncio
async def test_jwks_unavailable_expired_cache():
    async def mock_fetch(url):
        raise ConnectionError("unreachable")

    mgr = JWKSManager(jwks_url="http://fake", ttl_seconds=60)
    with patch("exampen_common.auth._fetch_jwks", side_effect=mock_fetch):
        with pytest.raises(HTTPException) as exc_info:
            await mgr.get_signing_key("any-kid")
        assert exc_info.value.status_code == 503


# -- U-COMMON-AUTH-07: Malformed token header -------------------------------


@pytest.mark.asyncio
async def test_malformed_token():
    with pytest.raises(HTTPException) as exc_info:
        await validate_token("not-a-jwt")
    assert exc_info.value.status_code == 401


# -- U-COMMON-AUTH-08: Multiple concurrent keysets --------------------------


@pytest.mark.asyncio
async def test_multiple_concurrent_keysets():
    priv1, pub1 = _generate_rsa_keypair()
    priv2, pub2 = _generate_rsa_keypair()

    from jwt.algorithms import RSAAlgorithm
    jwk1 = json.loads(RSAAlgorithm.to_jwk(pub1))
    jwk1.update({"kid": "k1", "use": "sig", "alg": "RS256"})
    jwk2 = json.loads(RSAAlgorithm.to_jwk(pub2))
    jwk2.update({"kid": "k2", "use": "sig", "alg": "RS256"})
    multi_jwks = {"keys": [jwk1, jwk2]}

    tok1 = _make_token(priv1, kid="k1")
    tok2 = _make_token(priv2, kid="k2")

    async def mock_fetch(url):
        return PyJWKSet.from_dict(multi_jwks)

    mgr = JWKSManager(jwks_url="http://fake", ttl_seconds=60)
    with patch("exampen_common.auth._fetch_jwks", side_effect=mock_fetch):
        user1 = await validate_token(tok1, manager=mgr)
        user2 = await validate_token(tok2, manager=mgr)

    assert user1.user_id == "user-42"
    assert user2.user_id == "user-42"


# -- U-COMMON-AUTH-09: Rotated key succeeds even when cache TTL is valid ----


@pytest.mark.asyncio
async def test_rotated_key_succeeds_within_ttl():
    """A newly rotated key (new kid) must be accepted even when the
    cache TTL has NOT expired. This is the core key-rotation fix."""
    priv, pub = _generate_rsa_keypair()
    old_jwks = _jwks_dict(pub, kid="old-kid")
    new_jwks = _jwks_dict(pub, kid="rotated-kid")
    new_token = _make_token(priv, kid="rotated-kid")
    call_count = 0

    async def mock_fetch(url):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return PyJWKSet.from_dict(old_jwks)
        return PyJWKSet.from_dict(new_jwks)

    mgr = JWKSManager(jwks_url="http://fake", ttl_seconds=86400)
    with patch("exampen_common.auth._fetch_jwks", side_effect=mock_fetch):
        await mgr.warmup()  # fetch #1 — populates cache with old-kid
        assert mgr._cache_valid()  # cache is still within TTL
        user = await validate_token(new_token, manager=mgr)

    assert user.user_id == "user-42"
    assert call_count == 2  # warmup + forced refresh on kid miss


# -- U-COMMON-AUTH-10: Forced refresh is rate-limited (30 s cooldown) -------


@pytest.mark.asyncio
async def test_forced_refresh_rate_limited():
    """A second kid-miss within the 30 s cooldown must NOT trigger
    another JWKS fetch, preventing DDoS via random kids."""
    priv, pub = _generate_rsa_keypair()
    old_jwks = _jwks_dict(pub, kid="old-kid")
    call_count = 0

    async def mock_fetch(url):
        nonlocal call_count
        call_count += 1
        return PyJWKSet.from_dict(old_jwks)

    mgr = JWKSManager(jwks_url="http://fake", ttl_seconds=86400)
    with patch("exampen_common.auth._fetch_jwks", side_effect=mock_fetch):
        await mgr.warmup()  # fetch #1
        assert call_count == 1

        # First unknown kid => force refresh (fetch #2)
        with pytest.raises(HTTPException) as exc_info:
            await mgr.get_signing_key("unknown-kid-1")
        assert exc_info.value.status_code == 401
        assert call_count == 2

        # Second unknown kid within cooldown => NO extra fetch
        with pytest.raises(HTTPException) as exc_info:
            await mgr.get_signing_key("unknown-kid-2")
        assert exc_info.value.status_code == 401
        assert call_count == 2  # cooldown prevented fetch #3
