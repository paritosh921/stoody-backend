"""RSA keypair generation for the Stoody mock server.

Generates an RSA-2048 keypair at import time and provides helpers
to export JWKS JSON and sign test JWTs.
"""

from __future__ import annotations

import json
import time
from typing import Any

import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from jwt.algorithms import RSAAlgorithm

# ---------------------------------------------------------------------------
# Keypair generation (module-level — generated once per process)
# ---------------------------------------------------------------------------

KID = "stoody-mock-kid-1"

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY = _PRIVATE_KEY.public_key()


def get_private_key() -> rsa.RSAPrivateKey:
    """Return the mock RSA private key."""
    return _PRIVATE_KEY


def get_public_key() -> rsa.RSAPublicKey:
    """Return the mock RSA public key."""
    return _PUBLIC_KEY


# ---------------------------------------------------------------------------
# JWKS export
# ---------------------------------------------------------------------------


def get_jwks_dict() -> dict[str, Any]:
    """Return a JWKS-format dict containing the public key."""
    # Use PyJWT's RSAAlgorithm to convert the public key to JWK
    jwk_dict = json.loads(RSAAlgorithm.to_jwk(_PUBLIC_KEY))
    jwk_dict["kid"] = KID
    jwk_dict["use"] = "sig"
    jwk_dict["alg"] = "RS256"
    return {"keys": [jwk_dict]}


# ---------------------------------------------------------------------------
# Token generation (for tests)
# ---------------------------------------------------------------------------


def make_token(
    user_id: str = "user-001",
    tenant_id: str = "tenant-001",
    role: str = "tutor",
    name: str = "Test User",
    email: str = "test@stoody.local",
    extra_claims: dict[str, Any] | None = None,
    ttl_seconds: int = 3600,
) -> str:
    """Generate a signed JWT for testing.

    Returns a compact JWS string.
    """
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "name": name,
        "email": email,
        "jti": f"jti-{user_id}-{now}",
        "iat": now,
        "exp": now + ttl_seconds,
    }
    if extra_claims:
        payload.update(extra_claims)

    return pyjwt.encode(
        payload,
        _PRIVATE_KEY,
        algorithm="RS256",
        headers={"kid": KID},
    )
