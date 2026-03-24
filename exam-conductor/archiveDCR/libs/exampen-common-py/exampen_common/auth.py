"""JWT validation against Stoody JWKS endpoint.

Responsibilities:
- Fetch and cache JWKS keyset (TTL 24 h)
- Re-fetch on ``kid`` mismatch (key rotation)
- Validate RS256 JWTs issued by Stoody
- Normalize claims to ExamPen format
- FastAPI dependency helper ``get_current_user()``
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import jwt
from fastapi import Depends, HTTPException, Request, status
from jwt import PyJWKClient, PyJWKSet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_JWKS_URL: str = os.getenv(
    "STOODY_JWKS_URL", "http://localhost:8000/.well-known/jwks.json"
)
_JWKS_TTL_SECONDS: int = int(os.getenv("JWKS_TTL_SECONDS", "86400"))  # 24 h
_JWT_ALGORITHMS: list[str] = ["RS256"]
_JWT_AUDIENCE: str | None = os.getenv("JWT_AUDIENCE")
_JWT_ISSUER: str | None = os.getenv("JWT_ISSUER")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

STOODY_TO_EXAMPEN_ROLE: dict[str, str] = {
    "tutor": "teacher",
    "student": "student",
    "parent": "parent",
}


@dataclass(frozen=True, slots=True)
class ExamPenUser:
    """Normalized user claims consumed by all ExamPen services."""

    user_id: str
    tenant_id: str
    stoody_role: str
    exampen_roles: list[str] = field(default_factory=list)
    name: str = ""
    email: str = ""


# ---------------------------------------------------------------------------
# JWKS cache
# ---------------------------------------------------------------------------


_FORCED_REFRESH_COOLDOWN: float = 30.0  # seconds between forced refreshes


class JWKSManager:
    """Fetches, caches, and rotates Stoody JWKS keysets.

    Supports multiple concurrent keysets so that a rotation does not
    invalidate tokens signed with the previous key.
    """

    def __init__(
        self,
        jwks_url: str = _JWKS_URL,
        ttl_seconds: int = _JWKS_TTL_SECONDS,
    ) -> None:
        self._jwks_url = jwks_url
        self._ttl = ttl_seconds
        self._keyset: PyJWKSet | None = None
        self._fetched_at: float = 0.0
        self._last_forced_at: float = 0.0
        self._lock = asyncio.Lock()

    # -- public API --------------------------------------------------------

    async def get_signing_key(self, kid: str) -> Any:
        """Return the signing key for *kid*, fetching JWKS if needed."""
        key = self._find_key(kid)
        if key is not None:
            return key

        # Cache empty or expired => normal TTL-based refresh first
        if not self._cache_valid():
            await self._refresh()
            key = self._find_key(kid)
            if key is not None:
                return key

        # kid absent from a valid cache => force refresh (key rotation)
        await self._refresh(force=True)

        key = self._find_key(kid)
        if key is not None:
            return key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unknown signing key: {kid}",
        )

    async def warmup(self) -> None:
        """Pre-fetch JWKS at startup so the first request is fast."""
        await self._refresh()

    # -- internals ---------------------------------------------------------

    def _cache_valid(self) -> bool:
        return (
            self._keyset is not None
            and (time.monotonic() - self._fetched_at) < self._ttl
        )

    def _find_key(self, kid: str) -> Any | None:
        if self._keyset is None:
            return None
        for jwk in self._keyset.keys:
            if jwk.key_id == kid:
                return jwk.key
        return None

    async def _refresh(self, *, force: bool = False) -> None:
        async with self._lock:
            if force:
                # Rate-limit forced refreshes to prevent DDoS via
                # tokens with random kids hammering the JWKS endpoint.
                now = time.monotonic()
                if (now - self._last_forced_at) < _FORCED_REFRESH_COOLDOWN:
                    return  # cooldown active — use whatever is cached
            else:
                # Normal TTL-based double-check inside lock
                if self._cache_valid():
                    return
            try:
                keyset = await _fetch_jwks(self._jwks_url)
                self._keyset = keyset
                self._fetched_at = time.monotonic()
                if force:
                    self._last_forced_at = self._fetched_at
            except Exception:
                if self._keyset is not None and self._cache_valid():
                    return  # stale but still within TTL
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="JWKS endpoint unavailable and cache expired",
                )


async def _fetch_jwks(url: str) -> PyJWKSet:
    """Fetch JWKS JSON and return a ``PyJWKSet``."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            resp.raise_for_status()
            data = await resp.json()
    return PyJWKSet.from_dict(data)


# Module-level singleton — services may also instantiate their own.
_default_manager: JWKSManager | None = None


def _get_manager() -> JWKSManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = JWKSManager()
    return _default_manager


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------


async def validate_token(
    token: str,
    manager: JWKSManager | None = None,
) -> ExamPenUser:
    """Validate a Stoody JWT and return normalized :class:`ExamPenUser`."""
    mgr = manager or _get_manager()

    try:
        unverified = jwt.get_unverified_header(token)
    except jwt.exceptions.DecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token header",
        ) from exc

    kid: str = unverified.get("kid", "")
    signing_key = await mgr.get_signing_key(kid)

    decode_opts: dict[str, Any] = {
        "algorithms": _JWT_ALGORITHMS,
        "key": signing_key,
    }
    if _JWT_AUDIENCE:
        decode_opts["audience"] = _JWT_AUDIENCE
    if _JWT_ISSUER:
        decode_opts["issuer"] = _JWT_ISSUER

    try:
        claims: dict[str, Any] = jwt.decode(token, **decode_opts)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from exc

    return _normalize_claims(claims)


def _normalize_claims(claims: dict[str, Any]) -> ExamPenUser:
    """Map raw Stoody JWT claims to :class:`ExamPenUser`."""
    stoody_role = claims.get("role", claims.get("stoody_role", ""))
    exampen_base = STOODY_TO_EXAMPEN_ROLE.get(stoody_role, "no_exampen_access")
    exampen_roles = claims.get("exampen_roles", [exampen_base])

    return ExamPenUser(
        user_id=str(claims.get("sub", claims.get("user_id", ""))),
        tenant_id=str(claims.get("tenant_id", "")),
        stoody_role=stoody_role,
        exampen_roles=exampen_roles,
        name=claims.get("name", ""),
        email=claims.get("email", ""),
    )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_current_user(request: Request) -> ExamPenUser:
    """FastAPI dependency that extracts and validates the Bearer token."""
    auth_header: str | None = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )
    token = auth_header.removeprefix("Bearer ").strip()
    return await validate_token(token)
