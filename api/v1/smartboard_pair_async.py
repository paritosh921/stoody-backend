"""
Smartboard Pairing API.

Replaces the broken multi-pen passcode flow. The teacher's phone calls
`POST /api/v1/smartboard-pair/register` with a tutor JWT; the backend mints
a 6-digit code and stores `{code → tutor_identity}` in Redis with a 5-minute
TTL. The smartboard tablet calls `POST /api/v1/smartboard-pair/redeem` with
the code and receives an 8-hour smartboard-scoped JWT (signed with
`JWT_SECRET_KEY` — same as a tutor login JWT, so all existing smartboard
endpoints accept it).

See `stoody-multi-pen/sb-android/PAIRING_API.md` for the full contract.
"""

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user
from core.auth import AuthManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/smartboard-pair", tags=["Smartboard Pairing"])

# Rate limiter (matches pattern in auth_async.py)
limiter = Limiter(key_func=get_remote_address)

# Constants
PAIR_CODE_TTL_SECONDS = 300          # 5 minutes
SMARTBOARD_TOKEN_HOURS = 8           # one teaching day
PAIR_CODE_DIGITS = 6
PAIR_CODE_MAX = 10 ** PAIR_CODE_DIGITS
PAIR_CODE_RETRIES = 2                # 1 retry on collision

# Redis key namespaces (raw, used directly via cache_manager.redis_client)
PREFIX = "skillbot"  # matches CacheManager._make_key default
KEY_CODE = f"{PREFIX}:sbpair:code:{{code}}"
KEY_TUTOR = f"{PREFIX}:sbpair:tutor:{{tutor_id}}"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RegisterPairCodeResponse(BaseModel):
    code: str = Field(..., min_length=PAIR_CODE_DIGITS, max_length=PAIR_CODE_DIGITS)
    expires_at: str  # ISO-8601 UTC


class RedeemPairCodeRequest(BaseModel):
    code: str = Field(..., min_length=PAIR_CODE_DIGITS, max_length=PAIR_CODE_DIGITS)


class TutorIdentity(BaseModel):
    tutor_id: str
    name: str = ""
    email: str = ""
    tenant_id: Optional[str] = None


class RedeemPairCodeResponse(BaseModel):
    access_token: str
    expires_at: str  # ISO-8601 UTC
    tutor: TutorIdentity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen_code() -> str:
    """Cryptographically secure 6-digit numeric code, zero-padded."""
    return f"{secrets.randbelow(PAIR_CODE_MAX):0{PAIR_CODE_DIGITS}d}"


def _redis_client(request: Request):
    """
    Return the underlying redis-py client from `app.state.cache.redis_client`,
    or None if Redis isn't initialized (in-memory fallback would defeat the
    purpose of pairing — we just 503 in that case).
    """
    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        return None
    return getattr(cache, "redis_client", None)


def _extract_tutor_identity(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull the fields needed to mint a smartboard-scoped JWT later. Mirrors
    what `core.auth.AuthManager.create_user_session` puts on a tutor token.
    """
    user_type = (user.get("user_type") or user.get("role") or "").lower()
    if user_type not in ("tutor", "teacher"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only tutors can pair smartboards",
        )

    tutor_id = (
        user.get("tutor_id")
        or user.get("user_id")
        or user.get("sub")
        or user.get("_id")
    )
    if not tutor_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing tutor identity",
        )

    return {
        "tutor_id": str(tutor_id),
        "user_id": str(tutor_id),
        "user_type": "tutor",
        "name": user.get("full_name") or user.get("name") or user.get("username") or "",
        "email": user.get("email") or "",
        "username": user.get("username") or user.get("email") or "",
        "tenant_id": user.get("tenant_id"),
        "admin_id": user.get("admin_id"),
        "subdomain": user.get("subdomain"),
        "db_name": user.get("db_name"),
        "institution_id": user.get("institution_id"),
        "permissions": user.get("permissions"),
        "enabled_features": user.get("enabled_features"),
        "enabled_features_v2": user.get("enabled_features_v2"),
    }


def _auth_manager(request: Request) -> AuthManager:
    auth = getattr(request.app.state, "auth", None)
    if auth is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service not initialized",
        )
    return auth


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=RegisterPairCodeResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a 6-digit smartboard pairing code",
)
@limiter.limit("10/minute")
async def register_pair_code(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> RegisterPairCodeResponse:
    """
    Tutor calls this from their phone. Returns a 6-digit code (5-min TTL)
    that the smartboard tablet redeems for an 8-hour JWT.
    """
    redis = _redis_client(request)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pairing service unavailable (cache offline)",
        )

    identity = _extract_tutor_identity(current_user)
    tutor_id = identity["tutor_id"]
    payload = json.dumps(identity, default=str)

    # Single-active-code-per-tutor: invalidate any previous code first.
    prev_key = KEY_TUTOR.format(tutor_id=tutor_id)
    try:
        prev_code = await redis.get(prev_key)
        if prev_code:
            prev_code_str = prev_code.decode() if isinstance(prev_code, bytes) else str(prev_code)
            await redis.delete(KEY_CODE.format(code=prev_code_str), prev_key)
    except Exception as exc:
        logger.warning("Failed to clear previous pair code for tutor %s: %s", tutor_id, exc)

    # Allocate a fresh code with one collision retry.
    code: Optional[str] = None
    last_error: Optional[Exception] = None
    for _ in range(PAIR_CODE_RETRIES):
        candidate = _gen_code()
        try:
            stored = await redis.set(
                KEY_CODE.format(code=candidate),
                payload,
                ex=PAIR_CODE_TTL_SECONDS,
                nx=True,
            )
        except Exception as exc:
            last_error = exc
            continue
        if stored:
            code = candidate
            break

    if code is None:
        logger.error(
            "Could not allocate pair code for tutor %s after %d retries (last error: %s)",
            tutor_id, PAIR_CODE_RETRIES, last_error,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not allocate pairing code, please retry",
        )

    try:
        await redis.set(prev_key, code, ex=PAIR_CODE_TTL_SECONDS)
    except Exception as exc:
        # Secondary index is best-effort; the primary key is enough for redeem.
        logger.warning("Failed to write tutor index for pair code: %s", exc)

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=PAIR_CODE_TTL_SECONDS)
    logger.info("[SBPAIR] Code issued for tutor %s, expires at %s", tutor_id, expires_at.isoformat())

    return RegisterPairCodeResponse(
        code=code,
        expires_at=expires_at.isoformat().replace("+00:00", "Z"),
    )


@router.post(
    "/redeem",
    response_model=RedeemPairCodeResponse,
    status_code=status.HTTP_200_OK,
    summary="Redeem a 6-digit smartboard pairing code",
)
@limiter.limit("20/minute")
async def redeem_pair_code(
    request: Request,
    body: RedeemPairCodeRequest = Body(...),
) -> RedeemPairCodeResponse:
    """
    Smartboard tablet calls this with the 6-digit code shown on the teacher's
    phone. Returns an 8-hour tutor JWT (signed with `JWT_SECRET_KEY` so all
    existing smartboard endpoints accept it) plus tutor identity for display.
    """
    code = body.code.strip()
    if len(code) != PAIR_CODE_DIGITS or not code.isdigit():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired pairing code",
        )

    redis = _redis_client(request)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pairing service unavailable (cache offline)",
        )

    code_key = KEY_CODE.format(code=code)
    try:
        raw = await redis.getdel(code_key)
    except AttributeError:
        # Fallback for redis-py < 4.2 which lacks GETDEL.
        raw = await redis.get(code_key)
        if raw is not None:
            await redis.delete(code_key)
    except Exception as exc:
        logger.error("[SBPAIR] Redis error during redeem: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pairing service error",
        ) from exc

    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired pairing code",
        )

    try:
        identity: Dict[str, Any] = json.loads(raw if isinstance(raw, (str, bytes)) else str(raw))
    except (TypeError, ValueError) as exc:
        logger.error("[SBPAIR] Corrupt pair-code payload for code %s: %s", code, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired pairing code",
        ) from exc

    tutor_id = str(identity.get("tutor_id") or identity.get("user_id") or "")
    if not tutor_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired pairing code",
        )

    # Best-effort cleanup of secondary index.
    try:
        await redis.delete(KEY_TUTOR.format(tutor_id=tutor_id))
    except Exception:
        pass

    # Mint the smartboard JWT — same shape as a tutor-login JWT plus
    # `device: "smartboard"` for audit trails.
    auth_manager = _auth_manager(request)
    expires_delta = timedelta(hours=SMARTBOARD_TOKEN_HOURS)
    token_payload = {
        "sub": tutor_id,
        "user_id": tutor_id,
        "tutor_id": tutor_id,
        "user_type": "tutor",
        "username": identity.get("username") or identity.get("email") or "",
        "email": identity.get("email"),
        "tenant_id": identity.get("tenant_id"),
        "admin_id": identity.get("admin_id"),
        "subdomain": identity.get("subdomain"),
        "db_name": identity.get("db_name"),
        "institution_id": identity.get("institution_id"),
        "permissions": identity.get("permissions"),
        "enabled_features": identity.get("enabled_features"),
        "enabled_features_v2": identity.get("enabled_features_v2"),
        "device": "smartboard",
    }
    # Drop None values so the JWT stays compact.
    token_payload = {k: v for k, v in token_payload.items() if v is not None}

    access_token = auth_manager.create_access_token(
        token_payload,
        expires_delta=expires_delta,
    )
    expires_at = datetime.now(timezone.utc) + expires_delta

    logger.info("[SBPAIR] Code %s redeemed for tutor %s", code, tutor_id)

    return RedeemPairCodeResponse(
        access_token=access_token,
        expires_at=expires_at.isoformat().replace("+00:00", "Z"),
        tutor=TutorIdentity(
            tutor_id=tutor_id,
            name=str(identity.get("name") or ""),
            email=str(identity.get("email") or ""),
            tenant_id=identity.get("tenant_id"),
        ),
    )
