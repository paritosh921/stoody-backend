"""
Smartboard Pairing API.

Replaces the broken multi-pen passcode flow. The teacher's phone calls
`POST /api/v1/smartboard-pair/register` with a tutor JWT; the backend mints
a 6-digit code and stores `{code → tutor_identity}` in Redis with a 5-minute
TTL. The smartboard tablet calls `POST /api/v1/smartboard-pair/redeem` with
the code and receives a tutor-selected, timer-bound smartboard-scoped JWT (signed with
`JWT_SECRET_KEY` — same as a tutor login JWT, so all existing smartboard
endpoints accept it).

See `stoody-multi-pen/sb-android/PAIRING_API.md` for the full contract.
"""

import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user
from core.auth import AuthManager
from core.tenant_features import is_feature_enabled, merge_tenant_features

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/smartboard-pair", tags=["Smartboard Pairing"])

# Rate limiter (matches pattern in auth_async.py)
limiter = Limiter(key_func=get_remote_address)

# Constants
PAIR_CODE_TTL_SECONDS = 300          # 5 minutes to enter the code
DEFAULT_SESSION_MINUTES = 60
MIN_SESSION_MINUTES = 5
MAX_SESSION_MINUTES = 480            # one teaching day
LIVE_HEARTBEAT_GRACE_SECONDS = 90
PAIR_CODE_DIGITS = 6
PAIR_CODE_MAX = 10 ** PAIR_CODE_DIGITS
PAIR_CODE_RETRIES = 2                # 1 retry on collision

# Redis key namespaces (raw, used directly via cache_manager.redis_client)
PREFIX = "skillbot"  # matches CacheManager._make_key default
KEY_CODE = f"{PREFIX}:sbpair:code:{{code}}"
KEY_TUTOR = f"{PREFIX}:sbpair:tutor:{{tutor_id}}"
KEY_SESSION = f"{PREFIX}:sbpair:session:{{session_id}}"
KEY_SESSION_TERMINAL = f"{PREFIX}:sbpair:session_terminal:{{session_id}}"
KEY_TUTOR_LIVE = f"{PREFIX}:sbpair:tutor_live:{{tutor_id}}"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RegisterPairCodeRequest(BaseModel):
    session_minutes: int = Field(DEFAULT_SESSION_MINUTES)

    @validator("session_minutes")
    def validate_session_minutes(cls, value: int) -> int:
        return max(MIN_SESSION_MINUTES, min(MAX_SESSION_MINUTES, int(value)))


class RegisterPairCodeResponse(BaseModel):
    code: str = Field(..., min_length=PAIR_CODE_DIGITS, max_length=PAIR_CODE_DIGITS)
    expires_at: str  # ISO-8601 UTC
    session_id: str
    session_expires_at: str  # ISO-8601 UTC
    session_minutes: int


class RedeemPairCodeRequest(BaseModel):
    code: str = Field(..., min_length=PAIR_CODE_DIGITS, max_length=PAIR_CODE_DIGITS)


class TutorIdentity(BaseModel):
    tutor_id: str
    name: str = ""
    email: str = ""
    tenant_id: Optional[str] = None


class CloudCapabilities(BaseModel):
    smartboard_core: bool = False
    smartboard_live_session: bool = False
    smartboard_cloud_access: bool = False


class RedeemPairCodeResponse(BaseModel):
    access_token: str
    expires_at: str  # ISO-8601 UTC
    session_id: str
    tutor: TutorIdentity
    capabilities: CloudCapabilities = CloudCapabilities()


class PairingStatusResponse(BaseModel):
    session_id: str
    status: str
    is_live: bool = False
    code_expires_at: Optional[str] = None
    session_expires_at: Optional[str] = None
    connected_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    signed_out_at: Optional[str] = None
    tutor: Optional[TutorIdentity] = None


class PairingHeartbeatResponse(BaseModel):
    session_id: str
    status: str
    is_live: bool = False
    session_expires_at: Optional[str] = None


class SignOutPairingRequest(BaseModel):
    session_id: str


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


def _decode_request_token_payload(request: Request) -> Dict[str, Any]:
    authorization = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return {}
    auth_manager = _auth_manager(request)
    return auth_manager.decode_access_token(token) or {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ttl_until(expires_at: datetime, floor: int = 300) -> int:
    return max(floor, int((expires_at - _utcnow()).total_seconds()) + 300)


async def _load_session(redis, session_id: str) -> Optional[Dict[str, Any]]:
    raw = await redis.get(KEY_SESSION.format(session_id=session_id))
    if raw is None:
        return None
    return json.loads(raw if isinstance(raw, (str, bytes)) else str(raw))


async def _store_session(redis, session: Dict[str, Any]) -> None:
    terminal_status = await _get_terminal_status(redis, str(session["session_id"]))
    if terminal_status and session.get("status") != terminal_status:
        return

    session_expires_at = _parse_dt(session.get("session_expires_at"))
    code_expires_at = _parse_dt(session.get("code_expires_at"))
    expires_at = session_expires_at or code_expires_at or (_utcnow() + timedelta(minutes=10))
    await redis.set(
        KEY_SESSION.format(session_id=session["session_id"]),
        json.dumps(session, default=str),
        ex=_ttl_until(expires_at),
    )


async def _get_terminal_status(redis, session_id: str) -> Optional[str]:
    raw = await redis.get(KEY_SESSION_TERMINAL.format(session_id=session_id))
    if raw is None:
        return None
    return raw.decode() if isinstance(raw, bytes) else str(raw)


async def _set_terminal_status(redis, session: Dict[str, Any], terminal_status: str) -> None:
    session_expires_at = _parse_dt(session.get("session_expires_at")) or (
        _utcnow() + timedelta(minutes=DEFAULT_SESSION_MINUTES)
    )
    await redis.set(
        KEY_SESSION_TERMINAL.format(session_id=session["session_id"]),
        terminal_status,
        ex=_ttl_until(session_expires_at),
    )


async def _apply_terminal_status(redis, session: Dict[str, Any]) -> Dict[str, Any]:
    terminal_status = await _get_terminal_status(redis, str(session["session_id"]))
    if terminal_status and session.get("status") != terminal_status:
        session = {**session, "status": terminal_status}
    return session


async def _mark_expired_if_needed(redis, session: Dict[str, Any]) -> Dict[str, Any]:
    session = await _apply_terminal_status(redis, session)
    if session.get("status") == "signed_out":
        return session

    now = _utcnow()
    status_value = session.get("status")
    session_expires_at = _parse_dt(session.get("session_expires_at"))
    code_expires_at = _parse_dt(session.get("code_expires_at"))
    should_expire = (
        status_value == "connected"
        and session_expires_at is not None
        and now >= session_expires_at
    ) or (
        status_value == "pending"
        and code_expires_at is not None
        and now >= code_expires_at
    )
    if should_expire:
        session["status"] = "expired"
        session["expired_at"] = _iso(now)
        await _store_session(redis, session)
        await _set_terminal_status(redis, session, "expired")
    return session


def _session_is_live(session: Dict[str, Any]) -> bool:
    if session.get("status") != "connected":
        return False
    session_expires_at = _parse_dt(session.get("session_expires_at"))
    last_seen_at = _parse_dt(session.get("last_seen_at"))
    now = _utcnow()
    if session_expires_at is not None and now >= session_expires_at:
        return False
    if last_seen_at is None:
        return False
    return (now - last_seen_at).total_seconds() <= LIVE_HEARTBEAT_GRACE_SECONDS


def _status_response(session: Dict[str, Any]) -> PairingStatusResponse:
    tutor = TutorIdentity(
        tutor_id=str(session.get("tutor_id") or ""),
        name=str(session.get("tutor_name") or ""),
        email=str(session.get("tutor_email") or ""),
        tenant_id=session.get("tenant_id"),
    ) if session.get("tutor_id") else None
    return PairingStatusResponse(
        session_id=str(session["session_id"]),
        status=str(session.get("status") or "unknown"),
        is_live=_session_is_live(session),
        code_expires_at=session.get("code_expires_at"),
        session_expires_at=session.get("session_expires_at"),
        connected_at=session.get("connected_at"),
        last_seen_at=session.get("last_seen_at"),
        signed_out_at=session.get("signed_out_at"),
        tutor=tutor,
    )


def _require_tutor_session_owner(current_user: Dict[str, Any], session: Dict[str, Any]) -> None:
    identity = _extract_tutor_identity(current_user)
    if str(identity["tutor_id"]) != str(session.get("tutor_id")):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pairing session not found",
        )


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
    body: RegisterPairCodeRequest = Body(default=RegisterPairCodeRequest()),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> RegisterPairCodeResponse:
    """
    Tutor calls this from their phone. Returns a 6-digit code (5-min TTL)
    that the smartboard tablet redeems for a timer-bound JWT.
    """
    # Feature gate: smartboard_cloud_access must be enabled
    if not is_feature_enabled(
        current_user.get("enabled_features"),
        "smartboard_cloud_access",
        current_user.get("enabled_features_v2"),
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Smartboard cloud access is not enabled for your institution",
        )

    redis = _redis_client(request)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pairing service unavailable (cache offline)",
        )

    identity = _extract_tutor_identity(current_user)
    tutor_id = identity["tutor_id"]
    now = _utcnow()
    code_expires_at = now + timedelta(seconds=PAIR_CODE_TTL_SECONDS)
    planned_session_expires_at = now + timedelta(minutes=body.session_minutes)
    session_id = str(uuid.uuid4())
    payload_dict = {
        **identity,
        "session_id": session_id,
        "session_minutes": body.session_minutes,
    }
    payload = json.dumps(payload_dict, default=str)

    # Single-active-code-per-tutor: invalidate any previous code first.
    prev_key = KEY_TUTOR.format(tutor_id=tutor_id)
    try:
        prev_code = await redis.get(prev_key)
        if prev_code:
            prev_code_str = prev_code.decode() if isinstance(prev_code, bytes) else str(prev_code)
            await redis.delete(KEY_CODE.format(code=prev_code_str), prev_key)
        prev_live = await redis.get(KEY_TUTOR_LIVE.format(tutor_id=tutor_id))
        if prev_live:
            prev_session_id = prev_live.decode() if isinstance(prev_live, bytes) else str(prev_live)
            prev_session = await _load_session(redis, prev_session_id)
            if prev_session and prev_session.get("status") == "connected":
                prev_session["status"] = "signed_out"
                prev_session["signed_out_at"] = _iso(now)
                await _set_terminal_status(redis, prev_session, "signed_out")
                await _store_session(redis, prev_session)
            await redis.delete(KEY_TUTOR_LIVE.format(tutor_id=tutor_id))
    except Exception as exc:
        logger.warning("Failed to clear previous smartboard pairing state for tutor %s: %s", tutor_id, exc)

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

    session = {
        "session_id": session_id,
        "status": "pending",
        "code": code,
        "tutor_id": tutor_id,
        "tutor_name": identity.get("name") or "",
        "tutor_email": identity.get("email") or "",
        "tenant_id": identity.get("tenant_id"),
        "created_at": _iso(now),
        "code_expires_at": _iso(code_expires_at),
        "session_minutes": body.session_minutes,
    }
    try:
        await _store_session(redis, session)
    except Exception as exc:
        await redis.delete(KEY_CODE.format(code=code), prev_key)
        logger.error("[SBPAIR] Failed to create session for tutor %s: %s", tutor_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not create pairing session, please retry",
        ) from exc

    logger.info("[SBPAIR] Code issued for tutor %s, expires at %s", tutor_id, code_expires_at.isoformat())

    return RegisterPairCodeResponse(
        code=code,
        expires_at=_iso(code_expires_at),
        session_id=session_id,
        session_expires_at=_iso(planned_session_expires_at),
        session_minutes=body.session_minutes,
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
    phone. Returns a timer-bound tutor JWT (signed with `JWT_SECRET_KEY` so all
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
    session_id = str(identity.get("session_id") or "")
    if not tutor_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired pairing code",
        )
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired pairing code",
        )

    try:
        session = await _load_session(redis, session_id)
    except Exception as exc:
        logger.error("[SBPAIR] Failed to load session %s: %s", session_id, exc)
        session = None
    if session is None:
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
    if session.get("status") == "expired":
        session["status"] = "expired"
        session["expired_at"] = _iso(_utcnow())
        await _store_session(redis, session)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired pairing code",
        )
    session_minutes = int(
        identity.get("session_minutes")
        or session.get("session_minutes")
        or DEFAULT_SESSION_MINUTES
    )
    session_minutes = max(MIN_SESSION_MINUTES, min(MAX_SESSION_MINUTES, session_minutes))
    session_expires_at = _utcnow() + timedelta(minutes=session_minutes)
    expires_delta = timedelta(minutes=session_minutes)
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
        "pair_session_id": session_id,
    }
    # Drop None values so the JWT stays compact.
    token_payload = {k: v for k, v in token_payload.items() if v is not None}

    access_token = auth_manager.create_access_token(
        token_payload,
        expires_delta=expires_delta,
    )
    expires_at = _utcnow() + expires_delta

    now_iso = _iso(_utcnow())
    session.update(
        {
            "status": "connected",
            "connected_at": now_iso,
            "last_seen_at": now_iso,
            "session_expires_at": _iso(expires_at),
        }
    )
    try:
        await _store_session(redis, session)
        await redis.set(KEY_TUTOR_LIVE.format(tutor_id=tutor_id), session_id, ex=_ttl_until(expires_at))
    except Exception as exc:
        logger.warning("[SBPAIR] Failed to store connected status for session %s: %s", session_id, exc)

    # Compute capabilities from the tutor's tenant feature state.
    raw_features = identity.get("enabled_features")
    raw_features_v2 = identity.get("enabled_features_v2")
    merged = merge_tenant_features(raw_features, raw_features_v2)

    capabilities = CloudCapabilities(
        smartboard_core=bool(merged.get("smartboard_core", False)),
        smartboard_live_session=bool(merged.get("smartboard_live_session", False)),
        smartboard_cloud_access=bool(merged.get("smartboard_cloud_access", False)),
    )

    logger.info("[SBPAIR] Code %s redeemed for tutor %s", code, tutor_id)

    return RedeemPairCodeResponse(
        access_token=access_token,
        expires_at=_iso(expires_at),
        session_id=session_id,
        tutor=TutorIdentity(
            tutor_id=tutor_id,
            name=str(identity.get("name") or ""),
            email=str(identity.get("email") or ""),
            tenant_id=identity.get("tenant_id"),
        ),
        capabilities=capabilities,
    )


@router.get(
    "/status/current",
    response_model=Optional[PairingStatusResponse],
    status_code=status.HTTP_200_OK,
    summary="Get the tutor's current live smartboard pairing session",
)
@limiter.limit("30/minute")
async def get_current_pairing_status(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Optional[PairingStatusResponse]:
    identity = _extract_tutor_identity(current_user)
    redis = _redis_client(request)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pairing service unavailable (cache offline)",
        )

    raw_session_id = await redis.get(KEY_TUTOR_LIVE.format(tutor_id=identity["tutor_id"]))
    if not raw_session_id:
        return None
    session_id = raw_session_id.decode() if isinstance(raw_session_id, bytes) else str(raw_session_id)
    session = await _load_session(redis, session_id)
    if session is None:
        return None
    session = await _mark_expired_if_needed(redis, session)
    _require_tutor_session_owner(current_user, session)
    return _status_response(session)


@router.get(
    "/status/{session_id}",
    response_model=PairingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get smartboard pairing session status",
)
@limiter.limit("60/minute")
async def get_pairing_status(
    session_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> PairingStatusResponse:
    redis = _redis_client(request)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pairing service unavailable (cache offline)",
        )
    session = await _load_session(redis, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pairing session not found")
    session = await _mark_expired_if_needed(redis, session)
    _require_tutor_session_owner(current_user, session)
    return _status_response(session)


@router.post(
    "/signout",
    response_model=PairingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Sign out a paired smartboard session",
)
@limiter.limit("20/minute")
async def signout_pairing_session(
    request: Request,
    body: SignOutPairingRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> PairingStatusResponse:
    redis = _redis_client(request)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pairing service unavailable (cache offline)",
        )
    session = await _load_session(redis, body.session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pairing session not found")
    _require_tutor_session_owner(current_user, session)
    session["status"] = "signed_out"
    session["signed_out_at"] = _iso(_utcnow())
    await _set_terminal_status(redis, session, "signed_out")
    await _store_session(redis, session)
    try:
        await redis.delete(KEY_TUTOR_LIVE.format(tutor_id=session.get("tutor_id")))
    except Exception:
        pass
    return _status_response(session)


@router.post(
    "/heartbeat",
    response_model=PairingHeartbeatResponse,
    status_code=status.HTTP_200_OK,
    summary="Smartboard heartbeat for live pairing status",
)
@limiter.limit("60/minute")
async def heartbeat_pairing_session(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> PairingHeartbeatResponse:
    token_payload = _decode_request_token_payload(request)
    session_id = str(current_user.get("pair_session_id") or token_payload.get("pair_session_id") or "")
    if not session_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token missing pairing session")

    redis = _redis_client(request)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pairing service unavailable (cache offline)",
        )

    session = await _load_session(redis, session_id)
    if session is None:
        return PairingHeartbeatResponse(session_id=session_id, status="expired", is_live=False)

    session = await _mark_expired_if_needed(redis, session)
    token_tutor_id = str(
        current_user.get("tutor_id")
        or current_user.get("user_id")
        or current_user.get("sub")
        or token_payload.get("tutor_id")
        or token_payload.get("user_id")
        or token_payload.get("sub")
        or ""
    )
    if token_tutor_id and str(session.get("tutor_id")) != token_tutor_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pairing session not found")

    if session.get("status") == "connected":
        session = await _apply_terminal_status(redis, session)
        if session.get("status") == "connected":
            session["last_seen_at"] = _iso(_utcnow())
            await _store_session(redis, session)

    return PairingHeartbeatResponse(
        session_id=session_id,
        status=str(session.get("status") or "unknown"),
        is_live=_session_is_live(session),
        session_expires_at=session.get("session_expires_at"),
    )
