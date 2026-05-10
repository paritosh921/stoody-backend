"""
ExamPen Hub Operations API — provisioning, heartbeat, and assignment contract.

Handles:
  - First-boot hub provisioning (hub_code → hub_id + invig codes + pen inventory)
  - Hub self-registration with capabilities/dongles
  - Periodic heartbeat with health status
  - Exam assignment and current assignment query
  - Session start/end reporting

Architecture:
    IMPLEMENTATION_PLAN.md §UP-003
    integration/HUB_DEPLOYMENT_SPEC.md

Ownership Declaration:
    - Writes:  exampen_hubs (hub registration, heartbeat, capabilities)
    - Reads from: exampen_exams (assignment validation), exampen_hubs
    - Never writes to: exampen_exams (exam lifecycle is owned by exam_orch)

Hard constraints:
    - C1: MongoDB only
    - Hub must be provisioned before receiving assignments
    - Heartbeat must arrive within 90s or hub is marked offline
    - Session start/end must be consistent with exam lifecycle in exam_orch
"""

from __future__ import annotations

import logging
import hashlib
import hmac
import json
import re
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Hub token utilities
# ---------------------------------------------------------------------------

HUB_SCOPE_MANIFEST_READ = "hub:manifest:read"
HUB_SCOPE_DATA_UPLOAD = "hub:data:upload"
HUB_SCOPE_HEARTBEAT = "hub:heartbeat"
HUB_BACKEND_SCOPES = [
    HUB_SCOPE_MANIFEST_READ,
    HUB_SCOPE_DATA_UPLOAD,
    HUB_SCOPE_HEARTBEAT,
]


def _create_hub_token(
    hub_id: str,
    db_name: str,
    *,
    institution_id: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> str:
    """Issue a long-lived JWT for a hub to authenticate with hub-facing endpoints."""
    import jwt as pyjwt
    from datetime import timedelta
    from core.auth import JWT_SECRET_KEY, JWT_ALGORITHM

    restricted_scopes = scopes or HUB_BACKEND_SCOPES
    now = datetime.now(timezone.utc)
    payload = {
        "sub": hub_id,
        "hub_id": hub_id,
        "db_name": db_name,
        "institution_id": institution_id,
        "scopes": restricted_scopes,
        "user_type": "hub",
        "user_id": hub_id,
        "exp": now + timedelta(days=365),
        "iat": now,
        "type": "access",
    }
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


MOBILE_ACCESS_SCOPES = [
    "hub:read",
    "hub:pens",
    "hub:storage",
    "smartboard:read",
    "smartboard:manage",
]
MOBILE_MANIFEST_DEFAULT_REFRESH_HOURS = 24
MOBILE_MANIFEST_MIN_REFRESH_HOURS = 1
MOBILE_MANIFEST_MAX_REFRESH_HOURS = 168
MOBILE_LOCAL_TOKEN_TTL_SECONDS = 5 * 60


def _canonical_json(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _manifest_signature(secret: str, payload: Dict[str, Any]) -> str:
    return hmac.new(secret.encode("utf-8"), _canonical_json(payload), hashlib.sha256).hexdigest()


def _safe_refresh_hours(value: Any) -> int:
    try:
        hours = int(value)
    except (TypeError, ValueError):
        return MOBILE_MANIFEST_DEFAULT_REFRESH_HOURS
    return min(MOBILE_MANIFEST_MAX_REFRESH_HOURS, max(MOBILE_MANIFEST_MIN_REFRESH_HOURS, hours))


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Admin-only for provisioning and assignment operations."""
    allowed = {"admin", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for hub operations",
        )
    return current_user


def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required",
        )
    return current_user


def require_hub_or_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Accept hub tokens (user_type=hub) or admin tokens."""
    allowed = {"hub", "admin", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hub or admin access required",
        )
    return current_user


def require_hub(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    if current_user.get("user_type") != "hub":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hub token required",
        )
    return current_user


def _hub_scopes(current_user: Dict[str, Any]) -> set[str]:
    raw_scopes = current_user.get("scopes") or []
    if isinstance(raw_scopes, str):
        return {raw_scopes}
    return {str(scope) for scope in raw_scopes}


def _require_hub_scope(current_user: Dict[str, Any], scope: str) -> None:
    if current_user.get("user_type") != "hub":
        return
    if scope not in _hub_scopes(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Hub token missing required scope: {scope}",
        )


def _require_hub_id_match(current_user: Dict[str, Any], hub_id: str) -> None:
    if current_user.get("user_type") != "hub":
        return
    token_hub_id = current_user.get("hub_id") or current_user.get("sub")
    if str(token_hub_id) != hub_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hub token does not match requested hub_id",
        )


# ---------------------------------------------------------------------------
# Tenant DB helper
# ---------------------------------------------------------------------------

async def _get_tenant_db(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Any:
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context missing from token",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DongleInfo(BaseModel):
    dongle_id: str
    hci_path: Optional[str] = None
    status: str = "ok"  # ok | error | missing


class ProvisionRequest(BaseModel):
    hub_code: str = Field(..., min_length=4, description="Provisioning code from admin")


class ProvisionResponse(BaseModel):
    hub_id: str
    institute_id: str
    hub_token: str = ""
    invig_codes: List[str]
    pen_inventory: List[Dict[str, Any]]
    backend_url: str
    provisioned_at: str


class LinkLocalHubRequest(BaseModel):
    hub_id: str = Field(..., min_length=3, max_length=80)
    hub_name: Optional[str] = Field(None, max_length=120)
    hostname: Optional[str] = Field(None, max_length=120)
    mac_address: Optional[str] = Field(None, max_length=64)
    ip_address: Optional[str] = Field(None, max_length=64)
    firmware_version: Optional[str] = Field(None, max_length=64)
    capabilities: List[str] = Field(default_factory=list)


class LinkLocalHubResponse(BaseModel):
    success: bool = True
    hub_id: str
    hub_name: Optional[str] = None
    tenant_id: str
    institution_id: Optional[str] = None
    db_name: str
    hub_token: str
    backend_url: str = ""
    scopes: List[str]
    linked_at: str


class RegisterRequest(BaseModel):
    hub_id: str
    firmware_version: Optional[str] = None
    dongles: List[DongleInfo] = Field(default_factory=list)
    storage_health: str = Field("ok", description="ok | degraded | unavailable")
    ip_address: Optional[str] = None


class RegisterResponse(BaseModel):
    hub_id: str
    registered: bool
    dongle_count: int


class HeartbeatRequest(BaseModel):
    health: str = Field("ok", description="ok | degraded | error")
    storage_health: str = Field("ok", description="ok | degraded | unavailable")
    active_exam_id: Optional[str] = None
    connected_pen_count: int = 0
    uplink_status: str = Field("connected", description="connected | disconnected | reconnecting")
    failed_upload_count: int = 0
    disk_usage_pct: Optional[float] = None


class HeartbeatResponse(BaseModel):
    hub_id: str
    ack: bool
    server_time: str
    pending_assignment: Optional[str] = None
    pending_manifest_refresh: bool = False
    manifest_refresh_requested_at: Optional[str] = None


class AssignRequest(BaseModel):
    exam_id: str


class AssignmentResponse(BaseModel):
    hub_id: str
    assigned_exam_id: Optional[str] = None
    exam_type: Optional[str] = None
    roster: List[str] = Field(default_factory=list)
    duration_minutes: Optional[int] = None
    lifecycle_state: Optional[str] = None


class SessionEventRequest(BaseModel):
    exam_id: str


class SessionEventResponse(BaseModel):
    hub_id: str
    exam_id: str
    event: str  # session_started | session_ended
    recorded_at: str


class HubListItem(BaseModel):
    hub_id: str
    hub_name: Optional[str] = None
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    firmware_version: Optional[str] = None
    linked_at: Optional[str] = None
    provisioned_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None
    online: bool
    health: str = "unknown"
    storage_health: str
    connected_pen_count: int
    assigned_exam_id: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=list)
    manifest_ready: bool = False
    manifest_id: Optional[str] = None
    manifest_issued_at: Optional[str] = None
    manifest_expires_at: Optional[str] = None
    manifest_updated_at: Optional[str] = None
    allowed_tutor_count: int = 0
    manifest_refresh_requested_at: Optional[str] = None
    manifest_refresh_ack_at: Optional[str] = None
    manifest_refresh_status: Optional[str] = None
    manifest_refresh_error: Optional[str] = None


class ManifestRegenerateResponse(BaseModel):
    success: bool = True
    hub_id: str
    manifest_id: str
    issued_at: str
    expires_at: str
    allowed_tutor_count: int
    refresh_requested_at: str
    refresh_status: str = "pending"


class ManifestTutor(BaseModel):
    tutor_id: str
    name: Optional[str] = None
    username: Optional[str] = None
    scopes: List[str] = Field(default_factory=lambda: MOBILE_ACCESS_SCOPES.copy())


class TutorManifestResponse(BaseModel):
    success: bool = True
    hub_id: str
    tenant_id: str
    manifest_id: str
    version: int = 1
    issued_at: str
    expires_at: str
    refresh_interval_hours: int
    allowed_tutors: List[ManifestTutor]
    scopes: List[str]
    signature: str
    local_access_secret: str = Field(
        ...,
        description="Per-hub HS256 verification secret. Returned only to authenticated hub tokens.",
    )


class LocalAccessRequest(BaseModel):
    manifest_id: Optional[str] = None
    device_label: Optional[str] = None


class LocalAccessResponse(BaseModel):
    success: bool = True
    hub_id: str
    tenant_id: str
    manifest_id: str
    tutor_id: str
    local_token: str
    expires_in_sec: int
    scopes: List[str]


class HubPenRegistryEntry(BaseModel):
    pen_mac: str
    pen_id: Optional[str] = None
    name: Optional[str] = None


class HubPenRegistryReplaceRequest(BaseModel):
    pens: List[Any] = Field(default_factory=list)


class HubPenRegistryResponse(BaseModel):
    success: bool = True
    hub_id: str
    pens: List[HubPenRegistryEntry]
    count: int
    version: int = 0
    has_registry: bool = False
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

_indexes_ensured = False


async def _ensure_indexes(collection) -> None:
    global _indexes_ensured
    if _indexes_ensured:
        return
    await collection.create_index("hub_id", unique=True)
    await collection.create_index("hub_code")
    await collection.create_index("last_heartbeat_at")
    await collection.create_index("assigned_exam_id")
    _indexes_ensured = True


async def _load_allowed_manifest_tutors(tenant_db) -> List[Dict[str, Any]]:
    cursor = tenant_db["tutors"].find(
        {"is_active": {"$ne": False}},
        {"_id": 1, "tutor_id": 1, "name": 1, "username": 1, "full_name": 1},
    )
    docs = await cursor.to_list(length=2000)
    tutors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        tutor_id = str(doc.get("tutor_id") or doc.get("_id") or "").strip()
        if not tutor_id or tutor_id in seen:
            continue
        seen.add(tutor_id)
        tutors.append({
            "tutor_id": tutor_id,
            "name": doc.get("name") or doc.get("full_name"),
            "username": doc.get("username"),
            "scopes": MOBILE_ACCESS_SCOPES.copy(),
        })
    return tutors


async def _ensure_mobile_manifest(
    hub_col,
    hub_doc: Dict[str, Any],
    tenant_db,
    *,
    rotate_if_expired: bool = True,
    force_rotate: bool = False,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    mobile_access = dict(hub_doc.get("mobile_access") or {})
    secret = mobile_access.get("local_access_secret") or secrets.token_urlsafe(32)
    refresh_hours = _safe_refresh_hours(mobile_access.get("refresh_interval_hours"))
    expires_at = _parse_dt(mobile_access.get("manifest_expires_at"))
    issued_at = _parse_dt(mobile_access.get("manifest_issued_at"))
    manifest_id = mobile_access.get("manifest_id")

    should_rotate = (
        force_rotate
        or
        not manifest_id
        or not issued_at
        or not expires_at
        or (rotate_if_expired and expires_at <= now)
    )
    if should_rotate:
        issued_at = now
        expires_at = now + timedelta(hours=refresh_hours)
        manifest_id = secrets.token_urlsafe(18)

    tutors = await _load_allowed_manifest_tutors(tenant_db)
    payload = {
        "hub_id": hub_doc["hub_id"],
        "tenant_id": hub_doc.get("institute_id") or "",
        "manifest_id": manifest_id,
        "version": 1,
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "refresh_interval_hours": refresh_hours,
        "allowed_tutors": tutors,
        "scopes": MOBILE_ACCESS_SCOPES,
    }
    signature = _manifest_signature(secret, payload)

    await hub_col.update_one(
        {"hub_id": hub_doc["hub_id"]},
        {
            "$set": {
                "mobile_access.local_access_secret": secret,
                "mobile_access.manifest_id": manifest_id,
                "mobile_access.manifest_issued_at": issued_at,
                "mobile_access.manifest_expires_at": expires_at,
                "mobile_access.refresh_interval_hours": refresh_hours,
                "mobile_access.allowed_tutors": tutors,
                "mobile_access.scopes": MOBILE_ACCESS_SCOPES,
                "mobile_access.signature": signature,
                "mobile_access.updated_at": now,
            }
        },
    )
    payload["signature"] = signature
    payload["local_access_secret"] = secret
    return payload


def _actor_tutor_id(current_user: Dict[str, Any]) -> Optional[str]:
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id") or current_user.get("sub")
    return str(tutor_id).strip() if tutor_id else None


# ---------------------------------------------------------------------------
# Helper: generate invigilator codes
# ---------------------------------------------------------------------------

def _generate_invig_codes(count: int = 3) -> List[str]:
    """Generate short alphanumeric invigilator codes."""
    chars = string.ascii_uppercase + string.digits
    return ["".join(secrets.choice(chars) for _ in range(6)) for _ in range(count)]


async def _get_master_db(db: DatabaseManager):
    """Get the master DB (skb_master) or raise 503."""
    master_db = await db.get_master_db()
    if master_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Master database unavailable",
        )
    return master_db


async def _validate_provision_code(
    db: DatabaseManager,
    hub_code: str,
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a provisioning code against the master DB.

    Checks that the code exists, is unused, has not expired, and that
    the admin's tenant matches the code's institution.

    Returns the code document on success.
    """
    master_db = await _get_master_db(db)

    code_doc = await master_db["exampen_hub_provision_codes"].find_one(
        {"code": hub_code, "used": False}
    )
    if not code_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or already-used provisioning code",
        )

    if code_doc["expires_at"].tzinfo is None:
        expires_at = code_doc["expires_at"].replace(tzinfo=timezone.utc)
    else:
        expires_at = code_doc["expires_at"]

    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Provisioning code has expired",
        )

    code_institution = code_doc["institution_id"]
    tenant = await master_db["tenants"].find_one(
        {"institution_id": code_institution, "status": {"$in": ["active", "approved"]}}
    )
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant associated with this code is no longer active",
        )

    tenant_db_name = tenant.get("db_name", "")
    admin_db_name = current_user.get("db_name", "")
    if tenant_db_name != admin_db_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Provisioning code is for a different institution",
        )

    return code_doc


def _fmt(v) -> Optional[str]:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if v is not None:
        return str(v)
    return None


_PEN_MAC_RE = re.compile(r"^[0-9A-F]{2}(:[0-9A-F]{2}){5}$")


def _normalize_pen_mac(value: Any) -> str:
    raw = str(value or "")
    hex_only = re.sub(r"[^0-9A-Fa-f]", "", raw).upper()
    if len(hex_only) != 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid pen MAC: {raw}",
        )
    mac = ":".join(hex_only[i:i + 2] for i in range(0, 12, 2))
    if not _PEN_MAC_RE.match(mac):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid pen MAC: {raw}",
        )
    return mac


def _default_pen_id(mac: str) -> str:
    return f"PEN-{mac.replace(':', '')[-6:]}"


def _normalise_pen_registry_entries(entries: List[Any]) -> List[Dict[str, str]]:
    deduped: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        if isinstance(entry, str):
            mac = _normalize_pen_mac(entry)
            pen_id = _default_pen_id(mac)
        elif isinstance(entry, dict):
            mac = _normalize_pen_mac(entry.get("pen_mac") or entry.get("mac"))
            pen_id = str(entry.get("pen_id") or entry.get("name") or _default_pen_id(mac))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Each pen must be a MAC string or an object with pen_mac",
            )
        deduped[mac] = {"pen_mac": mac, "pen_id": pen_id}
    return list(deduped.values())


def _hub_pen_registry_response(hub_doc: Dict[str, Any]) -> HubPenRegistryResponse:
    pens = [
        HubPenRegistryEntry(
            pen_mac=row.get("pen_mac", ""),
            pen_id=row.get("pen_id") or _default_pen_id(row.get("pen_mac", "")),
        )
        for row in hub_doc.get("registered_pens", [])
        if row.get("pen_mac")
    ]
    return HubPenRegistryResponse(
        hub_id=hub_doc.get("hub_id", ""),
        pens=pens,
        count=len(pens),
        version=int(hub_doc.get("pen_registry_version") or 0),
        has_registry=bool(hub_doc.get("pen_registry_uploaded_at")),
        updated_at=_fmt(hub_doc.get("pen_registry_uploaded_at")),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/link-local",
    summary="Link a local classroom hub to the authenticated admin tenant",
    responses={
        403: {"description": "Admin access required"},
        409: {"description": "Hub already linked to another tenant"},
        503: {"description": "Tenant or master database unavailable"},
    },
)
async def link_local_hub(
    body: LinkLocalHubRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
) -> LinkLocalHubResponse:
    """Bind an independently setup RPi hub to the admin's tenant.

    The admin JWT is used only for this link transaction. The returned token is
    a restricted hub JWT and is the only backend credential the hub should cache.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    master_db = await _get_master_db(db)
    collection = tenant_db["exampen_hubs"]
    await _ensure_indexes(collection)

    from config_async import settings

    now = datetime.now(timezone.utc)
    hub_id = body.hub_id.strip()
    if not hub_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="hub_id is required")
    db_name = str(current_user.get("db_name") or "")
    institution_id = current_user.get("institution_id") or db_name
    linked_by = current_user.get("user_id") or current_user.get("sub") or current_user.get("email")
    capabilities = [str(item).strip() for item in body.capabilities if str(item).strip()]

    existing_master = await master_db["exampen_hubs"].find_one({"hub_id": hub_id})
    if existing_master:
        existing_db_name = existing_master.get("db_name")
        existing_institution = existing_master.get("institution_id")
        linked_elsewhere = (
            existing_db_name
            and existing_db_name != db_name
        ) or (
            existing_institution
            and str(existing_institution) != str(institution_id)
        )
        if linked_elsewhere:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Hub is already linked to another tenant",
            )

    set_doc = {
        "hub_id": hub_id,
        "hub_name": body.hub_name,
        "institute_id": db_name,
        "institution_id": institution_id,
        "db_name": db_name,
        "hostname": body.hostname,
        "mac_address": body.mac_address,
        "ip_address": body.ip_address,
        "firmware_version": body.firmware_version,
        "capabilities": capabilities,
        "hub_scopes": HUB_BACKEND_SCOPES,
        "linked_by": linked_by,
        "linked_at": now,
        "link_method": "admin_login",
        "updated_at": now,
    }
    await collection.update_one(
        {"hub_id": hub_id},
        {
            "$set": set_doc,
            "$setOnInsert": {
                "hub_code": None,
                "invig_codes": [],
                "provisioned_at": now,
                "registered_at": None,
                "dongles": [],
                "registered_pens": [],
                "storage_health": "unknown",
                "last_heartbeat_at": None,
                "connected_pen_count": 0,
                "assigned_exam_id": None,
                "failed_upload_count": 0,
                "created_by": linked_by,
            },
        },
        upsert=True,
    )

    await master_db["exampen_hubs"].update_one(
        {"hub_id": hub_id},
        {
            "$set": {
                "hub_id": hub_id,
                "hub_name": body.hub_name,
                "institution_id": institution_id,
                "db_name": db_name,
                "status": "linked",
                "link_method": "admin_login",
                "last_seen_at": now,
                "updated_at": now,
            },
            "$setOnInsert": {"provisioned_at": now},
        },
        upsert=True,
    )

    hub_token = _create_hub_token(
        hub_id,
        db_name=db_name,
        institution_id=str(institution_id) if institution_id else None,
        scopes=HUB_BACKEND_SCOPES,
    )
    backend_url = getattr(settings, "BACKEND_URL", "") or ""
    logger.info("Hub %s linked locally to tenant %s by %s", hub_id, db_name, linked_by)

    return LinkLocalHubResponse(
        hub_id=hub_id,
        hub_name=body.hub_name,
        tenant_id=db_name,
        institution_id=str(institution_id) if institution_id else None,
        db_name=db_name,
        hub_token=hub_token,
        backend_url=backend_url,
        scopes=HUB_BACKEND_SCOPES,
        linked_at=now.isoformat(),
    )


@router.post(
    "/provision",
    status_code=status.HTTP_201_CREATED,
    summary="First-boot hub provisioning",
    responses={
        400: {"description": "Invalid hub code"},
        403: {"description": "Insufficient permissions or code is for different institution"},
        409: {"description": "Hub code already used"},
        410: {"description": "Provisioning code has expired"},
        500: {"description": "Server misconfigured (BACKEND_URL missing)"},
        503: {"description": "Tenant or master database unavailable"},
    },
)
async def provision_hub(
    body: ProvisionRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
) -> ProvisionResponse:
    """Provision a hub on first boot.

    Two-party flow (HUB_DEPLOYMENT_SPEC §7.2, SUPERADMIN_SPEC §5.1):
      1. Super-admin generates a provisioning code via
         ``POST /api/v1/superadmin/evalpen/hubs/provision-code``
      2. Admin on-site enters the code on the hub TUI
      3. TUI calls this endpoint with admin JWT
      4. Backend validates code against master DB, verifies tenant
         match, then creates the hub in the tenant DB

    Returns hub_id, invig codes, pen inventory, and hub_token for
    local caching.
    """
    # Pre-check: BACKEND_URL must be configured before any DB writes.
    from config_async import settings
    backend_url = getattr(settings, "BACKEND_URL", "") or ""
    if not backend_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_URL not configured on server — cannot complete provisioning",
        )

    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_hubs"]
    await _ensure_indexes(collection)

    # Step 1: Validate provisioning code against master DB
    code_doc = await _validate_provision_code(db, body.hub_code, current_user)

    # Step 2: Atomically consume the code before any writes. The used: False
    # filter ensures that if two requests race, only one wins the update.
    # The loser gets modified_count == 0 and exits before creating hub records.
    master_db = await _get_master_db(db)
    now = datetime.now(timezone.utc)
    hub_id = f"hub-{secrets.token_hex(8)}"
    consume_result = await master_db["exampen_hub_provision_codes"].update_one(
        {"_id": code_doc["_id"], "used": False},
        {"$set": {"used": True, "used_at": now, "used_by_hub_id": hub_id}},
    )
    if consume_result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Provisioning code was consumed by another request",
        )

    # Step 3: Check if this hub_code was already provisioned in tenant DB
    existing = await collection.find_one({"hub_code": body.hub_code})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Hub code {body.hub_code} has already been provisioned as hub_id={existing.get('hub_id')}",
        )

    institute_id = current_user.get("db_name", "unknown")
    invig_codes = _generate_invig_codes(3)

    # Fetch pen inventory from existing pen registry if available
    pen_inventory: List[Dict[str, Any]] = []
    try:
        pen_cursor = tenant_db["pen_registry"].find({})
        pen_docs = await pen_cursor.to_list(length=200)
        pen_inventory = [
            {"pen_mac": p.get("pen_mac", ""), "pen_id": p.get("pen_id", ""), "student_name": p.get("student_name", "")}
            for p in pen_docs
        ]
    except Exception:
        logger.debug("No pen_registry collection or empty — returning empty pen inventory")

    doc = {
        "hub_id": hub_id,
        "hub_code": body.hub_code,
        "hub_name": None,
        "institute_id": institute_id,
        "invig_codes": invig_codes,
        "provisioned_at": now,
        "registered_at": None,
        "firmware_version": None,
        "dongles": [],
        "registered_pens": [],
        "storage_health": "unknown",
        "ip_address": None,
        "last_heartbeat_at": None,
        "connected_pen_count": 0,
        "assigned_exam_id": None,
        "failed_upload_count": 0,
        "created_by": current_user.get("user_id", "unknown"),
    }

    await collection.insert_one(doc)

    # Write projection to master DB so super-admin fleet listing works.
    # Status is "provisioned" — "active" is set by registration/heartbeat.
    # Raw provision code is NOT stored in the master record.
    await master_db["exampen_hubs"].insert_one({
        "hub_id": hub_id,
        "institution_id": code_doc["institution_id"],
        "status": "provisioned",
        "provisioned_at": now,
        "last_seen_at": now,
    })

    hub_token = _create_hub_token(
        hub_id,
        db_name=institute_id,
        institution_id=str(code_doc.get("institution_id") or institute_id),
        scopes=HUB_BACKEND_SCOPES,
    )

    logger.info("Hub %s provisioned with code %s by %s", hub_id, body.hub_code, current_user.get("user_id"))

    return ProvisionResponse(
        hub_id=hub_id,
        institute_id=institute_id,
        hub_token=hub_token,
        invig_codes=invig_codes,
        pen_inventory=pen_inventory,
        backend_url=backend_url,
        provisioned_at=now.isoformat(),
    )


@router.post(
    "/register",
    summary="Hub registers capabilities and dongles",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Hub not found — must provision first"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def register_hub(
    body: RegisterRequest,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> RegisterResponse:
    """Hub registers its capabilities after provisioning.

    Updates firmware version, dongle list, storage health, and IP.
    """
    _require_hub_id_match(current_user, body.hub_id)
    _require_hub_scope(current_user, HUB_SCOPE_HEARTBEAT)

    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_hubs"]

    existing = await collection.find_one({"hub_id": body.hub_id})
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {body.hub_id} not found — provision first via POST /hubs/provision",
        )

    now = datetime.now(timezone.utc)
    await collection.update_one(
        {"hub_id": body.hub_id},
        {
            "$set": {
                "registered_at": now,
                "firmware_version": body.firmware_version,
                "dongles": [d.model_dump() for d in body.dongles],
                "storage_health": body.storage_health,
                "ip_address": body.ip_address,
                "last_heartbeat_at": now,
            }
        },
    )

    logger.info("Hub %s registered with %d dongles", body.hub_id, len(body.dongles))

    return RegisterResponse(
        hub_id=body.hub_id,
        registered=True,
        dongle_count=len(body.dongles),
    )


@router.post(
    "/{hub_id}/heartbeat",
    summary="Periodic hub heartbeat with health status",
    responses={
        404: {"description": "Hub not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def hub_heartbeat(
    hub_id: str,
    body: HeartbeatRequest,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> HeartbeatResponse:
    """Accept periodic heartbeat from a hub.

    Updates health status, pen count, storage, and uplink status.
    Returns server time and any pending exam assignment.
    """
    _require_hub_id_match(current_user, hub_id)
    _require_hub_scope(current_user, HUB_SCOPE_HEARTBEAT)

    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_hubs"]

    existing = await collection.find_one({"hub_id": hub_id})
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {hub_id} not found",
        )

    now = datetime.now(timezone.utc)
    await collection.update_one(
        {"hub_id": hub_id},
        {
            "$set": {
                "last_heartbeat_at": now,
                "health": body.health,
                "storage_health": body.storage_health,
                "active_exam_id": body.active_exam_id,
                "connected_pen_count": body.connected_pen_count,
                "uplink_status": body.uplink_status,
                "failed_upload_count": body.failed_upload_count,
                "disk_usage_pct": body.disk_usage_pct,
            }
        },
    )

    return HeartbeatResponse(
        hub_id=hub_id,
        ack=True,
        server_time=now.isoformat(),
        pending_assignment=existing.get("assigned_exam_id"),
        pending_manifest_refresh=(
            dict(existing.get("mobile_access") or {}).get("refresh_status") == "pending"
        ),
        manifest_refresh_requested_at=_fmt(
            dict(existing.get("mobile_access") or {}).get("refresh_requested_at")
        ),
    )


@router.post(
    "/{hub_id}/assign",
    summary="Assign an exam to a hub",
    responses={
        400: {"description": "Invalid assignment"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Hub or exam not found"},
        409: {"description": "Hub already assigned to another exam"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def assign_exam_to_hub(
    hub_id: str,
    body: AssignRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
) -> AssignmentResponse:
    """Assign an exam to a hub. The hub will receive this assignment on next heartbeat."""
    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]
    exam_col = tenant_db["exampen_exams"]

    hub_doc = await hub_col.find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {hub_id} not found",
        )

    # Check hub isn't already assigned to a different active exam
    current_assignment = hub_doc.get("assigned_exam_id")
    if current_assignment and current_assignment != body.exam_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Hub {hub_id} is already assigned to exam {current_assignment}. Unassign first.",
        )

    # Validate exam exists
    exam_doc = await exam_col.find_one({"exam_id": body.exam_id})
    if exam_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {body.exam_id} not found",
        )

    now = datetime.now(timezone.utc)
    await hub_col.update_one(
        {"hub_id": hub_id},
        {"$set": {"assigned_exam_id": body.exam_id, "assigned_at": now}},
    )

    logger.info("Hub %s assigned to exam %s by %s", hub_id, body.exam_id, current_user.get("user_id"))

    return AssignmentResponse(
        hub_id=hub_id,
        assigned_exam_id=body.exam_id,
        exam_type=exam_doc.get("exam_type"),
        roster=exam_doc.get("roster", []),
        duration_minutes=exam_doc.get("duration_minutes"),
        lifecycle_state=exam_doc.get("lifecycle_state"),
    )


@router.get(
    "/{hub_id}/assignment",
    summary="Fetch current or pending exam assignment",
    responses={
        404: {"description": "Hub not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_assignment(
    hub_id: str,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> AssignmentResponse:
    """Return the current exam assignment for a hub."""
    _require_hub_id_match(current_user, hub_id)
    _require_hub_scope(current_user, HUB_SCOPE_HEARTBEAT)

    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]
    exam_col = tenant_db["exampen_exams"]

    hub_doc = await hub_col.find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {hub_id} not found",
        )

    assigned_exam_id = hub_doc.get("assigned_exam_id")
    if not assigned_exam_id:
        return AssignmentResponse(hub_id=hub_id)

    exam_doc = await exam_col.find_one({"exam_id": assigned_exam_id})
    if exam_doc is None:
        return AssignmentResponse(hub_id=hub_id, assigned_exam_id=assigned_exam_id)

    return AssignmentResponse(
        hub_id=hub_id,
        assigned_exam_id=assigned_exam_id,
        exam_type=exam_doc.get("exam_type"),
        roster=exam_doc.get("roster", []),
        duration_minutes=exam_doc.get("duration_minutes"),
        lifecycle_state=exam_doc.get("lifecycle_state"),
    )


@router.get(
    "/{hub_id}/tutor-manifest",
    summary="Hub fetches signed tutor manifest for local mobile access",
    responses={
        403: {"description": "Hub token required or hub_id mismatch"},
        404: {"description": "Hub not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_tutor_manifest(
    hub_id: str,
    current_user: Dict[str, Any] = Depends(require_hub),
    db: DatabaseManager = Depends(get_database),
) -> TutorManifestResponse:
    """Return the 24-hour tutor manifest cached by the hub for local Wi-Fi access.

    This endpoint is deliberately hub-only. It returns a per-hub verification
    secret so the RPi can validate short-lived mobile local-access tokens
    without proxying arbitrary requests to the backend.
    """
    _require_hub_id_match(current_user, hub_id)
    _require_hub_scope(current_user, HUB_SCOPE_MANIFEST_READ)

    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]
    hub_doc = await hub_col.find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Hub {hub_id} not found")

    manifest = await _ensure_mobile_manifest(hub_col, hub_doc, tenant_db)
    mobile_access = dict(hub_doc.get("mobile_access") or {})
    if mobile_access.get("refresh_status") == "pending":
        now = datetime.now(timezone.utc)
        await hub_col.update_one(
            {"hub_id": hub_id},
            {
                "$set": {
                    "mobile_access.refresh_status": "synced",
                    "mobile_access.refresh_ack_at": now,
                    "mobile_access.acknowledged_manifest_id": manifest["manifest_id"],
                    "mobile_access.last_hub_fetch_at": now,
                    "mobile_access.refresh_error": "",
                }
            },
        )
    else:
        await hub_col.update_one(
            {"hub_id": hub_id},
            {"$set": {"mobile_access.last_hub_fetch_at": datetime.now(timezone.utc)}},
        )
    return TutorManifestResponse(**manifest)


@router.get(
    "/{hub_id}/auth-check",
    summary="Validate restricted hub token scope",
    responses={
        403: {"description": "Hub token required, hub_id mismatch, or missing scope"},
    },
)
async def hub_auth_check(
    hub_id: str,
    scope: str = Query(HUB_SCOPE_DATA_UPLOAD),
    current_user: Dict[str, Any] = Depends(require_hub),
) -> Dict[str, Any]:
    _require_hub_id_match(current_user, hub_id)
    _require_hub_scope(current_user, scope)
    return {
        "success": True,
        "hub_id": hub_id,
        "scope": scope,
    }


@router.post(
    "/{hub_id}/tutor-manifest/regenerate",
    summary="Admin regenerates the tutor manifest and requests hub refresh",
    responses={
        403: {"description": "Admin access required"},
        404: {"description": "Hub not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def regenerate_tutor_manifest(
    hub_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
) -> ManifestRegenerateResponse:
    """Regenerate the 24h tutor manifest and mark it for hub-side pull."""
    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]
    hub_doc = await hub_col.find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Hub {hub_id} not found")

    manifest = await _ensure_mobile_manifest(
        hub_col,
        hub_doc,
        tenant_db,
        force_rotate=True,
    )
    now = datetime.now(timezone.utc)
    await hub_col.update_one(
        {"hub_id": hub_id},
        {
            "$set": {
                "mobile_access.refresh_requested_at": now,
                "mobile_access.refresh_status": "pending",
                "mobile_access.refresh_error": "",
                "mobile_access.requested_manifest_id": manifest["manifest_id"],
                "mobile_access.refresh_requested_by": current_user.get("user_id") or current_user.get("sub"),
            },
            "$unset": {
                "mobile_access.refresh_ack_at": "",
                "mobile_access.acknowledged_manifest_id": "",
            },
        },
    )

    return ManifestRegenerateResponse(
        hub_id=hub_id,
        manifest_id=manifest["manifest_id"],
        issued_at=manifest["issued_at"],
        expires_at=manifest["expires_at"],
        allowed_tutor_count=len(manifest["allowed_tutors"]),
        refresh_requested_at=now.isoformat(),
    )


@router.get(
    "/{hub_id}/pen-registry",
    response_model=HubPenRegistryResponse,
    summary="Return the allowed pen registry for a hub",
)
async def get_hub_pen_registry(
    hub_id: str,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> HubPenRegistryResponse:
    if current_user.get("user_type") == "hub":
        _require_hub_id_match(current_user, hub_id)
        _require_hub_scope(current_user, HUB_SCOPE_HEARTBEAT)

    tenant_db = await _get_tenant_db(db, current_user)
    hub_doc = await tenant_db["exampen_hubs"].find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Hub {hub_id} not found")
    return _hub_pen_registry_response(hub_doc)


@router.put(
    "/{hub_id}/pen-registry",
    response_model=HubPenRegistryResponse,
    summary="Replace the allowed pen registry for a hub",
)
async def replace_hub_pen_registry(
    hub_id: str,
    body: HubPenRegistryReplaceRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> HubPenRegistryResponse:
    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]
    existing = await hub_col.find_one({"hub_id": hub_id})
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Hub {hub_id} not found")

    pens = _normalise_pen_registry_entries(body.pens)
    now = datetime.now(timezone.utc)
    await hub_col.update_one(
        {"hub_id": hub_id},
        {
            "$set": {
                "registered_pens": pens,
                "pen_registry_uploaded_at": now,
                "pen_registry_updated_by": str(current_user.get("user_id") or ""),
            },
            "$inc": {"pen_registry_version": 1},
        },
    )
    updated = await hub_col.find_one({"hub_id": hub_id}) or existing
    return _hub_pen_registry_response(updated)


@router.post(
    "/{hub_id}/local-access",
    summary="Tutor obtains short-lived local hub access token",
    responses={
        403: {"description": "Tutor not allowed for this hub"},
        404: {"description": "Hub not found"},
        409: {"description": "Hub manifest unavailable, expired, or mismatched"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def issue_local_access_token(
    hub_id: str,
    body: LocalAccessRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> LocalAccessResponse:
    """Issue a short-lived token for the mobile app to call local hub APIs.

    The backend validates the logged-in tutor against the current manifest.
    The token is signed with the hub-specific secret already cached on the RPi.
    """
    if current_user.get("user_type") != "tutor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor login required for local hub access",
        )

    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]
    hub_doc = await hub_col.find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Hub {hub_id} not found")

    mobile_access = hub_doc.get("mobile_access") or {}
    manifest_id = mobile_access.get("manifest_id")
    secret = mobile_access.get("local_access_secret")
    expires_at = _parse_dt(mobile_access.get("manifest_expires_at"))
    now = datetime.now(timezone.utc)
    if not manifest_id or not secret:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Hub tutor manifest is not ready. Refresh the manifest from hub TUI first.",
        )
    if expires_at and expires_at <= now:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Hub tutor manifest has expired. Refresh the manifest from hub TUI first.",
        )
    if body.manifest_id and body.manifest_id != manifest_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Scanned hub manifest is stale. Refresh the smartboard QR and try again.",
        )

    tutor_id = _actor_tutor_id(current_user)
    allowed_tutors = mobile_access.get("allowed_tutors") or []
    allowed = {str(t.get("tutor_id")): t for t in allowed_tutors if t.get("tutor_id")}
    if not tutor_id or tutor_id not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor is not authorised for this hub",
        )

    import jwt as pyjwt

    exp = now + timedelta(seconds=MOBILE_LOCAL_TOKEN_TTL_SECONDS)
    scopes = allowed[tutor_id].get("scopes") or MOBILE_ACCESS_SCOPES
    claims = {
        "type": "hub_local_access",
        "aud": "stoody-edge-hub-local",
        "hub_id": hub_id,
        "tenant_id": hub_doc.get("institute_id") or current_user.get("db_name"),
        "manifest_id": manifest_id,
        "tutor_id": tutor_id,
        "scopes": scopes,
        "device_label": body.device_label,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "jti": secrets.token_urlsafe(12),
    }
    token = pyjwt.encode(claims, secret, algorithm="HS256")

    await tenant_db["hub_local_access_audit"].insert_one({
        "hub_id": hub_id,
        "manifest_id": manifest_id,
        "tutor_id": tutor_id,
        "device_label": body.device_label,
        "issued_at": now,
        "expires_at": exp,
        "jti": claims["jti"],
    })

    return LocalAccessResponse(
        hub_id=hub_id,
        tenant_id=str(hub_doc.get("institute_id") or current_user.get("db_name") or ""),
        manifest_id=manifest_id,
        tutor_id=tutor_id,
        local_token=token,
        expires_in_sec=MOBILE_LOCAL_TOKEN_TTL_SECONDS,
        scopes=scopes,
    )


@router.post(
    "/{hub_id}/session-start",
    summary="Hub reports exam session started",
    responses={
        400: {"description": "Invalid state for session start"},
        404: {"description": "Hub not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def session_start(
    hub_id: str,
    body: SessionEventRequest,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> SessionEventResponse:
    """Hub reports that an exam session has started (collection begins)."""
    _require_hub_id_match(current_user, hub_id)
    _require_hub_scope(current_user, HUB_SCOPE_DATA_UPLOAD)

    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]
    exam_col = tenant_db["exampen_exams"]

    hub_doc = await hub_col.find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {hub_id} not found",
        )

    # Validate exam is in an appropriate state
    exam_doc = await exam_col.find_one({"exam_id": body.exam_id})
    if exam_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {body.exam_id} not found",
        )

    lifecycle = exam_doc.get("lifecycle_state", "draft")
    if lifecycle not in ("armed", "in_progress"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exam {body.exam_id} is in state '{lifecycle}' — must be 'armed' or 'in_progress' to start session",
        )

    now = datetime.now(timezone.utc)

    # Update hub's session timestamp
    await hub_col.update_one(
        {"hub_id": hub_id},
        {"$set": {"session_started_at": now, "active_exam_id": body.exam_id}},
    )

    # Update exam's hub_assignments with session_started_at for this hub
    await exam_col.update_one(
        {"exam_id": body.exam_id, "hub_assignments.hub_id": hub_id},
        {"$set": {"hub_assignments.$.session_started_at": now, "updated_at": now}},
    )

    logger.info("Hub %s started session for exam %s", hub_id, body.exam_id)

    return SessionEventResponse(
        hub_id=hub_id,
        exam_id=body.exam_id,
        event="session_started",
        recorded_at=now.isoformat(),
    )


@router.post(
    "/{hub_id}/session-end",
    summary="Hub reports collection closed",
    responses={
        400: {"description": "Invalid state for session end"},
        404: {"description": "Hub not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def session_end(
    hub_id: str,
    body: SessionEventRequest,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> SessionEventResponse:
    """Hub reports that collection has ended for an exam session."""
    _require_hub_id_match(current_user, hub_id)
    _require_hub_scope(current_user, HUB_SCOPE_DATA_UPLOAD)

    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]
    exam_col = tenant_db["exampen_exams"]

    hub_doc = await hub_col.find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {hub_id} not found",
        )

    now = datetime.now(timezone.utc)

    await hub_col.update_one(
        {"hub_id": hub_id},
        {"$set": {"session_ended_at": now}},
    )

    await exam_col.update_one(
        {"exam_id": body.exam_id, "hub_assignments.hub_id": hub_id},
        {"$set": {"hub_assignments.$.session_ended_at": now, "updated_at": now}},
    )

    logger.info("Hub %s ended session for exam %s", hub_id, body.exam_id)

    return SessionEventResponse(
        hub_id=hub_id,
        exam_id=body.exam_id,
        event="session_ended",
        recorded_at=now.isoformat(),
    )


@router.delete(
    "/{hub_id}/assignment",
    summary="Clear hub exam assignment",
    responses={
        404: {"description": "Hub not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def clear_assignment(
    hub_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
) -> AssignmentResponse:
    """Clear the current exam assignment from a hub."""
    tenant_db = await _get_tenant_db(db, current_user)
    hub_col = tenant_db["exampen_hubs"]

    hub_doc = await hub_col.find_one({"hub_id": hub_id})
    if hub_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {hub_id} not found",
        )

    await hub_col.update_one(
        {"hub_id": hub_id},
        {"$set": {"assigned_exam_id": None, "assigned_at": None}},
    )

    logger.info("Hub %s assignment cleared by %s", hub_id, current_user.get("user_id"))

    return AssignmentResponse(hub_id=hub_id)


@router.get(
    "",
    summary="List all provisioned hubs for this tenant",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def list_hubs(
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> List[HubListItem]:
    """Return all provisioned hubs for the current tenant."""
    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_hubs"]

    from datetime import timedelta
    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(seconds=90)

    cursor = collection.find({}).sort("provisioned_at", -1)
    docs = await cursor.to_list(length=200)

    items: List[HubListItem] = []
    for d in docs:
        mobile_access = dict(d.get("mobile_access") or {})
        last_heartbeat = d.get("last_heartbeat_at")
        items.append(
            HubListItem(
                hub_id=d.get("hub_id", ""),
                hub_name=d.get("hub_name"),
                hostname=d.get("hostname"),
                ip_address=d.get("ip_address"),
                firmware_version=d.get("firmware_version"),
                linked_at=_fmt(d.get("linked_at")),
                provisioned_at=_fmt(d.get("provisioned_at")),
                last_heartbeat_at=_fmt(last_heartbeat),
                online=bool(last_heartbeat and last_heartbeat > stale_threshold),
                health=d.get("health", "unknown"),
                storage_health=d.get("storage_health", "unknown"),
                connected_pen_count=d.get("connected_pen_count", 0),
                assigned_exam_id=d.get("assigned_exam_id"),
                capabilities=list(d.get("capabilities") or []),
                scopes=list(d.get("hub_scopes") or HUB_BACKEND_SCOPES),
                manifest_ready=bool(mobile_access.get("manifest_id")),
                manifest_id=mobile_access.get("manifest_id"),
                manifest_issued_at=_fmt(mobile_access.get("manifest_issued_at")),
                manifest_expires_at=_fmt(mobile_access.get("manifest_expires_at")),
                manifest_updated_at=_fmt(mobile_access.get("updated_at")),
                allowed_tutor_count=len(mobile_access.get("allowed_tutors") or []),
                manifest_refresh_requested_at=_fmt(mobile_access.get("refresh_requested_at")),
                manifest_refresh_ack_at=_fmt(mobile_access.get("refresh_ack_at")),
                manifest_refresh_status=mobile_access.get("refresh_status"),
                manifest_refresh_error=mobile_access.get("refresh_error"),
            )
        )
    return items
