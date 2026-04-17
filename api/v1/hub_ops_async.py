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
import secrets
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Hub token utilities
# ---------------------------------------------------------------------------

def _create_hub_token(hub_id: str, db_name: str) -> str:
    """Issue a long-lived JWT for a hub to authenticate with hub-facing endpoints."""
    import jwt as pyjwt
    from datetime import timedelta
    from core.auth import JWT_SECRET_KEY, JWT_ALGORITHM

    now = datetime.now(timezone.utc)
    payload = {
        "sub": hub_id,
        "hub_id": hub_id,
        "db_name": db_name,
        "user_type": "hub",
        "user_id": hub_id,
        "exp": now + timedelta(days=365),
        "iat": now,
        "type": "access",
    }
    return pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


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
    """Accept hub tokens (user_type=hub) or admin/tutor tokens."""
    allowed = {"hub", "admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hub or admin access required",
        )
    return current_user


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
    provisioned_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None
    online: bool
    storage_health: str
    connected_pen_count: int
    assigned_exam_id: Optional[str] = None


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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

    hub_token = _create_hub_token(hub_id, db_name=institute_id)

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
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
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

    return [
        HubListItem(
            hub_id=d.get("hub_id", ""),
            hub_name=d.get("hub_name"),
            provisioned_at=_fmt(d.get("provisioned_at")),
            last_heartbeat_at=_fmt(d.get("last_heartbeat_at")),
            online=bool(d.get("last_heartbeat_at") and d["last_heartbeat_at"] > stale_threshold),
            storage_health=d.get("storage_health", "unknown"),
            connected_pen_count=d.get("connected_pen_count", 0),
            assigned_exam_id=d.get("assigned_exam_id"),
        )
        for d in docs
    ]
