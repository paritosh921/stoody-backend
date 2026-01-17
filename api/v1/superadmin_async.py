"""
Super Admin API for Stoody Platform
Handles tenant management, registration approval, and feature flags
"""

import logging
import secrets
import string
import re
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from passlib.context import CryptContext
import jwt

from core.database import DatabaseManager
from config_async import settings

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings for super admin (separate from regular auth)
SUPERADMIN_JWT_SECRET = getattr(settings, "SUPERADMIN_JWT_SECRET", None) or getattr(settings, "JWT_SECRET_KEY", "")
SUPERADMIN_JWT_ALGORITHM = "HS256"
SUPERADMIN_JWT_EXPIRATION_HOURS = 24
SUPERADMIN_SETUP_KEY = getattr(settings, "SUPERADMIN_SETUP_KEY", "")


# ============ PYDANTIC MODELS ============

class SuperAdminLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class SuperAdminSetupRequest(BaseModel):
    name: str = Field(..., min_length=2)
    email: EmailStr
    password: str = Field(..., min_length=8)
    setup_key: str


class TenantFeatures(BaseModel):
    smartboard: bool = True
    online_class: bool = False
    ai_chat: bool = True
    stoody_pen: bool = False
    exam_mode: bool = True
    tutor_panel: bool = True
    analytics_dashboard: bool = True
    document_management: bool = True
    video_lessons: bool = False
    question_bank: bool = True
    leaderboard: bool = True
    student_monitoring: bool = True


class ApproveTenantRequest(BaseModel):
    institution_id: str = Field(..., min_length=14, max_length=14, pattern=r'^[A-Z]{4}-[A-Z]{4}-[0-9]{4}$')
    notes: Optional[str] = None
    features: Optional[TenantFeatures] = None


class RejectTenantRequest(BaseModel):
    reason: str = Field(..., min_length=10)


class SuspendTenantRequest(BaseModel):
    reason: str = Field(..., min_length=10)


class UpdateFeaturesRequest(BaseModel):
    features: TenantFeatures


class UpdateLimitsRequest(BaseModel):
    max_students: Optional[int] = None
    max_tutors: Optional[int] = None
    subscription_tier: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=8)


class SuperAdminResponse(BaseModel):
    _id: str
    email: str
    name: str
    role: str
    permissions: List[str]


# ============ HELPERS ============

async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


def convert_objectids(obj):
    """Recursively convert ObjectId fields to strings"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectids(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def generate_tenant_id() -> str:
    """Generate a tenant ID with pattern: AAAA-0000."""
    letters = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(4))
    digits = ''.join(secrets.choice(string.digits) for _ in range(4))
    return f"{letters}-{digits}"


def generate_institution_id() -> str:
    """Generate an institution ID with pattern: AAAA-BBBB-0000."""
    prefix = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(4))
    school = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(4))
    digits = ''.join(secrets.choice(string.digits) for _ in range(4))
    return f"{prefix}-{school}-{digits}"


INSTITUTION_ID_PATTERN = re.compile(r"^[A-Z]{4}-[A-Z]{4}-[0-9]{4}$")


def normalize_institution_id(institution_id: str) -> str:
    return institution_id.strip().upper()


def derive_tenant_id(institution_id: str) -> str:
    normalized = normalize_institution_id(institution_id)
    if not INSTITUTION_ID_PATTERN.match(normalized):
        raise HTTPException(
            status_code=400,
            detail="Institution ID must match AAAA-BBBB-0000 format"
        )
    parts = normalized.split("-")
    return f"{parts[1]}-{parts[2]}"


def build_db_name(institution_id: str) -> str:
    normalized = normalize_institution_id(institution_id)
    return f"skb_{normalized.lower()}"


async def ensure_tenant_indexes(tenant_db) -> None:
    await tenant_db["students"].create_index([("username_lower", 1)], unique=True, sparse=True, name="uniq_students_username_lower")
    await tenant_db["students"].create_index([("username", 1)], unique=True, name="uniq_students_username")
    await tenant_db["students"].create_index([("email", 1)], sparse=True, name="idx_students_email")
    await tenant_db["tutors"].create_index([("username_lower", 1)], unique=True, sparse=True, name="uniq_tutors_username_lower")
    await tenant_db["tutors"].create_index([("username", 1)], unique=True, name="uniq_tutors_username")
    await tenant_db["tutors"].create_index([("tutor_id", 1)], unique=True, name="uniq_tutors_tutor_id")
    await tenant_db["admins"].create_index([("email", 1)], unique=True, name="uniq_admins_email")
    await tenant_db["strokes"].create_index([("user_id", 1), ("timestamp", -1)], name="idx_strokes_user_ts")
    await tenant_db["smartboard_sessions"].create_index([("session_id", 1)], unique=True, name="uniq_smartboard_session_id")
    await tenant_db["smartboard_sessions"].create_index([("tutor_id", 1), ("status", 1)], name="idx_smartboard_tutor_status")


def create_superadmin_token(admin_id: str, email: str) -> str:
    """Create JWT token for super admin"""
    payload = {
        "sub": admin_id,
        "email": email,
        "type": "superadmin",
        "exp": datetime.utcnow() + timedelta(hours=SUPERADMIN_JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SUPERADMIN_JWT_SECRET, algorithm=SUPERADMIN_JWT_ALGORITHM)


async def verify_superadmin_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Verify super admin JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SUPERADMIN_JWT_SECRET, algorithms=[SUPERADMIN_JWT_ALGORITHM])

        if payload.get("type") != "superadmin":
            raise HTTPException(status_code=401, detail="Invalid token type")

        admin_id = payload.get("sub")
        if not admin_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Verify admin exists and is active
        master_db = await db.get_master_db()
        admin = await master_db["super_admins"].find_one({
            "_id": ObjectId(admin_id),
            "is_active": True,
        })

        if not admin:
            raise HTTPException(status_code=401, detail="Admin not found or inactive")

        return {
            "admin_id": str(admin["_id"]),
            "email": admin["email"],
            "name": admin["name"],
            "role": admin.get("role", "super_admin"),
            "permissions": admin.get("permissions", ["all"]),
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


# ============ AUTH ENDPOINTS ============

@router.get("/health")
async def health_check():
    """Health check for connection testing"""
    return {"status": "ok", "service": "superadmin"}


@router.post("/login")
async def superadmin_login(
    request: SuperAdminLoginRequest,
    db: DatabaseManager = Depends(get_database),
):
    """Authenticate super admin"""
    master_db = await db.get_master_db()

    admin = await master_db["super_admins"].find_one({
        "email": request.email,
        "is_active": True,
    })

    if not admin:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not pwd_context.verify(request.password, admin["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Update last login
    await master_db["super_admins"].update_one(
        {"_id": admin["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )

    token = create_superadmin_token(str(admin["_id"]), admin["email"])

    return {
        "access_token": token,
        "admin": {
            "_id": str(admin["_id"]),
            "email": admin["email"],
            "name": admin["name"],
            "role": admin.get("role", "super_admin"),
            "permissions": admin.get("permissions", ["all"]),
        }
    }


@router.post("/setup")
async def first_time_setup(
    request: SuperAdminSetupRequest,
    db: DatabaseManager = Depends(get_database),
):
    """Create first super admin (requires setup key)"""
    if request.setup_key != SUPERADMIN_SETUP_KEY:
        raise HTTPException(status_code=403, detail="Invalid setup key")

    master_db = await db.get_master_db()

    # Check if any super admin already exists
    existing = await master_db["super_admins"].find_one({"is_active": True})
    if existing:
        raise HTTPException(status_code=400, detail="Super admin already exists")

    # Check if email already exists
    existing_email = await master_db["super_admins"].find_one({"email": request.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create super admin
    admin_doc = {
        "email": request.email,
        "password_hash": pwd_context.hash(request.password),
        "name": request.name,
        "role": "super_admin",
        "permissions": ["all"],
        "created_at": datetime.utcnow(),
        "is_active": True,
    }

    result = await master_db["super_admins"].insert_one(admin_doc)

    return {"admin_id": str(result.inserted_id)}


# ============ DASHBOARD ENDPOINTS ============

@router.get("/dashboard/stats")
async def get_dashboard_stats(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Get dashboard statistics"""
    master_db = await db.get_master_db()

    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # Count tenants by status
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    status_counts = await master_db["tenants"].aggregate(pipeline).to_list(length=100)
    status_map = {item["_id"]: item["count"] for item in status_counts}

    # Count recent registrations
    week_count = await master_db["tenants"].count_documents({
        "created_at": {"$gte": week_ago}
    })
    month_count = await master_db["tenants"].count_documents({
        "created_at": {"$gte": month_ago}
    })

    total = sum(status_map.values())

    return {
        "total_tenants": total,
        "pending_registrations": status_map.get("pending", 0),
        "active_tenants": status_map.get("active", 0),
        "suspended_tenants": status_map.get("suspended", 0),
        "rejected_registrations": status_map.get("rejected", 0),
        "total_students": 0,  # Would need to aggregate across tenant DBs
        "total_tutors": 0,
        "registrations_this_week": week_count,
        "registrations_this_month": month_count,
    }


# ============ TENANT ENDPOINTS ============

@router.get("/tenants")
async def get_tenants(
    status: Optional[str] = None,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Get all tenants with optional status filter"""
    master_db = await db.get_master_db()

    query = {}
    if status:
        query["status"] = status

    tenants = await master_db["tenants"].find(query).sort("created_at", -1).to_list(length=1000)

    # Convert all ObjectIds and dates to JSON-serializable format
    result = []
    for tenant in tenants:
        tenant = convert_objectids(tenant)
        if "pending_admin" in tenant and tenant["pending_admin"]:
            # Remove password hash from response
            tenant["pending_admin"].pop("password_hash", None)
        result.append(tenant)

    return result


@router.get("/tenants/{tenant_id}")
async def get_tenant_by_id(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Get a specific tenant by ID"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    tenant = convert_objectids(tenant)
    if "pending_admin" in tenant and tenant["pending_admin"]:
        tenant["pending_admin"].pop("password_hash", None)

    return tenant


@router.get("/tenants/{tenant_id}/files")
async def get_tenant_files(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Get application files for a tenant"""
    master_db = await db.get_master_db()

    try:
        files = await master_db["tenant_application_files"].find({
            "tenant_request_id": ObjectId(tenant_id)
        }).to_list(length=100)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    return [convert_objectids(file) for file in files]


@router.post("/tenants/{tenant_id}/approve")
async def approve_tenant(
    tenant_id: str,
    request: ApproveTenantRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Approve a pending tenant registration"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if tenant["status"] != "pending":
        raise HTTPException(status_code=400, detail="Tenant is not in pending status")

    normalized_institution_id = normalize_institution_id(request.institution_id)
    new_tenant_id = derive_tenant_id(normalized_institution_id)
    db_name = build_db_name(normalized_institution_id)

    # Ensure unique institution_id, tenant_id, and db_name
    existing = await master_db["tenants"].find_one({
        "_id": {"$ne": ObjectId(tenant_id)},
        "$or": [
            {"institution_id": normalized_institution_id},
            {"tenant_id": new_tenant_id},
            {"db_name": db_name},
        ]
    })
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Institution ID or tenant ID already in use"
        )

    # Prepare subdomain
    subdomain = tenant.get("subdomain")
    if not subdomain:
        subdomain = tenant.get("organization", "").lower()
        subdomain = ''.join(c for c in subdomain if c.isalnum())[:20]

    # Default features
    default_features = {
        "smartboard": True,
        "online_class": False,
        "ai_chat": True,
        "stoody_pen": False,
        "exam_mode": True,
        "tutor_panel": True,
        "analytics_dashboard": True,
        "document_management": True,
        "video_lessons": False,
        "question_bank": True,
        "leaderboard": True,
        "student_monitoring": True,
    }

    if request.features:
        default_features.update(request.features.dict())

    # Approval action
    approval_action = {
        "action": "approved",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "notes": request.notes,
    }

    # Update tenant
    # Create tenant database + master admin
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Tenant database unavailable")

    pending_admin = tenant.get("pending_admin") or {}
    pending_password_hash = pending_admin.get("password_hash")
    if not pending_password_hash:
        raise HTTPException(status_code=400, detail="Pending admin credentials missing")

    admin_email = tenant.get("admin_email")
    existing_admin = await tenant_db["admins"].find_one({"email": admin_email})
    if not existing_admin:
        admin_doc = {
            "email": admin_email,
            "password_hash": pending_password_hash,
            "full_name": pending_admin.get("full_name") or tenant.get("admin_full_name"),
            "role": "master_admin",
            "permissions": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
            "created_by": None,
            "two_fa": pending_admin.get("two_fa") or {
                "enabled": False,
                "required": True,
                "secret_enc": None,
                "verified_at": None
            }
        }
        await tenant_db["admins"].insert_one(admin_doc)

    await ensure_tenant_indexes(tenant_db)

    update_result = await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {
                "status": "approved",
                "tenant_id": new_tenant_id,
                "db_name": db_name,
                "institution_id": normalized_institution_id,
                "subdomain": subdomain,
                "approved_at": datetime.utcnow(),
                "enabled_features": default_features,
                "max_students": 100,
                "max_tutors": 10,
                "subscription_tier": "standard",
            },
            "$push": {
                "approval_history": approval_action,
            }
        }
    )

    if update_result.modified_count != 1:
        raise HTTPException(status_code=500, detail="Failed to approve tenant")

    return {"success": True, "tenant_id": new_tenant_id, "institution_id": normalized_institution_id}


@router.post("/tenants/{tenant_id}/reject")
async def reject_tenant(
    tenant_id: str,
    request: RejectTenantRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Reject a tenant registration"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    rejection_action = {
        "action": "rejected",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "notes": request.reason,
    }

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {
                "status": "rejected",
                "rejected_at": datetime.utcnow(),
                "rejection_reason": request.reason,
            },
            "$push": {
                "approval_history": rejection_action,
            }
        }
    )

    return {"success": True}


@router.post("/tenants/{tenant_id}/activate")
async def activate_tenant(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Activate an approved tenant (after database setup)"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if tenant["status"] not in ["approved", "suspended"]:
        raise HTTPException(status_code=400, detail="Tenant must be approved or suspended to activate")

    activation_action = {
        "action": "activated",
        "by": admin["email"],
        "at": datetime.utcnow(),
    }

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {"status": "active"},
            "$push": {"approval_history": activation_action}
        }
    )

    return {"success": True}


@router.post("/tenants/{tenant_id}/suspend")
async def suspend_tenant(
    tenant_id: str,
    request: SuspendTenantRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Suspend an active tenant"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    suspend_action = {
        "action": "suspended",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "notes": request.reason,
    }

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {"status": "suspended"},
            "$push": {"approval_history": suspend_action}
        }
    )

    return {"success": True}


@router.put("/tenants/{tenant_id}/features")
async def update_tenant_features(
    tenant_id: str,
    request: UpdateFeaturesRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Update feature flags for a tenant"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    feature_action = {
        "action": "feature_changed",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "changes": request.features.dict(),
    }

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {"enabled_features": request.features.dict()},
            "$push": {"approval_history": feature_action}
        }
    )

    return {"success": True}


@router.put("/tenants/{tenant_id}/limits")
async def update_tenant_limits(
    tenant_id: str,
    request: UpdateLimitsRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Update limits for a tenant"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    update_fields = {}
    if request.max_students is not None:
        update_fields["max_students"] = request.max_students
    if request.max_tutors is not None:
        update_fields["max_tutors"] = request.max_tutors
    if request.subscription_tier is not None:
        update_fields["subscription_tier"] = request.subscription_tier

    if update_fields:
        await master_db["tenants"].update_one(
            {"_id": ObjectId(tenant_id)},
            {"$set": update_fields}
        )

    return {"success": True}


@router.post("/tenants/{tenant_id}/reset-admin-password")
async def reset_tenant_admin_password(
    tenant_id: str,
    request: ResetPasswordRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Reset the master admin password for a tenant"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    new_password_hash = pwd_context.hash(request.new_password)

    # If tenant is still pending, update pending_admin
    if tenant["status"] == "pending" and tenant.get("pending_admin"):
        await master_db["tenants"].update_one(
            {"_id": ObjectId(tenant_id)},
            {
                "$set": {"pending_admin.password_hash": new_password_hash},
                "$push": {
                    "approval_history": {
                        "action": "password_reset",
                        "by": admin["email"],
                        "at": datetime.utcnow(),
                        "notes": "Password reset by super admin",
                    }
                }
            }
        )
        return {"success": True}

    # For active tenants, update in their tenant database
    if tenant["status"] in ["active", "approved"] and tenant.get("db_name"):
        tenant_db = await db.get_tenant_db(tenant["db_name"])
        if tenant_db:
            # Find master admin in tenant database
            master_admin = await tenant_db["admins"].find_one({"role": "master_admin"})
            if master_admin:
                await tenant_db["admins"].update_one(
                    {"_id": master_admin["_id"]},
                    {"$set": {"password_hash": new_password_hash}}
                )

                # Log the action
                await master_db["tenants"].update_one(
                    {"_id": ObjectId(tenant_id)},
                    {
                        "$push": {
                            "approval_history": {
                                "action": "password_reset",
                                "by": admin["email"],
                                "at": datetime.utcnow(),
                                "notes": "Password reset by super admin",
                            }
                        }
                    }
                )
                return {"success": True}

    raise HTTPException(status_code=400, detail="Unable to reset password for this tenant")


class UpdateTenantIdRequest(BaseModel):
    institution_id: str = Field(..., min_length=14, max_length=14, pattern=r'^[A-Z]{4}-[A-Z]{4}-[0-9]{4}$')


class SendMessageRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1)
    priority: Optional[str] = Field(default="normal", pattern=r'^(low|normal|high|urgent)$')


class DeleteTenantRequest(BaseModel):
    confirmation: str = Field(..., description="Must match institution name to confirm deletion")


@router.put("/tenants/{tenant_id}/institution-id")
async def update_tenant_institution_id(
    tenant_id: str,
    request: UpdateTenantIdRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Update the institution/tenant ID for a tenant"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    new_id = normalize_institution_id(request.institution_id)
    new_tenant_id = derive_tenant_id(new_id)

    # Check if this ID is already in use by another tenant
    existing = await master_db["tenants"].find_one({
        "_id": {"$ne": ObjectId(tenant_id)},
        "$or": [
            {"institution_id": new_id},
            {"tenant_id": new_tenant_id},
        ]
    })
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Institution ID or tenant ID already in use by another tenant"
        )

    old_id = tenant.get("institution_id")

    # Update institution_id + derived tenant_id (db_name unchanged)
    update_action = {
        "action": "id_changed",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "notes": f"Institution ID changed from {old_id} to {new_id}",
        "changes": {"old_id": old_id, "new_id": new_id}
    }

    update_fields = {
        "institution_id": new_id,
        "tenant_id": new_tenant_id,
    }
    if not tenant.get("db_name"):
        update_fields["db_name"] = build_db_name(new_id)

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": update_fields,
            "$push": {"approval_history": update_action}
        }
    )

    return {"success": True, "institution_id": new_id, "tenant_id": new_tenant_id}


@router.post("/tenants/{tenant_id}/messages")
async def send_message_to_tenant(
    tenant_id: str,
    request: SendMessageRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Send a message from super admin to tenant master admin"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Create message document
    message_doc = {
        "tenant_id": ObjectId(tenant_id),
        "from_admin": admin["email"],
        "from_name": admin["name"],
        "to_email": tenant.get("admin_email"),
        "subject": request.subject,
        "message": request.message,
        "priority": request.priority,
        "created_at": datetime.utcnow(),
        "read": False,
        "read_at": None,
    }

    result = await master_db["superadmin_messages"].insert_one(message_doc)

    return {
        "success": True,
        "message_id": str(result.inserted_id)
    }


@router.get("/tenants/{tenant_id}/messages")
async def get_tenant_messages(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Get all messages sent to a specific tenant"""
    master_db = await db.get_master_db()

    try:
        messages = await master_db["superadmin_messages"].find({
            "tenant_id": ObjectId(tenant_id)
        }).sort("created_at", -1).to_list(length=100)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    return [convert_objectids(msg) for msg in messages]


@router.delete("/tenants/{tenant_id}")
async def delete_tenant(
    tenant_id: str,
    request: DeleteTenantRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Delete a tenant permanently (requires confirmation)"""
    master_db = await db.get_master_db()

    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Verify confirmation matches institution name
    if request.confirmation.lower() != tenant["institution_name"].lower():
        raise HTTPException(
            status_code=400,
            detail="Confirmation text does not match institution name"
        )

    # Log the deletion before removing
    logger.warning(
        f"Super admin {admin['email']} deleting tenant {tenant_id} "
        f"({tenant['institution_name']})"
    )

    # Delete associated files
    await master_db["tenant_application_files"].delete_many({
        "tenant_request_id": ObjectId(tenant_id)
    })

    # Delete associated messages
    await master_db["superadmin_messages"].delete_many({
        "tenant_id": ObjectId(tenant_id)
    })

    # Delete the tenant
    await master_db["tenants"].delete_one({"_id": ObjectId(tenant_id)})

    return {"success": True, "deleted": tenant["institution_name"]}
