"""
Async Admin API for SkillBot
Admin management endpoints with authentication and rate limiting
"""

import logging
import secrets
import string
import bcrypt
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from bson import ObjectId

from fastapi import (
    APIRouter,
    File,
    Form,
    Request,
    HTTPException,
    Depends,
    UploadFile,
    status,
    Query,
)
from fastapi.responses import Response as RawResponse
from pydantic import BaseModel, Field, EmailStr, field_validator

# Valid grades for students (class 6 to 12)
VALID_GRADES = ["6", "7", "8", "9", "10", "11", "12"]
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.permissions import has_permission, OPERATOR_PERMISSIONS
from core.cache import CacheManager
from core.security import MongoSanitizer, get_security_logger
from api.v1.auth_async import get_current_user, get_database, get_cache
from config_async import settings, MONGODB_URL, DISABLE_MONGODB

# Security logger for audit trail
security_logger = get_security_logger()

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# Utility functions
def generate_secure_password(length: int = 12) -> str:
    """Generate a cryptographically secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = "".join(secrets.choice(alphabet) for _ in range(length))
    # Ensure password has at least one digit and one special char
    if not any(c.isdigit() for c in password):
        password = password[:-1] + secrets.choice(string.digits)
    if not any(c in "!@#$%^&*" for c in password):
        password = password[:-1] + secrets.choice("!@#$%^&*")
    return password


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    # Bcrypt has a 72 byte limit, truncate if necessary
    password_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")


# Pydantic models
class CreateStudentRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: str = Field(..., min_length=2, max_length=100)
    password: Optional[str] = Field(
        None, min_length=6
    )  # Optional - will auto-generate if not provided
    email: Optional[EmailStr] = None
    date_of_birth: Optional[str] = None  # Format: YYYY-MM-DD
    gender: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    stream: Optional[str] = None
    grade: Optional[str] = Field(None, description="Student grade/class (6-12)")
    phone: Optional[str] = None
    section: Optional[str] = Field(None, description="Section (e.g. A, B, C)")
    plan_types: Optional[List[str]] = None
    subjects: Optional[List[str]] = None

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, v):
        if v is not None and v not in VALID_GRADES:
            raise ValueError(f"Grade must be one of {VALID_GRADES}, got '{v}'")
        return v


class UpdateStudentRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    stream: Optional[str] = None
    phone: Optional[str] = None
    plan_types: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    grade: Optional[str] = Field(
        None, description="New grade/class for the student (6-12)"
    )
    section: Optional[str] = Field(None, description="New section for the student")
    is_active: Optional[bool] = None

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, v):
        if v is not None and v not in VALID_GRADES:
            raise ValueError(f"Grade must be one of {VALID_GRADES}, got '{v}'")
        return v


class SessionPromotionRequest(BaseModel):
    """Request model for promoting students to next session"""

    new_session: str = Field(..., description="New academic session e.g., '2025-26'")
    grade_mappings: Dict[str, str] = Field(
        ...,
        description="Mapping of current grade to new grade e.g., {'10': '11', '11': '12'}",
    )
    # Filters to select which students to promote
    grade_filter: Optional[List[str]] = Field(
        None,
        description="Only promote students from these grades. If empty/None, apply to all grades in mappings",
    )
    section_filter: Optional[List[str]] = Field(
        None,
        description="Only promote students from these sections. If empty/None, apply to all sections",
    )
    student_ids: Optional[List[str]] = Field(
        None,
        description="Specific student IDs to promote. If provided, only these students are promoted (overrides grade/section filters)",
    )
    section_updates: Optional[Dict[str, str]] = Field(
        None,
        description="Optional section updates for specific students. Key is student_id, value is new section",
    )
    deactivate_old_content: bool = Field(
        True, description="Whether to mark old notes, assignments, tests as inactive"
    )
    preview_only: bool = Field(
        False, description="If true, only return preview without making changes"
    )

    @field_validator("grade_mappings")
    @classmethod
    def validate_grade_mappings(cls, v):
        for from_grade, to_grade in v.items():
            if from_grade not in VALID_GRADES:
                raise ValueError(
                    f"Source grade must be one of {VALID_GRADES}, got '{from_grade}'"
                )
            if to_grade not in VALID_GRADES:
                raise ValueError(
                    f"Target grade must be one of {VALID_GRADES}, got '{to_grade}'"
                )
        return v

    @field_validator("grade_filter")
    @classmethod
    def validate_grade_filter(cls, v):
        if v is not None:
            for grade in v:
                if grade not in VALID_GRADES:
                    raise ValueError(
                        f"Grade filter must only contain values from {VALID_GRADES}, got '{grade}'"
                    )
        return v


class SessionPromotionResponse(BaseModel):
    """Response model for session promotion"""

    success: bool
    message: str
    new_session: str
    students_promoted: int
    students_skipped: int
    content_deactivated: int
    details: Optional[List[Dict[str, Any]]] = None


class StudentResponse(BaseModel):
    id: str
    student_id: str
    username: str
    full_name: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    stream: Optional[str] = None
    grade: Optional[str] = None
    phone: Optional[str] = None
    plan_types: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    is_active: bool
    requires_password_change: Optional[bool] = None
    password_reset_requested: Optional[bool] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    generated_password: Optional[str] = (
        None  # Only included on creation if auto-generated
    )


class StudentsListResponse(BaseModel):
    students: List[StudentResponse]
    total: int
    page: int
    limit: int


class DashboardStats(BaseModel):
    total_students: int
    valid_students: int  # Students with is_active = true
    active_students: int  # Students who have logged in (have last_login)
    practice_sets_count: int
    test_series_count: int
    chapter_notes_count: int


class AdminSummary(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str
    permissions: List[str] = []
    is_active: bool
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None


class CreateOperatorAdminRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)
    permissions: List[str] = Field(default_factory=list)

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, value: List[str]) -> List[str]:
        invalid = [perm for perm in value if perm not in OPERATOR_PERMISSIONS]
        if invalid:
            raise ValueError(f"Invalid permissions: {invalid}")
        return value


class UpdateOperatorAdminRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        invalid = [perm for perm in value if perm not in OPERATOR_PERMISSIONS]
        if invalid:
            raise ValueError(f"Invalid permissions: {invalid}")
        return value


class ResetOperatorPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=8)


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access (regular or B2C)"""
    if current_user.get("user_type") not in ["admin", "b2c_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return current_user


def require_admin_permission(permission: str):
    """Require admin access with a specific permission for operator admins."""

    def dependency(current_user: Dict[str, Any] = Depends(require_admin)):
        if not has_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return current_user

    return dependency


def require_master_admin(current_user: Dict[str, Any] = Depends(require_admin)):
    """Require master admin role for tenant admin management."""
    if current_user.get("user_type") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    role = current_user.get("admin_role") or "master_admin"
    if role != "master_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Master admin access required"
        )
    return current_user


def require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Allow admin, B2C admin, and tutor roles"""
    if current_user.get("user_type") not in ["admin", "b2c_admin", "tutor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Tutor access required",
        )
    return current_user


def require_admin_or_tutor_permission(permission: str):
    """Require admin/tutor access with a specific permission for operator admins."""

    def dependency(current_user: Dict[str, Any] = Depends(require_admin_or_tutor)):
        if current_user.get("user_type") in [
            "admin",
            "b2c_admin",
        ] and not has_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return current_user

    return dependency


def is_b2c_admin(current_user: Dict[str, Any]) -> bool:
    """Check if the current user is a B2C admin"""
    return current_user.get("user_type") == "b2c_admin"


async def check_registration_limit(
    db: DatabaseManager,
    current_user: Dict[str, Any],
    collection: str,
    limit_field: str,
    additional: int = 1,
) -> None:
    """Raise 403 if adding records would exceed tenant registration limit."""
    if is_b2c_admin(current_user):
        return  # B2C admins are not subject to tenant limits

    tenant_id = current_user.get("tenant_id") or current_user.get("institution_id")
    if not tenant_id:
        return  # No tenant context - skip check

    try:
        master_db = await db.get_master_db()
        if master_db is None:
            return  # Master DB unavailable - fail open
        tenant_doc = await master_db["tenants"].find_one(
            {"$or": [{"tenant_id": tenant_id}, {"institution_id": tenant_id}]}
        )
        if not tenant_doc:
            return  # Tenant not found in master - skip

        limit = tenant_doc.get(limit_field, 0)
        if not limit or limit <= 0:
            return  # No limit configured

        current_count = await db.mongo_count_documents(collection, {})
        if current_count + additional > limit:
            entity_name = "Student" if collection == "students" else "Tutor"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{entity_name} registration limit reached ({current_count}/{limit}). "
                f"Contact your administrator to increase the limit.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Registration limit check failed (allowing request): %s", e)


async def db_find_one(
    db: DatabaseManager,
    collection: str,
    query: dict,
    current_user: Dict[str, Any],
    **kwargs,
):
    """Route find_one to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_find_one(collection, query, **kwargs)
    return await db.mongo_find_one(collection, query, **kwargs)


async def db_find(
    db: DatabaseManager,
    collection: str,
    query: dict,
    current_user: Dict[str, Any],
    **kwargs,
):
    """Route find to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_find(collection, query, **kwargs)
    return await db.mongo_find(collection, query, **kwargs)


async def db_insert_one(
    db: DatabaseManager, collection: str, document: dict, current_user: Dict[str, Any]
):
    """Route insert_one to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_insert_one(collection, document)
    return await db.mongo_insert_one(collection, document)


async def db_update_one(
    db: DatabaseManager,
    collection: str,
    query: dict,
    update: dict,
    current_user: Dict[str, Any],
    **kwargs,
):
    """Route update_one to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_update_one(collection, query, update, **kwargs)
    return await db.mongo_update_one(collection, query, update, **kwargs)


async def db_delete_one(
    db: DatabaseManager, collection: str, query: dict, current_user: Dict[str, Any]
):
    """Route delete_one to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_delete_one(collection, query)
    return await db.mongo_delete_one(collection, query)


async def get_tenant_db_or_403(db: DatabaseManager, current_user: Dict[str, Any]):
    """Resolve tenant DB for the current admin user."""
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Tenant context required"
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


async def calculate_streak_days(student_id: ObjectId, db: DatabaseManager) -> int:
    """Calculate consecutive login days for a student"""
    try:
        # Get all login activities sorted by timestamp descending
        login_activities = await db.mongo_find(
            "student_activity_log",
            {"student_id": student_id, "action": "login"},
            sort=[("timestamp", -1)],
            limit=365,  # Only check last year
        )

        if not login_activities:
            return 0

        # Extract unique login dates (ignore time)
        login_dates = []
        for activity in login_activities:
            timestamp = activity.get("timestamp")
            if timestamp:
                login_date = timestamp.date()
                if not login_dates or login_dates[-1] != login_date:
                    login_dates.append(login_date)

        if not login_dates:
            return 0

        # Calculate streak from most recent date backwards
        today = datetime.utcnow().date()
        streak = 0

        # Check if user logged in today or yesterday (streak is still active)
        most_recent = login_dates[0]
        if most_recent == today or most_recent == today - timedelta(days=1):
            streak = 1
            expected_date = most_recent - timedelta(days=1)

            # Count consecutive days
            for i in range(1, len(login_dates)):
                if login_dates[i] == expected_date:
                    streak += 1
                    expected_date -= timedelta(days=1)
                elif login_dates[i] < expected_date:
                    # Gap in streak, stop counting
                    break

        return streak

    except Exception as e:
        logger.error(f"Calculate streak error: {str(e)}")
        return 0


async def calculate_student_level(
    student_id: ObjectId, db: DatabaseManager
) -> tuple[int, int]:
    """Calculate student level and XP based on activities"""
    try:
        # Get all question attempts
        attempts = await db.mongo_find("question_attempts", {"student_id": student_id})

        # Calculate XP
        total_xp = 0
        for attempt in attempts:
            if attempt.get("is_correct"):
                # Award XP based on difficulty
                difficulty = attempt.get("metadata", {}).get("difficulty", "medium")
                if difficulty == "easy":
                    total_xp += 5
                elif difficulty == "medium":
                    total_xp += 10
                elif difficulty == "hard":
                    total_xp += 20
                else:
                    total_xp += 10  # Default
            else:
                # Small XP for attempting even if wrong
                total_xp += 1

        # Get chat sessions for bonus XP
        sessions = await db.mongo_find("chat_sessions", {"student_id": student_id})
        # 2 XP per chat session
        total_xp += len(sessions) * 2

        # Calculate level (100 XP per level)
        level = max(1, (total_xp // 100) + 1)

        return level, total_xp

    except Exception as e:
        logger.error(f"Calculate level error: {str(e)}")
        return 1, 0


# ==================== Operator Admin Management ====================


@router.get("/admins", response_model=List[AdminSummary])
@limiter.limit("30/minute")
async def list_admins(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_master_admin),
    db: DatabaseManager = Depends(get_database),
):
    """List admins within the tenant (master + operator)."""
    tenant_db = await get_tenant_db_or_403(db, current_user)
    cursor = tenant_db["admins"].find({}, {"password_hash": 0}).sort("created_at", 1)
    admins = await cursor.to_list(length=100)

    results: List[AdminSummary] = []
    for admin in admins:
        results.append(
            AdminSummary(
                id=str(admin.get("_id")),
                email=admin.get("email", ""),
                full_name=admin.get("full_name") or admin.get("name"),
                role=admin.get("role", "master_admin"),
                permissions=admin.get("permissions") or [],
                is_active=admin.get("is_active", True),
                created_at=admin.get("created_at"),
                created_by=str(admin.get("created_by"))
                if admin.get("created_by")
                else None,
            )
        )

    return results


@router.post("/admins/operator", response_model=AdminSummary, status_code=201)
@limiter.limit("10/minute")
async def create_operator_admin(
    request: Request,
    payload: CreateOperatorAdminRequest,
    current_user: Dict[str, Any] = Depends(require_master_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Create the operator admin for this tenant (master admin only)."""
    tenant_db = await get_tenant_db_or_403(db, current_user)

    existing_operator = await tenant_db["admins"].find_one({"role": "operator_admin"})
    if existing_operator:
        raise HTTPException(
            status_code=400, detail="Operator admin already exists for this tenant"
        )

    existing_email = await tenant_db["admins"].find_one({"email": payload.email})
    if existing_email:
        raise HTTPException(
            status_code=400, detail="Email already in use for this tenant"
        )

    admin_doc = {
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "full_name": payload.full_name,
        "role": "operator_admin",
        "permissions": payload.permissions or [],
        "is_active": True,
        "created_at": datetime.utcnow(),
        "created_by": ObjectId(current_user.get("user_id")),
        "two_fa": {
            "enabled": False,
            "required": True,
            "secret_enc": None,
            "verified_at": None,
        },
    }

    result = await tenant_db["admins"].insert_one(admin_doc)
    admin_doc["_id"] = result.inserted_id

    return AdminSummary(
        id=str(result.inserted_id),
        email=admin_doc["email"],
        full_name=admin_doc.get("full_name"),
        role=admin_doc["role"],
        permissions=admin_doc.get("permissions") or [],
        is_active=admin_doc.get("is_active", True),
        created_at=admin_doc.get("created_at"),
        created_by=str(admin_doc.get("created_by")),
    )


@router.put("/admins/operator/{admin_id}", response_model=AdminSummary)
@limiter.limit("10/minute")
async def update_operator_admin(
    request: Request,
    admin_id: str,
    payload: UpdateOperatorAdminRequest,
    current_user: Dict[str, Any] = Depends(require_master_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Update operator admin details and permissions (master admin only)."""
    tenant_db = await get_tenant_db_or_403(db, current_user)

    try:
        admin_oid = ObjectId(admin_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid admin_id")

    admin_doc = await tenant_db["admins"].find_one(
        {"_id": admin_oid, "role": "operator_admin"}
    )
    if not admin_doc:
        raise HTTPException(status_code=404, detail="Operator admin not found")

    update_fields: Dict[str, Any] = {}
    if payload.full_name is not None:
        update_fields["full_name"] = payload.full_name
    if payload.email is not None and payload.email != admin_doc.get("email"):
        existing = await tenant_db["admins"].find_one(
            {"email": payload.email, "_id": {"$ne": admin_oid}}
        )
        if existing:
            raise HTTPException(
                status_code=400, detail="Email already in use for this tenant"
            )
        update_fields["email"] = payload.email
    if payload.permissions is not None:
        update_fields["permissions"] = payload.permissions
    if payload.is_active is not None:
        update_fields["is_active"] = payload.is_active

    if update_fields:
        update_fields["updated_at"] = datetime.utcnow()
        await tenant_db["admins"].update_one(
            {"_id": admin_oid}, {"$set": update_fields}
        )

    refreshed = await tenant_db["admins"].find_one({"_id": admin_oid})
    return AdminSummary(
        id=str(refreshed.get("_id")),
        email=refreshed.get("email", ""),
        full_name=refreshed.get("full_name") or refreshed.get("name"),
        role=refreshed.get("role", "operator_admin"),
        permissions=refreshed.get("permissions") or [],
        is_active=refreshed.get("is_active", True),
        created_at=refreshed.get("created_at"),
        created_by=str(refreshed.get("created_by"))
        if refreshed.get("created_by")
        else None,
    )


@router.post("/admins/operator/{admin_id}/reset-password")
@limiter.limit("10/minute")
async def reset_operator_admin_password(
    request: Request,
    admin_id: str,
    payload: ResetOperatorPasswordRequest,
    current_user: Dict[str, Any] = Depends(require_master_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Reset operator admin password (master admin only)."""
    tenant_db = await get_tenant_db_or_403(db, current_user)

    try:
        admin_oid = ObjectId(admin_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid admin_id")

    admin_doc = await tenant_db["admins"].find_one(
        {"_id": admin_oid, "role": "operator_admin"}
    )
    if not admin_doc:
        raise HTTPException(status_code=404, detail="Operator admin not found")

    await tenant_db["admins"].update_one(
        {"_id": admin_oid},
        {
            "$set": {
                "password_hash": hash_password(payload.new_password),
                "password_changed_at": datetime.utcnow(),
            }
        },
    )

    return {"success": True, "message": "Operator admin password updated"}


@router.delete("/admins/operator/{admin_id}")
@limiter.limit("10/minute")
async def delete_operator_admin(
    request: Request,
    admin_id: str,
    current_user: Dict[str, Any] = Depends(require_master_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Delete operator admin account (master admin only)."""
    tenant_db = await get_tenant_db_or_403(db, current_user)

    try:
        admin_oid = ObjectId(admin_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid admin_id")

    admin_doc = await tenant_db["admins"].find_one({"_id": admin_oid})
    if not admin_doc or admin_doc.get("role") != "operator_admin":
        raise HTTPException(status_code=404, detail="Operator admin not found")

    await tenant_db["admins"].delete_one({"_id": admin_oid})
    return {"success": True, "message": "Operator admin deleted"}


@router.get("/students", response_model=StudentsListResponse)
@limiter.limit("30/minute")
async def get_students(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, max_length=100),
    is_active: Optional[bool] = Query(None),
    grade: Optional[str] = Query(None, max_length=20),
    recently_active_days: Optional[int] = Query(
        None, ge=1, le=365, description="Filter students who logged in within N days"
    ),
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """Get paginated list of students"""
    try:
        # Determine which collection to use based on user type
        # B2C admin uses 'users' collection in STOODY-b2c database
        # Regular admin uses 'students' collection in tenant database
        is_b2c = is_b2c_admin(current_user)
        collection = "users" if is_b2c else "students"

        # Get admin_id from JWT token - filter by tenant
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Build filter - B2C uses different filter (no admin_id for B2C users since they're self-registered)
        if is_b2c:
            filter_dict = {}  # B2C admin sees all B2C users
        else:
            filter_dict = {
                "admin_id": admin_id
            }  # Multi-tenancy filter for regular admin

        if search:
            # SECURITY: Escape regex special characters to prevent NoSQL injection
            safe_search = MongoSanitizer.escape_for_like_search(search)
            if safe_search:
                filter_dict["$or"] = [
                    {"student_id": {"$regex": safe_search, "$options": "i"}},
                    {"username": {"$regex": safe_search, "$options": "i"}},
                    {"full_name": {"$regex": safe_search, "$options": "i"}},
                    {"name": {"$regex": safe_search, "$options": "i"}},
                    {"email": {"$regex": safe_search, "$options": "i"}},
                    {"phone": {"$regex": safe_search, "$options": "i"}},
                ]
        if is_active is not None:
            filter_dict["is_active"] = is_active
        if grade is not None:
            filter_dict["grade"] = grade
        if recently_active_days is not None:
            from datetime import timedelta

            cutoff = datetime.utcnow() - timedelta(days=recently_active_days)
            filter_dict["last_login"] = {"$gte": cutoff}

        # Check cache first (include admin_id and is_b2c for tenant isolation)
        cache_key = f"students:{'b2c' if is_b2c else str(admin_id)}:{page}:{limit}:{search}:{is_active}:{grade}:{recently_active_days}"
        cached_result = await cache.get(cache_key, "admin")

        if cached_result:
            return StudentsListResponse(**cached_result)

        # Get data using appropriate database
        if is_b2c:
            all_students = await db.b2c_find(collection, filter_dict)
            total_students = len(all_students)
            skip = (page - 1) * limit
            students_data = await db.b2c_find(
                collection,
                filter_dict,
                projection={"password_hash": 0},
                sort=[("created_at", -1)],
                skip=skip,
                limit=limit,
            )
        else:
            total_students = len(await db.mongo_find(collection, filter_dict))
            skip = (page - 1) * limit
            students_data = await db.mongo_find(
                collection,
                filter_dict,
                projection={"password_hash": 0},
                sort=[("created_at", -1)],
                skip=skip,
                limit=limit,
            )

        students = []
        for student in students_data:
            students.append(
                StudentResponse(
                    id=str(student.get("_id") or student.get("id")),
                    student_id=str(
                        student.get("student_id")
                        or student.get("_id")
                        or student.get("id")
                    ),
                    username=student.get("username", ""),
                    full_name=(
                        student.get("full_name")
                        or student.get("name")
                        or student.get("username", "")
                    ),
                    email=student.get("email"),
                    date_of_birth=student.get("date_of_birth"),
                    gender=student.get("gender"),
                    location=student.get("location"),
                    school=student.get("school"),
                    stream=student.get("stream"),
                    grade=student.get("grade"),
                    phone=student.get("phone"),
                    plan_types=student.get("plan_types"),
                    subjects=student.get("subjects"),
                    is_active=student.get("is_active", True),
                    requires_password_change=student.get(
                        "requires_password_change", False
                    ),
                    password_reset_requested=student.get(
                        "password_reset_requested", False
                    ),
                    created_at=student.get("created_at", datetime.utcnow()),
                    last_login=student.get("last_login"),
                )
            )

        response_data = {
            "students": [s.dict() for s in students],
            "total": total_students,
            "page": page,
            "limit": limit,
        }

        # Cache the result
        await cache.set(cache_key, response_data, 300, "admin")  # 5 minute cache

        return StudentsListResponse(**response_data)

    except Exception as e:
        logger.error(f"Get students error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get students",
        )


@router.put("/students/{student_id}", response_model=StudentResponse)
@limiter.limit("20/minute")
async def update_student(
    request: Request,
    student_id: str,
    update: UpdateStudentRequest,
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """Update student details or status"""
    try:
        # Get admin_id for tenant isolation
        admin_id = current_user.get("admin_id") or current_user.get("user_id")
        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Tenant context required"
            )

        # Find by custom student_id first, then by ObjectId - WITH admin_id filter for tenant isolation
        query: Dict[str, Any] = {
            "student_id": student_id,
            "admin_id": ObjectId(admin_id),
        }
        student = await db.mongo_find_one("students", query)
        if not student:
            try:
                oid = ObjectId(student_id)
                query = {"_id": oid, "admin_id": ObjectId(admin_id)}
                student = await db.mongo_find_one("students", query)
            except Exception:
                student = None

        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Student not found"
            )

        update_fields: Dict[str, Any] = {}
        if update.full_name is not None:
            update_fields["full_name"] = update.full_name
            # also keep legacy name in sync
            update_fields["name"] = update.full_name
        if update.email is not None:
            update_fields["email"] = update.email
        if update.date_of_birth is not None:
            update_fields["date_of_birth"] = update.date_of_birth
        if update.gender is not None:
            update_fields["gender"] = update.gender
        if update.location is not None:
            update_fields["location"] = update.location
        if update.school is not None:
            update_fields["school"] = update.school
        if update.stream is not None:
            update_fields["stream"] = update.stream
        if update.phone is not None:
            update_fields["phone"] = update.phone
        if update.plan_types is not None:
            update_fields["plan_types"] = update.plan_types
        if update.subjects is not None:
            update_fields["subjects"] = update.subjects
        if update.grade is not None:
            update_fields["grade"] = update.grade
        if update.section is not None:
            update_fields["section"] = update.section
        if update.is_active is not None:
            update_fields["is_active"] = update.is_active

        if update_fields:
            await db.mongo_update_one("students", query, {"$set": update_fields})

        # Invalidate cached students lists and dashboard stats
        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

        # Return updated document
        updated = await db.mongo_find_one(
            "students", query, projection={"password_hash": 0}
        )
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update student",
            )

        return StudentResponse(
            id=str(updated.get("_id")),
            username=updated.get("username", "unknown"),
            full_name=updated.get("full_name") or updated.get("name"),
            name=updated.get("name") or updated.get("full_name"),
            student_id=updated.get("student_id") or str(updated.get("_id")),
            email=updated.get("email"),
            date_of_birth=updated.get("date_of_birth"),
            gender=updated.get("gender"),
            location=updated.get("location"),
            school=updated.get("school"),
            stream=updated.get("stream"),
            grade=updated.get("grade"),
            phone=updated.get("phone"),
            plan_types=updated.get("plan_types"),
            subjects=updated.get("subjects"),
            is_active=updated.get("is_active", True),
            requires_password_change=updated.get("requires_password_change", False),
            password_reset_requested=updated.get("password_reset_requested", False),
            created_at=updated.get("created_at") or datetime.utcnow(),
            last_login=updated.get("last_login"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update student error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update student",
        )


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)


@router.post("/students/{student_id}/reset-password")
@limiter.limit("10/minute")
async def reset_student_password(
    request: Request,
    student_id: str,
    payload: ResetPasswordRequest,
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
):
    """Reset student password"""
    try:
        from core.auth import AuthManager

        auth = AuthManager()

        # Find student
        query: Dict[str, Any] = {"student_id": student_id}
        student = await db.mongo_find_one("students", query)
        if not student:
            try:
                oid = ObjectId(student_id)
                query = {"_id": oid}
                student = await db.mongo_find_one("students", query)
            except Exception:
                student = None

        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Student not found"
            )

        # Truncate password to 72 bytes before hashing (bcrypt limit)
        new_password = payload.new_password
        password_bytes = new_password.encode("utf-8")
        if len(password_bytes) > 72:
            logger.warning(
                f"Password too long ({len(password_bytes)} bytes), truncating to 72 bytes"
            )
            new_password = password_bytes[:72].decode("utf-8", errors="ignore")

        # Update password hash and reset flags
        password_hash = auth.get_password_hash(new_password)
        ok = await db.mongo_update_one(
            "students",
            query,
            {
                "$set": {
                    "password_hash": password_hash,
                    "requires_password_change": True,  # Force password change on next login
                    "password_reset_requested": False,  # Clear reset request flag
                    "password_reset_by_admin_at": datetime.utcnow(),
                }
            },
        )
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset password",
            )

        return {"success": True, "message": "Password reset successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset student password error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password",
        )


@router.post("/students", response_model=StudentResponse)
@limiter.limit("10/minute")
async def create_student(
    request: Request,
    student_data: CreateStudentRequest,
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """Create a new student"""
    try:
        # Ensure MongoDB is available
        if db.mongo_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB is not configured or unavailable",
            )

        # Check registration limit before proceeding
        await check_registration_limit(db, current_user, "students", "max_students")

        from core.auth import AuthManager

        auth_manager = AuthManager()

        # Get admin_id from JWT token
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Check if username already exists (case-insensitive)
        if student_data.username:
            normalized_username = student_data.username.strip()
            username_lower = normalized_username.lower()
            existing_student = await db.mongo_find_one(
                "students", {"username_lower": username_lower}
            )
            if not existing_student:
                existing_student = await db.mongo_find_one(
                    "students",
                    {"username": normalized_username},
                    collation={"locale": "en", "strength": 2},
                )
            if existing_student:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists. Please choose a different username.",
                )
            student_data.username = normalized_username
        else:
            # Auto-generate simple username: {first_name}{4-digit-random}
            # Extract first name from full_name
            import re
            import random

            # Get first word only, lowercase, alphanumeric only
            first_name = re.sub(
                r"[^a-zA-Z0-9]",
                "",
                student_data.full_name.split()[0]
                if student_data.full_name.split()
                else "student",
            ).lower()

            # Use first name (max 8 chars) + random 4 digits
            short_name = first_name[:8] if first_name else "student"

            # Generate unique username with random suffix
            max_attempts = 100
            for _ in range(max_attempts):
                random_suffix = random.randint(1000, 9999)
                username = f"{short_name}{random_suffix}"

                # Check if username exists within this tenant
                existing = await db.mongo_find_one(
                    "students",
                    {"username_lower": username.lower(), "admin_id": admin_id},
                )
                if not existing:
                    student_data.username = username
                    break
            else:
                # Fallback if all attempts fail (extremely rare)
                student_data.username = (
                    f"{short_name}{int(datetime.utcnow().timestamp()) % 10000}"
                )

        # Check if email already exists within this admin's tenant
        # Email can be duplicated across different admins, but not within same admin
        if student_data.email:
            existing_email = await db.mongo_find_one(
                "students", {"admin_id": admin_id, "email": student_data.email}
            )
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already exists in your organization",
                )

        # Student_id = username (simplified - username is already unique)
        auto_student_id = student_data.username

        # Auto-generate password if not provided
        plain_password = student_data.password or generate_secure_password()
        password_hash = hash_password(plain_password)

        # Get School Settings to auto-assign Plan Types and Subjects
        # Fetch directly from DB to avoid circular dependency or extra import
        school_settings = await db.mongo_find_one(
            "school_settings", {"admin_id": str(admin_id)}
        )

        # Determine plan_types and subjects
        # Priority: 1. Provided in request (if any) 2. From Settings 3. Empty list
        # Since stream is removed/fluid, we default to ALL available in settings if not provided
        assigned_plan_types = student_data.plan_types
        if not assigned_plan_types and school_settings:
            assigned_plan_types = school_settings.get("plan_types", [])

        assigned_subjects = student_data.subjects
        if not assigned_subjects and school_settings:
            assigned_subjects = school_settings.get("subjects", [])

        # Create student document
        new_student = {
            "admin_id": admin_id,  # Links student to admin for multi-tenancy
            "student_id": auto_student_id,  # Auto-generated unique ID
            "username": student_data.username,
            "full_name": student_data.full_name,
            "name": student_data.full_name,  # Keep legacy field for compatibility
            "email": student_data.email,
            "username_lower": student_data.username.lower(),
            "password_hash": password_hash,
            "date_of_birth": student_data.date_of_birth,  # Store as string YYYY-MM-DD
            "gender": student_data.gender,
            "location": student_data.location,
            "school": student_data.school,
            "stream": None,  # Stream is removed/fluid
            "grade": student_data.grade,
            "section": student_data.section or "A",
            "phone": student_data.phone,
            "plan_types": assigned_plan_types,
            "subjects": assigned_subjects,
            "is_active": True,
            "requires_password_change": True,  # NEW - Force password change on first login
            "created_at": datetime.utcnow(),
            "created_by": current_user["user_id"],
        }

        # Insert student
        inserted_id = await db.mongo_insert_one("students", new_student)
        if not inserted_id:
            # Distinguish likely causes
            if db.mongo_client is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database unavailable",
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create student",
            )

        # Invalidate cached students lists and dashboard stats
        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

        # Return created student with generated password (if applicable)
        return StudentResponse(
            id=inserted_id,
            student_id=auto_student_id,
            username=student_data.username,
            full_name=student_data.full_name,
            email=student_data.email,
            date_of_birth=student_data.date_of_birth,
            gender=student_data.gender,
            location=student_data.location,
            school=student_data.school,
            stream=student_data.stream,
            grade=student_data.grade,
            phone=student_data.phone,
            plan_types=student_data.plan_types,
            subjects=student_data.subjects,
            is_active=True,
            requires_password_change=True,
            password_reset_requested=False,
            created_at=new_student["created_at"],
            generated_password=plain_password
            if not student_data.password
            else None,  # Only return if auto-generated
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create student error: {str(e)}", exc_info=True)

        # Check for duplicate key error from MongoDB unique index
        if "duplicate key error" in str(e).lower() or "E11000" in str(e):
            if "username" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists. Please choose a different username.",
                )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create student",
        )


@router.get("/students/{student_id}", response_model=StudentResponse)
@limiter.limit("60/minute")
async def get_student(
    request: Request,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
):
    """Get student by ID"""
    try:
        # Canonical: look up by business student_id only

        student = await db.mongo_find_one(
            "students", {"student_id": student_id}, projection={"password_hash": 0}
        )
        if not student:
            # Try by MongoDB ObjectId
            try:
                oid = ObjectId(student_id)
                student = await db.mongo_find_one(
                    "students", {"_id": oid}, projection={"password_hash": 0}
                )
            except Exception:
                student = None

        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Student not found"
            )

        return StudentResponse(
            id=str(student.get("_id")),
            username=student.get("username", "unknown"),
            full_name=student.get("full_name") or student.get("name"),
            name=student.get("name") or student.get("full_name"),
            student_id=student.get("student_id") or str(student.get("_id")),
            email=student.get("email"),
            is_active=student.get("is_active", True),
            created_at=student.get("created_at") or datetime.utcnow(),
            last_login=student.get("last_login"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get student error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get student",
        )


@router.put("/students/{student_id}/v2", response_model=StudentResponse)
@limiter.limit("20/minute")
async def update_student_v2(
    request: Request,
    student_id: str,
    update_data: UpdateStudentRequest,
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """Update student by ID (v2 - with cache invalidation)"""
    try:
        # Get admin_id for tenant isolation
        admin_id = current_user.get("admin_id") or current_user.get("user_id")
        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Tenant context required"
            )

        # Build update dict from provided fields only
        update_dict = {}
        if update_data.full_name is not None:
            update_dict["full_name"] = update_data.full_name
        if update_data.email is not None:
            update_dict["email"] = update_data.email
        if update_data.grade is not None:
            update_dict["grade"] = update_data.grade
        if update_data.section is not None:
            update_dict["section"] = update_data.section
        if update_data.is_active is not None:
            update_dict["is_active"] = update_data.is_active

        if not update_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update"
            )

        # Update student by student_id WITH admin_id filter for tenant isolation
        updated = await db.mongo_update_one(
            "students",
            {"student_id": student_id, "admin_id": ObjectId(admin_id)},
            {"$set": update_dict},
        )

        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Student not found"
            )

        # Get updated student
        student = await db.mongo_find_one(
            "students", {"student_id": student_id, "admin_id": ObjectId(admin_id)}
        )

        # Invalidate cached students lists and dashboard stats
        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

        # Format response
        return StudentResponse(
            id=str(student["_id"]),
            student_id=student["student_id"],
            username=student["username"],
            full_name=student.get("full_name", student.get("name", "")),
            email=student.get("email"),
            age=student.get("age"),
            gender=student.get("gender"),
            location=student.get("location"),
            school=student.get("school"),
            stream=student.get("stream"),
            grade=student.get("grade"),
            phone=student.get("phone"),
            plan_types=student.get("plan_types"),
            subjects=student.get("subjects"),
            is_active=student.get("is_active", True),
            created_at=student.get("created_at", datetime.utcnow()),
            last_login=student.get("last_login"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update student error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update student",
        )


@router.delete("/students/{student_id}")
@limiter.limit("10/minute")
async def delete_student(
    request: Request,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """Delete student by ID"""
    try:
        # Get admin_id from JWT token for multi-tenancy
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Try to find and delete by student_id (business ID) within this admin's tenant
        query = {"admin_id": admin_id, "student_id": student_id}
        deleted = await db.mongo_delete_one("students", query)

        # If not found by student_id, try by ObjectId within this admin's tenant
        if not deleted:
            try:
                oid = ObjectId(student_id)
                query = {"admin_id": admin_id, "_id": oid}
                deleted = await db.mongo_delete_one("students", query)
            except Exception:
                pass

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Student not found"
            )

        # Invalidate cached students lists and dashboard stats
        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

        logger.info(f"Student deleted successfully: {student_id} by admin {admin_id}")
        return {"message": "Student deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete student error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete student",
        )


@router.get("/dashboard/stats", response_model=DashboardStats)
@limiter.limit("30/minute")
async def get_dashboard_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_permission("view_analytics")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """Get admin dashboard statistics"""
    try:
        # Determine if B2C admin
        is_b2c = is_b2c_admin(current_user)

        # Get admin_id for data isolation
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Check cache first (include admin_id and is_b2c for isolation)
        cache_key = f"dashboard_stats_{'b2c' if is_b2c else str(admin_id)}"
        cached_stats = await cache.get(cache_key, "admin")
        if cached_stats:
            return DashboardStats(**cached_stats)

        if is_b2c:
            # B2C Admin - query STOODY-b2c database 'users' collection
            all_users = await db.b2c_find("users", {})
            total_students = len(all_users)

            # Valid students = students with is_active = true
            valid_students = len([s for s in all_users if s.get("is_active", True)])

            # Active students = students who have logged in at least once
            active_students = len(
                [s for s in all_users if s.get("last_login") is not None]
            )

            # Get document counts from B2C database
            practice_sets = await db.b2c_find(
                "documents", {"document_type": "Practice Sets"}
            )
            test_series = await db.b2c_find(
                "documents", {"document_type": "Test Series"}
            )
            chapter_notes = await db.b2c_find(
                "documents", {"document_type": "Chapter Notes"}
            )
        else:
            # Regular Admin - query tenant database
            admin_students = await db.mongo_find("students", {"admin_id": admin_id})
            total_students = len(admin_students)

            # Valid students = students with is_active = true
            valid_students = len(
                [s for s in admin_students if s.get("is_active", True)]
            )

            # Active students = students who have logged in at least once (have last_login field)
            active_students = len(
                [s for s in admin_students if s.get("last_login") is not None]
            )

            # Get document counts by type (filtered by admin_id)
            practice_sets = await db.mongo_find(
                "documents", {"document_type": "Practice Sets", "admin_id": admin_id}
            )
            test_series = await db.mongo_find(
                "documents", {"document_type": "Test Series", "admin_id": admin_id}
            )
            chapter_notes = await db.mongo_find(
                "documents", {"document_type": "Chapter Notes", "admin_id": admin_id}
            )

        stats_data = {
            "total_students": total_students,
            "valid_students": valid_students,
            "active_students": active_students,
            "practice_sets_count": len(practice_sets),
            "test_series_count": len(test_series),
            "chapter_notes_count": len(chapter_notes),
        }

        # Cache for 5 minutes
        await cache.set(cache_key, stats_data, 300, "admin")

        return DashboardStats(**stats_data)

    except Exception as e:
        logger.error(f"Dashboard stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dashboard statistics",
        )


@router.get("/dashboard/school-stats")
@limiter.limit("30/minute")
async def get_school_dashboard_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_permission("view_analytics")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """
    Get school-level dashboard statistics with:
    1. Class/Division level breakdown
    2. Teacher summary (teachers per subject per class, documents uploaded)
    """
    try:
        # Get admin_id for data isolation
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Check cache first
        cache_key = f"school_dashboard_stats_{admin_id}"
        cached_stats = await cache.get(cache_key, "admin")
        if cached_stats:
            return cached_stats

        # Get all students for this admin
        admin_students = await db.mongo_find("students", {"admin_id": admin_id})

        # Get all tutors for this admin
        admin_tutors = await db.mongo_find("tutors", {"created_by": str(admin_id)})

        # Get all documents for this admin
        admin_documents = await db.mongo_find("documents", {"admin_id": admin_id})

        # ===== CLASS/DIVISION LEVEL BREAKDOWN =====
        class_division_stats = {}
        for student in admin_students:
            # Handle None/null values explicitly
            grade = student.get("grade") or "Unknown"
            section = student.get("section") or "Unknown"
            is_active = student.get("is_active", True)
            has_logged_in = student.get("last_login") is not None

            key = f"{grade}"
            if key not in class_division_stats:
                class_division_stats[key] = {
                    "grade": grade,
                    "sections": {},
                    "total_students": 0,
                    "active_students": 0,
                    "logged_in_students": 0,
                }

            class_division_stats[key]["total_students"] += 1
            if is_active:
                class_division_stats[key]["active_students"] += 1
            if has_logged_in:
                class_division_stats[key]["logged_in_students"] += 1

            # Section-level stats
            if section not in class_division_stats[key]["sections"]:
                class_division_stats[key]["sections"][section] = {
                    "section": section,
                    "total_students": 0,
                    "active_students": 0,
                    "logged_in_students": 0,
                    "subjects": {},
                }

            class_division_stats[key]["sections"][section]["total_students"] += 1
            if is_active:
                class_division_stats[key]["sections"][section]["active_students"] += 1
            if has_logged_in:
                class_division_stats[key]["sections"][section][
                    "logged_in_students"
                ] += 1

            # Track subjects per section
            student_subjects = student.get("subjects", []) or []
            for subj in student_subjects:
                if (
                    subj
                    not in class_division_stats[key]["sections"][section]["subjects"]
                ):
                    class_division_stats[key]["sections"][section]["subjects"][subj] = 0
                class_division_stats[key]["sections"][section]["subjects"][subj] += 1

        # Convert to list format for frontend
        class_division_list = []
        for grade_key, grade_data in class_division_stats.items():
            sections_list = []
            for section_key, section_data in grade_data["sections"].items():
                sections_list.append(
                    {
                        "section": section_data["section"],
                        "total_students": section_data["total_students"],
                        "active_students": section_data["active_students"],
                        "logged_in_students": section_data["logged_in_students"],
                        "subjects": section_data["subjects"],
                    }
                )
            # Sort sections alphabetically
            sections_list.sort(key=lambda x: x["section"])
            class_division_list.append(
                {
                    "grade": grade_data["grade"],
                    "total_students": grade_data["total_students"],
                    "active_students": grade_data["active_students"],
                    "logged_in_students": grade_data["logged_in_students"],
                    "sections": sections_list,
                }
            )
        # Sort by grade
        class_division_list.sort(key=lambda x: str(x["grade"]))

        # ===== TEACHER SUMMARY =====
        teacher_summary = []
        for tutor in admin_tutors:
            tutor_id = tutor.get("tutor_id")
            tutor_name = tutor.get("name", tutor.get("username", "Unknown"))
            tutor_subjects = tutor.get("subjects", []) or []
            tutor_standards = tutor.get("standards", []) or []
            tutor_sections = tutor.get("sections", []) or []

            # Count documents uploaded by this teacher
            tutor_docs = [
                d
                for d in admin_documents
                if tutor_id in (d.get("teacher_ids", []) or [])
            ]
            notes_count = len(
                [d for d in tutor_docs if d.get("document_type") == "Chapter Notes"]
            )
            tests_count = len(
                [d for d in tutor_docs if d.get("document_type") == "Test Series"]
            )
            practice_count = len(
                [d for d in tutor_docs if d.get("document_type") == "Practice Sets"]
            )

            # Count assigned students
            assigned_ids = tutor.get("assigned_student_ids", []) or []

            teacher_summary.append(
                {
                    "tutor_id": tutor_id,
                    "name": tutor_name,
                    "email": tutor.get("email"),
                    "subjects": tutor_subjects,
                    "standards": tutor_standards,
                    "sections": tutor_sections,
                    "is_active": tutor.get("is_active", True),
                    "assigned_students_count": len(assigned_ids),
                    "documents_uploaded": {
                        "notes": notes_count,
                        "tests": tests_count,
                        "practice": practice_count,
                        "total": notes_count + tests_count + practice_count,
                    },
                }
            )

        # Sort teachers by name
        teacher_summary.sort(key=lambda x: x["name"].lower())

        # ===== DOCUMENT STATS BY CLASS =====
        docs_by_class = {}
        for doc in admin_documents:
            standard = doc.get("standard", "Unknown")
            doc_type = doc.get("document_type", "Unknown")

            if standard not in docs_by_class:
                docs_by_class[standard] = {
                    "grade": standard,
                    "Chapter Notes": 0,
                    "Test Series": 0,
                    "Practice Sets": 0,
                }
            if doc_type in docs_by_class[standard]:
                docs_by_class[standard][doc_type] += 1

        docs_by_class_list = list(docs_by_class.values())
        docs_by_class_list.sort(key=lambda x: str(x["grade"]))

        # ===== SCHOOL STATISTICS (new overview cards) =====
        now = datetime.utcnow()
        active_student_cutoff = now - timedelta(days=15)
        active_teacher_cutoff = now - timedelta(days=7)

        # --- Total Students by class/section ---
        total_students_by_class = {}
        active_students_by_class = {}
        for student in admin_students:
            grade = str(student.get("grade") or "Unknown")
            section = str(student.get("section") or "Unknown")

            # Total students
            if grade not in total_students_by_class:
                total_students_by_class[grade] = {"total": 0, "sections": {}}
            total_students_by_class[grade]["total"] += 1
            total_students_by_class[grade]["sections"][section] = (
                total_students_by_class[grade]["sections"].get(section, 0) + 1
            )

            # Active students (logged in within last 15 days)
            last_login = student.get("last_login")
            if last_login and last_login >= active_student_cutoff:
                if grade not in active_students_by_class:
                    active_students_by_class[grade] = {"total": 0, "sections": {}}
                active_students_by_class[grade]["total"] += 1
                active_students_by_class[grade]["sections"][section] = (
                    active_students_by_class[grade]["sections"].get(section, 0) + 1
                )

        total_active_students = sum(
            v["total"] for v in active_students_by_class.values()
        )

        # --- Teachers by highest grade taught ---
        teachers_by_class = {}
        total_active_teachers = 0
        teachers_with_docs = set()

        for tutor in admin_tutors:
            tutor_id = tutor.get("tutor_id")
            standards = tutor.get("standards", []) or []
            # Find highest grade
            numeric_grades = [s for s in standards if s.isdigit()]
            if numeric_grades:
                highest_grade = str(max(int(g) for g in numeric_grades))
            else:
                highest_grade = "Unknown"

            if highest_grade not in teachers_by_class:
                teachers_by_class[highest_grade] = {"total": 0, "active": 0}
            teachers_by_class[highest_grade]["total"] += 1

            # Check if active (logged in within last 7 days)
            tutor_last_login = tutor.get("last_login")
            if tutor_last_login and tutor_last_login >= active_teacher_cutoff:
                teachers_by_class[highest_grade]["active"] += 1
                total_active_teachers += 1

            # Check if teacher has uploaded any documents
            tutor_docs = [
                d
                for d in admin_documents
                if tutor_id in (d.get("teacher_ids", []) or [])
            ]
            if tutor_docs:
                teachers_with_docs.add(tutor_id)

        # --- Multi-feature Adoption rates ---
        # Teacher features: upload_content, generate_paper, check_paper, share_feedback
        # Student features: practice, test_attempt, login, chat

        total_teachers_count = len(admin_tutors)

        # Build student ID lists for querying activity collections
        # (some collections don't store admin_id, so query by student IDs)
        all_student_oids = [s["_id"] for s in admin_students]
        all_student_id_strs = [str(s["_id"]) for s in admin_students]
        student_ids_combined = all_student_oids + all_student_id_strs

        # Fetch additional collections for adoption tracking
        try:
            admin_papers = await db.mongo_find(
                "question_papers", {"admin_id": admin_id}
            )
        except Exception:
            admin_papers = []

        # Practice activity: spread across multiple collections with different ID formats
        # practice_attempts (string), practice_sessions (string), question_attempts (ObjectId), mcq_attempts (string)
        practice_student_ids = set()
        for coll_name in [
            "practice_attempts",
            "practice_sessions",
            "question_attempts",
            "mcq_attempts",
        ]:
            try:
                docs = await db.mongo_find(
                    coll_name, {"student_id": {"$in": student_ids_combined}}
                )
                for doc in docs:
                    sid = doc.get("student_id")
                    if sid:
                        practice_student_ids.add(str(sid))
            except Exception:
                pass

        # student_test_attempts: student_id stored as string, NO admin_id field
        try:
            admin_test_attempts = await db.mongo_find(
                "student_test_attempts", {"student_id": {"$in": student_ids_combined}}
            )
        except Exception:
            admin_test_attempts = []

        # chat_sessions: student_id stored as ObjectId, admin_id may be missing
        try:
            admin_chat_sessions = await db.mongo_find(
                "chat_sessions", {"student_id": {"$in": student_ids_combined}}
            )
        except Exception:
            admin_chat_sessions = []

        # student_activity_log: tracks chapter_viewed events (student_id as ObjectId)
        try:
            note_views = await db.mongo_find(
                "student_activity_log",
                {
                    "student_id": {"$in": student_ids_combined},
                    "action": "chapter_viewed",
                },
            )
        except Exception:
            note_views = []

        # --- Teacher feature sets ---
        # 1. Upload content: tutor in document teacher_ids or uploaded_by
        teachers_with_docs_set = set()
        for doc in admin_documents:
            for tid in doc.get("teacher_ids", []) or []:
                teachers_with_docs_set.add(str(tid))
            ub = doc.get("uploaded_by")
            if ub:
                teachers_with_docs_set.add(str(ub))

        # 2. Generate paper: created_by in question_papers
        teachers_paper_creators = set()
        for paper in admin_papers:
            creator = paper.get("created_by")
            if creator:
                teachers_paper_creators.add(str(creator))

        # 3. Check paper: graded_by/tutor_id in test attempts
        teachers_paper_checkers = set()
        for attempt in admin_test_attempts:
            graded_at = attempt.get("graded_at")
            grader = attempt.get("tutor_id") or attempt.get("graded_by")
            if graded_at and grader:
                teachers_paper_checkers.add(str(grader))

        # 4. Share feedback: tutor has active (shared) documents
        teachers_with_active_docs = set()
        for doc in admin_documents:
            if doc.get("is_active", False):
                for tid in doc.get("teacher_ids", []) or []:
                    teachers_with_active_docs.add(str(tid))
                ub = doc.get("uploaded_by")
                if ub:
                    teachers_with_active_docs.add(str(ub))

        TEACHER_FEATURES = [
            "upload_content",
            "generate_paper",
            "check_paper",
            "share_feedback",
        ]

        def get_teacher_features(tutor):
            """Return dict of feature_name -> bool for a teacher."""
            tutor_id = str(tutor.get("tutor_id", ""))
            user_id = str(tutor.get("_id", ""))
            all_ids = {tutor_id, user_id} - {""}
            features = {}
            features["upload_content"] = bool(all_ids & teachers_with_docs_set)
            features["generate_paper"] = bool(all_ids & teachers_paper_creators)
            features["check_paper"] = bool(all_ids & teachers_paper_checkers)
            features["share_feedback"] = bool(all_ids & teachers_with_active_docs)
            return features

        # Compute overall teacher adoption (avg of feature completion)
        teacher_adoption_scores = []
        for tutor in admin_tutors:
            features = get_teacher_features(tutor)
            score = sum(1 for v in features.values() if v) / len(TEACHER_FEATURES) * 100
            teacher_adoption_scores.append(score)

        teacher_adoption_pct = (
            round(sum(teacher_adoption_scores) / len(teacher_adoption_scores), 1)
            if teacher_adoption_scores
            else 0.0
        )

        # --- Student feature sets ---
        # practice_student_ids already built above from all practice collections

        students_with_tests = set()
        for attempt in admin_test_attempts:
            sid = attempt.get("student_id")
            if sid:
                students_with_tests.add(str(sid))

        students_with_chat = set()
        for session in admin_chat_sessions:
            sid = session.get("student_id")
            if sid:
                students_with_chat.add(str(sid))

        students_with_notes = set()
        for view in note_views:
            sid = view.get("student_id")
            if sid:
                students_with_notes.add(str(sid))

        STUDENT_FEATURES = ["notes", "practice", "test_attempt", "login", "chat"]

        def get_student_features(student):
            """Return dict of feature_name -> bool for a student."""
            sid = str(student["_id"])
            features = {}
            features["notes"] = sid in students_with_notes
            features["practice"] = sid in practice_student_ids
            features["test_attempt"] = sid in students_with_tests
            features["login"] = student.get("last_login") is not None
            features["chat"] = sid in students_with_chat
            return features

        # Build student lookup by grade for adoption
        students_by_grade = {}
        for student in admin_students:
            grade = str(student.get("grade") or "Unknown")
            if grade not in students_by_grade:
                students_by_grade[grade] = []
            students_by_grade[grade].append(student)

        total_student_count = len(admin_students)
        student_adoption_scores = []
        for student in admin_students:
            features = get_student_features(student)
            score = sum(1 for v in features.values() if v) / len(STUDENT_FEATURES) * 100
            student_adoption_scores.append(score)

        student_adoption_pct = (
            round(sum(student_adoption_scores) / len(student_adoption_scores), 1)
            if student_adoption_scores
            else 0.0
        )

        # Per-class teacher adoption
        teacher_adoption_by_class_pct = {}
        for tutor in admin_tutors:
            standards = tutor.get("standards", []) or []
            features = get_teacher_features(tutor)
            score = sum(1 for v in features.values() if v) / len(TEACHER_FEATURES) * 100
            for std in standards:
                grade = str(std)
                if grade not in teacher_adoption_by_class_pct:
                    teacher_adoption_by_class_pct[grade] = {"sum": 0, "count": 0}
                teacher_adoption_by_class_pct[grade]["sum"] += score
                teacher_adoption_by_class_pct[grade]["count"] += 1

        for grade in teacher_adoption_by_class_pct:
            data = teacher_adoption_by_class_pct[grade]
            teacher_adoption_by_class_pct[grade] = (
                round(data["sum"] / data["count"], 1) if data["count"] > 0 else 0.0
            )

        # Per-class student adoption
        student_adoption_by_class = {}
        for grade, grade_students in students_by_grade.items():
            scores = []
            for student in grade_students:
                features = get_student_features(student)
                score = (
                    sum(1 for v in features.values() if v) / len(STUDENT_FEATURES) * 100
                )
                scores.append(score)
            student_adoption_by_class[grade] = (
                round(sum(scores) / len(scores), 1) if scores else 0.0
            )

        # --- Active-only document counts (Practice Papers & Class Notes) ---
        active_practice_papers = {"total": 0, "by_class": {}}
        active_class_notes = {"total": 0, "by_class": {}}
        for doc in admin_documents:
            if not doc.get("is_active", True):
                continue
            standard = str(doc.get("standard", "Unknown"))
            doc_type = doc.get("document_type", "")
            if doc_type == "Practice Sets":
                active_practice_papers["total"] += 1
                active_practice_papers["by_class"][standard] = (
                    active_practice_papers["by_class"].get(standard, 0) + 1
                )
            elif doc_type == "Chapter Notes":
                active_class_notes["total"] += 1
                active_class_notes["by_class"][standard] = (
                    active_class_notes["by_class"].get(standard, 0) + 1
                )

        school_statistics = {
            "total_students": {
                "total": len(admin_students),
                "by_class": total_students_by_class,
            },
            "active_students": {
                "total": total_active_students,
                "by_class": active_students_by_class,
            },
            "total_teachers": {
                "total": total_teachers_count,
                "active": total_active_teachers,
                "by_class": teachers_by_class,
            },
            "adoption": {
                "teacher_pct": teacher_adoption_pct,
                "student_pct": student_adoption_pct,
                "teacher_by_class": teacher_adoption_by_class_pct,
                "student_by_class": student_adoption_by_class,
                "teacher_features": TEACHER_FEATURES,
                "student_features": STUDENT_FEATURES,
            },
            "practice_papers": active_practice_papers,
            "class_notes": active_class_notes,
        }

        # ===== CONTENT SUMMARY (all documents + videos + question papers) =====
        # Notes (Chapter Notes - ALL)
        content_notes = {"total": 0, "by_class": {}}
        # Practice Papers (Practice Sets - ALL)
        content_practice = {"total": 0, "by_class": {}}
        # Exam Tests (Test Series - ALL)
        content_exam_tests = {"total": 0, "by_class": {}}
        # Class Notes (Chapter Notes - ACTIVE only) - same as active_class_notes
        content_class_notes = {
            "total": active_class_notes["total"],
            "by_class": dict(active_class_notes["by_class"]),
        }

        for doc in admin_documents:
            standard = str(doc.get("standard", "Unknown"))
            doc_type = doc.get("document_type", "")
            if doc_type == "Chapter Notes":
                content_notes["total"] += 1
                content_notes["by_class"][standard] = (
                    content_notes["by_class"].get(standard, 0) + 1
                )
            elif doc_type == "Practice Sets":
                content_practice["total"] += 1
                content_practice["by_class"][standard] = (
                    content_practice["by_class"].get(standard, 0) + 1
                )
            elif doc_type == "Test Series":
                content_exam_tests["total"] += 1
                content_exam_tests["by_class"][standard] = (
                    content_exam_tests["by_class"].get(standard, 0) + 1
                )

        # Videos - query from videos collection
        content_videos = {"total": 0, "by_class": {}}
        try:
            admin_id_str = str(admin_id)
            tutor_user_ids = [
                str(t.get("user_id", "")) for t in admin_tutors if t.get("user_id")
            ]
            all_user_ids = [admin_id_str] + tutor_user_ids
            admin_videos = await db.mongo_find(
                "videos", {"uploaded_by": {"$in": all_user_ids}}
            )
            for video in admin_videos:
                standard = str(video.get("standard", "Unknown"))
                content_videos["total"] += 1
                content_videos["by_class"][standard] = (
                    content_videos["by_class"].get(standard, 0) + 1
                )
        except Exception:
            pass

        # Question Papers (Class Tests)
        content_class_tests = {"total": 0, "by_class": {}}
        try:
            admin_question_papers = await db.mongo_find(
                "question_papers", {"admin_id": str(admin_id)}
            )
            for qp in admin_question_papers:
                standard = str(qp.get("standard", "Unknown"))
                content_class_tests["total"] += 1
                content_class_tests["by_class"][standard] = (
                    content_class_tests["by_class"].get(standard, 0) + 1
                )
        except Exception:
            pass

        content_summary = {
            "notes": content_notes,
            "videos": content_videos,
            "practice_papers": content_practice,
            "exam_tests": content_exam_tests,
            "class_notes": content_class_notes,
            "class_tests": content_class_tests,
        }

        result = {
            "success": True,
            "class_division_stats": class_division_list,
            "teacher_summary": teacher_summary,
            "documents_by_class": docs_by_class_list,
            "totals": {
                "total_classes": len(class_division_list),
                "total_teachers": len(teacher_summary),
                "total_students": len(admin_students),
                "total_documents": len(admin_documents),
            },
            "school_statistics": school_statistics,
            "content_summary": content_summary,
        }

        # Cache for 5 minutes
        await cache.set(cache_key, result, 300, "admin")

        return result

    except Exception as e:
        logger.error(f"School dashboard stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get school dashboard statistics",
        )


@router.get("/dashboard/adoption-details")
@limiter.limit("30/minute")
async def get_adoption_details(
    request: Request,
    grade: str = Query(..., description="Grade/class to get adoption details for"),
    role: str = Query(..., description="'teachers' or 'students'"),
    current_user: Dict[str, Any] = Depends(
        require_admin_or_tutor_permission("view_analytics")
    ),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """
    Get per-person adoption feature usage for a specific grade.
    Returns each teacher/student with their feature completion status.
    """
    try:
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
        admin_id_str = str(admin_id)

        if role == "teachers":
            # Fetch tutors for this admin
            admin_tutors = await db.mongo_find("tutors", {"created_by": admin_id_str})
            # Filter to tutors who teach this grade (standards may be int or str)
            grade_tutors = [
                t
                for t in admin_tutors
                if grade in [str(s) for s in (t.get("standards", []) or [])]
            ]

            # Fetch documents (has admin_id field)
            admin_documents = await db.mongo_find("documents", {"admin_id": admin_id})
            try:
                admin_papers = await db.mongo_find(
                    "question_papers", {"admin_id": admin_id}
                )
            except Exception:
                admin_papers = []

            # For check_paper: student_test_attempts does NOT store admin_id,
            # so query by student IDs belonging to this admin
            all_students = await db.mongo_find("students", {"admin_id": admin_id})
            student_oids = [s["_id"] for s in all_students]
            student_id_strs = [str(s["_id"]) for s in all_students]
            try:
                admin_test_attempts = await db.mongo_find(
                    "student_test_attempts", {"student_id": {"$in": student_id_strs}}
                )
            except Exception:
                admin_test_attempts = []

            # Build feature lookup sets
            # 1. Upload content: tutor_id in document teacher_ids or uploaded_by
            teachers_with_docs = set()
            for doc in admin_documents:
                for tid in doc.get("teacher_ids", []) or []:
                    teachers_with_docs.add(str(tid))
                ub = doc.get("uploaded_by")
                if ub:
                    teachers_with_docs.add(str(ub))

            # 2. Generate paper: created_by in question_papers
            teachers_paper_creators = set()
            for paper in admin_papers:
                creator = paper.get("created_by")
                if creator:
                    teachers_paper_creators.add(str(creator))

            # 3. Check paper: graded_by/tutor_id in test attempts
            teachers_paper_checkers = set()
            for attempt in admin_test_attempts:
                graded_at = attempt.get("graded_at")
                grader = attempt.get("tutor_id") or attempt.get("graded_by")
                if graded_at and grader:
                    teachers_paper_checkers.add(str(grader))

            # 4. Share feedback: tutor has active (shared) documents
            teachers_with_active_docs = set()
            for doc in admin_documents:
                if doc.get("is_active", False):
                    for tid in doc.get("teacher_ids", []) or []:
                        teachers_with_active_docs.add(str(tid))
                    ub = doc.get("uploaded_by")
                    if ub:
                        teachers_with_active_docs.add(str(ub))

            features_list = [
                "upload_content",
                "generate_paper",
                "check_paper",
                "share_feedback",
            ]
            feature_labels = {
                "upload_content": "Upload Content",
                "generate_paper": "Generate Paper",
                "check_paper": "Check Paper",
                "share_feedback": "Share Feedback",
            }

            persons = []
            for tutor in grade_tutors:
                tutor_id = str(tutor.get("tutor_id", ""))
                user_id = str(tutor.get("_id", ""))
                display_name = (
                    tutor.get("full_name")
                    or tutor.get("name")
                    or tutor.get("username")
                    or "Unknown"
                )
                # Collect all possible IDs for this tutor
                all_ids = {tutor_id, user_id} - {""}

                features = {
                    "upload_content": bool(all_ids & teachers_with_docs),
                    "generate_paper": bool(all_ids & teachers_paper_creators),
                    "check_paper": bool(all_ids & teachers_paper_checkers),
                    "share_feedback": bool(all_ids & teachers_with_active_docs),
                }
                score = (
                    sum(1 for v in features.values() if v) / len(features_list) * 100
                )

                persons.append(
                    {
                        "id": tutor_id or user_id,
                        "name": display_name,
                        "email": tutor.get("email", ""),
                        "subjects": tutor.get("subjects", []),
                        "features": features,
                        "score": round(score, 1),
                    }
                )

            return {
                "success": True,
                "grade": grade,
                "role": role,
                "features": features_list,
                "feature_labels": feature_labels,
                "persons": sorted(persons, key=lambda x: x["score"], reverse=True),
            }

        elif role == "students":
            # Fetch all students and filter by grade (grade may be int or str in DB)
            all_students = await db.mongo_find("students", {"admin_id": admin_id})
            admin_students = [
                s for s in all_students if str(s.get("grade", "")) == grade
            ]

            # Build student ID lists for querying activity collections directly
            # (these collections may not have admin_id)
            student_oids = [s["_id"] for s in admin_students]
            student_id_strs = [str(s["_id"]) for s in admin_students]
            # Combine both formats for $in queries (ObjectId + string)
            student_ids_combined = student_oids + student_id_strs

            # Query activity collections by student IDs directly
            # Practice: check all 4 practice-related collections
            practice_student_ids = set()
            for coll_name in [
                "practice_attempts",
                "practice_sessions",
                "question_attempts",
                "mcq_attempts",
            ]:
                try:
                    docs = await db.mongo_find(
                        coll_name, {"student_id": {"$in": student_ids_combined}}
                    )
                    for doc in docs:
                        sid = doc.get("student_id")
                        if sid:
                            practice_student_ids.add(str(sid))
                except Exception:
                    pass

            try:
                test_attempts = await db.mongo_find(
                    "student_test_attempts",
                    {"student_id": {"$in": student_ids_combined}},
                )
            except Exception:
                test_attempts = []
            try:
                chat_sessions = await db.mongo_find(
                    "chat_sessions", {"student_id": {"$in": student_ids_combined}}
                )
            except Exception:
                chat_sessions = []
            try:
                note_views = await db.mongo_find(
                    "student_activity_log",
                    {
                        "student_id": {"$in": student_ids_combined},
                        "action": "chapter_viewed",
                    },
                )
            except Exception:
                note_views = []

            # Build sets of student IDs (as strings) who used each feature
            students_with_notes = set()
            for v in note_views:
                sid = v.get("student_id")
                if sid:
                    students_with_notes.add(str(sid))

            students_with_tests = set()
            for a in test_attempts:
                sid = a.get("student_id")
                if sid:
                    students_with_tests.add(str(sid))

            students_with_chat = set()
            for s in chat_sessions:
                sid = s.get("student_id")
                if sid:
                    students_with_chat.add(str(sid))

            features_list = ["notes", "practice", "test_attempt", "login", "chat"]
            feature_labels = {
                "notes": "Notes",
                "practice": "Practice",
                "test_attempt": "Tests",
                "login": "Login",
                "chat": "Chat",
            }

            persons = []
            for student in admin_students:
                sid_str = str(student["_id"])
                display_name = (
                    student.get("name") or student.get("username") or "Unknown"
                )
                section = student.get("section", "")

                features = {
                    "notes": sid_str in students_with_notes,
                    "practice": sid_str in practice_student_ids,
                    "test_attempt": sid_str in students_with_tests,
                    "login": student.get("last_login") is not None,
                    "chat": sid_str in students_with_chat,
                }
                score = (
                    sum(1 for v in features.values() if v) / len(features_list) * 100
                )

                persons.append(
                    {
                        "id": student.get("student_id") or sid_str,
                        "name": display_name,
                        "section": section,
                        "features": features,
                        "score": round(score, 1),
                    }
                )

            return {
                "success": True,
                "grade": grade,
                "role": role,
                "features": features_list,
                "feature_labels": feature_labels,
                "persons": sorted(persons, key=lambda x: x["score"], reverse=True),
            }

        else:
            raise HTTPException(
                status_code=400, detail="role must be 'teachers' or 'students'"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adoption details error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get adoption details",
        )


@router.get("/monitoring/class-section-stats")
@limiter.limit("30/minute")
async def get_class_section_monitoring_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(
        require_admin_or_tutor_permission("view_analytics")
    ),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """
    Get class-section level monitoring statistics:
    - Number of students active
    - Notes/Assignments/Tests uploaded per class-section
    - Average usage in minutes
    - Average % completion
    - Average Score

    Admins see all students under them.
    Tutors see only students assigned/mapped to them.
    """
    try:
        user_type = current_user.get("user_type")
        admin_id = None

        # Determine which students to include based on user type
        if user_type == "admin":
            admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
            cache_key = f"class_section_monitoring_admin_{admin_id}"
            cached_stats = await cache.get(cache_key, "admin")
            if cached_stats:
                return cached_stats
            admin_students = await db.mongo_find("students", {"admin_id": admin_id})
        else:
            # Tutor scoping: assigned + teacher_ids + class/section matching
            from utils.tutor_scoping import get_tutor_scoped_students

            tutor_id = current_user.get("tutor_id")
            admin_id_str = current_user.get("admin_id")
            try:
                admin_id = ObjectId(admin_id_str) if admin_id_str else None
            except Exception:
                admin_id = None

            cache_key = f"class_section_monitoring_tutor_{tutor_id}"
            cached_stats = await cache.get(cache_key, "admin")
            if cached_stats:
                return cached_stats

            admin_students = await get_tutor_scoped_students(
                tutor_id=tutor_id,
                admin_oid=admin_id,
                db=db,
                projection={"password_hash": 0},
            )

        # Get all documents for this admin (tutors see documents for their admin)
        admin_documents = []
        if admin_id:
            admin_documents = await db.mongo_find("documents", {"admin_id": admin_id})

        # Get all question attempts for scoped students
        student_ids = [s["_id"] for s in admin_students]
        all_attempts = (
            await db.mongo_find(
                "question_attempts", {"student_id": {"$in": student_ids}}
            )
            if student_ids
            else []
        )

        # Get all chat sessions for students of this admin
        all_sessions = (
            await db.mongo_find("chat_sessions", {"student_id": {"$in": student_ids}})
            if student_ids
            else []
        )

        # Build class-section level stats
        class_section_stats = {}

        for student in admin_students:
            # Handle None/null values explicitly (not just missing keys)
            grade = student.get("grade") or "Unknown"
            section = student.get("section") or "Unknown"
            key = f"{grade}_{section}"

            if key not in class_section_stats:
                class_section_stats[key] = {
                    "grade": grade,
                    "section": section,
                    "total_students": 0,
                    "active_students": 0,  # has logged in
                    "online_students": 0,  # currently online
                    "documents": {"notes": 0, "tests": 0, "practice": 0},
                    "total_time_minutes": 0,
                    "total_problems_attempted": 0,
                    "total_correct": 0,
                    "total_score_sum": 0,
                    "score_count": 0,
                    "students_with_activity": 0,
                }

            class_section_stats[key]["total_students"] += 1

            # Active = has logged in at least once
            if student.get("last_login") is not None:
                class_section_stats[key]["active_students"] += 1

            # Online = currently online
            if student.get("is_online", False):
                class_section_stats[key]["online_students"] += 1

            # Calculate student-specific metrics
            student_id = student["_id"]

            # Sessions and time
            student_sessions = [
                s for s in all_sessions if s.get("student_id") == student_id
            ]
            student_time = (
                sum(s.get("duration", 0) for s in student_sessions) / 60
            )  # convert to minutes
            class_section_stats[key]["total_time_minutes"] += student_time

            # Attempts and scores
            student_attempts = [
                a for a in all_attempts if a.get("student_id") == student_id
            ]
            if student_attempts:
                class_section_stats[key]["students_with_activity"] += 1
                class_section_stats[key]["total_problems_attempted"] += len(
                    student_attempts
                )
                class_section_stats[key]["total_correct"] += sum(
                    1 for a in student_attempts if a.get("is_correct", False)
                )

                scores = [a.get("score", 0) for a in student_attempts if "score" in a]
                if scores:
                    class_section_stats[key]["total_score_sum"] += sum(scores)
                    class_section_stats[key]["score_count"] += len(scores)

        # Add document counts per class-section
        for doc in admin_documents:
            standard = doc.get("standard") or "Unknown"
            section = doc.get("section") or "Unknown"
            doc_type = doc.get("document_type") or ""
            key = f"{standard}_{section}"

            if key in class_section_stats:
                if doc_type == "Chapter Notes":
                    class_section_stats[key]["documents"]["notes"] += 1
                elif doc_type == "Test Series":
                    class_section_stats[key]["documents"]["tests"] += 1
                elif doc_type == "Practice Sets":
                    class_section_stats[key]["documents"]["practice"] += 1

        # Convert to list and calculate averages
        result_list = []
        for key, stats in class_section_stats.items():
            total = stats["total_students"]
            with_activity = stats["students_with_activity"]
            score_count = stats["score_count"]

            avg_time = round(stats["total_time_minutes"] / total, 1) if total > 0 else 0
            avg_problems = (
                round(stats["total_problems_attempted"] / total, 1) if total > 0 else 0
            )

            # Completion % = (students with activity / total students) * 100
            completion_pct = round((with_activity / total) * 100, 1) if total > 0 else 0

            # Average score from all attempts
            avg_score = (
                round(stats["total_score_sum"] / score_count, 1)
                if score_count > 0
                else 0
            )

            # Accuracy = correct / attempted
            accuracy = (
                round(
                    (stats["total_correct"] / stats["total_problems_attempted"]) * 100,
                    1,
                )
                if stats["total_problems_attempted"] > 0
                else 0
            )

            result_list.append(
                {
                    "grade": stats["grade"],
                    "section": stats["section"],
                    "total_students": stats["total_students"],
                    "active_students": stats["active_students"],
                    "online_students": stats["online_students"],
                    "documents": stats["documents"],
                    "avg_usage_minutes": avg_time,
                    "avg_problems_per_student": avg_problems,
                    "avg_completion_pct": completion_pct,
                    "avg_score": avg_score,
                    "avg_accuracy": accuracy,
                }
            )

        # Sort by grade, then section
        result_list.sort(key=lambda x: (str(x["grade"]), x["section"]))

        result = {
            "success": True,
            "class_section_stats": result_list,
            "totals": {
                "total_classes": len(set(s["grade"] for s in result_list)),
                "total_sections": len(result_list),
                "total_students": sum(s["total_students"] for s in result_list),
                "total_active": sum(s["active_students"] for s in result_list),
                "total_documents": len(admin_documents),
            },
        }

        # Cache for 5 minutes
        await cache.set(cache_key, result, 300, "admin")

        return result

    except Exception as e:
        logger.error(f"Class section monitoring stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get class section monitoring statistics",
        )


@router.get("/monitoring/student-progress")
@limiter.limit("30/minute")
async def get_student_progress(
    request: Request,
    current_user: Dict[str, Any] = Depends(
        require_admin_or_tutor_permission("view_analytics")
    ),
    db: DatabaseManager = Depends(get_database),
):
    """Get student progress monitoring data"""
    try:
        user_type = current_user.get("user_type")
        students = []
        if user_type == "admin":
            admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
            students = await db.mongo_find(
                "students", {"admin_id": admin_id}, projection={"password_hash": 0}
            )
        else:
            # Tutor scoping: assigned + teacher_ids + class/section matching
            from utils.tutor_scoping import get_tutor_scoped_students

            tutor_id = current_user.get("tutor_id")
            admin_id_str = current_user.get("admin_id")
            admin_oid = None
            try:
                admin_oid = ObjectId(admin_id_str) if admin_id_str else None
            except Exception:
                admin_oid = None

            students = await get_tutor_scoped_students(
                tutor_id=tutor_id,
                admin_oid=admin_oid,
                db=db,
                projection={"password_hash": 0},
            )

        progress_data = []
        for student in students:
            student_oid = student["_id"]

            # Get student's chat sessions
            sessions = await db.mongo_find("chat_sessions", {"student_id": student_oid})

            # Get student's question attempts
            attempts = await db.mongo_find(
                "question_attempts", {"student_id": student_oid}
            )

            # Calculate total time spent (in minutes)
            total_time = (
                sum(session.get("duration", 0) for session in sessions) / 60
                if sessions
                else 0
            )

            # Calculate average score from attempts
            scores = [
                attempt.get("score", 0) for attempt in attempts if "score" in attempt
            ]
            avg_score = sum(scores) / len(scores) if scores else 0

            # Calculate problems solved (correct attempts)
            problems_solved = sum(
                1 for attempt in attempts if attempt.get("is_correct", False)
            )

            # Calculate streak days
            streak_days = await calculate_streak_days(student_oid, db)

            # Calculate level and XP
            level, xp = await calculate_student_level(student_oid, db)

            progress_data.append(
                {
                    "student_id": str(student_oid),
                    "student_name": student.get("full_name")
                    or student.get("name")
                    or "Unknown",
                    "email": student.get("email") or "",
                    "grade": student.get("grade") or "Unknown",
                    "section": student.get("section") or "Unknown",
                    "total_sessions": len(sessions),
                    "total_time_spent": int(total_time),
                    "problems_solved": problems_solved,
                    "average_score": round(avg_score, 1),
                    "last_active_at": student.get(
                        "last_login", student.get("updated_at")
                    ),
                    "streak_days": streak_days,
                    "level": level,
                    "xp": xp,
                    "is_online": student.get("is_online", False),
                }
            )

        return {"success": True, "data": progress_data}

    except Exception as e:
        logger.error(f"Get student progress error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get student progress",
        )


@router.get("/monitoring/recent-activities")
@limiter.limit("30/minute")
async def get_recent_activities(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(
        require_admin_or_tutor_permission("view_analytics")
    ),
    db: DatabaseManager = Depends(get_database),
):
    """Get recent student activities"""
    try:
        user_type = current_user.get("user_type")
        # Determine scoped student IDs
        scoped_student_ids = []
        if user_type == "admin":
            admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
            admin_students = await db.mongo_find(
                "students", {"admin_id": admin_id}, projection={"_id": 1}
            )
            scoped_student_ids = [s["_id"] for s in admin_students]
        else:
            # Tutor scoping: assigned + teacher_ids + class/section matching
            from utils.tutor_scoping import get_tutor_scoped_students

            tutor_id = current_user.get("tutor_id")
            admin_id_str = current_user.get("admin_id")
            admin_oid = None
            try:
                admin_oid = ObjectId(admin_id_str) if admin_id_str else None
            except Exception:
                admin_oid = None

            scoped_students = await get_tutor_scoped_students(
                tutor_id=tutor_id,
                admin_oid=admin_oid,
                db=db,
                projection={"_id": 1},
            )
            scoped_student_ids = [s["_id"] for s in scoped_students]

        # Get recent activities for scoped students
        recent_activity_logs = await db.mongo_find(
            "student_activity_log",
            {"student_id": {"$in": scoped_student_ids}},
            sort=[("timestamp", -1)],
            limit=limit * 2,
        )

        # Get recent chat sessions for scoped students
        recent_sessions = await db.mongo_find(
            "chat_sessions",
            {"student_id": {"$in": scoped_student_ids}},
            sort=[("created_at", -1)],
            limit=limit,
        )

        # Get recent question attempts for scoped students
        recent_attempts = await db.mongo_find(
            "question_attempts",
            {"student_id": {"$in": scoped_student_ids}},
            sort=[("created_at", -1)],
            limit=limit,
        )

        activities = []

        # Process activity logs (login, logout, chat_message)
        for log in recent_activity_logs:
            student_id = log.get("student_id")
            student = await db.mongo_find_one("students", {"_id": student_id})

            activity = {
                "id": str(log.get("_id")),
                "student_id": str(student_id) if student_id else None,
                "student_name": student.get("full_name", student.get("name", "Unknown"))
                if student
                else "Unknown",
                "action": log.get("action"),
                "timestamp": log.get("timestamp"),
            }

            # Add metadata based on action type
            metadata = log.get("metadata", {})
            if log.get("action") == "login":
                activity["metadata"] = {
                    "ip_address": metadata.get("ip_address"),
                    "user_agent": metadata.get("user_agent"),
                }
            elif log.get("action") == "session_end":
                activity["session_duration"] = metadata.get("session_duration")
            elif log.get("action") == "chat_message":
                activity["metadata"] = {
                    "session_id": metadata.get("session_id"),
                    "mode": metadata.get("mode"),
                }

            activities.append(activity)

        # Process chat sessions
        for session in recent_sessions:
            student_id = session.get("student_id")
            student = await db.mongo_find_one("students", {"_id": student_id})
            activities.append(
                {
                    "id": str(session.get("_id")),
                    "student_id": str(student_id) if student_id else None,
                    "student_name": student.get(
                        "full_name", student.get("name", "Unknown")
                    )
                    if student
                    else "Unknown",
                    "action": "problem_solving",
                    "timestamp": session.get("created_at"),
                    "session_duration": session.get("duration"),
                    "problem_details": {"session_id": str(session.get("_id"))},
                }
            )

        # Process question attempts
        for attempt in recent_attempts:
            student_id = attempt.get("student_id")
            student = await db.mongo_find_one("students", {"_id": student_id})
            activities.append(
                {
                    "id": str(attempt.get("_id")),
                    "student_id": str(student_id) if student_id else None,
                    "student_name": student.get(
                        "full_name", student.get("name", "Unknown")
                    )
                    if student
                    else "Unknown",
                    "action": "question_attempted",
                    "timestamp": attempt.get("created_at"),
                    "score": attempt.get("score"),
                    "problem_details": {
                        "correct": attempt.get("is_correct", False),
                        "question_id": str(attempt.get("question_id")),
                    },
                }
            )

        # Sort by timestamp descending
        activities.sort(key=lambda x: x["timestamp"] or datetime.min, reverse=True)

        return {"success": True, "data": activities[:limit]}

    except Exception as e:
        logger.error(f"Get recent activities error: {str(e)}")
        return {"success": True, "data": []}


@router.get("/database/status")
@limiter.limit("30/minute")
async def get_database_status(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Get database connection status"""
    try:
        # Determine MongoDB status without forcing a ping (to avoid SSL timeouts)
        mongo_connected = (
            getattr(db, "mongo_client", None) is not None and not DISABLE_MONGODB
        )
        mongo_status = (
            "online"
            if mongo_connected
            else ("disabled" if DISABLE_MONGODB or not MONGODB_URL else "offline")
        )

        # Get questions count from MongoDB
        questions_count = 0
        if mongo_connected:
            try:
                questions_count = await db.mongo_count("questions")
            except Exception as e:
                logger.warning(f"Failed to count questions: {str(e)}")

        return {
            "success": True,
            "status": {
                "mongodb": {
                    "connected": mongo_connected,
                    "status": mongo_status,
                    "questions_count": questions_count,
                }
            },
        }

    except Exception as e:
        logger.error(f"Get database status error: {str(e)}")
        return {
            "success": False,
            "status": {
                "mongodb": {
                    "connected": False,
                    "status": "offline",
                    "questions_count": 0,
                }
            },
        }


@router.post("/validate-test-series/{document_id}")
@limiter.limit("30/minute")
async def validate_test_series(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(
        require_admin_permission("manage_questions")
    ),
    db: DatabaseManager = Depends(get_database),
):
    """
    Validate test series before making it available to students:
    1. Total points matches sum of question points
    2. Total minutes > 0
    3. All questions have correct_answer set
    """
    try:
        # Get document
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.get("document_type") != "Test Series":
            raise HTTPException(
                status_code=400, detail="Only Test Series can be validated"
            )

        # Get questions
        questions = await db.mongo_find("questions", {"document_id": document_id})

        if not questions or len(questions) == 0:
            raise HTTPException(
                status_code=400,
                detail="No questions found for this test series. Please process OCR first.",
            )

        # Validation 1: Total points (auto-calculated from questions, default 4 marks each)
        total_allocated = sum(
            q.get("points", 4) for q in questions
        )  # Default 4 marks per question
        total_points = document.get("total_points", 0)
        if abs(total_allocated - total_points) >= 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Total points ({total_points}) doesn't match allocated points ({total_allocated})",
            )

        # Validation 2: Total minutes
        total_minutes = document.get("total_minutes", 0)
        if not total_minutes or total_minutes <= 0:
            raise HTTPException(
                status_code=400, detail="Total minutes must be greater than 0"
            )

        # Validation 3: All questions have correct_answer
        questions_without_answer = [
            q
            for q in questions
            if not q.get("correct_answer") or q.get("correct_answer", "").strip() == ""
        ]
        if questions_without_answer:
            raise HTTPException(
                status_code=400,
                detail=f"{len(questions_without_answer)} question(s) don't have correct answer set",
            )

        # Mark as validated
        await db.mongo_update_one(
            "documents",
            {"document_id": document_id},
            {"$set": {"is_validated": True, "validated_at": datetime.utcnow()}},
        )

        logger.info(f"Test series {document_id} validated successfully")

        return {
            "success": True,
            "message": "Test series validated successfully",
            "data": {
                "total_questions": len(questions),
                "total_points": total_points,
                "total_minutes": total_minutes,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate test series: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-attempts")
@limiter.limit("30/minute")
async def get_all_test_attempts(
    request: Request,
    current_user: Dict[str, Any] = Depends(
        require_admin_or_tutor_permission("view_analytics")
    ),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get all test attempts across all students for admin leaderboard
    """
    try:
        # Scope attempts to either admin's students or tutor's students
        from typing import List as _List

        user_type = current_user.get("user_type")
        student_ids: _List[str] = []

        if user_type == "admin":
            admin_oid = ObjectId(current_user.get("admin_id", current_user["user_id"]))
            admin_students = await db.mongo_find(
                "students", {"admin_id": admin_oid}, projection={"_id": 1}
            )
            student_ids = [str(s["_id"]) for s in admin_students]
        else:
            # Tutor scoping: assigned + teacher_ids + class/section matching
            from utils.tutor_scoping import get_tutor_scoped_students

            tutor_id = current_user.get("tutor_id")
            admin_id_str = current_user.get("admin_id")
            admin_oid = None
            try:
                admin_oid = ObjectId(admin_id_str) if admin_id_str else None
            except Exception:
                admin_oid = None

            scoped_students = await get_tutor_scoped_students(
                tutor_id=tutor_id,
                admin_oid=admin_oid,
                db=db,
                projection={"_id": 1},
            )
            student_ids = [str(s["_id"]) for s in scoped_students]

        attempts = await db.mongo_find(
            "student_test_attempts",
            {"student_id": {"$in": student_ids}},
            sort=[("submitted_at", -1)],
        )

        # Format response
        formatted_attempts = []
        for attempt in attempts:
            formatted_attempts.append(
                {
                    "attempt_id": str(attempt.get("_id")),
                    "student_id": attempt.get("student_id"),
                    "student_name": attempt.get("student_name"),
                    "student_grade": attempt.get("student_grade"),
                    "document_id": attempt.get("document_id"),
                    "document_title": attempt.get("document_title"),
                    "subject": attempt.get("subject"),
                    "score": attempt.get("score", 0),
                    "total_points": attempt.get("total_points", 0),
                    "percentage": attempt.get("percentage", 0),
                    "total_questions": attempt.get("total_questions", 0),
                    "correct_count": attempt.get("correct_count", 0),
                    "incorrect_count": attempt.get("incorrect_count", 0),
                    "unanswered_count": attempt.get("unanswered_count", 0),
                    "time_taken": attempt.get("time_taken", 0),
                    "total_minutes": attempt.get("total_minutes", 0),
                    "can_reattempt": attempt.get("can_reattempt", False),
                    "submitted_at": attempt.get("submitted_at").isoformat()
                    if attempt.get("submitted_at")
                    else None,
                }
            )

        logger.info(f"Retrieved {len(formatted_attempts)} test attempts")

        return {
            "success": True,
            "data": {"attempts": formatted_attempts, "total": len(formatted_attempts)},
        }

    except Exception as e:
        logger.error(f"Failed to get test attempts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-attempts/{attempt_id}/toggle-reattempt")
@limiter.limit("30/minute")
async def toggle_reattempt(
    request: Request,
    attempt_id: str,
    current_user: Dict[str, Any] = Depends(
        require_admin_permission("manage_questions")
    ),
    db: DatabaseManager = Depends(get_database),
):
    """
    Toggle the can_reattempt flag for a test attempt
    Allows/disallows student to re-attempt the test
    """
    try:
        from bson import ObjectId

        # Get the attempt
        attempt = await db.mongo_find_one(
            "student_test_attempts", {"_id": ObjectId(attempt_id)}
        )
        if not attempt:
            raise HTTPException(status_code=404, detail="Test attempt not found")

        # Toggle the flag
        new_value = not attempt.get("can_reattempt", False)

        # Update in database
        await db.mongo_update_one(
            "student_test_attempts",
            {"_id": ObjectId(attempt_id)},
            {"$set": {"can_reattempt": new_value}},
        )

        logger.info(f"Toggled re-attempt for attempt {attempt_id} to {new_value}")

        return {
            "success": True,
            "message": f"Re-attempt {'enabled' if new_value else 'disabled'} successfully",
            "data": {"can_reattempt": new_value},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle re-attempt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/promote", response_model=SessionPromotionResponse)
@limiter.limit("5/minute")
async def promote_session(
    request: Request,
    promotion_data: SessionPromotionRequest,
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """
    Promote all students to the next academic session.

    This endpoint:
    1. Updates all students' grades based on grade_mappings
    2. Optionally updates sections for specific students
    3. Optionally deactivates old content (notes, assignments, tests)
    4. Stores the session history for reference

    Use preview_only=True to see changes without actually applying them.
    """
    try:
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Get all active students for this admin
        students = await db.mongo_find(
            "students", {"admin_id": admin_id, "is_active": True}
        )

        if not students:
            return SessionPromotionResponse(
                success=True,
                message="No active students found to promote",
                new_session=promotion_data.new_session,
                students_promoted=0,
                students_skipped=0,
                content_deactivated=0,
                details=[],
            )

        promoted_count = 0
        skipped_count = 0
        content_deactivated = 0
        details = []

        for student in students:
            student_id = str(student.get("_id"))
            current_grade = student.get("grade", "")
            current_section = student.get("section", "")
            student_name = (
                student.get("full_name")
                or student.get("name")
                or student.get("username", "Unknown")
            )
            business_id = student.get("student_id", student_id)

            # Apply filters
            # 1. If specific student_ids are provided, only those students are considered
            if promotion_data.student_ids:
                if (
                    business_id not in promotion_data.student_ids
                    and student_id not in promotion_data.student_ids
                ):
                    # This student is not in the selected list, skip without adding to details
                    continue
            else:
                # 2. Apply grade filter if specified
                if (
                    promotion_data.grade_filter
                    and current_grade not in promotion_data.grade_filter
                ):
                    continue

                # 3. Apply section filter if specified
                if (
                    promotion_data.section_filter
                    and current_section not in promotion_data.section_filter
                ):
                    continue

            # Check if grade is in the mapping
            if current_grade in promotion_data.grade_mappings:
                new_grade = promotion_data.grade_mappings[current_grade]

                # Check for section update
                new_section = current_section
                if (
                    promotion_data.section_updates
                    and business_id in promotion_data.section_updates
                ):
                    new_section = promotion_data.section_updates[business_id]

                detail = {
                    "student_id": business_id,
                    "student_name": student_name,
                    "old_grade": current_grade,
                    "new_grade": new_grade,
                    "old_section": current_section,
                    "new_section": new_section,
                    "status": "promoted",
                }

                # Only update if not preview mode
                if not promotion_data.preview_only:
                    update_fields = {
                        "grade": new_grade,
                        "section": new_section,
                        "last_session": f"{current_grade}-{current_section}",
                        "current_session": promotion_data.new_session,
                        "promoted_at": datetime.utcnow(),
                        "promoted_by": str(admin_id),
                    }

                    await db.mongo_update_one(
                        "students", {"_id": student["_id"]}, {"$set": update_fields}
                    )

                details.append(detail)
                promoted_count += 1
            else:
                # Grade not in mapping - skip
                details.append(
                    {
                        "student_id": business_id,
                        "student_name": student_name,
                        "old_grade": current_grade,
                        "new_grade": current_grade,
                        "old_section": current_section,
                        "new_section": current_section,
                        "status": "skipped",
                        "reason": "Grade not in mapping",
                    }
                )
                skipped_count += 1

        # Deactivate old content if requested
        if promotion_data.deactivate_old_content and not promotion_data.preview_only:
            # Deactivate documents (notes, tests, practice sets) for the old session
            result = await db.mongo_update_many(
                "documents",
                {
                    "admin_id": admin_id,
                    "is_active": True,
                    "session": {"$ne": promotion_data.new_session},
                },
                {
                    "$set": {
                        "is_active": False,
                        "deactivated_at": datetime.utcnow(),
                        "deactivated_reason": f"Session promotion to {promotion_data.new_session}",
                    }
                },
            )
            content_deactivated = result if isinstance(result, int) else 0

        # Store session promotion history if not preview
        if not promotion_data.preview_only:
            await db.mongo_insert_one(
                "session_promotions",
                {
                    "admin_id": admin_id,
                    "new_session": promotion_data.new_session,
                    "grade_mappings": promotion_data.grade_mappings,
                    "students_promoted": promoted_count,
                    "students_skipped": skipped_count,
                    "content_deactivated": content_deactivated,
                    "promoted_at": datetime.utcnow(),
                    "promoted_by": str(admin_id),
                },
            )

            # Invalidate caches
            try:
                await cache.clear_pattern("students:*", "admin")
                await cache.delete("dashboard_stats", "admin")
            except Exception:
                pass

        message = (
            f"Preview: {promoted_count} students would be promoted"
            if promotion_data.preview_only
            else f"Successfully promoted {promoted_count} students to session {promotion_data.new_session}"
        )

        logger.info(
            f"Session promotion {'preview' if promotion_data.preview_only else 'completed'} for admin {admin_id}: {promoted_count} promoted, {skipped_count} skipped"
        )

        return SessionPromotionResponse(
            success=True,
            message=message,
            new_session=promotion_data.new_session,
            students_promoted=promoted_count,
            students_skipped=skipped_count,
            content_deactivated=content_deactivated,
            details=details,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session promotion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote session: {str(e)}",
        )


@router.get("/session/promotions")
@limiter.limit("30/minute")
async def get_session_promotions(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
):
    """Get history of session promotions for this admin"""
    try:
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        promotions = await db.mongo_find(
            "session_promotions",
            {"admin_id": admin_id},
            sort=[("promoted_at", -1)],
            limit=20,
        )

        # Format for response
        formatted_promotions = []
        for promo in promotions:
            formatted_promotions.append(
                {
                    "id": str(promo.get("_id")),
                    "new_session": promo.get("new_session"),
                    "grade_mappings": promo.get("grade_mappings"),
                    "students_promoted": promo.get("students_promoted", 0),
                    "students_skipped": promo.get("students_skipped", 0),
                    "content_deactivated": promo.get("content_deactivated", 0),
                    "promoted_at": promo.get("promoted_at").isoformat()
                    if promo.get("promoted_at")
                    else None,
                }
            )

        return {"success": True, "promotions": formatted_promotions}

    except Exception as e:
        logger.error(f"Get session promotions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ SUPER ADMIN MESSAGES ============


@router.get("/superadmin-messages")
@limiter.limit("30/minute")
async def get_superadmin_messages(
    request: Request,
    unread_only: bool = Query(False, description="Only return unread messages"),
    current_user: Dict[str, Any] = Depends(require_admin_permission("view_admin")),
    db: DatabaseManager = Depends(get_database),
):
    """Get all messages for this tenant (both directions: SA→admin and admin→SA)"""
    try:
        # Get tenant_id from current user's context
        tenant_id = current_user.get("tenant_id")
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant context not found")

        # Query the master database for messages to this tenant
        master_db = await db.get_master_db()

        # Find tenant by institution_id
        tenant = await master_db["tenants"].find_one(
            {"$or": [{"institution_id": tenant_id}, {"tenant_id": tenant_id}]}
        )

        if not tenant:
            return {"success": True, "messages": [], "unread_count": 0}

        # Build query — bidirectional: return all messages for this tenant
        query: Dict[str, Any] = {"tenant_id": tenant["_id"]}
        if unread_only:
            query["read"] = False

        # Fetch messages
        messages_cursor = (
            master_db["superadmin_messages"].find(query).sort("created_at", -1)
        )
        messages = await messages_cursor.to_list(length=100)

        # Count unread — only incoming (superadmin_to_admin) messages count as unread
        unread_count = await master_db["superadmin_messages"].count_documents(
            {
                "tenant_id": tenant["_id"],
                "$or": [
                    {"direction": {"$exists": False}},
                    {"direction": "superadmin_to_admin"},
                ],
                "read": False,
            }
        )

        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(
                {
                    "id": str(msg["_id"]),
                    "from_name": msg.get("from_name", "Super Admin"),
                    "from_email": msg.get("from_admin"),
                    "subject": msg.get("subject"),
                    "message": msg.get("message"),
                    "priority": msg.get("priority", "normal"),
                    "direction": msg.get("direction", "superadmin_to_admin"),
                    "created_at": msg.get("created_at").isoformat()
                    if msg.get("created_at")
                    else None,
                    "read": msg.get("read", False),
                    "read_at": msg.get("read_at").isoformat()
                    if msg.get("read_at")
                    else None,
                    "attachments": [
                        {
                            "filename": att.get("filename"),
                            "content_type": att.get("content_type"),
                            "size_bytes": att.get("size_bytes"),
                        }
                        for att in (msg.get("attachments") or [])
                        if isinstance(att, dict)
                    ],
                }
            )

        return {
            "success": True,
            "messages": formatted_messages,
            "unread_count": unread_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get superadmin messages error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/superadmin-messages/{message_id}/read")
@limiter.limit("30/minute")
async def mark_superadmin_message_read(
    message_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_permission("view_admin")),
    db: DatabaseManager = Depends(get_database),
):
    """Mark a super admin message as read"""
    try:
        tenant_id = current_user.get("tenant_id")
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant context not found")

        master_db = await db.get_master_db()

        # Find tenant
        tenant = await master_db["tenants"].find_one(
            {"$or": [{"institution_id": tenant_id}, {"tenant_id": tenant_id}]}
        )

        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")

        # Update message as read (only if it belongs to this tenant)
        result = await master_db["superadmin_messages"].update_one(
            {"_id": ObjectId(message_id), "tenant_id": tenant["_id"]},
            {"$set": {"read": True, "read_at": datetime.utcnow()}},
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Message not found")

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mark message read error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ ADMIN → SUPER-ADMIN MESSAGING ============


@router.post("/superadmin-messages")
@limiter.limit("10/minute")
async def send_message_to_superadmin(
    request: Request,
    subject: str = Form(..., min_length=1, max_length=200),
    message: str = Form(..., min_length=1),
    priority: str = Form("normal", pattern=r"^(low|normal|high|urgent)$"),
    attachments: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(require_admin_permission("view_admin")),
    db: DatabaseManager = Depends(get_database),
):
    """Send a message from tenant admin to their assigned super-admin."""
    from utils.message_attachments import upload_message_attachments

    try:
        tenant_id = current_user.get("tenant_id") or current_user.get("institution_id")
        if not tenant_id:
            raise HTTPException(status_code=400, detail="No tenant context")

        master_db = await db.get_master_db()
        if master_db is None:
            raise HTTPException(status_code=503, detail="Master database unavailable")

        tenant_doc = await master_db["tenants"].find_one(
            {"$or": [{"tenant_id": tenant_id}, {"institution_id": tenant_id}]}
        )
        if not tenant_doc:
            raise HTTPException(status_code=404, detail="Tenant not found")

        superadmin_id = tenant_doc.get("assigned_superadmin_id")
        if not superadmin_id:
            raise HTTPException(
                status_code=400, detail="No super-admin assigned to this tenant"
            )

        # Upload attachments to S3
        attachment_metadata = await upload_message_attachments(
            [f for f in attachments if f.filename]
        )

        message_doc = {
            "tenant_id": tenant_doc["_id"],
            "superadmin_id": ObjectId(str(superadmin_id)),
            "from_admin": current_user.get("email", ""),
            "from_name": current_user.get("full_name", current_user.get("name", "")),
            "to_email": "",
            "subject": subject,
            "message": message,
            "priority": priority or "normal",
            "direction": "admin_to_superadmin",
            "attachments": attachment_metadata,
            "created_at": datetime.utcnow(),
            "read": False,
            "read_at": None,
        }

        result = await master_db["superadmin_messages"].insert_one(message_doc)
        return {"success": True, "message_id": str(result.inserted_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send message to superadmin error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send message")


@router.get("/superadmin-messages/{message_id}/attachments/{attachment_index}")
@limiter.limit("30/minute")
async def download_message_attachment(
    message_id: str,
    attachment_index: int,
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_permission("view_admin")),
    db: DatabaseManager = Depends(get_database),
):
    """Download an attachment from a superadmin message."""
    import base64
    from utils.s3_storage import download_file as s3_download_file

    try:
        tenant_id = current_user.get("tenant_id")
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Tenant context not found")

        master_db = await db.get_master_db()

        tenant = await master_db["tenants"].find_one(
            {"$or": [{"institution_id": tenant_id}, {"tenant_id": tenant_id}]}
        )
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")

        msg = await master_db["superadmin_messages"].find_one(
            {
                "_id": ObjectId(message_id),
                "tenant_id": tenant["_id"],
            }
        )
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")

        attachments = msg.get("attachments") or []
        if attachment_index < 0 or attachment_index >= len(attachments):
            raise HTTPException(status_code=404, detail="Attachment not found")

        att = attachments[attachment_index]
        filename = att.get("filename", "attachment")
        content_type = att.get("content_type", "application/octet-stream")

        # S3-stored attachment (new format)
        if att.get("storage_path"):
            file_data = await s3_download_file(att["storage_path"])
            if file_data is None:
                raise HTTPException(
                    status_code=404, detail="Attachment file not found in storage"
                )
            return RawResponse(
                content=file_data,
                media_type=content_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        # Legacy base64-encoded attachment
        if att.get("data_base64"):
            file_data = base64.b64decode(att["data_base64"])
            return RawResponse(
                content=file_data,
                media_type=content_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        raise HTTPException(status_code=404, detail="Attachment has no file data")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download message attachment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download attachment")
