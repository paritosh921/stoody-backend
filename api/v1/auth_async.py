"""
Async Authentication API for SkillBot
JWT-based authentication with rate limiting and caching
"""

import base64
import logging
import mimetypes
import os
import re
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from bson import ObjectId

import jwt as pyjwt

from fastapi import APIRouter, Request, HTTPException, Depends, status, Form, UploadFile, File, Query
from fastapi.responses import JSONResponse, Response as RawResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from core.auth import AuthManager
from core.pen_tokens import create_pen_token
from core.tenant_registry import (
    get_tenant_by_subdomain,
    get_tenant_by_tenant_id,
    normalize_tenant_id,
)
from core.tenant_features import (
    build_enabled_features_v2,
    is_feature_enabled,
    merge_tenant_features,
)
from core.security import (
    get_security_logger,
    get_password_validator,
)
from core.password_reset_otp import PasswordResetOtpManager
from core.transactional_email import send_password_reset_otp_email
from core.cookie_auth import get_current_user_dual_auth, cookie_auth_manager
from core.observability import record_auth_login
from config_async import settings, JWT_SECRET_KEY, JWT_ALGORITHM

# Security logger for audit trail
security_logger = get_security_logger()

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
TENANT_ID_PATTERN = r'^[A-Za-z]{4}-?[0-9]{4}$'
SUPERADMIN_AUTH_CODE_PATTERN = r'^[A-Z0-9]{6}$'
ALLOWED_SUPPORTING_FILE_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/jpg"}
MAX_REGISTRATION_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB per file
MAX_REGISTRATION_FILES = 10
MAX_STATUS_REPLY_FILES = 5

class AdminLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    tenant_id: Optional[str] = Field(None, pattern=TENANT_ID_PATTERN)

class StudentLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    tenant_id: str = Field(..., pattern=TENANT_ID_PATTERN)

class TutorLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    tenant_id: str = Field(..., pattern=TENANT_ID_PATTERN)

class StudentChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=6)
    new_password: str = Field(..., min_length=8)

class StudentForgotPasswordRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    date_of_birth: str  # Format: YYYY-MM-DD
    phone: str
    tenant_id: str = Field(..., pattern=TENANT_ID_PATTERN)

class TokenResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any]

class UserResponse(BaseModel):
    user_id: str
    user_type: str
    email: Optional[str] = None
    username: Optional[str] = None
    full_name: Optional[str] = None

class AdminRegistrationResponse(BaseModel):
    success: bool
    message: str
    status: str

class AdminRegistrationStatusAuthRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)

class TenantLookupResponse(BaseModel):
    tenant_id: str
    institution_name: Optional[str] = None
    status: Optional[str] = None

class StudentPasswordResetOtpRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    tenant_id: str = Field(..., pattern=TENANT_ID_PATTERN)


class RolePasswordResetOtpRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    tenant_id: str = Field(..., pattern=TENANT_ID_PATTERN)


class StudentPasswordResetOtpCompleteRequest(StudentPasswordResetOtpRequest):
    otp: str = Field(..., min_length=6, max_length=8)
    new_password: str = Field(..., min_length=8, max_length=72)


class RolePasswordResetOtpCompleteRequest(RolePasswordResetOtpRequest):
    otp: str = Field(..., min_length=6, max_length=8)
    new_password: str = Field(..., min_length=8, max_length=72)

# Dependency injection
async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db

async def get_cache(request: Request) -> CacheManager:
    return request.app.state.cache

async def get_auth_manager(request: Request) -> AuthManager:
    return request.app.state.auth

def _get_request_subdomain(request: Request) -> Optional[str]:
    return getattr(request.state, "subdomain", None)

async def _resolve_tenant_for_auth(
    db: DatabaseManager,
    request: Request,
    tenant_id: Optional[str],
    require_active: bool = True,
) -> Dict[str, Any]:
    subdomain = _get_request_subdomain(request)
    tenant = None
    if subdomain:
        tenant = await get_tenant_by_subdomain(db, subdomain, include_inactive=not require_active)
    if not tenant and tenant_id:
        tenant = await get_tenant_by_tenant_id(
            db,
            tenant_id,
            include_inactive=not require_active
        )
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant not found for this request"
        )
    if require_active and tenant.get("status") != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant is not active"
        )
    if tenant.get("platform_suspended"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Platform access suspended. Contact your service provider."
        )
    return tenant

async def _get_tenant_db_or_503(db: DatabaseManager, tenant: Dict[str, Any]):
    db_name = tenant.get("db_name")
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available"
        )
    return tenant_db

async def _get_tenant_db_from_user(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Optional[Any]:
    db_name = current_user.get("db_name")
    if not db_name:
        return None
    return await db.get_tenant_db(db_name)


def _password_reset_otp_manager() -> PasswordResetOtpManager:
    return PasswordResetOtpManager(
        length=settings.PASSWORD_RESET_OTP_LENGTH,
        expire_minutes=settings.PASSWORD_RESET_OTP_EXPIRE_MINUTES,
        max_attempts=settings.PASSWORD_RESET_OTP_MAX_ATTEMPTS,
    )


def _generic_otp_request_response() -> Dict[str, Any]:
    return {
        "success": True,
        "message": "If the account information matches, a password reset code has been sent to the registered email.",
    }


def _raise_no_records_found() -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No records found")


async def _find_reset_identity_record(collection, *, username: str, email: str):
    username_clean = username.strip()
    username_lower = username_clean.lower()
    email_lower = email.lower().strip()
    user = await collection.find_one({
        "email": email_lower,
        "username_lower": username_lower,
        "is_active": True,
    })
    if user:
        return user
    return await collection.find_one(
        {
            "email": email_lower,
            "username": username_clean,
            "is_active": True,
        },
        collation={"locale": "en", "strength": 2},
    )


async def _insert_password_reset_otp(
    tenant_db,
    *,
    user: Dict[str, Any],
    role: str,
    tenant_id: str,
    username: str,
) -> None:
    user_email = (user.get("email") or "").strip().lower()
    if not user_email:
        return
    if not await _can_issue_password_reset_otp(
        tenant_db,
        user_id=str(user["_id"]),
        role=role,
        tenant_id=tenant_id,
    ):
        return
    manager = _password_reset_otp_manager()
    created = manager.create_otp_record(
        user_id=str(user["_id"]),
        email=user_email,
        role=role,
        tenant_id=tenant_id,
    )
    await tenant_db["password_reset_otps"].insert_one(created["record"])
    await send_password_reset_otp_email(
        to_email=user_email,
        otp=created["otp"],
        username=username,
        role=role,
        expire_minutes=settings.PASSWORD_RESET_OTP_EXPIRE_MINUTES,
    )


async def _find_latest_password_reset_otp(tenant_db, *, user_id: str, role: str, tenant_id: str):
    return await tenant_db["password_reset_otps"].find_one(
        {
            "user_id": user_id,
            "role": role,
            "tenant_id": tenant_id,
            "used": False,
        },
        sort=[("created_at", -1)],
    )


async def _can_issue_password_reset_otp(tenant_db, *, user_id: str, role: str, tenant_id: str) -> bool:
    collection = tenant_db["password_reset_otps"]
    now = datetime.utcnow()
    base_query = {
        "user_id": user_id,
        "role": role,
        "tenant_id": tenant_id,
    }
    latest = await collection.find_one(base_query, sort=[("created_at", -1)])
    created_at = latest.get("created_at") if latest else None
    cooldown_seconds = max(0, int(settings.PASSWORD_RESET_OTP_RESEND_COOLDOWN_SECONDS))
    if created_at and cooldown_seconds and now - created_at < timedelta(seconds=cooldown_seconds):
        return False

    max_per_hour = max(0, int(settings.PASSWORD_RESET_OTP_MAX_REQUESTS_PER_HOUR))
    if max_per_hour:
        recent_count = await collection.count_documents({
            **base_query,
            "created_at": {"$gte": now - timedelta(hours=1)},
        })
        if recent_count >= max_per_hour:
            return False

    return True


async def _mark_otp_attempt(tenant_db, record: Dict[str, Any], *, used: bool = False) -> None:
    update: Dict[str, Any] = {"$inc": {"attempts": 1}}
    if used:
        update["$set"] = {"used": True, "used_at": datetime.utcnow()}
    await tenant_db["password_reset_otps"].update_one({"_id": record["_id"]}, update)


def _validate_new_password(new_password: str) -> None:
    password_validator = get_password_validator()
    is_valid, errors = password_validator.validate(new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Password does not meet requirements", "errors": errors},
        )


async def _get_enabled_features_for_tenant(
    db: DatabaseManager,
    tenant: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
) -> Dict[str, bool]:
    if tenant and (
        isinstance(tenant.get("enabled_features"), dict)
        or isinstance(tenant.get("enabled_features_v2"), dict)
    ):
        return merge_tenant_features(
            tenant.get("enabled_features"),
            tenant.get("enabled_features_v2"),
        )

    tenant_lookup_id = (tenant_id or (tenant or {}).get("tenant_id") or "").strip().upper()
    if not tenant_lookup_id:
        return merge_tenant_features(None)

    tenants = await db.get_master_collection("tenants")
    if tenants is None:
        return merge_tenant_features(None)

    tenant_doc = await tenants.find_one(
        {"$or": [{"tenant_id": tenant_lookup_id}, {"institution_id": tenant_lookup_id}]},
        {"enabled_features": 1, "enabled_features_v2": 1},
    )
    return merge_tenant_features(
        (tenant_doc or {}).get("enabled_features"),
        (tenant_doc or {}).get("enabled_features_v2"),
    )

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        
        # CRITICAL: Check if token is revoked (for portal auto-logout)
        from core.token_blacklist import token_blacklist
        if token_blacklist.is_revoked(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked. Please log in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_data = await auth_manager.verify_token_and_get_user(token)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check user-level revocation in Redis (cross-client logout)
        from core.token_blacklist import is_user_session_revoked
        uid = user_data.get("user_id")
        if uid and await is_user_session_revoked(
            auth_manager.cache_manager, uid
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session has been revoked. Please log in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate tenant claims against request context (if available)
        subdomain = _get_request_subdomain(request)
        token_subdomain = user_data.get("subdomain")
        if subdomain and token_subdomain != subdomain:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant mismatch for this session",
                headers={"WWW-Authenticate": "Bearer"},
            )

        tenant_ctx = getattr(request.state, "tenant", None)
        if tenant_ctx:
            if user_data.get("tenant_id") != tenant_ctx.get("tenant_id"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Tenant mismatch for this session",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return user_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/student/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def student_login(
    request: Request,
    login_data: StudentLoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Student login endpoint with tenant-scoped usernames

    - Username is unique per tenant (tenant_id required)
    - Subdomain is optional for non-web clients
    """
    login_recorded = False
    try:
        tenant_id = normalize_tenant_id(login_data.tenant_id)
        tenant = await _resolve_tenant_for_auth(db, request, tenant_id)
        tenant_db = await _get_tenant_db_or_503(db, tenant)
        enabled_features = merge_tenant_features(
            tenant.get("enabled_features"),
            tenant.get("enabled_features_v2"),
        )
        enabled_features_v2 = build_enabled_features_v2(
            tenant.get("enabled_features_v2"),
            tenant.get("enabled_features"),
        )

        normalized_username = login_data.username.strip()
        username_lower = normalized_username.lower()

        student = await tenant_db["students"].find_one({
            "username_lower": username_lower,
            "is_active": True
        })
        if not student:
            student = await tenant_db["students"].find_one(
                {"username": normalized_username, "is_active": True},
                collation={"locale": "en", "strength": 2}
            )

        if not student:
            logger.warning(f"Student {normalized_username} not found")
            record_auth_login("student", False)
            login_recorded = True
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        if not auth_manager.verify_password(login_data.password, student.get("password_hash", "")):
            logger.warning(f"Invalid password for student {normalized_username}")
            record_auth_login("student", False)
            login_recorded = True
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        admin_id = student.get("admin_id")
        subdomain = tenant.get("subdomain")

        student_data = {
            "user_id": str(student["_id"]),
            "user_type": "student",
            "username": student.get("username"),
            "email": student.get("email"),
            "full_name": student.get("full_name", student.get("name")),
            "admin_id": str(admin_id) if admin_id else None,
            "subdomain": subdomain,
            "tenant_id": tenant.get("tenant_id"),
            "db_name": tenant.get("db_name"),
            "institution_id": tenant.get("institution_id"),
            "enabled_features": enabled_features,
            "enabled_features_v2": enabled_features_v2,
        }

        session_data = await auth_manager.create_user_session(student_data)
        record_auth_login("student", True)
        login_recorded = True

        # Clear any user-level revocation so the new token is accepted
        from core.token_blacklist import clear_user_session_revocation
        await clear_user_session_revocation(auth_manager.cache_manager, str(student["_id"]))

        pen_token = None
        pen_token_expires_at = None
        try:
            pen_token, pen_token_expires_at = create_pen_token(
                str(student["_id"]),
                tenant.get("tenant_id"),
                tenant.get("db_name"),
                tenant.get("institution_id")
            )
            await tenant_db["pen_tokens"].insert_one({
                "token": pen_token,
                "student_id": student["_id"],
                "pen_mac": None,
                "tenant_id": tenant.get("tenant_id"),
                "issued_at": datetime.utcnow(),
                "expires_at": pen_token_expires_at,
                "active": True
            })
        except Exception as e:
            logger.warning(f"Failed to create pen token: {str(e)}")

        try:
            await tenant_db["students"].update_one(
                {"_id": student["_id"]},
                {"$set": {"is_online": True, "last_login": datetime.utcnow()}}
            )

            await tenant_db["student_activity_log"].insert_one({
                "student_id": student["_id"],
                "admin_id": admin_id,
                "action": "login",
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "subdomain": subdomain,
                    "ip_address": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown")
                }
            })
        except Exception as e:
            logger.warning(f"Failed to track student login: {str(e)}")

        response_data = {
            "success": True,
            "data": {
                "access_token": session_data["access_token"],
                "user_type": "student",
                "user": session_data["user"],
                "requires_password_change": student.get("requires_password_change", False),
                "pen_token": pen_token,
                "pen_token_expires_at": pen_token_expires_at.isoformat() if pen_token_expires_at else None
            }
        }

        response = JSONResponse(content=response_data)
        cookie_auth_manager.set_auth_cookie(response, session_data["access_token"])
        return response

    except HTTPException:
        if not login_recorded:
            record_auth_login("student", False)
        raise
    except Exception as e:
        if not login_recorded:
            record_auth_login("student", False)
        logger.error(f"Student login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/student/change-password")
async def student_change_password(
    request: Request,
    password_data: StudentChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Student changes their password (must be logged in)"""
    try:
        # Ensure user is a student
        if current_user.get("user_type") != "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only students can use this endpoint"
            )

        student_id = ObjectId(current_user["user_id"])
        tenant_db = await _get_tenant_db_from_user(db, current_user)
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context required"
            )

        student = await tenant_db["students"].find_one({"_id": student_id})
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        # Verify current password
        if not auth_manager.verify_password(
            password_data.current_password,
            student.get("password_hash", "")
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )

        # Validate new password strength
        password_validator = get_password_validator()
        is_valid, errors = password_validator.validate(password_data.new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password too weak: {'; '.join(errors)}"
            )

        # Hash new password
        new_password_hash = auth_manager.get_password_hash(password_data.new_password)

        # Update password and clear requires_password_change flag
        await tenant_db["students"].update_one(
            {"_id": student_id},
            {"$set": {
                "password_hash": new_password_hash,
                "requires_password_change": False,
                "password_changed_at": datetime.utcnow()
            }}
        )

        # Invalidate all sessions for security (forces re-login)
        user_id_str = str(student_id)
        await auth_manager.invalidate_user_session(user_id_str)

        logger.info(f"Student {student.get('username')} changed their password")

        return {
            "success": True,
            "message": "Password changed successfully. Please log in again."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.post("/tutor/change-password")
async def tutor_change_password(
    request: Request,
    password_data: StudentChangePasswordRequest,  # Reuse same schema (current_password, new_password)
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Tutor changes their password (must be logged in)"""
    try:
        # Ensure user is a tutor
        if current_user.get("user_type") != "tutor":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only tutors can use this endpoint"
            )

        tutor_id = ObjectId(current_user["user_id"])
        tenant_db = await _get_tenant_db_from_user(db, current_user)
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context required"
            )

        tutor = await tenant_db["tutors"].find_one({"_id": tutor_id})
        if not tutor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tutor not found"
            )

        # Verify current password
        if not auth_manager.verify_password(
            password_data.current_password,
            tutor.get("password_hash", "")
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )

        # Validate new password strength
        password_validator = get_password_validator()
        is_valid, errors = password_validator.validate(password_data.new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password too weak: {'; '.join(errors)}"
            )

        # Hash new password
        new_password_hash = auth_manager.get_password_hash(password_data.new_password)

        # Update password and clear requires_password_change flag
        await tenant_db["tutors"].update_one(
            {"_id": tutor_id},
            {"$set": {
                "password_hash": new_password_hash,
                "requires_password_change": False,
                "password_changed_at": datetime.utcnow()
            }}
        )

        # Invalidate all sessions for security (forces re-login)
        user_id_str = str(tutor_id)
        await auth_manager.invalidate_user_session(user_id_str)

        logger.info(f"Tutor {tutor.get('username')} changed their password")

        return {
            "success": True,
            "message": "Password changed successfully. Please log in again."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tutor change password error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.post("/admin/change-password")
async def admin_change_password(
    request: Request,
    password_data: StudentChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Admin changes their password after first login or a super-admin reset."""
    try:
        if current_user.get("user_type") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can use this endpoint"
            )

        admin_id = ObjectId(current_user["user_id"])
        tenant_db = await _get_tenant_db_from_user(db, current_user)
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context required"
            )

        admin = await tenant_db["admins"].find_one({"_id": admin_id})
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )

        if not auth_manager.verify_password(
            password_data.current_password,
            admin.get("password_hash", "")
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )

        password_validator = get_password_validator()
        is_valid, errors = password_validator.validate(password_data.new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password too weak: {'; '.join(errors)}"
            )

        new_password_hash = auth_manager.get_password_hash(password_data.new_password)

        await tenant_db["admins"].update_one(
            {"_id": admin_id},
            {"$set": {
                "password_hash": new_password_hash,
                "requires_password_change": False,
                "password_changed_at": datetime.utcnow()
            }}
        )

        await auth_manager.invalidate_user_session(str(admin_id))

        logger.info("Admin %s changed their password", admin.get("email"))

        return {
            "success": True,
            "message": "Password changed successfully. Please log in again."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin change password error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.post("/student/password-reset/request")
@limiter.limit("3/hour")
async def request_student_password_reset_otp(
    request: Request,
    reset_data: StudentPasswordResetOtpRequest,
    db: DatabaseManager = Depends(get_database),
):
    """Request a student password reset OTP. Searches students only."""
    tenant_id = normalize_tenant_id(reset_data.tenant_id)
    tenant = await _resolve_tenant_for_auth(db, request, tenant_id, require_active=True)
    tenant_db = await _get_tenant_db_or_503(db, tenant)
    email_lower = reset_data.email.lower().strip()
    student = await _find_reset_identity_record(
        tenant_db["students"],
        username=reset_data.username,
        email=email_lower,
    )
    if not student:
        _raise_no_records_found()
    await _insert_password_reset_otp(
        tenant_db,
        user=student,
        role="student",
        tenant_id=tenant_id,
        username=student.get("full_name") or student.get("username") or email_lower,
    )

    return _generic_otp_request_response()


@router.post("/tutor/password-reset/request")
@limiter.limit("3/hour")
async def request_tutor_password_reset_otp(
    request: Request,
    reset_data: RolePasswordResetOtpRequest,
    db: DatabaseManager = Depends(get_database),
):
    """Request a tutor password reset OTP. Searches tutors only."""
    tenant_id = normalize_tenant_id(reset_data.tenant_id)
    tenant = await _resolve_tenant_for_auth(db, request, tenant_id, require_active=True)
    tenant_db = await _get_tenant_db_or_503(db, tenant)
    email_lower = reset_data.email.lower().strip()
    tutor = await _find_reset_identity_record(
        tenant_db["tutors"],
        username=reset_data.username,
        email=email_lower,
    )
    if not tutor:
        _raise_no_records_found()
    await _insert_password_reset_otp(
        tenant_db,
        user=tutor,
        role="tutor",
        tenant_id=tenant_id,
        username=tutor.get("full_name") or tutor.get("username") or email_lower,
    )
    return _generic_otp_request_response()


@router.post("/admin/password-reset/request")
@limiter.limit("3/hour")
async def request_admin_password_reset_otp(
    request: Request,
    reset_data: RolePasswordResetOtpRequest,
    db: DatabaseManager = Depends(get_database),
):
    """Request an admin password reset OTP. Searches admins only."""
    tenant_id = normalize_tenant_id(reset_data.tenant_id)
    tenant = await _resolve_tenant_for_auth(db, request, tenant_id, require_active=True)
    tenant_db = await _get_tenant_db_or_503(db, tenant)
    email_lower = reset_data.email.lower().strip()
    admin = await _find_reset_identity_record(
        tenant_db["admins"],
        username=reset_data.username,
        email=email_lower,
    )
    if not admin:
        _raise_no_records_found()
    await _insert_password_reset_otp(
        tenant_db,
        user=admin,
        role="admin",
        tenant_id=tenant_id,
        username=admin.get("full_name") or admin.get("username") or email_lower,
    )
    return _generic_otp_request_response()


@router.post("/student/password-reset/complete")
@limiter.limit("5/minute")
async def complete_student_password_reset_otp(
    request: Request,
    reset_data: StudentPasswordResetOtpCompleteRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """Complete a student password reset with OTP and a new password."""
    _validate_new_password(reset_data.new_password)
    tenant_id = normalize_tenant_id(reset_data.tenant_id)
    tenant = await _resolve_tenant_for_auth(db, request, tenant_id, require_active=True)
    tenant_db = await _get_tenant_db_or_503(db, tenant)
    email_lower = reset_data.email.lower().strip()
    student = await _find_reset_identity_record(
        tenant_db["students"],
        username=reset_data.username,
        email=email_lower,
    )
    if not student:
        _raise_no_records_found()

    record = await _find_latest_password_reset_otp(
        tenant_db,
        user_id=str(student["_id"]),
        role="student",
        tenant_id=tenant_id,
    )
    manager = _password_reset_otp_manager()
    valid, reason = manager.validate_record(record, reset_data.otp)
    if not valid:
        if record and reason == "invalid_otp":
            await _mark_otp_attempt(tenant_db, record)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset code")

    password_hash = auth_manager.get_password_hash(reset_data.new_password)
    await tenant_db["students"].update_one(
        {"_id": student["_id"]},
        {"$set": {
            "password_hash": password_hash,
            "password_changed_at": datetime.utcnow(),
            "password_reset_requested": False,
            "requires_password_change": False,
        }},
    )
    await _mark_otp_attempt(tenant_db, record, used=True)
    return {"success": True, "message": "Password reset successfully"}


async def _complete_role_password_reset_otp(
    *,
    request: Request,
    reset_data: RolePasswordResetOtpCompleteRequest,
    db: DatabaseManager,
    auth_manager: AuthManager,
    role: str,
):
    _validate_new_password(reset_data.new_password)
    tenant_id = normalize_tenant_id(reset_data.tenant_id)
    tenant = await _resolve_tenant_for_auth(db, request, tenant_id, require_active=True)
    tenant_db = await _get_tenant_db_or_503(db, tenant)
    collection_name = "tutors" if role == "tutor" else "admins"
    email_lower = reset_data.email.lower().strip()
    user = await _find_reset_identity_record(
        tenant_db[collection_name],
        username=reset_data.username,
        email=email_lower,
    )
    if not user:
        _raise_no_records_found()

    record = await _find_latest_password_reset_otp(
        tenant_db,
        user_id=str(user["_id"]),
        role=role,
        tenant_id=tenant_id,
    )
    manager = _password_reset_otp_manager()
    valid, reason = manager.validate_record(record, reset_data.otp)
    if not valid:
        if record and reason == "invalid_otp":
            await _mark_otp_attempt(tenant_db, record)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset code")

    password_hash = auth_manager.get_password_hash(reset_data.new_password)
    await tenant_db[collection_name].update_one(
        {"_id": user["_id"]},
        {"$set": {
            "password_hash": password_hash,
            "password_changed_at": datetime.utcnow(),
            "password_reset_requested": False,
            "requires_password_change": False,
        }},
    )
    await _mark_otp_attempt(tenant_db, record, used=True)
    return {"success": True, "message": "Password reset successfully"}


@router.post("/tutor/password-reset/complete")
@limiter.limit("5/minute")
async def complete_tutor_password_reset_otp(
    request: Request,
    reset_data: RolePasswordResetOtpCompleteRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    return await _complete_role_password_reset_otp(
        request=request,
        reset_data=reset_data,
        db=db,
        auth_manager=auth_manager,
        role="tutor",
    )


@router.post("/admin/password-reset/complete")
@limiter.limit("5/minute")
async def complete_admin_password_reset_otp(
    request: Request,
    reset_data: RolePasswordResetOtpCompleteRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    return await _complete_role_password_reset_otp(
        request=request,
        reset_data=reset_data,
        db=db,
        auth_manager=auth_manager,
        role="admin",
    )


@router.post("/student/forgot-password")
@limiter.limit("3/hour")
async def student_forgot_password(
    request: Request,
    forgot_data: StudentForgotPasswordRequest,
    db: DatabaseManager = Depends(get_database)
):
    """Student requests password reset using tenant-scoped username"""
    try:
        tenant_id = normalize_tenant_id(forgot_data.tenant_id)
        tenant = await _resolve_tenant_for_auth(db, request, tenant_id)
        tenant_db = await _get_tenant_db_or_503(db, tenant)

        normalized_username = forgot_data.username.strip()
        username_lower = normalized_username.lower()
        student = await tenant_db["students"].find_one({
            "username_lower": username_lower
        })
        if not student:
            student = await tenant_db["students"].find_one(
                {"username": normalized_username},
                collation={"locale": "en", "strength": 2}
            )

        if not student:
            # Don't reveal if user exists
            return {
                "success": True,
                "message": "If your information matches, a password reset request has been sent to your administrator"
            }

        # Verify DOB and phone
        if (student.get("date_of_birth") != forgot_data.date_of_birth or
            student.get("phone") != forgot_data.phone):
            # Don't reveal which field is wrong
            return {
                "success": True,
                "message": "If your information matches, a password reset request has been sent to your administrator"
            }

        # Set password reset request flag
        await tenant_db["students"].update_one(
            {"_id": student["_id"]},
            {"$set": {
                "password_reset_requested": True,
                "password_reset_requested_at": datetime.utcnow()
            }}
        )

        logger.info(f"Password reset requested for student: {normalized_username}")

        return {
            "success": True,
            "message": "Password reset request has been sent to your administrator"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forgot password error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process request"
        )


@router.get("/verify")
async def verify_token(
    request: Request,
    auth_manager: AuthManager = Depends(get_auth_manager),
    db: DatabaseManager = Depends(get_database),
):
    """
    Verify JWT token and return user data
    Supports both cookie-based and header-based authentication for backward compatibility
    """
    # Try dual authentication (cookie first, then header)
    try:
        current_user = await get_current_user_dual_auth(request, auth_manager)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Verify token error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    enabled_features = await _get_enabled_features_for_tenant(
        db,
        tenant_id=current_user.get("tenant_id"),
    )
    enabled_features_v2 = build_enabled_features_v2(
        current_user.get("enabled_features_v2"),
        current_user.get("enabled_features") or enabled_features,
    )

    return {
        "success": True,
        "data": {
            "user_id": current_user.get("user_id"),
            "user_type": current_user.get("user_type"),
            "email": current_user.get("email"),
            "username": current_user.get("username"),
            "full_name": current_user.get("full_name"),
            "requires_password_change": current_user.get("requires_password_change", False),
            "enabled_features": enabled_features,
            "enabled_features_v2": enabled_features_v2,
        }
    }

@router.post("/admin/register", response_model=AdminRegistrationResponse)
@limiter.limit("3/hour")
async def register_admin(
    request: Request,
    full_name: str = Form(..., min_length=2, max_length=100),
    email: EmailStr = Form(...),
    password: str = Form(..., min_length=6),
    organization: str = Form(..., min_length=2, max_length=100),
    institution_name: Optional[str] = Form(None, max_length=150),
    contact_email: Optional[EmailStr] = Form(None),
    phone_country_code: Optional[str] = Form(None, max_length=10),
    phone_number: Optional[str] = Form(None, max_length=20),
    attachments: List[UploadFile] = File(default=[]),
    attachment_types: List[str] = Form(default=[]),
    subdomain: Optional[str] = Form(None, min_length=3, max_length=50),
    requested_tenant_id: Optional[str] = Form(None, max_length=9),
    authorization_code: str = Form(..., min_length=6, max_length=6),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Register a new admin
    Creates a pending tenant request for super-admin review
    """
    try:
        normalized_auth_code = authorization_code.strip().upper()
        if not re.match(SUPERADMIN_AUTH_CODE_PATTERN, normalized_auth_code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Authorization code must be 6 uppercase alphanumeric characters"
            )

        # Validate requested_tenant_id format if provided
        if requested_tenant_id:
            requested_tenant_id = requested_tenant_id.strip()
            if not re.match(r'^[A-Z]{4}-[0-9]{4}$', requested_tenant_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Requested tenant ID must match format XXXX-0000"
                )

        tenants = await db.get_master_collection("tenants")
        super_admins = await db.get_master_collection("super_admins")
        if tenants is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tenant registry not available"
            )
        if super_admins is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Super admin registry not available"
            )

        assigned_super_admin = await super_admins.find_one({
            "authorization_code": normalized_auth_code,
            "is_active": True,
        })
        if not assigned_super_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid authorization code"
            )

        # Check if email already exists in tenant registry
        existing_email = await tenants.find_one({"admin_email": email})
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        normalized_subdomain = subdomain.lower().strip() if subdomain else None
        if normalized_subdomain:
            # Check if subdomain already exists
            existing_subdomain = await tenants.find_one({"subdomain": normalized_subdomain})
            if existing_subdomain:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Subdomain already taken. Please choose another."
                )

            # Validate subdomain format again
            if not re.match(r'^[a-z0-9\-]+$', normalized_subdomain):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Subdomain can only contain lowercase letters, numbers, and hyphens"
                )

            # Check reserved subdomains
            reserved = ['www', 'app', 'admin', 'api', 'demo', 'test', 'staging', 'dev', 'mail', 'ftp']
            if normalized_subdomain in reserved:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This subdomain is reserved"
                )

        # Hash password
        password_hash = auth_manager.get_password_hash(password)

        institution_name_value = institution_name or organization
        contact_email_value = contact_email or email
        contact_phone_value = None
        if phone_country_code and phone_number:
            contact_phone_value = f"{phone_country_code.strip()} {phone_number.strip()}"

        files_collection = None
        prepared_file_docs: List[Dict[str, Any]] = []
        if attachments:
            files_collection = await db.get_master_collection("tenant_application_files")
            if files_collection is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Tenant files collection not available"
                )

            uploads = [upload for upload in attachments if upload and upload.filename]
            if len(uploads) > MAX_REGISTRATION_FILES:
                raise HTTPException(status_code=400, detail=f"Maximum {MAX_REGISTRATION_FILES} files allowed")

            if uploads and len(attachment_types) != len(uploads):
                raise HTTPException(
                    status_code=400,
                    detail="Please provide a document type for every uploaded file"
                )

            for index, upload in enumerate(uploads):
                if not upload.filename:
                    continue

                content_type = (upload.content_type or "").lower().strip()
                if content_type == "application/octet-stream" or not content_type:
                    guessed_type, _ = mimetypes.guess_type(upload.filename)
                    if guessed_type:
                        content_type = guessed_type.lower()

                if content_type not in ALLOWED_SUPPORTING_FILE_TYPES:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.content_type or upload.filename}")

                content = await upload.read()
                if len(content) > MAX_REGISTRATION_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=400, detail=f"File too large: {upload.filename}")

                document_type = attachment_types[index].strip() if index < len(attachment_types) else ""
                if not document_type:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Document type missing for file: {upload.filename}"
                    )

                prepared_file_docs.append({
                    "admin_email": email,
                    "institution_name": institution_name_value,
                    "document_type": document_type[:80],
                    "filename": upload.filename,
                    "content_type": content_type,
                    "size_bytes": len(content),
                    "data_base64": base64.b64encode(content).decode("utf-8"),
                })

        tenant_doc = {
            "tenant_id": None,
            "requested_tenant_id": requested_tenant_id or None,
            "db_name": None,
            "institution_id": None,
            "subdomain": normalized_subdomain,
            "organization": organization,
            "institution_name": institution_name_value,
            "contact_email": contact_email_value,
            "phone_country_code": phone_country_code.strip() if phone_country_code else None,
            "phone_number": phone_number.strip() if phone_number else None,
            "contact_phone": contact_phone_value,
            "status": "pending",
            "admin_email": email,
            "admin_full_name": full_name,
            "assigned_superadmin_id": assigned_super_admin["_id"],
            "authorization_code_used": normalized_auth_code,
            "pending_admin": {
                "email": email,
                "password_hash": password_hash,
                "full_name": full_name,
                "role": "master_admin",
                "created_at": datetime.utcnow(),
                "two_fa": {
                    "enabled": False,
                    "required": True,
                    "secret_enc": None,
                    "verified_at": None
                }
            },
            "created_at": datetime.utcnow()
        }

        result = await tenants.insert_one(tenant_doc)
        if not result.inserted_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create tenant request"
            )

        if prepared_file_docs and files_collection is not None:
            timestamp = datetime.utcnow()
            file_docs = [
                {
                    **file_doc,
                    "tenant_request_id": result.inserted_id,
                    "uploaded_at": timestamp,
                }
                for file_doc in prepared_file_docs
            ]
            insert_result = await files_collection.insert_many(file_docs)
            if insert_result.inserted_ids:
                file_ids = list(insert_result.inserted_ids)
                await tenants.update_one(
                    {"_id": result.inserted_id},
                    {"$set": {"application_files": file_ids}}
                )

        logger.info("New tenant request: %s", email)

        return AdminRegistrationResponse(
            success=True,
            message="Registration submitted. Awaiting tenant ID assignment.",
            status="pending"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

async def _authenticate_registration_status_admin(
    db: DatabaseManager,
    auth_manager: AuthManager,
    email: str,
    password: str,
) -> Dict[str, Any]:
    tenants = await db.get_master_collection("tenants")
    if tenants is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

    email_regex = {"$regex": f"^{re.escape(email.strip())}$", "$options": "i"}
    candidate_tenants = await tenants.find(
        {"admin_email": email_regex}
    ).sort("created_at", -1).to_list(length=20)

    if not candidate_tenants:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    matched_tenants: List[Dict[str, Any]] = []
    for tenant in candidate_tenants:
        password_hash = (tenant.get("pending_admin") or {}).get("password_hash")
        if not password_hash:
            tenant_db = None
            if tenant.get("db_name"):
                tenant_db = await db.get_tenant_db(tenant.get("db_name"))
            if tenant_db is not None:
                admin_doc = await tenant_db["admins"].find_one({
                    "email": email_regex,
                    "is_active": True,
                })
                if admin_doc:
                    password_hash = admin_doc.get("password_hash")

        if password_hash and auth_manager.verify_password(password, password_hash):
            matched_tenants.append(tenant)

    if not matched_tenants:
        # No matching tenant/password pair found
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if len(matched_tenants) == 1:
        return matched_tenants[0]

    # Duplicate-email safety: choose the most likely tenant based on message anchors.
    email_lower = email.strip().lower()
    master_db = await db.get_master_db()
    if master_db is not None:
        scored: List[tuple[int, datetime, Dict[str, Any]]] = []
        for tenant in matched_tenants:
            tenant_oid = tenant.get("_id")
            if not tenant_oid:
                continue

            created_at = tenant.get("created_at") or datetime.min
            anchor_queries: List[Dict[str, Any]] = [{"tenant_id": tenant_oid}]

            auth_code = (tenant.get("authorization_code_used") or "").strip().upper()
            if auth_code:
                query: Dict[str, Any] = {
                    "authorization_code": auth_code,
                    "to_email": email_lower,
                }
                if tenant.get("created_at"):
                    query["created_at"] = {"$gte": tenant.get("created_at")}
                anchor_queries.append(query)

            assigned_superadmin_id = tenant.get("assigned_superadmin_id")
            if assigned_superadmin_id:
                superadmin_oid = assigned_superadmin_id if isinstance(assigned_superadmin_id, ObjectId) else None
                if superadmin_oid is None:
                    try:
                        superadmin_oid = ObjectId(str(assigned_superadmin_id))
                    except Exception:
                        superadmin_oid = None
                if superadmin_oid is not None:
                    query = {
                        "superadmin_id": superadmin_oid,
                        "to_email": email_lower,
                    }
                    if tenant.get("created_at"):
                        query["created_at"] = {"$gte": tenant.get("created_at")}
                    anchor_queries.append(query)

            score = await master_db["superadmin_messages"].count_documents({"$or": anchor_queries})
            scored.append((score, created_at, tenant))

        if scored:
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return scored[0][2]

    # Final fallback: most recently created matching tenant.
    return matched_tenants[0]


def _serialize_registration_status_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    attachments = msg.get("attachments") or []
    serialized_attachments: List[Dict[str, Any]] = []
    for file_doc in attachments:
        if not isinstance(file_doc, dict):
            continue
        att_entry: Dict[str, Any] = {
            "filename": file_doc.get("filename"),
            "content_type": file_doc.get("content_type"),
            "size_bytes": file_doc.get("size_bytes"),
        }
        # Include data_base64 for legacy inline attachments
        if file_doc.get("data_base64"):
            att_entry["data_base64"] = file_doc["data_base64"]
        # Flag S3-stored attachments (don't expose the actual storage path)
        if file_doc.get("storage_path"):
            att_entry["stored_on_server"] = True
        serialized_attachments.append(att_entry)

    return {
        "_id": str(msg.get("_id")),
        "subject": msg.get("subject"),
        "message": msg.get("message"),
        "priority": msg.get("priority", "normal"),
        "direction": msg.get("direction", "superadmin_to_admin"),
        "from_name": msg.get("from_name"),
        "from_admin": msg.get("from_admin"),
        "created_at": msg.get("created_at").isoformat() if msg.get("created_at") else None,
        "attachments": serialized_attachments,
        "read": bool(msg.get("read", False)),
    }


async def _get_registration_thread_messages(
    db: DatabaseManager,
    tenant: Dict[str, Any],
) -> List[Dict[str, Any]]:
    master_db = await db.get_master_db()
    if master_db is None:
        return []

    tenant_oid = tenant.get("_id")
    if not tenant_oid:
        return []

    admin_email = (tenant.get("admin_email") or "").strip()
    auth_code = (tenant.get("authorization_code_used") or "").strip().upper()
    assigned_superadmin_id = tenant.get("assigned_superadmin_id")
    superadmin_oid = None
    if assigned_superadmin_id:
        if isinstance(assigned_superadmin_id, ObjectId):
            superadmin_oid = assigned_superadmin_id
        else:
            try:
                superadmin_oid = ObjectId(str(assigned_superadmin_id))
            except Exception:
                superadmin_oid = None

    created_at_filter = {"$gte": tenant.get("created_at")} if tenant.get("created_at") else None
    email_filter_value = {"$regex": f"^{re.escape(admin_email)}$", "$options": "i"} if admin_email else None

    async def fetch_messages(query: Dict[str, Any]) -> List[Dict[str, Any]]:
        return await master_db["superadmin_messages"].find(query).sort("created_at", -1).to_list(length=100)

    # Priority 1: strict tenant_id match (most deterministic)
    tenant_id_query: Dict[str, Any] = {"tenant_id": tenant_oid}
    messages = await fetch_messages(tenant_id_query)
    if messages:
        return [_serialize_registration_status_message(msg) for msg in messages]

    # Legacy safety where tenant_id may have been stored as string
    try:
        legacy_tenant_id_query: Dict[str, Any] = {"tenant_id": str(tenant_oid)}
        messages = await fetch_messages(legacy_tenant_id_query)
        if messages:
            return [_serialize_registration_status_message(msg) for msg in messages]
    except Exception:
        pass

    # Priority 2: authorization-code anchored match
    if auth_code and email_filter_value:
        code_query: Dict[str, Any] = {
            "authorization_code": auth_code,
            "to_email": email_filter_value,
        }
        if created_at_filter:
            code_query["created_at"] = created_at_filter
        if superadmin_oid is not None:
            code_query["superadmin_id"] = superadmin_oid
        messages = await fetch_messages(code_query)
        if messages:
            return [_serialize_registration_status_message(msg) for msg in messages]

    # Priority 3: owner + email fallback
    if superadmin_oid is not None and email_filter_value:
        owner_query: Dict[str, Any] = {
            "superadmin_id": superadmin_oid,
            "to_email": email_filter_value,
        }
        if created_at_filter:
            owner_query["created_at"] = created_at_filter
        messages = await fetch_messages(owner_query)
        if messages:
            return [_serialize_registration_status_message(msg) for msg in messages]

    # Priority 4: email-only fallback (least strict, kept for backward compatibility)
    if email_filter_value:
        email_query: Dict[str, Any] = {"to_email": email_filter_value}
        if created_at_filter:
            email_query["created_at"] = created_at_filter
        messages = await fetch_messages(email_query)
        if messages:
            return [_serialize_registration_status_message(msg) for msg in messages]

    return []


def _build_registration_status_payload(tenant: Dict[str, Any], messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "status": tenant.get("status", "pending"),
        "institution_name": tenant.get("institution_name") or tenant.get("organization"),
        "created_at": tenant.get("created_at").isoformat() if tenant.get("created_at") else None,
    }

    if tenant.get("status") == "rejected":
        response["rejection_reason"] = tenant.get("rejection_reason")

    if tenant.get("status") in ["approved", "active", "verification", "suspended"]:
        response["approved_at"] = tenant.get("approved_at").isoformat() if tenant.get("approved_at") else None
        response["tenant_id"] = tenant.get("tenant_id")

    if messages is not None:
        response["messages"] = messages

    return response


@router.post("/admin/registration-status-auth")
@limiter.limit("10/minute")
async def get_registration_status_authenticated(
    request: Request,
    status_data: AdminRegistrationStatusAuthRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Check registration status using admin credentials.
    This endpoint is intended for institution admins before activation.
    """
    try:
        tenant = await _authenticate_registration_status_admin(
            db=db,
            auth_manager=auth_manager,
            email=status_data.email,
            password=status_data.password,
        )

        if not is_feature_enabled(
            tenant.get("enabled_features"),
            "admin_registration_status_tools",
            tenant.get("enabled_features_v2"),
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Registration status tools are not enabled for this institution",
            )

        thread_messages = await _get_registration_thread_messages(db, tenant)
        return _build_registration_status_payload(tenant, thread_messages)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authenticated registration status check error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check status"
        )


@router.post("/admin/registration-status-message")
@limiter.limit("10/minute")
async def send_registration_status_message(
    request: Request,
    email: EmailStr = Form(...),
    password: str = Form(..., min_length=6),
    subject: str = Form(..., min_length=1, max_length=200),
    message: str = Form(..., min_length=1),
    attachments: List[UploadFile] = File(default=[]),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Send a message from institution admin to assigned super admin
    while the institution is in registration flow.
    """
    try:
        tenant = await _authenticate_registration_status_admin(
            db=db,
            auth_manager=auth_manager,
            email=email,
            password=password,
        )

        if not is_feature_enabled(
            tenant.get("enabled_features"),
            "admin_registration_status_tools",
            tenant.get("enabled_features_v2"),
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Registration status tools are not enabled for this institution",
            )

        clean_subject = subject.strip()
        clean_message = message.strip()
        if not clean_subject or not clean_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Subject and message are required"
            )

        master_db = await db.get_master_db()
        if master_db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable"
            )

        superadmin_oid = tenant.get("assigned_superadmin_id")
        if not superadmin_oid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No super admin is assigned to this registration yet"
            )

        if not isinstance(superadmin_oid, ObjectId):
            try:
                superadmin_oid = ObjectId(str(superadmin_oid))
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid super admin assignment"
                )

        super_admin = await master_db["super_admins"].find_one({"_id": superadmin_oid})

        uploads = [upload for upload in attachments if upload and upload.filename]
        if len(uploads) > MAX_STATUS_REPLY_FILES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {MAX_STATUS_REPLY_FILES} attachments are allowed"
            )

        serialized_attachments: List[Dict[str, Any]] = []
        for upload in uploads:
            content_type = (upload.content_type or "").lower().strip()
            if content_type == "application/octet-stream" or not content_type:
                guessed_type, _ = mimetypes.guess_type(upload.filename)
                if guessed_type:
                    content_type = guessed_type.lower()

            if content_type not in ALLOWED_SUPPORTING_FILE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {upload.content_type or upload.filename}"
                )

            content = await upload.read()
            if len(content) > MAX_REGISTRATION_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File too large: {upload.filename}"
                )

            serialized_attachments.append({
                "filename": upload.filename,
                "content_type": content_type,
                "size_bytes": len(content),
                "data_base64": base64.b64encode(content).decode("utf-8"),
            })

        message_doc = {
            "tenant_id": tenant.get("_id"),
            "superadmin_id": superadmin_oid,
            "authorization_code": (tenant.get("authorization_code_used") or "").strip().upper() or None,
            "from_admin": tenant.get("admin_email"),
            "from_name": tenant.get("admin_full_name") or (tenant.get("pending_admin") or {}).get("full_name"),
            "to_email": (super_admin or {}).get("email"),
            "subject": clean_subject,
            "message": clean_message,
            "priority": "normal",
            "direction": "admin_to_superadmin",
            "attachments": serialized_attachments,
            "created_at": datetime.utcnow(),
            "read": False,
            "read_at": None,
        }

        insert_result = await master_db["superadmin_messages"].insert_one(message_doc)

        return {
            "success": True,
            "message_id": str(insert_result.inserted_id),
            "message": "Message sent successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration status message send error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send message"
        )


@router.post("/admin/registration-message-attachment")
@limiter.limit("30/minute")
async def download_registration_message_attachment(
    request: Request,
    email: EmailStr = Form(...),
    password: str = Form(..., min_length=6),
    message_id: str = Form(...),
    attachment_index: int = Form(...),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Download an attachment from a registration-phase message.
    Uses email+password auth since tenants are not logged in during registration.
    """
    from utils.s3_storage import download_file as s3_download_file

    try:
        tenant = await _authenticate_registration_status_admin(
            db=db,
            auth_manager=auth_manager,
            email=email,
            password=password,
        )

        master_db = await db.get_master_db()
        if master_db is None:
            raise HTTPException(status_code=503, detail="Master database unavailable")

        # Find the message and verify it belongs to this tenant
        msg = await master_db["superadmin_messages"].find_one({
            "_id": ObjectId(message_id),
            "tenant_id": tenant["_id"],
        })
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
                raise HTTPException(status_code=404, detail="Attachment file not found in storage")
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
        logger.error(f"Registration attachment download error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download attachment"
        )


@router.get("/admin/registration-status")
@limiter.limit("30/minute")
async def get_registration_status(
    request: Request,
    email: str,
    db: DatabaseManager = Depends(get_database),
):
    """
    Deprecated public endpoint.
    Registration status must be checked via credential-gated endpoint.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Use POST /auth/admin/registration-status-auth with registered email and password",
    )


@router.get("/tenant/check-availability")
@limiter.limit("30/minute")
async def check_tenant_availability(
    request: Request,
    tenant_id: str = Query(..., max_length=9),
    db: DatabaseManager = Depends(get_database),
):
    """Check if a requested tenant ID is available."""
    if not re.match(r'^[A-Z]{4}-[0-9]{4}$', tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID must match format XXXX-0000 (4 uppercase letters, hyphen, 4 digits)"
        )

    tenants = await db.get_master_collection("tenants")
    if tenants is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

    existing = await tenants.find_one({
        "$or": [
            {"tenant_id": tenant_id},
            {"requested_tenant_id": tenant_id},
        ]
    })

    return {"available": existing is None}


@router.get("/tenant/lookup", response_model=TenantLookupResponse)
@limiter.limit("60/minute")
async def lookup_tenant_name(
    request: Request,
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
):
    """Lookup institution name by tenant ID for login assistance."""
    normalized = normalize_tenant_id(tenant_id)
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant ID is required"
        )

    tenant = await get_tenant_by_tenant_id(db, normalized, include_inactive=True)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )

    return {
        "tenant_id": tenant.get("tenant_id") or normalized,
        "institution_name": (
            tenant.get("institution_name")
            or tenant.get("organization")
            or tenant.get("admin_full_name")
            or tenant.get("admin_email")
        ),
        "status": tenant.get("status"),
    }


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Logout user and invalidate session.
    
    This endpoint:
    1. Revokes the JWT token (adds to blacklist)
    2. Invalidates the session cache  
    3. Tracks student logout activity
    
    This ensures portal sessions are automatically logged out when desktop client logs out.
    """
    try:
        user_id = current_user.get("user_id")
        user_type = current_user.get("user_type")
        tenant_db = await _get_tenant_db_from_user(db, current_user)
        
        # CRITICAL: Revoke JWT token to force portal logout
        # This is called by desktop client when user logs out
        from core.token_blacklist import token_blacklist
        token = credentials.credentials
        token_blacklist.revoke(token, expiry_seconds=86400)  # Keep in blacklist for 24 hours

        # User-level revocation (Redis): invalidate ALL tokens for this user
        # so that other clients (portal, other browsers) are forced out
        from core.token_blacklist import revoke_user_session
        await revoke_user_session(auth_manager.cache_manager, user_id)
        logger.info(f"Token + user-level revocation for user {user_id}")

        # For students, track session end
        if user_type == "student":
            try:
                if tenant_db is not None:
                    # Set offline status
                    await tenant_db["students"].update_one(
                        {"_id": ObjectId(user_id)},
                        {"$set": {"is_online": False}}
                    )

                    # Get last login to calculate session duration
                    student = await tenant_db["students"].find_one({"_id": ObjectId(user_id)})
                    last_login = student.get("last_login") if student else None

                    session_duration = 0
                    if last_login:
                        session_duration = (datetime.utcnow() - last_login).total_seconds()

                    # Log session end activity
                    await tenant_db["student_activity_log"].insert_one({
                        "student_id": ObjectId(user_id),
                        "action": "session_end",
                        "timestamp": datetime.utcnow(),
                        "metadata": {
                            "session_duration": session_duration
                        }
                    })
                else:
                    logger.warning(f"No tenant_db for student {user_id} during logout — skipping activity tracking")
            except Exception as e:
                logger.warning(f"Failed to track student logout: {str(e)}")

        # Invalidate session
        await auth_manager.invalidate_user_session(user_id)

        return {"success": True, "message": "Successfully logged out"}

    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.post("/remote-logout")
async def remote_logout(
    request: Request,
    auth_manager: AuthManager = Depends(get_auth_manager),
    db: DatabaseManager = Depends(get_database),
):
    """
    Cross-client logout endpoint for the desktop agent.

    The agent's user token is signed with USER_JWT_SECRET (pen backend),
    which differs from the main backend's JWT_SECRET_KEY.  The normal
    /logout endpoint rejects it because get_current_user cannot decode it.

    This endpoint tries both secrets so the agent can trigger user-level
    revocation that forces the web portal out on the next /verify poll.

    IMPORTANT: The pen backend stores the student's *username* as user_id
    in its tokens (because _auth_user prefers username over ObjectId).
    The main backend uses the MongoDB ObjectId as user_id.  We must
    resolve the username → ObjectId so the Redis revocation key matches
    what the /verify path checks.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    token = auth_header.split(" ", 1)[1]

    # Try decoding with each known secret (main backend, then agent)
    payload: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    jwt_secrets = [
        JWT_SECRET_KEY,
        os.getenv("USER_JWT_SECRET", ""),
    ]
    for secret in jwt_secrets:
        if not secret:
            continue
        try:
            payload = pyjwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
            user_id = (
                payload.get("sub")
                or payload.get("user_id")
                or payload.get("username")
            )
            if user_id:
                break
        except pyjwt.InvalidTokenError:
            continue

    if not user_id or not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate token with any known secret",
        )

    from core.token_blacklist import token_blacklist, revoke_user_session

    # Token-level (in-memory) revocation
    token_blacklist.revoke(token, expiry_seconds=86400)

    # Collect all user_id variants to revoke.
    # The pen backend may store the username as user_id, while the main
    # backend uses the MongoDB ObjectId.  We revoke ALL known identifiers
    # so the /verify check (which uses ObjectId) always finds the flag.
    ids_to_revoke = {
        str(candidate)
        for candidate in (
            payload.get("sub"),
            payload.get("user_id"),
            payload.get("username"),
            user_id,
        )
        if candidate
    }

    lookup_username = None
    for candidate in (payload.get("username"), payload.get("user_id")):
        if isinstance(candidate, str) and candidate:
            lookup_username = candidate
            break

    # Resolve username → MongoDB ObjectId via tenant DB lookup
    db_name = payload.get("db_name")
    if db_name:
        try:
            tenant_db = await db.get_tenant_db(db_name)
            if tenant_db is not None:
                student = None
                if lookup_username:
                    # Try to find student by username (case-insensitive)
                    student = await tenant_db["students"].find_one(
                        {"username_lower": lookup_username.lower()},
                        {"_id": 1}
                    )
                    if not student:
                        # Fallback: try exact username match
                        student = await tenant_db["students"].find_one(
                            {"username": lookup_username},
                            {"_id": 1}
                        )
                if student:
                    ids_to_revoke.add(str(student["_id"]))
                elif lookup_username:
                    logger.warning(
                        "Remote logout: could not resolve username '%s' in %s",
                        lookup_username,
                        db_name,
                    )

                # Also check if any token identifier is already an ObjectId
                # (students/admins/tutors). This covers tokens that already carry
                # the canonical MongoDB id in `sub`.
                if not student:
                    found_canonical_id = False
                    for candidate in list(ids_to_revoke):
                        try:
                            oid = ObjectId(candidate)
                        except Exception:
                            continue
                        for coll_name in ("students", "admins", "tutors"):
                            doc = await tenant_db[coll_name].find_one({"_id": oid}, {"_id": 1})
                            if doc:
                                ids_to_revoke.add(str(doc["_id"]))
                                found_canonical_id = True
                                break
                        if found_canonical_id:
                            break
            else:
                logger.warning("Remote logout: tenant DB '%s' not found", db_name)
        except Exception as e:
            logger.warning("Remote logout: tenant DB lookup failed: %s", e)
    else:
        logger.warning("Remote logout: no db_name in token for user '%s'", user_id)

    # User-level (Redis) revocation for ALL known identifiers
    for uid in ids_to_revoke:
        await revoke_user_session(auth_manager.cache_manager, uid)
        await auth_manager.invalidate_user_session(uid)
    return {"success": True, "message": "Successfully logged out"}


@router.post("/invalidate-all-sessions")
async def invalidate_all_sessions(
    request: Request,
    db: DatabaseManager = Depends(get_database),
):
    """
    Invalidate all existing JWT sessions by setting a minimum issued-at
    timestamp.  Called by the deploy pipeline after a frontend release so
    every user is forced to re-login and pick up the new JS bundles.

    Requires X-Deploy-Secret header matching DEPLOY_SESSION_SECRET env var.
    """
    deploy_secret = os.environ.get("DEPLOY_SESSION_SECRET", "")
    if not deploy_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session invalidation not configured",
        )

    provided = request.headers.get("X-Deploy-Secret", "")
    if not provided or provided != deploy_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid deploy secret",
        )

    master_db = await db.get_master_db()
    if master_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        )

    import time as _time
    now_epoch = _time.time()
    await master_db["system_config"].update_one(
        {"key": "min_token_issued_at"},
        {"$set": {"key": "min_token_issued_at", "value": now_epoch}},
        upsert=True,
    )

    logger.info("All sessions invalidated at epoch=%s by deploy pipeline", now_epoch)
    return {"success": True, "min_token_issued_at": now_epoch}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current user information"""
    return UserResponse(
        user_id=current_user.get("user_id"),
        user_type=current_user.get("user_type"),
        email=current_user.get("email"),
        username=current_user.get("username"),
        full_name=current_user.get("full_name")
    )

@router.get("/user")
async def get_full_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Get full user profile from tenant database"""
    try:
        user_type = current_user.get("user_type")
        user_id = current_user.get("user_id")

        tenant_db = await _get_tenant_db_from_user(db, current_user)
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context required"
            )

        if user_type == "student":
            student = await tenant_db["students"].find_one({"_id": ObjectId(user_id)})
            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )
            student["_id"] = str(student["_id"])
            if "admin_id" in student:
                student["admin_id"] = str(student["admin_id"])
            return {"success": True, "data": student}

        elif user_type == "admin":
            admin = await tenant_db["admins"].find_one({"_id": ObjectId(user_id)})
            if not admin:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Admin profile not found"
                )
            admin["_id"] = str(admin["_id"])
            return {"success": True, "data": admin}

        elif user_type == "tutor":
            tutor = await tenant_db["tutors"].find_one({"_id": ObjectId(user_id)})
            if not tutor:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Tutor profile not found"
                )
            tutor["_id"] = str(tutor["_id"])
            return {"success": True, "data": tutor}

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unknown user type"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user profile"
        )
