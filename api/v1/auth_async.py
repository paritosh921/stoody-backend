"""
Async Authentication API for SkillBot
JWT-based authentication with rate limiting and caching
"""

import base64
import logging
import mimetypes
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, Form, UploadFile, File, Query
from fastapi.responses import JSONResponse
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
from core.security import (
    PasswordResetTokenManager,
    get_security_logger,
    get_password_validator,
    SecurityEventType,
    get_client_ip,
    get_user_agent,
)
from core.email_service import get_email_service
from core.cookie_auth import get_current_user_dual_auth, cookie_auth_manager
from core.observability import record_auth_login
from config_async import settings, PASSWORD_RESET_TOKEN_EXPIRE_MINUTES

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

# Secure Password Reset Models
class PasswordResetRequestModel(BaseModel):
    """Request a password reset link via email"""
    email: EmailStr
    tenant_id: str = Field(..., pattern=TENANT_ID_PATTERN)

class PasswordResetVerifyRequest(BaseModel):
    """Verify a password reset token is valid"""
    token: str = Field(..., min_length=32)

class PasswordResetCompleteRequest(BaseModel):
    """Complete password reset with new password"""
    token: str = Field(..., min_length=32)
    new_password: str = Field(..., min_length=8, max_length=72)
    institution_name: Optional[str] = None
    status: Optional[str] = None

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

@router.post("/admin/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def admin_login(
    request: Request,
    login_data: AdminLoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Admin login endpoint"""
    login_recorded = False
    try:
        tenant_id = normalize_tenant_id(login_data.tenant_id) if login_data.tenant_id else None
        tenant = await _resolve_tenant_for_auth(db, request, tenant_id)
        tenant_db = await _get_tenant_db_or_503(db, tenant)

        admin_doc = await tenant_db["admins"].find_one({
            "email": login_data.email,
            "is_active": True
        })

        if not admin_doc:
            record_auth_login("admin", False)
            login_recorded = True
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        if not auth_manager.verify_password(login_data.password, admin_doc.get("password_hash", "")):
            record_auth_login("admin", False)
            login_recorded = True
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        await tenant_db["admins"].update_one(
            {"_id": admin_doc["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )

        admin_data = {
            "user_id": str(admin_doc["_id"]),
            "admin_id": str(admin_doc["_id"]),
            "email": admin_doc.get("email"),
            "full_name": admin_doc.get("full_name") or admin_doc.get("name"),
            "user_type": "admin",
            "subdomain": tenant.get("subdomain"),
            "tenant_id": tenant.get("tenant_id"),
            "db_name": tenant.get("db_name"),
            "institution_id": tenant.get("institution_id"),
            "admin_role": admin_doc.get("role", "master_admin"),
            "permissions": admin_doc.get("permissions") or []
        }

        session_data = await auth_manager.create_user_session(admin_data)
        record_auth_login("admin", True)
        login_recorded = True

        response_data = {
            "success": True,
            "data": {
                "access_token": session_data["access_token"],
                "user_type": "admin",
                "user": session_data["user"]
            }
        }

        response = JSONResponse(content=response_data)
        cookie_auth_manager.set_auth_cookie(response, session_data["access_token"])
        return response

    except HTTPException:
        if not login_recorded:
            record_auth_login("admin", False)
        raise
    except Exception as e:
        if not login_recorded:
            record_auth_login("admin", False)
        logger.error(f"Admin login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
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
        }

        session_data = await auth_manager.create_user_session(student_data)
        record_auth_login("student", True)
        login_recorded = True

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

@router.post("/tutor/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def tutor_login(
    request: Request,
    login_data: TutorLoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Tutor login endpoint"""
    login_recorded = False
    try:
        tenant_id = normalize_tenant_id(login_data.tenant_id)
        tenant = await _resolve_tenant_for_auth(db, request, tenant_id)
        tenant_db = await _get_tenant_db_or_503(db, tenant)

        tutor_data = await auth_manager.authenticate_tutor(
            login_data.username, login_data.password, db, tenant_db
        )

        if not tutor_data:
            record_auth_login("tutor", False)
            login_recorded = True
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        tutor_data.update({
            "subdomain": tenant.get("subdomain"),
            "tenant_id": tenant.get("tenant_id"),
            "db_name": tenant.get("db_name"),
            "institution_id": tenant.get("institution_id"),
        })

        session_data = await auth_manager.create_user_session(tutor_data)
        record_auth_login("tutor", True)
        login_recorded = True

        response_data = {
            "success": True,
            "data": {
                "access_token": session_data["access_token"],
                "user_type": "tutor",
                "user": session_data["user"]
            }
        }

        response = JSONResponse(content=response_data)
        cookie_auth_manager.set_auth_cookie(response, session_data["access_token"])
        return response

    except HTTPException:
        if not login_recorded:
            record_auth_login("tutor", False)
        raise
    except Exception as e:
        if not login_recorded:
            record_auth_login("tutor", False)
        logger.error(f"Tutor login error: {str(e)}", exc_info=True)
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

        logger.info(f"Student {student.get('username')} changed their password")

        return {
            "success": True,
            "message": "Password changed successfully"
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

        logger.info(f"Tutor {tutor.get('username')} changed their password")

        return {
            "success": True,
            "message": "Password changed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tutor change password error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
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


# =============================================================================
# SECURE PASSWORD RESET ENDPOINTS (Token-Based)
# =============================================================================

@router.post("/password-reset/request")
@limiter.limit("3/hour")
async def request_password_reset(
    request: Request,
    reset_data: PasswordResetRequestModel,
    db: DatabaseManager = Depends(get_database)
):
    """
    Request a password reset link via email.

    Security features:
    - Rate limited to 3 requests per hour per IP
    - Sends email with time-limited, single-use token
    - Generic response to prevent email enumeration
    - Logs all reset requests for security audit
    """
    ip_address = get_client_ip(request)
    user_agent = get_user_agent(request)

    try:
        tenant_id = normalize_tenant_id(reset_data.tenant_id)
        tenant = await _resolve_tenant_for_auth(db, request, tenant_id, require_active=True)
        tenant_db = await _get_tenant_db_or_503(db, tenant)

        # Look up user by email (check admins, then students)
        email_lower = reset_data.email.lower().strip()
        user = None
        user_type = None
        collection_name = None

        # Check admins
        admin = await tenant_db["admins"].find_one({"email": email_lower, "is_active": True})
        if admin:
            user = admin
            user_type = "admin"
            collection_name = "admins"

        # Check students if not found in admins
        if not user:
            student = await tenant_db["students"].find_one({"email": email_lower, "is_active": True})
            if student:
                user = student
                user_type = "student"
                collection_name = "students"

        # Check tutors if not found
        if not user:
            tutor = await tenant_db["tutors"].find_one({"email": email_lower, "is_active": True})
            if tutor:
                user = tutor
                user_type = "tutor"
                collection_name = "tutors"

        # Generic response to prevent email enumeration
        generic_response = {
            "success": True,
            "message": "If an account with this email exists, you will receive a password reset link shortly."
        }

        if not user:
            # Log failed attempt for security monitoring
            security_logger.log_event(
                event_type=SecurityEventType.PASSWORD_RESET_REQUEST,
                email=reset_data.email,
                ip_address=ip_address,
                user_agent=user_agent,
                tenant_id=tenant_id,
                success=False,
                details={"reason": "email_not_found"}
            )
            return generic_response

        # Check if user has email
        user_email = user.get("email")
        if not user_email:
            security_logger.log_event(
                event_type=SecurityEventType.PASSWORD_RESET_REQUEST,
                user_id=str(user["_id"]),
                ip_address=ip_address,
                tenant_id=tenant_id,
                success=False,
                details={"reason": "no_email_on_account"}
            )
            return generic_response

        # Generate secure reset token
        token_manager = PasswordResetTokenManager(expire_minutes=PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
        reset_data = token_manager.create_reset_record(
            user_id=str(user["_id"]),
            email=user_email
        )

        # Store reset record in database
        reset_record = reset_data["record"]
        reset_record["user_type"] = user_type
        reset_record["tenant_id"] = tenant_id
        reset_record["ip_address"] = ip_address
        reset_record["user_agent"] = user_agent

        await tenant_db["password_reset_tokens"].insert_one(reset_record)

        # Send reset email
        email_service = get_email_service()
        username = user.get("username") or user.get("full_name") or user.get("email", "User")

        frontend_url = tenant.get("frontend_url") or "https://app.stoody.in"
        email_sent = await email_service.send_password_reset_email(
            to_email=user_email,
            reset_token=reset_data["token"],
            username=username,
            frontend_url=frontend_url,
            expire_minutes=PASSWORD_RESET_TOKEN_EXPIRE_MINUTES
        )

        # Log the request
        security_logger.log_event(
            event_type=SecurityEventType.PASSWORD_RESET_REQUEST,
            user_id=str(user["_id"]),
            email=user_email,
            ip_address=ip_address,
            user_agent=user_agent,
            tenant_id=tenant_id,
            success=email_sent,
            details={"user_type": user_type, "email_sent": email_sent}
        )

        logger.info(f"Password reset requested for {user_type}: {user_email}")

        return generic_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset request error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request"
        )


@router.post("/password-reset/verify")
@limiter.limit("10/minute")
async def verify_password_reset_token(
    request: Request,
    verify_data: PasswordResetVerifyRequest,
    db: DatabaseManager = Depends(get_database)
):
    """
    Verify if a password reset token is valid.

    Used by frontend to check token before showing reset form.
    """
    try:
        # Get master DB to search across tenants (or use request context)
        subdomain = getattr(request.state, "subdomain", None)
        tenant = None

        if subdomain:
            tenant = await get_tenant_by_subdomain(db, subdomain)

        if not tenant:
            # Try to find token across all tenants (expensive but necessary for email links)
            master_db = await db.get_master_db()
            if master_db is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service unavailable"
                )

            # Get all active tenants and search for the token
            tenants = await master_db["tenants"].find({"status": "active"}).to_list(length=1000)

            token_manager = PasswordResetTokenManager()
            token_hash = token_manager.hash_token(verify_data.token)

            for t in tenants:
                db_name = t.get("db_name")
                if not db_name:
                    continue

                tenant_db = await db.get_tenant_db(db_name)
                if tenant_db is None:
                    continue

                reset_record = await tenant_db["password_reset_tokens"].find_one({
                    "token_hash": token_hash
                })

                if reset_record:
                    is_valid, reason = token_manager.is_token_valid(reset_record)
                    if is_valid:
                        return {
                            "success": True,
                            "valid": True,
                            "message": "Token is valid",
                            "expires_in_minutes": max(0, int(
                                (reset_record["expires_at"] - datetime.utcnow()).total_seconds() / 60
                            ))
                        }
                    else:
                        return {
                            "success": True,
                            "valid": False,
                            "message": reason
                        }

            return {
                "success": True,
                "valid": False,
                "message": "Invalid or expired reset token"
            }

        # Tenant found via subdomain
        tenant_db = await db.get_tenant_db(tenant.get("db_name"))
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tenant database not available"
            )

        token_manager = PasswordResetTokenManager()
        token_hash = token_manager.hash_token(verify_data.token)

        reset_record = await tenant_db["password_reset_tokens"].find_one({
            "token_hash": token_hash
        })

        if not reset_record:
            return {
                "success": True,
                "valid": False,
                "message": "Invalid or expired reset token"
            }

        is_valid, reason = token_manager.is_token_valid(reset_record)

        return {
            "success": True,
            "valid": is_valid,
            "message": "Token is valid" if is_valid else reason,
            "expires_in_minutes": max(0, int(
                (reset_record["expires_at"] - datetime.utcnow()).total_seconds() / 60
            )) if is_valid else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify token"
        )


@router.post("/password-reset/complete")
@limiter.limit("5/minute")
async def complete_password_reset(
    request: Request,
    reset_data: PasswordResetCompleteRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Complete password reset with new password.

    Security features:
    - Validates password strength
    - Marks token as used (single-use)
    - Sends confirmation email
    - Logs password change for audit
    """
    ip_address = get_client_ip(request)
    user_agent = get_user_agent(request)

    try:
        # Validate new password
        password_validator = get_password_validator()
        is_valid, errors = password_validator.validate(reset_data.new_password)

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Password does not meet requirements",
                    "errors": errors
                }
            )

        # Find the reset token across tenants
        master_db = await db.get_master_db()
        if master_db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable"
            )

        tenants = await master_db["tenants"].find({"status": "active"}).to_list(length=1000)

        token_manager = PasswordResetTokenManager()
        token_hash = token_manager.hash_token(reset_data.token)

        reset_record = None
        tenant_db = None
        tenant_info = None

        for t in tenants:
            db_name = t.get("db_name")
            if not db_name:
                continue

            tdb = await db.get_tenant_db(db_name)
            if tdb is None:
                continue

            record = await tdb["password_reset_tokens"].find_one({
                "token_hash": token_hash
            })

            if record:
                reset_record = record
                tenant_db = tdb
                tenant_info = t
                break

        if not reset_record:
            security_logger.log_event(
                event_type=SecurityEventType.PASSWORD_RESET_COMPLETE,
                ip_address=ip_address,
                success=False,
                details={"reason": "invalid_token"}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )

        # Validate token
        is_valid, reason = token_manager.is_token_valid(reset_record)
        if not is_valid:
            security_logger.log_event(
                event_type=SecurityEventType.PASSWORD_RESET_COMPLETE,
                user_id=reset_record.get("user_id"),
                ip_address=ip_address,
                success=False,
                details={"reason": reason}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=reason
            )

        # Get user info
        user_id = reset_record["user_id"]
        user_type = reset_record.get("user_type", "student")
        collection_name = {
            "admin": "admins",
            "student": "students",
            "tutor": "tutors"
        }.get(user_type, "students")

        user = await tenant_db[collection_name].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User account not found"
            )

        # Hash new password
        new_password_hash = auth_manager.get_password_hash(reset_data.new_password)

        # Update password
        await tenant_db[collection_name].update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "password_hash": new_password_hash,
                "password_changed_at": datetime.utcnow(),
                "password_reset_requested": False,
                "requires_password_change": False
            }}
        )

        # Mark token as used
        await tenant_db["password_reset_tokens"].update_one(
            {"_id": reset_record["_id"]},
            {"$set": {
                "used": True,
                "used_at": datetime.utcnow(),
                "used_ip": ip_address
            }}
        )

        # Invalidate any existing sessions for security
        await auth_manager.invalidate_user_session(user_id)

        # Send confirmation email
        email_service = get_email_service()
        user_email = user.get("email")
        username = user.get("username") or user.get("full_name") or "User"

        if user_email:
            await email_service.send_password_changed_notification(
                to_email=user_email,
                username=username
            )

        # Log successful reset
        security_logger.log_event(
            event_type=SecurityEventType.PASSWORD_RESET_COMPLETE,
            user_id=user_id,
            email=user_email,
            ip_address=ip_address,
            user_agent=user_agent,
            tenant_id=tenant_info.get("tenant_id") if tenant_info else None,
            success=True,
            details={"user_type": user_type}
        )

        logger.info(f"Password reset completed for {user_type}: {user_id}")

        return {
            "success": True,
            "message": "Password has been reset successfully. You can now log in with your new password."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset completion error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password"
        )


@router.get("/verify")
async def verify_token(
    request: Request,
    auth_manager: AuthManager = Depends(get_auth_manager)
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

    return {
        "success": True,
        "data": {
            "user_id": current_user.get("user_id"),
            "user_type": current_user.get("user_type"),
            "email": current_user.get("email"),
            "username": current_user.get("username"),
            "full_name": current_user.get("full_name")
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
            return tenant

    # No matching tenant/password pair found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password"
    )


def _serialize_registration_status_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    attachments = msg.get("attachments") or []
    serialized_attachments: List[Dict[str, Any]] = []
    for file_doc in attachments:
        if not isinstance(file_doc, dict):
            continue
        serialized_attachments.append({
            "filename": file_doc.get("filename"),
            "content_type": file_doc.get("content_type"),
            "size_bytes": file_doc.get("size_bytes"),
            "data_base64": file_doc.get("data_base64"),
        })

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
        logger.info(f"Token revoked for user {user_id}")

        # For students, track session end
        if user_type == "student":
            try:
                # Set offline status
                if tenant_db is not None:
                    await tenant_db["students"].update_one(
                        {"_id": ObjectId(user_id)},
                        {"$set": {"is_online": False}}
                    )
                else:
                    await db.mongo_update_one(
                        "students",
                        {"_id": ObjectId(user_id)},
                        {"$set": {"is_online": False}}
                    )

                # Get last login to calculate session duration
                if tenant_db is not None:
                    student = await tenant_db["students"].find_one({"_id": ObjectId(user_id)})
                else:
                    student = await db.mongo_find_one("students", {"_id": ObjectId(user_id)})
                last_login = student.get("last_login") if student else None

                session_duration = 0
                if last_login:
                    session_duration = (datetime.utcnow() - last_login).total_seconds()

                # Log session end activity
                log_doc = {
                    "student_id": ObjectId(user_id),
                    "action": "session_end",
                    "timestamp": datetime.utcnow(),
                    "metadata": {
                        "session_duration": session_duration
                    }
                }
                if tenant_db is not None:
                    await tenant_db["student_activity_log"].insert_one(log_doc)
                else:
                    await db.mongo_insert_one("student_activity_log", log_doc)
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

@router.post("/init-admin")
@limiter.limit("5/minute")
async def init_admin(
    request: Request,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Initialize default admin account (dev/testing only)"""
    if not settings.DEBUG_MODE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found"
        )

    try:
        # Check if admin already exists
        existing_admin = await db.mongo_find_one("admins", {"email": "admin@skillbot.app"})

        if existing_admin:
            return {"message": "Admin already exists"}

        # Create default admin
        admin_data = {
            "email": "admin@skillbot.app",
            "password_hash": auth_manager.get_password_hash("admin123"),
            "full_name": "System Administrator",
            "is_active": True,
            "created_at": datetime.utcnow()
        }

        admin_id = await db.mongo_insert_one("admins", admin_data)

        if admin_id:
            return {"message": "Default admin created successfully", "admin_id": admin_id}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create admin"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Init admin error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize admin"
        )

@router.post("/init-demo-student")
@limiter.limit("5/minute")
async def init_demo_student(
    request: Request,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Initialize demo student account (dev/testing only)"""
    if not settings.DEBUG_MODE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found"
        )

    try:
        # Check if demo student already exists
        existing_student = await db.mongo_find_one("students", {"username": "demo_student"})

        if existing_student:
            return {"message": "Demo student already exists"}

        # Create demo student
        student_data = {
            "username": "demo_student",
            "password_hash": auth_manager.get_password_hash("student123"),
            "full_name": "Demo Student",
            "email": "demo@student.com",
            "is_active": True,
            "created_at": datetime.utcnow()
        }

        student_id = await db.mongo_insert_one("students", student_data)

        if student_id:
            return {"message": "Demo student created successfully", "student_id": student_id}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create demo student"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Init demo student error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize demo student"
        )

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
    """Get full user profile from database"""
    try:
        user_type = current_user.get("user_type")
        user_id = current_user.get("user_id")

        if user_type == "student":
            # Fetch full student profile from database
            student = await db.mongo_find_one("students", {"_id": ObjectId(user_id)})
            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )

            # Convert ObjectId to string for JSON serialization
            student["_id"] = str(student["_id"])
            if "admin_id" in student:
                student["admin_id"] = str(student["admin_id"])

            return {
                "success": True,
                "data": student
            }

        elif user_type == "admin":
            # Fetch admin profile from database
            admin = await db.mongo_find_one("admins", {"_id": ObjectId(user_id)})
            if not admin:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Admin profile not found"
                )

            # Convert ObjectId to string
            admin["_id"] = str(admin["_id"])

            return {
                "success": True,
                "data": admin
            }

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
