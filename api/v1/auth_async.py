"""
Async Authentication API for SkillBot
JWT-based authentication with rate limiting and caching
"""

import base64
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, Form, UploadFile, File
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
from config_async import settings

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
TENANT_ID_PATTERN = r'^[A-Za-z]{4}[0-9]{4}$'

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
    try:
        tenant_id = normalize_tenant_id(login_data.tenant_id) if login_data.tenant_id else None
        tenant = await _resolve_tenant_for_auth(db, request, tenant_id)
        tenant_db = await _get_tenant_db_or_503(db, tenant)

        admin_doc = await tenant_db["admins"].find_one({
            "email": login_data.email,
            "is_active": True
        })

        if not admin_doc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        if not auth_manager.verify_password(login_data.password, admin_doc.get("password_hash", "")):
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

        return TokenResponse(
            success=True,
            data={
                "access_token": session_data["access_token"],
                "user_type": "admin",
                "user": session_data["user"]
            }
        )

    except HTTPException:
        raise
    except Exception as e:
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
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        if not auth_manager.verify_password(login_data.password, student.get("password_hash", "")):
            logger.warning(f"Invalid password for student {normalized_username}")
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

        return TokenResponse(
            success=True,
            data={
                "access_token": session_data["access_token"],
                "user_type": "student",
                "user": session_data["user"],
                "requires_password_change": student.get("requires_password_change", False),
                "pen_token": pen_token,
                "pen_token_expires_at": pen_token_expires_at.isoformat() if pen_token_expires_at else None
            }
        )

    except HTTPException:
        raise
    except Exception as e:
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
    try:
        tenant_id = normalize_tenant_id(login_data.tenant_id)
        tenant = await _resolve_tenant_for_auth(db, request, tenant_id)
        tenant_db = await _get_tenant_db_or_503(db, tenant)

        tutor_data = await auth_manager.authenticate_tutor(
            login_data.username, login_data.password, db, tenant_db
        )

        if not tutor_data:
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

        # Update tutor last_login is already done; optionally log activity if needed
        return TokenResponse(
            success=True,
            data={
                "access_token": session_data["access_token"],
                "user_type": "tutor",
                "user": session_data["user"]
            }
        )

    except HTTPException:
        raise
    except Exception as e:
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

@router.get("/verify")
async def verify_token(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Verify JWT token and return user data"""
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
    subdomain: Optional[str] = Form(None, min_length=3, max_length=50),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Register a new admin
    Creates a pending tenant request for super-admin review
    """
    try:
        tenants = await db.get_master_collection("tenants")
        if tenants is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tenant registry not available"
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
            import re
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

        tenant_doc = {
            "tenant_id": None,
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

        if attachments:
            files_collection = await db.get_master_collection("tenant_application_files")
            if files_collection is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Tenant files collection not available"
                )
            allowed_types = {"application/pdf", "image/png", "image/jpeg", "image/jpg", "image/webp"}
            max_file_size = 5 * 1024 * 1024  # 5MB
            if len(attachments) > 10:
                raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

            file_docs = []
            for upload in attachments:
                if not upload.filename:
                    continue
                if upload.content_type not in allowed_types:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.content_type}")
                content = await upload.read()
                if len(content) > max_file_size:
                    raise HTTPException(status_code=400, detail=f"File too large: {upload.filename}")
                file_docs.append({
                    "tenant_request_id": result.inserted_id,
                    "admin_email": email,
                    "institution_name": institution_name_value,
                    "filename": upload.filename,
                    "content_type": upload.content_type,
                    "size_bytes": len(content),
                    "data_base64": base64.b64encode(content).decode("utf-8"),
                    "uploaded_at": datetime.utcnow(),
                })

            if file_docs:
                insert_result = await files_collection.insert_many(file_docs)
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


@router.get("/admin/registration-status")
@limiter.limit("30/minute")
async def get_registration_status(
    request: Request,
    email: str,
    db: DatabaseManager = Depends(get_database),
):
    """
    Check the status of a tenant registration by admin email.
    Returns the current status and relevant details for the registration page.
    """
    try:
        tenants = await db.get_master_collection("tenants")
        if tenants is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable"
            )

        tenant = await tenants.find_one({"admin_email": email})
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No application found for this email"
            )

        # Build response based on status
        response = {
            "status": tenant.get("status", "pending"),
            "institution_name": tenant.get("institution_name") or tenant.get("organization"),
            "created_at": tenant.get("created_at").isoformat() if tenant.get("created_at") else None,
        }

        # Add status-specific fields
        if tenant.get("status") == "rejected":
            response["rejection_reason"] = tenant.get("rejection_reason")

        if tenant.get("status") in ["approved", "active"]:
            response["approved_at"] = tenant.get("approved_at").isoformat() if tenant.get("approved_at") else None
            response["tenant_id"] = tenant.get("tenant_id")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration status check error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check status"
        )


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
