"""
Async Authentication API for SkillBot
JWT-based authentication with rate limiting and caching
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from core.auth import AuthManager
from config_async import settings

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
class AdminLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)

class AdminRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2, max_length=100)
    subdomain: str = Field(..., min_length=3, max_length=50, pattern=r'^[a-z0-9\-]+$')
    organization: str = Field(..., min_length=2, max_length=100)

class StudentLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class TutorLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class StudentChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=6)
    new_password: str = Field(..., min_length=8)

class StudentForgotPasswordRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    date_of_birth: str  # Format: YYYY-MM-DD
    phone: str

class TokenResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any]

class UserResponse(BaseModel):
    user_id: str
    user_type: str
    email: Optional[str] = None
    username: Optional[str] = None
    full_name: Optional[str] = None

# Dependency injection
async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db

async def get_cache(request: Request) -> CacheManager:
    return request.app.state.cache

async def get_auth_manager(request: Request) -> AuthManager:
    return request.app.state.auth

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        user_data = await auth_manager.verify_token_and_get_user(token)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
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
        # Authenticate admin
        admin_data = await auth_manager.authenticate_admin(
            login_data.email, login_data.password, db
        )

        if not admin_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Create session
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
    Student login endpoint with globally unique usernames

    - Username is globally unique across all students
    - No subdomain dependency
    - Student automatically mapped to correct admin via admin_id field
    """
    try:
        # Find student by globally unique username
        student = await db.mongo_find_one("students", {
            "username": login_data.username
        })

        if not student:
            logger.warning(f"Student {login_data.username} not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Verify password
        if not auth_manager.verify_password(login_data.password, student.get("password_hash", "")):
            logger.warning(f"Invalid password for student {login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Get admin info for the student's admin_id
        admin_id = student.get("admin_id")
        if not admin_id:
            logger.error(f"Student {login_data.username} has no admin_id")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Student account is not properly configured"
            )

        admin = await db.mongo_find_one("admins", {"_id": admin_id})
        if not admin:
            logger.error(f"Admin not found for student {login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Associated school/organization not found"
            )

        subdomain = admin.get("subdomain", "demo")

        # Create student_data for session
        student_data = {
            "user_id": str(student["_id"]),
            "user_type": "student",
            "username": student.get("username"),
            "email": student.get("email"),
            "full_name": student.get("full_name", student.get("name")),
            "admin_id": str(admin_id),  # IMPORTANT: Include admin_id in JWT
            "subdomain": subdomain
        }

        # Create session (JWT token)
        session_data = await auth_manager.create_user_session(student_data)

        # Update student last_login
        try:
            await db.mongo_update_one(
                "students",
                {"_id": student["_id"]},
                {
                    "$set": {
                        "is_online": True,
                        "last_login": datetime.utcnow()
                    }
                }
            )

            # Log login activity
            await db.mongo_insert_one("student_activity_log", {
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
                "requires_password_change": student.get("requires_password_change", False)
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
        tutor_data = await auth_manager.authenticate_tutor(
            login_data.username, login_data.password, db
        )

        if not tutor_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Create session (JWT token)
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

        # Get student
        student = await db.mongo_find_one("students", {"_id": student_id})
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
        await db.mongo_update_one(
            "students",
            {"_id": student_id},
            {
                "$set": {
                    "password_hash": new_password_hash,
                    "requires_password_change": False,
                    "password_changed_at": datetime.utcnow()
                }
            }
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

@router.post("/student/forgot-password")
@limiter.limit("3/hour")
async def student_forgot_password(
    request: Request,
    forgot_data: StudentForgotPasswordRequest,
    db: DatabaseManager = Depends(get_database)
):
    """Student requests password reset using globally unique username"""
    try:
        # Find student by globally unique username (no subdomain needed)
        student = await db.mongo_find_one("students", {
            "username": forgot_data.username
        })

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
        await db.mongo_update_one(
            "students",
            {"_id": student["_id"]},
            {
                "$set": {
                    "password_reset_requested": True,
                    "password_reset_requested_at": datetime.utcnow()
                }
            }
        )

        logger.info(f"Password reset requested for student: {forgot_data.username}")

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

@router.get("/admin/subdomain/check/{subdomain}")
@limiter.limit("10/minute")
async def check_subdomain_availability(
    subdomain: str,
    request: Request,
    db: DatabaseManager = Depends(get_database)
):
    """
    Check if subdomain is available for registration
    Returns: {available: boolean, message: string}
    """
    try:
        # Validate subdomain format
        if not subdomain or len(subdomain) < 3 or len(subdomain) > 50:
            return {
                "success": True,
                "data": {
                    "available": False,
                    "message": "Subdomain must be between 3 and 50 characters"
                }
            }

        # Check if subdomain contains only lowercase letters, numbers, and hyphens
        import re
        if not re.match(r'^[a-z0-9\-]+$', subdomain):
            return {
                "success": True,
                "data": {
                    "available": False,
                    "message": "Subdomain can only contain lowercase letters, numbers, and hyphens"
                }
            }

        # Check reserved subdomains
        reserved = ['www', 'app', 'admin', 'api', 'demo', 'test', 'staging', 'dev', 'mail', 'ftp']
        if subdomain in reserved:
            return {
                "success": True,
                "data": {
                    "available": False,
                    "message": "This subdomain is reserved"
                }
            }

        # Check if subdomain already exists
        existing = await db.mongo_find_one("admins", {"subdomain": subdomain})

        if existing:
            return {
                "success": True,
                "data": {
                    "available": False,
                    "message": "This subdomain is already taken"
                }
            }

        return {
            "success": True,
            "data": {
                "available": True,
                "message": "Subdomain is available!"
            }
        }

    except Exception as e:
        logger.error(f"Subdomain check error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check subdomain availability"
        )

@router.post("/admin/register", response_model=TokenResponse)
@limiter.limit("3/hour")
async def register_admin(
    request: Request,
    register_data: AdminRegisterRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Register a new admin with subdomain
    Creates admin account and returns JWT token
    """
    try:
        # Check if email already exists
        existing_email = await db.mongo_find_one("admins", {"email": register_data.email})
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Check if subdomain already exists
        existing_subdomain = await db.mongo_find_one("admins", {"subdomain": register_data.subdomain})
        if existing_subdomain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Subdomain already taken. Please choose another."
            )

        # Validate subdomain format again
        import re
        if not re.match(r'^[a-z0-9\-]+$', register_data.subdomain):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Subdomain can only contain lowercase letters, numbers, and hyphens"
            )

        # Check reserved subdomains
        reserved = ['www', 'app', 'admin', 'api', 'demo', 'test', 'staging', 'dev', 'mail', 'ftp']
        if register_data.subdomain in reserved:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This subdomain is reserved"
            )

        # Hash password
        password_hash = auth_manager.get_password_hash(register_data.password)

        # Create admin document
        admin_doc = {
            "email": register_data.email,
            "password_hash": password_hash,
            "name": register_data.full_name,
            "subdomain": register_data.subdomain,
            "organization": register_data.organization,
            "role": "admin",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "google_id": None
        }

        # Insert admin
        admin_id = await db.mongo_insert_one("admins", admin_doc)

        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create admin account"
            )

        # Create JWT token
        admin_data = {
            "user_id": admin_id,
            "user_type": "admin",
            "admin_id": admin_id,  # Add admin_id for consistency
            "email": register_data.email,
            "full_name": register_data.full_name,
            "subdomain": register_data.subdomain
        }

        session_data = await auth_manager.create_user_session(admin_data)

        logger.info(f"New admin registered: {register_data.email} with subdomain: {register_data.subdomain}")

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
        logger.error(f"Admin registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/logout")
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Logout user and invalidate session"""
    try:
        user_id = current_user.get("user_id")
        user_type = current_user.get("user_type")

        # For students, track session end
        if user_type == "student":
            try:
                # Set offline status
                await db.mongo_update_one(
                    "students",
                    {"_id": ObjectId(user_id)},
                    {"$set": {"is_online": False}}
                )

                # Get last login to calculate session duration
                student = await db.mongo_find_one("students", {"_id": ObjectId(user_id)})
                last_login = student.get("last_login") if student else None

                session_duration = 0
                if last_login:
                    session_duration = (datetime.utcnow() - last_login).total_seconds()

                # Log session end activity
                await db.mongo_insert_one("student_activity_log", {
                    "student_id": ObjectId(user_id),
                    "action": "session_end",
                    "timestamp": datetime.utcnow(),
                    "metadata": {
                        "session_duration": session_duration
                    }
                })
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
