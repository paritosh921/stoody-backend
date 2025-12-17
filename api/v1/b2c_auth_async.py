"""
B2C Authentication API for Stoody
Google OAuth-based authentication for B2C users
Uses stoody-b2c database (completely separate from skillbot_db)
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
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from core.database import DatabaseManager
from core.cache import CacheManager
from core.auth import AuthManager
from config_async import settings, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# ==================== Pydantic Models ====================

class GoogleLoginRequest(BaseModel):
    """Request model for Google OAuth login"""
    credential: str = Field(..., description="Google OAuth ID token")


class B2CUserResponse(BaseModel):
    """Response model for B2C user data"""
    user_id: str
    email: str
    full_name: str
    picture: Optional[str] = None
    user_type: str = "b2c_user"


class B2CTokenResponse(BaseModel):
    """Response model for B2C authentication"""
    success: bool = True
    data: Dict[str, Any]


# ==================== Dependency Injection ====================

async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


async def get_cache(request: Request) -> CacheManager:
    return request.app.state.cache


async def get_auth_manager(request: Request) -> AuthManager:
    return request.app.state.auth


async def get_current_b2c_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Dict[str, Any]:
    """Get current authenticated B2C user"""
    try:
        token = credentials.credentials
        user_data = await auth_manager.verify_token_and_get_user(token)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Ensure this is a B2C user
        if user_data.get("user_type") != "b2c_user":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is for B2C users only"
            )

        return user_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ==================== Helper Functions ====================

async def verify_google_token(id_token_str: str) -> Optional[Dict[str, Any]]:
    """Verify Google OAuth ID token and return user info"""
    try:
        # Verify the token with clock skew tolerance (5 seconds)
        # This handles slight clock differences between local machine and Google servers
        idinfo = id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=5  # Allow 5 seconds of clock skew
        )

        # Verify issuer
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Invalid issuer')

        # Token is valid, extract user info
        return {
            'google_id': idinfo['sub'],
            'email': idinfo['email'],
            'email_verified': idinfo.get('email_verified', False),
            'full_name': idinfo.get('name', ''),
            'given_name': idinfo.get('given_name', ''),
            'family_name': idinfo.get('family_name', ''),
            'picture': idinfo.get('picture', ''),
            'locale': idinfo.get('locale', 'en')
        }

    except ValueError as e:
        logger.error(f"Google token verification failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error verifying Google token: {str(e)}")
        return None


# ==================== B2C Authentication Routes ====================

@router.post("/google/login", response_model=B2CTokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def b2c_google_login(
    request: Request,
    login_data: GoogleLoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    B2C user login/signup via Google OAuth
    
    - Verifies Google ID token
    - Creates new user in stoody-b2c database if first login
    - Returns JWT token for session
    
    Note: Uses stoody-b2c database (completely separate from skillbot_db)
    """
    try:
        # Verify Google token
        google_user = await verify_google_token(login_data.credential)
        
        if not google_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google token"
            )

        if not google_user.get('email_verified'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not verified with Google"
            )

        google_id = google_user['google_id']
        email = google_user['email']

        # Check if user exists in B2C database
        existing_user = await db.b2c_find_one("users", {"google_id": google_id})

        if existing_user:
            # Existing user - update last login
            await db.b2c_update_one(
                "users",
                {"_id": existing_user["_id"]},
                {
                    "$set": {
                        "last_login": datetime.utcnow(),
                        "picture": google_user.get('picture', existing_user.get('picture'))
                    }
                }
            )
            
            user_id = str(existing_user["_id"])
            full_name = existing_user.get("full_name", google_user['full_name'])
            
            logger.info(f"B2C user logged in: {email}")
        else:
            # New user - create account in B2C database
            new_user = {
                "google_id": google_id,
                "email": email,
                "full_name": google_user['full_name'],
                "given_name": google_user.get('given_name', ''),
                "family_name": google_user.get('family_name', ''),
                "picture": google_user.get('picture', ''),
                "locale": google_user.get('locale', 'en'),
                "is_active": True,
                "user_type": "b2c_user",
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                # B2C users don't have admin association
                "admin_id": None,
                "subdomain": None
            }
            
            user_id = await db.b2c_insert_one("users", new_user)
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user account"
                )
            
            full_name = google_user['full_name']
            logger.info(f"New B2C user created: {email}")

        # Create JWT session
        user_data = {
            "user_id": user_id,
            "user_type": "b2c_user",
            "email": email,
            "full_name": full_name,
            "google_id": google_id,
            "is_b2c": True  # Flag to identify B2C users
        }

        session_data = await auth_manager.create_user_session(user_data)

        # Log login activity in B2C database
        try:
            await db.b2c_insert_one("user_activity_log", {
                "user_id": ObjectId(user_id),
                "action": "login",
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "ip_address": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown"),
                    "login_method": "google_oauth"
                }
            })
        except Exception as e:
            logger.warning(f"Failed to log B2C user activity: {str(e)}")

        return B2CTokenResponse(
            success=True,
            data={
                "access_token": session_data["access_token"],
                "user_type": "b2c_user",
                "user": {
                    "user_id": user_id,
                    "email": email,
                    "full_name": full_name,
                    "picture": google_user.get('picture', ''),
                    "is_b2c": True
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C Google login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=B2CUserResponse)
async def get_b2c_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """Get current B2C user profile"""
    try:
        user_id = current_user.get("user_id")
        
        # Fetch full profile from B2C database
        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return B2CUserResponse(
            user_id=str(user["_id"]),
            email=user.get("email", ""),
            full_name=user.get("full_name", ""),
            picture=user.get("picture"),
            user_type="b2c_user"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get B2C profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch profile"
        )


@router.post("/logout")
async def b2c_logout(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """B2C user logout"""
    try:
        user_id = current_user.get("user_id")

        # Log logout activity in B2C database
        try:
            await db.b2c_insert_one("user_activity_log", {
                "user_id": ObjectId(user_id),
                "action": "logout",
                "timestamp": datetime.utcnow(),
                "metadata": {}
            })
        except Exception as e:
            logger.warning(f"Failed to log B2C logout: {str(e)}")

        # Invalidate session
        await auth_manager.invalidate_user_session(user_id)

        return {"success": True, "message": "Successfully logged out"}

    except Exception as e:
        logger.error(f"B2C logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/verify")
async def verify_b2c_token(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user)
):
    """Verify B2C JWT token and return user data"""
    return {
        "success": True,
        "data": {
            "user_id": current_user.get("user_id"),
            "user_type": current_user.get("user_type"),
            "email": current_user.get("email"),
            "full_name": current_user.get("full_name"),
            "is_b2c": True
        }
    }


# ==================== B2C Admin Authentication ====================

class B2CAdminLoginRequest(BaseModel):
    """Request model for B2C Admin login"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class B2CAdminResponse(BaseModel):
    """Response model for B2C Admin data"""
    admin_id: str
    username: str
    email: Optional[str] = None
    full_name: str
    user_type: str = "b2c_admin"


class B2CAdminSetupRequest(BaseModel):
    """Request model for initial B2C Admin setup"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: Optional[EmailStr] = None
    full_name: str = Field(..., min_length=2, max_length=100)
    setup_key: str = Field(..., description="Secret key to authorize admin creation")


async def get_current_b2c_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Dict[str, Any]:
    """Get current authenticated B2C Admin"""
    try:
        token = credentials.credentials
        user_data = await auth_manager.verify_token_and_get_user(token)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Ensure this is a B2C admin
        if user_data.get("user_type") != "b2c_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="B2C Admin access required"
            )

        return user_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C Admin authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/admin/login")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def b2c_admin_login(
    request: Request,
    login_data: B2CAdminLoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    B2C Admin login with username and password
    
    - Uses stoody-b2c database only
    - Separate from the main admin system
    - Returns JWT token for session
    """
    try:
        import bcrypt
        
        # Find admin in B2C database
        admin = await db.b2c_find_one("admins", {"username": login_data.username})
        
        if not admin:
            logger.warning(f"B2C Admin login attempt with unknown username: {login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Verify password
        stored_hash = admin.get("password_hash", "")
        if not stored_hash or not bcrypt.checkpw(
            login_data.password.encode('utf-8'),
            stored_hash.encode('utf-8') if isinstance(stored_hash, str) else stored_hash
        ):
            logger.warning(f"B2C Admin login failed - incorrect password: {login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Check if admin is active
        if not admin.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin account is disabled"
            )
        
        admin_id = str(admin["_id"])
        
        # Update last login
        await db.b2c_update_one(
            "admins",
            {"_id": admin["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Create JWT session
        user_data = {
            "user_id": admin_id,
            "user_type": "b2c_admin",
            "username": admin.get("username"),
            "email": admin.get("email", ""),
            "full_name": admin.get("full_name", "B2C Admin"),
            "is_b2c": True,
            "is_b2c_admin": True  # Flag to identify B2C admin
        }
        
        session_data = await auth_manager.create_user_session(user_data)
        
        # Log admin activity
        try:
            await db.b2c_insert_one("admin_activity_log", {
                "admin_id": admin["_id"],
                "action": "login",
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "ip_address": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown")
                }
            })
        except Exception as e:
            logger.warning(f"Failed to log B2C admin activity: {str(e)}")
        
        logger.info(f"B2C Admin logged in: {login_data.username}")
        
        return {
            "success": True,
            "data": {
                "access_token": session_data["access_token"],
                "user_type": "b2c_admin",
                "admin": {
                    "admin_id": admin_id,
                    "username": admin.get("username"),
                    "email": admin.get("email", ""),
                    "full_name": admin.get("full_name", "B2C Admin"),
                    "is_b2c_admin": True
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C Admin login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/admin/setup")
@limiter.limit("3/hour")
async def setup_b2c_admin(
    request: Request,
    setup_data: B2CAdminSetupRequest,
    db: DatabaseManager = Depends(get_database)
):
    """
    Initial B2C Admin account setup
    
    - Creates the single B2C admin account
    - Requires a setup key for security
    - Can only be used once (only one admin allowed)
    
    The setup key should be set in environment variable B2C_ADMIN_SETUP_KEY
    """
    import os
    import bcrypt
    
    try:
        # Verify setup key
        expected_key = os.getenv("B2C_ADMIN_SETUP_KEY", "stoody-b2c-admin-setup-2024")
        if setup_data.setup_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid setup key"
            )
        
        # Check if admin already exists
        existing_admin = await db.b2c_find_one("admins", {})
        if existing_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="B2C Admin already exists. Only one admin is allowed."
            )
        
        # Hash password
        password_hash = bcrypt.hashpw(
            setup_data.password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
        
        # Create admin document
        admin_doc = {
            "username": setup_data.username,
            "password_hash": password_hash,
            "email": setup_data.email,
            "full_name": setup_data.full_name,
            "user_type": "b2c_admin",
            "is_active": True,
            "is_b2c_admin": True,
            "created_at": datetime.utcnow(),
            "last_login": None,
            "permissions": {
                "manage_students": True,
                "manage_content": True,
                "view_analytics": True,
                "manage_settings": True
            }
        }
        
        admin_id = await db.b2c_insert_one("admins", admin_doc)
        
        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create admin account"
            )
        
        logger.info(f"B2C Admin account created: {setup_data.username}")
        
        return {
            "success": True,
            "message": "B2C Admin account created successfully",
            "data": {
                "admin_id": admin_id,
                "username": setup_data.username,
                "full_name": setup_data.full_name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C Admin setup error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin setup failed"
        )


@router.get("/admin/me", response_model=B2CAdminResponse)
async def get_b2c_admin_profile(
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get current B2C Admin profile"""
    try:
        admin_id = current_admin.get("user_id")
        
        # Fetch full profile from B2C database
        admin = await db.b2c_find_one("admins", {"_id": ObjectId(admin_id)})
        
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        return B2CAdminResponse(
            admin_id=str(admin["_id"]),
            username=admin.get("username", ""),
            email=admin.get("email"),
            full_name=admin.get("full_name", "B2C Admin"),
            user_type="b2c_admin"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get B2C Admin profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch admin profile"
        )


@router.get("/admin/dashboard/stats")
async def get_b2c_admin_dashboard_stats(
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get B2C Admin dashboard statistics"""
    try:
        # Get user count from B2C database
        users = await db.b2c_find("users", {})
        total_users = len(users)
        active_users = len([u for u in users if u.get("is_active", True)])
        
        # Get recent signups (last 7 days)
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_users = len([u for u in users if u.get("created_at", datetime.min) >= week_ago])
        
        # Get activity logs count
        activity_logs = await db.b2c_find("user_activity_log", {})
        total_logins = len([l for l in activity_logs if l.get("action") == "login"])
        
        return {
            "success": True,
            "data": {
                "total_students": total_users,
                "active_students": active_users,
                "new_signups_7d": recent_users,
                "total_logins": total_logins,
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"B2C Admin dashboard stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch dashboard statistics"
        )


@router.get("/admin/students")
async def get_b2c_students(
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database),
    page: int = 1,
    limit: int = 20
):
    """Get list of B2C students"""
    try:
        # Get all users from B2C database
        all_users = await db.b2c_find("users", {"user_type": "b2c_user"})
        
        logger.info(f"B2C Admin fetching students - found {len(all_users)} users")
        
        # Apply pagination
        total = len(all_users)
        start = (page - 1) * limit
        end = start + limit
        users = all_users[start:end]
        
        # Format response with full details
        students = []
        for user in users:
            students.append({
                "user_id": str(user["_id"]),
                "email": user.get("email", ""),
                "full_name": user.get("full_name", ""),
                "picture": user.get("picture"),
                "phone": user.get("phone"),
                "is_active": user.get("is_active", True),
                # Plan details
                "exam_type": user.get("exam_type"),
                "class_level": user.get("class_level"),
                "standard": user.get("standard"),
                "subjects": user.get("subjects", []),
                "plan_types": user.get("plan_types", []),
                # Onboarding status
                "onboarding_complete": user.get("onboarding_complete", False),
                "onboarding_completed_at": user.get("onboarding_completed_at").isoformat() if user.get("onboarding_completed_at") else None,
                # Timestamps
                "created_at": user.get("created_at", datetime.utcnow()).isoformat(),
                "last_login": user.get("last_login", datetime.utcnow()).isoformat() if user.get("last_login") else None
            })
        
        return {
            "success": True,
            "data": {
                "students": students,
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": (total + limit - 1) // limit
            }
        }
        
    except Exception as e:
        logger.error(f"Get B2C students error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch students"
        )



# ==================== B2C User Onboarding & Profile ====================

class B2COnboardingRequest(BaseModel):
    """Request model for B2C user onboarding"""
    exam_type: str = Field(..., description="JEE or NEET")
    class_level: str = Field(..., description="9, 10, 11, 12, or Dropper")
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=15)
    school_name: Optional[str] = None
    city: Optional[str] = None


class B2CProfileUpdateRequest(BaseModel):
    """Request model for B2C profile update"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = Field(None, min_length=10, max_length=15)
    school_name: Optional[str] = None
    city: Optional[str] = None
    exam_type: Optional[str] = None
    class_level: Optional[str] = None


@router.post("/profile/onboarding")
@limiter.limit("10/minute")
async def b2c_onboarding(
    request: Request,
    onboarding_data: B2COnboardingRequest,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Complete B2C user onboarding with plan selection and personal details
    
    - Updates user profile with exam type (JEE/NEET) and class level
    - Stores personal details (phone, school, city)
    - Marks onboarding as complete
    
    Note: Uses stoody-b2c database
    """
    try:
        user_id = current_user.get("user_id")
        
        # Validate exam type
        if onboarding_data.exam_type not in ["JEE", "NEET"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid exam type. Must be 'JEE' or 'NEET'"
            )
        
        # Validate class level
        valid_classes = ["9", "10", "11", "12", "Dropper"]
        if onboarding_data.class_level not in valid_classes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid class level. Must be one of: {', '.join(valid_classes)}"
            )
        
        # Prepare update document
        update_data = {
            "full_name": onboarding_data.full_name,
            "phone": onboarding_data.phone,
            "exam_type": onboarding_data.exam_type,
            "class_level": onboarding_data.class_level,
            "onboarding_complete": True,
            "onboarding_completed_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Add optional fields if provided
        if onboarding_data.school_name:
            update_data["school_name"] = onboarding_data.school_name
        if onboarding_data.city:
            update_data["city"] = onboarding_data.city
        
        # Determine subjects based on exam type
        if onboarding_data.exam_type == "JEE":
            update_data["subjects"] = ["Physics", "Chemistry", "Mathematics"]
        else:  # NEET
            update_data["subjects"] = ["Physics", "Chemistry", "Biology"]
        
        # Map class level to standard for content filtering
        if onboarding_data.class_level == "Dropper":
            update_data["standard"] = "12"  # Droppers see Class 12 content
            update_data["is_dropper"] = True
        else:
            update_data["standard"] = onboarding_data.class_level
            update_data["is_dropper"] = False
        
        # Create plan_types array for content filtering
        update_data["plan_types"] = [onboarding_data.exam_type]
        
        # Update user in B2C database
        result = await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update profile"
            )
        
        # Log onboarding activity
        try:
            await db.b2c_insert_one("user_activity_log", {
                "user_id": ObjectId(user_id),
                "action": "onboarding_complete",
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "exam_type": onboarding_data.exam_type,
                    "class_level": onboarding_data.class_level
                }
            })
        except Exception as e:
            logger.warning(f"Failed to log onboarding activity: {str(e)}")
        
        logger.info(f"B2C user onboarding complete: {user_id} - {onboarding_data.exam_type}/{onboarding_data.class_level}")
        
        return {
            "success": True,
            "message": "Onboarding completed successfully",
            "data": {
                "user_id": user_id,
                "exam_type": onboarding_data.exam_type,
                "class_level": onboarding_data.class_level,
                "subjects": update_data["subjects"],
                "standard": update_data["standard"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C onboarding error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Onboarding failed"
        )


@router.get("/profile")
async def get_b2c_user_full_profile(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """Get full B2C user profile including plan and personal details"""
    try:
        user_id = current_user.get("user_id")
        
        # Fetch full profile from B2C database
        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "data": {
                "user_id": str(user["_id"]),
                "email": user.get("email", ""),
                "full_name": user.get("full_name", ""),
                "picture": user.get("picture"),
                "phone": user.get("phone"),
                "school_name": user.get("school_name"),
                "city": user.get("city"),
                "exam_type": user.get("exam_type"),
                "class_level": user.get("class_level"),
                "standard": user.get("standard"),
                "subjects": user.get("subjects", []),
                "plan_types": user.get("plan_types", []),
                "is_dropper": user.get("is_dropper", False),
                "onboarding_complete": user.get("onboarding_complete", False),
                "created_at": user.get("created_at", datetime.utcnow()).isoformat(),
                "last_login": user.get("last_login", datetime.utcnow()).isoformat() if user.get("last_login") else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get B2C profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch profile"
        )


@router.put("/profile")
@limiter.limit("10/minute")
async def update_b2c_user_profile(
    request: Request,
    profile_data: B2CProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """Update B2C user profile"""
    try:
        user_id = current_user.get("user_id")
        
        # Build update document with only provided fields
        update_data = {"updated_at": datetime.utcnow()}
        
        if profile_data.full_name:
            update_data["full_name"] = profile_data.full_name
        if profile_data.phone:
            update_data["phone"] = profile_data.phone
        if profile_data.school_name is not None:
            update_data["school_name"] = profile_data.school_name
        if profile_data.city is not None:
            update_data["city"] = profile_data.city
        
        # Handle exam type change
        if profile_data.exam_type:
            if profile_data.exam_type not in ["JEE", "NEET"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid exam type. Must be 'JEE' or 'NEET'"
                )
            update_data["exam_type"] = profile_data.exam_type
            update_data["plan_types"] = [profile_data.exam_type]
            
            # Update subjects based on exam type
            if profile_data.exam_type == "JEE":
                update_data["subjects"] = ["Physics", "Chemistry", "Mathematics"]
            else:
                update_data["subjects"] = ["Physics", "Chemistry", "Biology"]
        
        # Handle class level change
        if profile_data.class_level:
            valid_classes = ["9", "10", "11", "12", "Dropper"]
            if profile_data.class_level not in valid_classes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid class level. Must be one of: {', '.join(valid_classes)}"
                )
            update_data["class_level"] = profile_data.class_level
            
            if profile_data.class_level == "Dropper":
                update_data["standard"] = "12"
                update_data["is_dropper"] = True
            else:
                update_data["standard"] = profile_data.class_level
                update_data["is_dropper"] = False
        
        # Update user in B2C database
        result = await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update profile"
            )
        
        logger.info(f"B2C user profile updated: {user_id}")
        
        return {
            "success": True,
            "message": "Profile updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update B2C profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.get("/profile/check-onboarding")
async def check_b2c_onboarding_status(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """Check if B2C user has completed onboarding"""
    try:
        user_id = current_user.get("user_id")
        
        user = await db.b2c_find_one(
            "users", 
            {"_id": ObjectId(user_id)},
            {"onboarding_complete": 1, "exam_type": 1, "class_level": 1}
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        is_complete = user.get("onboarding_complete", False)
        
        return {
            "success": True,
            "data": {
                "onboarding_complete": is_complete,
                "exam_type": user.get("exam_type") if is_complete else None,
                "class_level": user.get("class_level") if is_complete else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Check onboarding status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check onboarding status"
        )

