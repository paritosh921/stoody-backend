"""
B2C Authentication API for Stoody
Google OAuth-based authentication for B2C users
Uses stoody-b2c database (completely separate from skillbot_db)
"""

import logging
import os
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

        # Clear any user-level revocation so the new token is accepted
        from core.token_blacklist import token_blacklist
        token_blacklist.clear_user_revocation(user_id)

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

    except Exception as e:
        logger.error(f"B2C Google login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


# ==================== Desktop Agent Redirect Flow ====================

@router.get("/google/authorize")
async def b2c_google_authorize(redirect_uri: str):
    """
    Initiates Google OAuth flow for Desktop Agent
    Redirects user to Google's consent page
    """
    from fastapi.responses import RedirectResponse
    import urllib.parse
    
    # Base Google Auth URL
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    
    # Use the production API URL (https://api.stoody.in is the deployed backend)
    import os
    api_base = os.getenv("PUBLIC_API_URL", "https://api.stoody.in/api/v1")
    callback_url = f"{api_base}/b2c/google/callback"
    
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": callback_url,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        # Pass the agent's local callback as state so we know where to send the token
        "state": redirect_uri
    }
    
    url = f"{auth_url}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url)


@router.get("/google/callback")
async def b2c_google_callback(
    code: str, 
    state: str, 
    request: Request,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Handles callback from Google, exchanges code for token,
    and redirects back to the Desktop Agent
    """
    from fastapi.responses import RedirectResponse
    import httpx
    
    agent_callback = state  # e.g., http://localhost:8001/api/v1/auth/callback
    
    try:
        # Exchange code for tokens
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            # Use same production URL as authorize endpoint
            "redirect_uri": f"{os.getenv('PUBLIC_API_URL', 'https://api.stoody.in/api/v1')}/b2c/google/callback"
        }
        
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(token_url, data=data)
            token_resp.raise_for_status()
            tokens = token_resp.json()
            
        id_token_str = tokens.get("id_token")
        
        # Verify token and get user info
        google_user = await verify_google_token(id_token_str)
        if not google_user:
             return RedirectResponse(f"{agent_callback}?error=invalid_token")

        google_id = google_user['google_id']
        email = google_user['email']
        
        # --- Database Logic (Same as b2c_google_login) ---
        existing_user = await db.b2c_find_one("users", {"google_id": google_id})
        
        if existing_user:
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
        else:
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
                "admin_id": None,
                "subdomain": None
            }
            user_id = await db.b2c_insert_one("users", new_user)
            full_name = google_user['full_name']

        # Create JWT session
        user_data = {
            "user_id": user_id,
            "user_type": "b2c_user",
            "email": email,
            "full_name": full_name,
            "google_id": google_id,
            "is_b2c": True
        }
        session_data = await auth_manager.create_user_session(user_data)
        access_token = session_data["access_token"]

        # Clear any user-level revocation so the new token is accepted
        from core.token_blacklist import token_blacklist
        token_blacklist.clear_user_revocation(user_id)

        # Redirect back to Agent with token (URL-encode to handle special chars)
        import urllib.parse
        encoded_token = urllib.parse.quote(access_token, safe='')
        encoded_email = urllib.parse.quote(email, safe='')
        return RedirectResponse(f"{agent_callback}?token={encoded_token}&user_id={user_id}&email={encoded_email}")
        
    except Exception as e:
        logger.error(f"Google Callback Error: {e}")
        # Return error detail to help debugging (in production be careful exposing internal errors, 
        # but for now we need to know why it failed)
        import urllib.parse
        error_msg = urllib.parse.quote(str(e))
        return RedirectResponse(f"{agent_callback}?error=login_failed&detail={error_msg}")


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

        # User-level revocation: invalidate ALL tokens for this user
        from core.token_blacklist import token_blacklist
        token_blacklist.revoke_user(user_id)

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
        
        normalized_username = login_data.username.strip()
        username_lower = normalized_username.lower()

        # Find admin in B2C database (case-insensitive)
        admin = await db.b2c_find_one("admins", {"username_lower": username_lower})
        if not admin:
            admin = await db.b2c_find_one(
                "admins",
                {"username": normalized_username},
                collation={"locale": "en", "strength": 2}
            )
        
        if not admin:
            logger.warning(f"B2C Admin login attempt with unknown username: {normalized_username}")
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
            logger.warning(f"B2C Admin login failed - incorrect password: {normalized_username}")
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

        # Clear any user-level revocation so the new token is accepted
        from core.token_blacklist import token_blacklist
        token_blacklist.clear_user_revocation(admin_id)

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
        
        logger.info(f"B2C Admin logged in: {normalized_username}")
        
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
            "username": setup_data.username.strip(),
            "username_lower": setup_data.username.strip().lower(),
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
        
        logger.info(f"B2C Admin account created: {setup_data.username.strip()}")
        
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
                # Personal details
                "school_name": user.get("school_name"),
                "city": user.get("city"),
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

                # GDPR Consent fields
                "consent_completed": user.get("consent_completed", False),
                "is_minor": user.get("is_minor", False),
                "has_parental_consent": user.get("has_parental_consent", False),
                "parent_info": user.get("parent_info"),
                "gdpr_consent": user.get("gdpr_consent"),
                "ai_personalization_consent": user.get("ai_personalization_consent"),
                "marketing_consent": user.get("marketing_consent"),
                "consent_timestamp": user.get("consent_timestamp"),
                # Timestamps
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


# ==================== B2C Admin Leaderboard & Progress ====================

@router.get("/admin/leaderboard/progress")
@limiter.limit("30/minute")
async def get_b2c_student_progress(
    request: Request,
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get B2C student progress for leaderboard
    Uses stoody-b2c database only
    """
    try:
        # Get all B2C users (students) from the b2c database
        users = await db.b2c_find(
            "users",
            {"is_active": True},
            projection={"google_id": 0}
        )
        
        progress_data = []
        for user in users:
            user_id = user.get("_id")
            user_id_str = str(user_id)
            
            # Get user's test attempts from b2c database
            attempts = await db.b2c_find(
                "student_test_attempts",
                {"student_id": user_id_str}
            )
            
            # Calculate statistics
            total_attempts = len(attempts)
            total_score = sum(a.get("score", 0) for a in attempts)
            total_points = sum(a.get("total_points", 0) for a in attempts)
            avg_score = (total_score / total_points * 100) if total_points > 0 else 0
            
            # Calculate problems solved from question_progress
            question_progress = await db.b2c_find(
                "question_progress",
                {"user_id": user_id_str, "is_correct": True}
            )
            problems_solved = len(question_progress)
            
            # Calculate total time spent
            total_time = sum(a.get("time_taken", 0) for a in attempts) / 60  # Convert to minutes
            
            # Get streak and XP from user document
            streak_days = user.get("streak_days", 0)
            level = user.get("level", 1)
            xp = user.get("xp", 0)
            
            progress_data.append({
                "student_id": user_id_str,
                "student_name": user.get("full_name", user.get("given_name", "Unknown")),
                "email": user.get("email", ""),
                "grade": user.get("class_level", "Unknown"),
                "section": user.get("exam_type", "Unknown"),
                "total_sessions": total_attempts,
                "total_time_spent": int(total_time),
                "problems_solved": problems_solved,
                "average_score": round(avg_score, 1),
                "last_active_at": user.get("last_login"),
                "streak_days": streak_days,
                "level": level,
                "xp": xp,
                "is_online": user.get("is_online", False)
            })
        
        # Sort by average score descending
        progress_data.sort(key=lambda x: x["average_score"], reverse=True)
        
        return {
            "success": True,
            "data": progress_data
        }
        
    except Exception as e:
        logger.error(f"Get B2C student progress error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get student progress"
        )


@router.get("/admin/leaderboard/test-attempts")
@limiter.limit("30/minute")
async def get_b2c_test_attempts(
    request: Request,
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all B2C test attempts for admin leaderboard
    Uses stoody-b2c database only
    """
    try:
        # Get all test attempts from B2C database
        attempts = await db.b2c_find(
            "student_test_attempts",
            {},
            sort=[("submitted_at", -1)]
        )
        
        # Format response
        formatted_attempts = []
        for attempt in attempts:
            # Get student info
            student_id = attempt.get("student_id")
            student = await db.b2c_find_one("users", {"_id": ObjectId(student_id)}) if student_id else None
            
            formatted_attempts.append({
                "attempt_id": str(attempt.get("_id")),
                "student_id": student_id,
                "student_name": student.get("full_name", "Unknown") if student else attempt.get("student_name", "Unknown"),
                "student_grade": student.get("class_level", "Unknown") if student else attempt.get("student_grade", "Unknown"),
                "document_id": attempt.get("document_id"),
                "document_title": attempt.get("document_title", "Unknown Test"),
                "subject": attempt.get("subject", "Unknown"),
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
                "submitted_at": attempt.get("submitted_at").isoformat() if attempt.get("submitted_at") else None
            })
        
        logger.info(f"Retrieved {len(formatted_attempts)} B2C test attempts")
        
        return {
            "success": True,
            "data": {
                "attempts": formatted_attempts,
                "total": len(formatted_attempts)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get B2C test attempts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/leaderboard/test-attempts/{attempt_id}/toggle-reattempt")
@limiter.limit("30/minute")
async def toggle_b2c_reattempt(
    request: Request,
    attempt_id: str,
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Toggle the can_reattempt flag for a B2C test attempt
    Uses stoody-b2c database only
    """
    try:
        from bson import ObjectId
        
        # Get the attempt from B2C database
        attempt = await db.b2c_find_one("student_test_attempts", {"_id": ObjectId(attempt_id)})
        if not attempt:
            raise HTTPException(status_code=404, detail="Test attempt not found")
        
        # Toggle the flag
        new_value = not attempt.get("can_reattempt", False)
        
        # Update in B2C database
        await db.b2c_update_one(
            "student_test_attempts",
            {"_id": ObjectId(attempt_id)},
            {"$set": {"can_reattempt": new_value}}
        )
        
        logger.info(f"Toggled B2C re-attempt for attempt {attempt_id} to {new_value}")
        
        return {
            "success": True,
            "message": f"Re-attempt {'enabled' if new_value else 'disabled'} successfully",
            "data": {
                "can_reattempt": new_value
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle B2C re-attempt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== GDPR Consent Endpoints ====================

class AdultConsentRequest(BaseModel):
    """Request model for adult GDPR consent"""
    is_minor: bool = False
    gdpr_consent: bool = Field(..., description="Privacy policy consent")
    ai_personalization_consent: bool = Field(..., description="AI processing consent")
    marketing_consent: bool = Field(False, description="Marketing communications consent")
    consent_timestamp: str = Field(..., description="ISO timestamp of consent")
    consent_version: str = Field("1.0", description="Version of consent document")


class ParentInfoModel(BaseModel):
    """Model for parent/guardian information"""
    full_name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., description="Parent's email address")
    phone: str = Field(..., min_length=10, max_length=20)
    country: str = Field(..., description="Country of residence")
    relationship: str = Field(..., description="Relationship to child")


class ParentalConsentModel(BaseModel):
    """Model for parental consent checkboxes"""
    is_legal_guardian: bool
    consent_data_processing: bool
    consent_ai_analysis: bool
    consent_international_transfers: bool


class ParentalConsentRequest(BaseModel):
    """Request model for parental GDPR consent"""
    is_minor: bool = True
    parent_info: ParentInfoModel
    parental_consent: ParentalConsentModel
    digital_signature: str = Field(..., description="Parent's full name as digital signature")
    signature_date: str = Field(..., description="ISO timestamp of signature")
    consent_timestamp: str = Field(..., description="ISO timestamp of consent")
    consent_version: str = Field("1.0", description="Version of consent document")
    scc_version: str = Field("2021/914", description="EU SCC version")


@router.post("/consent/adult")
@limiter.limit("5/minute")
async def submit_adult_consent(
    request: Request,
    consent_data: AdultConsentRequest,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Submit GDPR consent for adult users (16+)
    
    - Records user's consent for privacy policy and AI processing
    - Stores consent timestamp and version for audit trail
    - Logs IP address for compliance
    """
    try:
        user_id = current_user.get("user_id")
        
        # Validate required consents
        if not consent_data.gdpr_consent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Privacy policy consent is required"
            )
        if not consent_data.ai_personalization_consent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="AI personalization consent is required for our service"
            )
        
        # Prepare consent record
        consent_record = {
            "consent_completed": True,
            "is_minor": False,
            "has_parental_consent": False,
            "gdpr_consent": consent_data.gdpr_consent,
            "ai_personalization_consent": consent_data.ai_personalization_consent,
            "marketing_consent": consent_data.marketing_consent,
            "consent_timestamp": consent_data.consent_timestamp,
            "consent_version": consent_data.consent_version,
            "consent_ip": request.client.host if request.client else "unknown",
            "consent_user_agent": request.headers.get("user-agent", "unknown"),
            "updated_at": datetime.utcnow()
        }
        
        # Update user in B2C database
        await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": consent_record}
        )
        
        # Log consent activity for audit trail
        await db.b2c_insert_one("consent_audit_log", {
            "user_id": ObjectId(user_id),
            "consent_type": "adult",
            "action": "consent_granted",
            "timestamp": datetime.utcnow(),
            "ip_address": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "consent_version": consent_data.consent_version,
            "consent_details": {
                "gdpr_consent": consent_data.gdpr_consent,
                "ai_personalization_consent": consent_data.ai_personalization_consent,
                "marketing_consent": consent_data.marketing_consent,
            }
        })
        
        logger.info(f"Adult GDPR consent recorded for user: {user_id}")
        
        return {
            "success": True,
            "message": "Consent recorded successfully",
            "data": {
                "consent_completed": True,
                "is_minor": False
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adult consent error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record consent"
        )


@router.post("/consent/parental")
@limiter.limit("5/minute")
async def submit_parental_consent(
    request: Request,
    consent_data: ParentalConsentRequest,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Submit parental consent for minors (under 16) under GDPR Article 8
    
    - Records parent/guardian information
    - Stores all consent checkboxes
    - Records digital signature and audit trail
    - Compliant with EU SCC requirements
    """
    try:
        user_id = current_user.get("user_id")
        
        # Validate all required parental consents
        pc = consent_data.parental_consent
        if not all([pc.is_legal_guardian, pc.consent_data_processing, 
                    pc.consent_ai_analysis, pc.consent_international_transfers]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All parental consent checkboxes must be checked"
            )
        
        # Validate digital signature matches parent name
        if consent_data.digital_signature.lower().strip() != consent_data.parent_info.full_name.lower().strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Digital signature must match parent's full name"
            )
        
        # Prepare parent info
        parent_info = {
            "full_name": consent_data.parent_info.full_name,
            "email": consent_data.parent_info.email,
            "phone": consent_data.parent_info.phone,
            "country": consent_data.parent_info.country,
            "relationship": consent_data.parent_info.relationship,
        }
        
        # Prepare consent record
        consent_record = {
            "consent_completed": True,
            "is_minor": True,
            "has_parental_consent": True,
            "parent_info": parent_info,
            "parental_consent": {
                "is_legal_guardian": pc.is_legal_guardian,
                "consent_data_processing": pc.consent_data_processing,
                "consent_ai_analysis": pc.consent_ai_analysis,
                "consent_international_transfers": pc.consent_international_transfers,
            },
            "digital_signature": consent_data.digital_signature,
            "signature_date": consent_data.signature_date,
            "consent_timestamp": consent_data.consent_timestamp,
            "consent_version": consent_data.consent_version,
            "scc_version": consent_data.scc_version,
            "consent_ip": request.client.host if request.client else "unknown",
            "consent_user_agent": request.headers.get("user-agent", "unknown"),
            "updated_at": datetime.utcnow()
        }
        
        # Update user in B2C database
        await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": consent_record}
        )
        
        # Log consent for audit trail (critical for GDPR compliance)
        await db.b2c_insert_one("consent_audit_log", {
            "user_id": ObjectId(user_id),
            "consent_type": "parental",
            "action": "consent_granted",
            "timestamp": datetime.utcnow(),
            "ip_address": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "consent_version": consent_data.consent_version,
            "scc_version": consent_data.scc_version,
            "parent_info": parent_info,
            "consent_details": {
                "is_legal_guardian": pc.is_legal_guardian,
                "consent_data_processing": pc.consent_data_processing,
                "consent_ai_analysis": pc.consent_ai_analysis,
                "consent_international_transfers": pc.consent_international_transfers,
            },
            "digital_signature": consent_data.digital_signature,
            "signature_date": consent_data.signature_date,
        })
        
        logger.info(f"Parental consent recorded for minor user: {user_id}")
        
        return {
            "success": True,
            "message": "Parental consent recorded successfully",
            "data": {
                "consent_completed": True,
                "is_minor": True,
                "has_parental_consent": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parental consent error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record parental consent"
        )


@router.get("/parent/dashboard")
async def get_parent_dashboard(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get parent dashboard data for minor accounts
    
    Returns:
    - Parent/guardian info
    - Child's data summary
    - Consent record details
    """
    try:
        user_id = current_user.get("user_id")
        
        # Fetch user profile
        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        if not user.get("is_minor") or not user.get("has_parental_consent"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is only for accounts with parental consent"
            )
        
        # Calculate data summary
        # Count learning sessions
        learning_sessions = await db.b2c_find("user_activity_log", {"user_id": ObjectId(user_id)})
        
        # Prepare data summary
        data_summary = {
            "learning_sessions": len([l for l in learning_sessions if l.get("action") == "learning_session"]),
            "handwriting_samples": 0,  # Placeholder - integrate with actual data
            "audio_recordings": 0,  # Placeholder - integrate with actual data
            "practice_tests": len([l for l in learning_sessions if l.get("action") in ["test_submitted", "practice_submitted"]]),
            "total_study_hours": 0,  # Placeholder - calculate from sessions
            "last_active": user.get("last_login", datetime.utcnow()).isoformat() if user.get("last_login") else "Never"
        }
        
        # Get consent record
        consent_record = None
        consent_log = await db.b2c_find_one(
            "consent_audit_log",
            {"user_id": ObjectId(user_id), "action": "consent_granted"},
            sort=[("timestamp", -1)]
        )
        if consent_log:
            consent_record = {
                "consent_timestamp": consent_log.get("timestamp", datetime.utcnow()).isoformat(),
                "consent_version": consent_log.get("consent_version", "1.0"),
                "scc_version": consent_log.get("scc_version", "2021/914"),
                "ip_address": "Logged securely"  # Don't expose actual IP
            }
        
        return {
            "success": True,
            "data": {
                "parent_info": user.get("parent_info"),
                "data_summary": data_summary,
                "consent_record": consent_record
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parent dashboard error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch parent dashboard"
        )


@router.post("/parent/export-data")
@limiter.limit("3/hour")
async def request_data_export(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Request data export (GDPR data portability)
    
    Creates an export request that will be processed and emailed to the parent
    """
    try:
        user_id = current_user.get("user_id")
        
        # Verify user has parental consent
        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})
        
        if not user or not user.get("has_parental_consent"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Data export requires parental consent"
            )
        
        # Create export request
        await db.b2c_insert_one("data_export_requests", {
            "user_id": ObjectId(user_id),
            "parent_email": user.get("parent_info", {}).get("email"),
            "status": "pending",
            "requested_at": datetime.utcnow(),
            "ip_address": request.client.host if request.client else "unknown"
        })
        
        # Log activity
        await db.b2c_insert_one("consent_audit_log", {
            "user_id": ObjectId(user_id),
            "consent_type": "parental",
            "action": "data_export_requested",
            "timestamp": datetime.utcnow(),
            "ip_address": request.client.host if request.client else "unknown"
        })
        
        logger.info(f"Data export requested for user: {user_id}")
        
        return {
            "success": True,
            "message": "Data export request submitted. You will receive an email within 24 hours."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data export request error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit data export request"
        )


@router.post("/parent/withdraw-consent")
@limiter.limit("3/hour")
async def withdraw_consent(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Withdraw parental consent
    
    - Suspends the child's account
    - Queues data for deletion
    - Sends confirmation email
    """
    try:
        user_id = current_user.get("user_id")
        
        # Update user to suspended state
        await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": {
                "is_active": False,
                "consent_withdrawn": True,
                "consent_withdrawn_at": datetime.utcnow(),
                "deletion_scheduled": True,
                "deletion_scheduled_at": datetime.utcnow()
            }}
        )
        
        # Log consent withdrawal
        await db.b2c_insert_one("consent_audit_log", {
            "user_id": ObjectId(user_id),
            "consent_type": "parental",
            "action": "consent_withdrawn",
            "timestamp": datetime.utcnow(),
            "ip_address": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")
        })
        
        # Invalidate user session
        await auth_manager.invalidate_user_session(user_id)
        
        logger.info(f"Parental consent withdrawn for user: {user_id}")
        
        return {
            "success": True,
            "message": "Consent withdrawn. Account suspended and data queued for deletion."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Consent withdrawal error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to withdraw consent"
        )


@router.delete("/parent/delete-data")
@limiter.limit("1/hour")
async def delete_all_data(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Delete all user data (GDPR Right to be Forgotten)
    
    - Queues all data for permanent deletion
    - Keeps only legal compliance logs
    - Sends confirmation email
    """
    try:
        user_id = current_user.get("user_id")
        
        # Verify user has parental consent
        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})
        
        if not user or not user.get("has_parental_consent"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This action requires parental consent verification"
            )
        
        # Create deletion request (actual deletion would be processed by a background job)
        await db.b2c_insert_one("data_deletion_requests", {
            "user_id": ObjectId(user_id),
            "parent_email": user.get("parent_info", {}).get("email"),
            "status": "pending",
            "requested_at": datetime.utcnow(),
            "ip_address": request.client.host if request.client else "unknown"
        })
        
        # Update user status
        await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": {
                "is_active": False,
                "deletion_requested": True,
                "deletion_requested_at": datetime.utcnow()
            }}
        )
        
        # Log for compliance (this log is kept even after deletion)
        await db.b2c_insert_one("consent_audit_log", {
            "user_id": ObjectId(user_id),
            "consent_type": "parental",
            "action": "data_deletion_requested",
            "timestamp": datetime.utcnow(),
            "ip_address": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "note": "Compliance log - retained for legal purposes"
        })
        
        # Invalidate user session
        await auth_manager.invalidate_user_session(user_id)
        
        logger.info(f"Data deletion requested for user: {user_id}")
        
        return {
            "success": True,
            "message": "Data deletion request submitted. All data will be permanently deleted within 30 days."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit data deletion request"
        )

