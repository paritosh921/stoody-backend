"""
Cookie-Based Authentication Endpoints for Stoody
Provides secure httpOnly cookie authentication as alternative to localStorage
"""

import logging
from datetime import datetime
from typing import Dict, Any
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr

from core.database import DatabaseManager
from core.auth import AuthManager
from core.cookie_auth import (
    cookie_auth_manager,
    create_response_with_cookie,
    get_current_user_dual_auth
)
from core.csrf_protection import csrf_protection, csrf_protect
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
from api.v1.auth_async import (
    get_database,
    get_auth_manager,
    StudentLoginRequest,
    _resolve_tenant_for_auth,
    _get_tenant_db_or_503,
    _get_request_subdomain,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class SSOExchangeRequest(BaseModel):
    """SSO token exchange request (POST instead of URL parameter)"""
    token: str = Field(..., min_length=10)
    user_id: str = Field(..., min_length=1)


async def _get_current_user_dual_auth_optional(
    request: Request,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> Dict[str, Any] | None:
    """
    Best-effort auth resolver for idempotent logout.

    Cookie logout is called defensively during login/logout cleanup, so lack of a
    current session should not turn into a 401 that pollutes the browser console.
    """
    try:
        return await get_current_user_dual_auth(request, auth_manager)
    except HTTPException:
        return None


@router.get("/csrf-token")
async def get_csrf_token(request: Request):
    """
    Get CSRF token for the current session
    This endpoint generates a new CSRF token and sets it as a cookie
    
    Returns:
        JSON response with CSRF token
    """
    # Generate CSRF token
    csrf_token = csrf_protection.generate_csrf_token()
    
    # Create response with CSRF token in body
    response_data = {
        "success": True,
        "csrf_token": csrf_token
    }
    
    response = JSONResponse(content=response_data)
    
    # Set CSRF token in cookie (double-submit pattern)
    csrf_protection.set_csrf_cookie(response, csrf_token)
    
    logger.debug("CSRF token generated and set")
    
    return response


@router.post("/student/cookie-login")
async def student_cookie_login(
    request: Request,
    login_data: StudentLoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Student login with cookie-based authentication
    Sets JWT token in httpOnly cookie instead of returning it in response
    
    Returns:
        JSON response with user data (NO token in body)
    """
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
            "enabled_features": enabled_features,
            "enabled_features_v2": enabled_features_v2,
        }

        session_data = await auth_manager.create_user_session(student_data)

        # Generate pen token
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

        # Update student status
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
                    "user_agent": request.headers.get("user-agent", "unknown"),
                    "auth_method": "cookie"
                }
            })
        except Exception as e:
            logger.warning(f"Failed to track student login: {str(e)}")

        # Generate CSRF token
        csrf_token = csrf_protection.generate_csrf_token()

        # Create response data (NO token in body)
        response_data = {
            "success": True,
            "data": {
                "user_type": "student",
                "user": session_data["user"],
                "requires_password_change": student.get("requires_password_change", False),
                "pen_token": pen_token,
                "pen_token_expires_at": pen_token_expires_at.isoformat() if pen_token_expires_at else None,
                "csrf_token": csrf_token
            }
        }

        # Create response with auth cookie
        response = create_response_with_cookie(
            response_data,
            session_data["access_token"]
        )
        
        # Set CSRF cookie
        csrf_protection.set_csrf_cookie(response, csrf_token)

        logger.info(f"Student logged in via cookie: {normalized_username}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Student cookie login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/sso-exchange")
async def sso_token_exchange(
    request: Request,
    sso_data: SSOExchangeRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    SSO token exchange endpoint (POST-based for security)
    Desktop app POSTs token, backend validates and sets httpOnly cookie
    
    This replaces the insecure URL parameter approach
    
    Args:
        sso_data: SSO token and user ID
        
    Returns:
        JSON response with success status
    """
    try:
        # Verify SSO token
        user_data = await auth_manager.verify_token_and_get_user(sso_data.token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid SSO token"
            )
            
        # Verify user ID matches
        if user_data.get("user_id") != sso_data.user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID mismatch"
            )
        
        # Generate CSRF token
        csrf_token = csrf_protection.generate_csrf_token()
        
        # Create response data
        response_data = {
            "success": True,
            "data": {
                "user": user_data,
                "csrf_token": csrf_token
            }
        }
        
        # Create response with auth cookie
        response = create_response_with_cookie(
            response_data,
            sso_data.token
        )
        
        # Set CSRF cookie
        csrf_protection.set_csrf_cookie(response, csrf_token)
        
        logger.info(f"SSO token exchange successful for user: {sso_data.user_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SSO token exchange error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SSO exchange failed"
        )


@router.post("/cookie-logout")
async def cookie_logout(
    request: Request,
    current_user: Dict[str, Any] | None = Depends(_get_current_user_dual_auth_optional),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Logout endpoint that clears httpOnly cookie
    
    Returns:
        JSON response with success status
    """
    try:
        user_id = current_user.get("user_id") if current_user else None
        user_type = current_user.get("user_type") if current_user else None
        
        # Get token from cookie for blacklisting
        token = cookie_auth_manager.get_token_from_cookie(request)
        
        # Revoke token in blacklist
        from core.token_blacklist import token_blacklist
        if token:
            token_blacklist.revoke(token, expiry_seconds=86400)

        # User-level revocation (Redis): invalidate ALL tokens for this user
        from core.token_blacklist import revoke_user_session
        if user_id:
            await revoke_user_session(auth_manager.cache_manager, user_id)
            logger.info(f"Token + user-level revocation for user {user_id}")
        
        # For students, update status
        if user_type == "student":
            try:
                from api.v1.auth_async import _get_tenant_db_from_user
                tenant_db = await _get_tenant_db_from_user(db, current_user)
                
                if tenant_db is not None:
                    await tenant_db["students"].update_one(
                        {"_id": ObjectId(user_id)},
                        {"$set": {"is_online": False}}
                    )
                    
                    # Log session end
                    student = await tenant_db["students"].find_one({"_id": ObjectId(user_id)})
                    last_login = student.get("last_login") if student else None
                    
                    session_duration = 0
                    if last_login:
                        session_duration = (datetime.utcnow() - last_login).total_seconds()
                    
                    await tenant_db["student_activity_log"].insert_one({
                        "student_id": ObjectId(user_id),
                        "action": "logout",
                        "timestamp": datetime.utcnow(),
                        "metadata": {
                            "session_duration": session_duration,
                            "auth_method": "cookie"
                        }
                    })
            except Exception as e:
                logger.warning(f"Failed to track student logout: {str(e)}")
        
        # Invalidate session
        if user_id:
            await auth_manager.invalidate_user_session(user_id)
        
        # Create response
        response_data = {
            "success": True,
            "message": "Successfully logged out"
        }
        
        response = JSONResponse(content=response_data)
        
        # Clear auth cookie
        cookie_auth_manager.clear_auth_cookie(response)
        
        # Clear CSRF cookie
        response.delete_cookie(
            key="stoody_csrf_token",
            path="/",
            domain=None
        )
        
        if user_id:
            logger.info(f"User logged out via cookie: {user_id}")
        else:
            logger.debug("Cookie logout called without an active authenticated session")
        
        return response
        
    except Exception as e:
        logger.error(f"Cookie logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/refresh")
async def refresh_token(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user_dual_auth),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Refresh JWT token and update cookie
    Extends the session without requiring re-authentication
    
    Returns:
        JSON response with success status
    """
    try:
        # Create new session with refreshed token
        session_data = await auth_manager.create_user_session(current_user)
        
        # Generate new CSRF token
        csrf_token = csrf_protection.generate_csrf_token()
        
        # Create response data
        response_data = {
            "success": True,
            "message": "Token refreshed successfully",
            "csrf_token": csrf_token
        }
        
        # Create response with new auth cookie
        response = create_response_with_cookie(
            response_data,
            session_data["access_token"]
        )
        
        # Set new CSRF cookie
        csrf_protection.set_csrf_cookie(response, csrf_token)
        
        logger.debug(f"Token refreshed for user: {current_user.get('user_id')}")
        
        return response
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )
