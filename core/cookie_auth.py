"""
Cookie-Based Authentication Module for Stoody
Provides httpOnly cookie support for secure JWT token storage
"""

import logging
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse

from config_async import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    DEBUG_MODE
)

logger = logging.getLogger(__name__)

# Cookie configuration
COOKIE_NAME = "stoody_auth_token"
COOKIE_MAX_AGE = JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
COOKIE_HTTPONLY = True
COOKIE_SECURE = not DEBUG_MODE  # Only send over HTTPS in production
COOKIE_SAMESITE = "strict"  # CSRF protection
COOKIE_DOMAIN = None  # Set to ".stoody.in" for subdomain support in production
COOKIE_PATH = "/"

class CookieAuthManager:
    """Manage cookie-based authentication"""
    
    def __init__(self):
        self.cookie_name = COOKIE_NAME
        self.max_age = COOKIE_MAX_AGE
        
    def set_auth_cookie(
        self,
        response: Response,
        token: str,
        max_age: Optional[int] = None
    ) -> None:
        """
        Set authentication token in httpOnly cookie
        
        Args:
            response: FastAPI Response object
            token: JWT token to store
            max_age: Optional custom max age in seconds
        """
        if not token:
            raise ValueError("Token cannot be empty")
            
        cookie_max_age = max_age or self.max_age
        
        # Set httpOnly cookie with security settings
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            max_age=cookie_max_age,
            expires=datetime.now(timezone.utc) + timedelta(seconds=cookie_max_age),
            path=COOKIE_PATH,
            domain=COOKIE_DOMAIN,
            secure=COOKIE_SECURE,  # HTTPS only in production
            httponly=COOKIE_HTTPONLY,  # Prevents JavaScript access
            samesite=COOKIE_SAMESITE  # CSRF protection
        )
        
        logger.debug(f"Set auth cookie with max_age={cookie_max_age}s")
        
    def clear_auth_cookie(self, response: Response) -> None:
        """
        Clear authentication cookie (logout)
        
        Args:
            response: FastAPI Response object
        """
        response.delete_cookie(
            key=self.cookie_name,
            path=COOKIE_PATH,
            domain=COOKIE_DOMAIN,
            secure=COOKIE_SECURE,
            httponly=COOKIE_HTTPONLY,
            samesite=COOKIE_SAMESITE
        )
        
        logger.debug("Cleared auth cookie")
        
    def get_token_from_cookie(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from cookie
        
        Args:
            request: FastAPI Request object
            
        Returns:
            JWT token string or None if not found
        """
        token = request.cookies.get(self.cookie_name)
        
        if token:
            logger.debug("Token found in cookie")
        else:
            logger.debug("No token found in cookie")
            
        return token
        
    async def get_current_user_from_cookie(
        self,
        request: Request,
        auth_manager
    ) -> Optional[Dict[str, Any]]:
        """
        Get current user from cookie token

        Args:
            request: FastAPI Request object
            auth_manager: AuthManager instance for token verification

        Returns:
            User data dict or None if invalid/missing token
        """
        token = self.get_token_from_cookie(request)

        if not token:
            return None

        # Check token-level blacklist (same check as Bearer path)
        from core.token_blacklist import token_blacklist
        if token_blacklist.is_revoked(token):
            logger.info("Cookie token is revoked (token-level)")
            return None

        # Verify token and get user data
        try:
            user_data = await auth_manager.verify_token_and_get_user(token)

            if not user_data:
                logger.warning("Invalid token in cookie")
                return None

            # Check user-level revocation in Redis (cross-client logout)
            from core.token_blacklist import is_user_session_revoked
            user_id = user_data.get("user_id")
            if user_id and await is_user_session_revoked(
                auth_manager.cache_manager, user_id
            ):
                logger.info(f"Cookie token rejected: user {user_id} revoked (user-level)")
                return None

            return user_data

        except Exception as e:
            logger.error(f"Cookie auth error: {str(e)}")
            return None


# Global instance
cookie_auth_manager = CookieAuthManager()


async def get_current_user_dual_auth(
    request: Request,
    auth_manager
) -> Dict[str, Any]:
    """
    Dual authentication: Try cookie first, then Authorization header
    This provides backward compatibility during migration
    
    Args:
        request: FastAPI Request object
        auth_manager: AuthManager instance
        
    Returns:
        User data dict
        
    Raises:
        HTTPException: If authentication fails
    """
    # Try cookie authentication first (preferred)
    user_data = await cookie_auth_manager.get_current_user_from_cookie(
        request, auth_manager
    )
    
    if user_data:
        logger.debug("Authenticated via cookie")
        return user_data
        
    # Fallback to Authorization header (for backward compatibility)
    auth_header = request.headers.get("Authorization", "")
    
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    token = auth_header.split(" ", 1)[1]
    
    # Check if token is revoked
    from core.token_blacklist import token_blacklist
    if token_blacklist.is_revoked(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify token
    user_data = await auth_manager.verify_token_and_get_user(token)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check user-level revocation in Redis (cross-client logout)
    from core.token_blacklist import is_user_session_revoked
    user_id = user_data.get("user_id")
    if user_id and await is_user_session_revoked(
        auth_manager.cache_manager, user_id
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has been revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.warning("Authenticated via header (deprecated). Migrate to cookie auth.")

    return user_data


def create_response_with_cookie(
    data: Dict[str, Any],
    token: str,
    status_code: int = 200
) -> JSONResponse:
    """
    Create JSON response with auth cookie set
    
    Args:
        data: Response data dict
        token: JWT token to set in cookie
        status_code: HTTP status code
        
    Returns:
        JSONResponse with cookie set
    """
    response = JSONResponse(
        content=data,
        status_code=status_code
    )
    
    cookie_auth_manager.set_auth_cookie(response, token)
    
    return response
