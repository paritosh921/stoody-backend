"""
Shared dependencies for B2C authentication endpoints.
"""

import logging
from typing import Any, Dict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.auth import AuthManager
from core.cache import CacheManager
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

security = HTTPBearer()


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
    """Get current authenticated B2C user."""
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


async def get_current_b2c_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Dict[str, Any]:
    """Get current authenticated B2C Admin."""
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
