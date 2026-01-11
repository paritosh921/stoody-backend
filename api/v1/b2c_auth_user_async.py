"""
B2C user auth endpoints.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.v1.b2c_auth_dependencies import get_auth_manager, get_current_b2c_user, get_database
from api.v1.b2c_auth_schemas import B2CUserResponse
from core.auth import AuthManager
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/me", response_model=B2CUserResponse)
async def get_b2c_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get current B2C user profile."""
    try:
        user_id = current_user.get("user_id")

        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        return B2CUserResponse(
            user_id=str(user["_id"]),
            email=user.get("email", ""),
            full_name=user.get("full_name", ""),
            picture=user.get("picture"),
            user_type="b2c_user",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get B2C profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch profile",
        )


@router.post("/logout")
async def b2c_logout(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """B2C user logout."""
    try:
        user_id = current_user.get("user_id")

        try:
            await db.b2c_insert_one(
                "user_activity_log",
                {
                    "user_id": ObjectId(user_id),
                    "action": "logout",
                    "timestamp": datetime.utcnow(),
                    "metadata": {},
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log B2C logout: {str(e)}")

        await auth_manager.invalidate_user_session(user_id)

        return {"success": True, "message": "Successfully logged out"}

    except Exception as e:
        logger.error(f"B2C logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed",
        )


@router.get("/verify")
async def verify_b2c_token(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
):
    """Verify B2C JWT token and return user data."""
    return {
        "success": True,
        "data": {
            "user_id": current_user.get("user_id"),
            "user_type": current_user.get("user_type"),
            "email": current_user.get("email"),
            "full_name": current_user.get("full_name"),
            "is_b2c": True,
        },
    }
