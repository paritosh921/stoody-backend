"""
B2C admin authentication and dashboard endpoints.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.b2c_auth_dependencies import get_auth_manager, get_current_b2c_admin, get_database
from api.v1.b2c_auth_schemas import B2CAdminLoginRequest, B2CAdminResponse, B2CAdminSetupRequest
from config_async import settings
from core.auth import AuthManager
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/admin/login")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def b2c_admin_login(
    request: Request,
    login_data: B2CAdminLoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    B2C Admin login with username and password.

    - Uses stoody-b2c database only
    - Separate from the main admin system
    - Returns JWT token for session
    """
    try:
        import bcrypt

        admin = await db.b2c_find_one("admins", {"username": login_data.username})

        if not admin:
            logger.warning(
                f"B2C Admin login attempt with unknown username: {login_data.username}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        stored_hash = admin.get("password_hash", "")
        if not stored_hash or not bcrypt.checkpw(
            login_data.password.encode("utf-8"),
            stored_hash.encode("utf-8") if isinstance(stored_hash, str) else stored_hash,
        ):
            logger.warning(
                f"B2C Admin login failed - incorrect password: {login_data.username}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        if not admin.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin account is disabled",
            )

        admin_id = str(admin["_id"])

        await db.b2c_update_one(
            "admins",
            {"_id": admin["_id"]},
            {"$set": {"last_login": datetime.utcnow()}},
        )

        user_data = {
            "user_id": admin_id,
            "user_type": "b2c_admin",
            "username": admin.get("username"),
            "email": admin.get("email", ""),
            "full_name": admin.get("full_name", "B2C Admin"),
            "is_b2c": True,
            "is_b2c_admin": True,
        }

        session_data = await auth_manager.create_user_session(user_data)

        try:
            await db.b2c_insert_one(
                "admin_activity_log",
                {
                    "admin_id": admin["_id"],
                    "action": "login",
                    "timestamp": datetime.utcnow(),
                    "metadata": {
                        "ip_address": request.client.host if request.client else "unknown",
                        "user_agent": request.headers.get("user-agent", "unknown"),
                    },
                },
            )
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
                    "is_b2c_admin": True,
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C Admin login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        )


@router.post("/admin/setup")
@limiter.limit("3/hour")
async def setup_b2c_admin(
    request: Request,
    setup_data: B2CAdminSetupRequest,
    db: DatabaseManager = Depends(get_database),
):
    """
    Initial B2C Admin account setup.

    - Creates the single B2C admin account
    - Requires a setup key for security
    - Can only be used once (only one admin allowed)
    """
    import bcrypt

    try:
        expected_key = os.getenv("B2C_ADMIN_SETUP_KEY", "stoody-b2c-admin-setup-2024")
        if setup_data.setup_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid setup key",
            )

        existing_admin = await db.b2c_find_one("admins", {})
        if existing_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="B2C Admin already exists. Only one admin is allowed.",
            )

        password_hash = bcrypt.hashpw(
            setup_data.password.encode("utf-8"),
            bcrypt.gensalt(),
        ).decode("utf-8")

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
                "manage_settings": True,
            },
        }

        admin_id = await db.b2c_insert_one("admins", admin_doc)

        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create admin account",
            )

        logger.info(f"B2C Admin account created: {setup_data.username}")

        return {
            "success": True,
            "message": "B2C Admin account created successfully",
            "data": {
                "admin_id": admin_id,
                "username": setup_data.username,
                "full_name": setup_data.full_name,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C Admin setup error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin setup failed",
        )


@router.get("/admin/me", response_model=B2CAdminResponse)
async def get_b2c_admin_profile(
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Get current B2C Admin profile."""
    try:
        admin_id = current_admin.get("user_id")

        admin = await db.b2c_find_one("admins", {"_id": ObjectId(admin_id)})

        if not admin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found",
            )

        return B2CAdminResponse(
            admin_id=str(admin["_id"]),
            username=admin.get("username", ""),
            email=admin.get("email"),
            full_name=admin.get("full_name", "B2C Admin"),
            user_type="b2c_admin",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get B2C Admin profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch admin profile",
        )


@router.get("/admin/dashboard/stats")
async def get_b2c_admin_dashboard_stats(
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Get B2C Admin dashboard statistics."""
    try:
        users = await db.b2c_find("users", {})
        total_users = len(users)
        active_users = len([u for u in users if u.get("is_active", True)])

        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_users = len(
            [u for u in users if u.get("created_at", datetime.min) >= week_ago]
        )

        activity_logs = await db.b2c_find("user_activity_log", {})
        total_logins = len([l for l in activity_logs if l.get("action") == "login"])

        return {
            "success": True,
            "data": {
                "total_students": total_users,
                "active_students": active_users,
                "new_signups_7d": recent_users,
                "total_logins": total_logins,
                "last_updated": datetime.utcnow().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"B2C Admin dashboard stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch dashboard statistics",
        )


@router.get("/admin/students")
async def get_b2c_students(
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database),
    page: int = 1,
    limit: int = 20,
):
    """Get list of B2C students."""
    try:
        all_users = await db.b2c_find("users", {"user_type": "b2c_user"})

        logger.info(f"B2C Admin fetching students - found {len(all_users)} users")

        total = len(all_users)
        start = (page - 1) * limit
        end = start + limit
        users = all_users[start:end]

        students = []
        for user in users:
            students.append(
                {
                    "user_id": str(user["_id"]),
                    "email": user.get("email", ""),
                    "full_name": user.get("full_name", ""),
                    "picture": user.get("picture"),
                    "phone": user.get("phone"),
                    "is_active": user.get("is_active", True),
                    "exam_type": user.get("exam_type"),
                    "class_level": user.get("class_level"),
                    "standard": user.get("standard"),
                    "subjects": user.get("subjects", []),
                    "plan_types": user.get("plan_types", []),
                    "school_name": user.get("school_name"),
                    "city": user.get("city"),
                    "onboarding_complete": user.get("onboarding_complete", False),
                    "onboarding_completed_at": user.get("onboarding_completed_at").isoformat()
                    if user.get("onboarding_completed_at")
                    else None,
                    "created_at": user.get("created_at", datetime.utcnow()).isoformat(),
                    "last_login": user.get("last_login", datetime.utcnow()).isoformat()
                    if user.get("last_login")
                    else None,
                }
            )

        return {
            "success": True,
            "data": {
                "students": students,
                "total": total,
                "page": page,
                "limit": limit,
                "total_pages": (total + limit - 1) // limit,
            },
        }

    except Exception as e:
        logger.error(f"Get B2C students error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch students",
        )
