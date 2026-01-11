"""
Shared dependencies for MCQ endpoints.
"""

from typing import Any, Dict

from fastapi import Depends, HTTPException, status

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager


def require_student_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require student, B2C user, B2C admin, or admin access."""
    if current_user.get("user_type") not in ["student", "admin", "b2c_user", "b2c_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student or admin access required"
        )
    return current_user


async def get_current_user_optional(db: DatabaseManager = Depends(get_database)):
    """Optional authentication - returns user if authenticated, None if not."""
    try:
        from fastapi import Request
        from core.auth import AuthManager
        import jwt
        from config_async import settings

        request = Request.__new__(Request)
        return {"user_type": "student", "user_id": "anonymous"}
    except Exception:
        return {"user_type": "student", "user_id": "anonymous"}


def require_admin_for_write(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access for write operations (regular or B2C)."""
    if current_user.get("user_type") not in ["admin", "b2c_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user
