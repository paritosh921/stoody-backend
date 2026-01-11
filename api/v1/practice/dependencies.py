"""
Practice Module - Dependencies
FastAPI dependencies for access control and validation
"""

from typing import Dict, Any
from fastapi import Depends, HTTPException, status
from api.v1.auth_async import get_current_user


def require_student_or_admin(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Dependency to require student, admin, or B2C user access.

    Args:
        current_user: Current authenticated user from JWT

    Returns:
        The current user dict if authorized

    Raises:
        HTTPException: If user type is not allowed
    """
    allowed_types = ["student", "admin", "b2c_user", "b2c_admin"]
    if current_user.get("user_type") not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student, admin, or B2C user access required"
        )
    return current_user
