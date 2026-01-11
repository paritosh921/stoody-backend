from typing import Dict, Any
from fastapi import HTTPException, Depends, status
from api.v1.auth_async import get_current_user

def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access (regular or B2C)"""
    if current_user.get("user_type") not in ["admin", "b2c_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Allow admin, B2C admin, and tutor roles"""
    if current_user.get("user_type") not in ["admin", "b2c_admin", "tutor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Tutor access required"
        )
    return current_user
