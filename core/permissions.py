"""
Permission helpers for admin role enforcement.
"""

from typing import Dict, Any, List

OPERATOR_PERMISSIONS: List[str] = [
    "manage_students",
    "manage_tutors",
    "view_analytics",
    "manage_smartboard",
    "manage_questions",
    "view_strokes",
]


def has_permission(current_user: Dict[str, Any], permission: str) -> bool:
    """Return True if the current user can access a permission."""
    if not permission:
        return True
    role = current_user.get("admin_role")
    if role == "master_admin":
        return True
    if current_user.get("user_type") == "b2c_admin":
        return True
    permissions = current_user.get("permissions") or []
    if isinstance(permissions, dict):
        return bool(permissions.get(permission))
    return permission in permissions
