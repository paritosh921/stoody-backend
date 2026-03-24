"""JWT validation and RBAC enforcement for teacher BFF endpoints.

All endpoints in this BFF require teacher-level access:
teacher, evaluator, hod, or principal. Students and parents get 403.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status

from exampen_common.auth import ExamPenUser, get_current_user

TEACHER_ROLES = frozenset({
    "teacher",
    "evaluator",
    "hod",
    "principal",
    "super_admin",
})


async def require_teacher(
    user: ExamPenUser = Depends(get_current_user),
) -> ExamPenUser:
    """FastAPI dependency: reject if the caller lacks a teacher-level role.

    Raises 403 for students, parents, and unknown roles.
    """
    if not TEACHER_ROLES.intersection(user.exampen_roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher, evaluator, HOD, or principal role required",
        )
    return user
