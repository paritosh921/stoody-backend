"""JWT validation + RBAC enforcement for student BFF.

Allowed roles:
  - student: sees own data only (user_id from JWT)
  - parent: sees linked children's data only (resolved via Stoody API)

Denied roles:
  - tutor, evaluator, hod, principal, super_admin -> 403 (use teacher-bff)

Parent child resolution:
  On every parent request, the middleware resolves the parent's linked
  children via ``GET /api/parents/{user_id}/children`` on the Stoody API
  (cached on app.state.stoody_client).  The resolved child IDs are
  attached to ``request.state.allowed_student_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request, status

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.logging import get_logger

_log = get_logger(__name__)

# Roles that may use the student BFF
_STUDENT_BFF_ROLES = frozenset({"student", "parent"})

# Roles that must use the teacher BFF instead
_TEACHER_BFF_ROLES = frozenset({
    "teacher", "evaluator", "reviewer", "tutor",
    "hod", "principal", "super_admin", "invigilator",
})


@dataclass(frozen=True, slots=True)
class StudentBFFIdentity:
    """Resolved identity for student BFF requests.

    Attributes
    ----------
    user:
        The authenticated ExamPenUser from the JWT.
    role:
        Primary BFF role: ``"student"`` or ``"parent"``.
    allowed_student_ids:
        Student IDs this actor may view. For a student, it is their own
        user_id. For a parent, it is their linked children.
    """

    user: ExamPenUser
    role: str
    allowed_student_ids: list[str] = field(default_factory=list)


def _effective_bff_role(user: ExamPenUser) -> str:
    """Determine the BFF role from the user's ExamPen roles."""
    roles = set(user.exampen_roles)
    if roles & _TEACHER_BFF_ROLES:
        return "teacher_bff_redirect"
    if "parent" in roles:
        return "parent"
    if "student" in roles:
        return "student"
    # Fallback: check stoody_role
    if user.stoody_role == "parent":
        return "parent"
    if user.stoody_role == "student":
        return "student"
    return "denied"


async def require_student_or_parent(
    request: Request,
) -> StudentBFFIdentity:
    """FastAPI dependency that enforces student/parent RBAC.

    Raises ``403`` for teacher/admin roles, ``401`` for missing auth.
    Attaches resolved ``allowed_student_ids`` for scope filtering.
    """
    user = await get_current_user(request)
    bff_role = _effective_bff_role(user)

    if bff_role == "teacher_bff_redirect":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher/admin roles must use the teacher BFF",
        )

    if bff_role == "denied":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient role for student portal access",
        )

    if bff_role == "student":
        return StudentBFFIdentity(
            user=user,
            role="student",
            allowed_student_ids=[user.user_id],
        )

    # Parent: resolve children via Stoody API
    stoody_client = request.app.state.stoody_client
    child_ids = await stoody_client.get_parent_children(user.user_id)
    if child_ids is None:
        _log.warning(
            "Parent %s: could not resolve children (Stoody unreachable)",
            user.user_id,
        )
        child_ids = []

    if not child_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No linked children found for parent account",
        )

    return StudentBFFIdentity(
        user=user,
        role="parent",
        allowed_student_ids=child_ids,
    )


def require_own_data(
    identity: StudentBFFIdentity,
    student_id: str | None = None,
) -> str:
    """Resolve which student_id to use and verify access.

    For students, returns their own user_id. For parents, validates that
    the requested student_id is among their linked children.

    Returns the effective student_id.
    """
    if identity.role == "student":
        return identity.user.user_id

    # Parent must specify which child
    if student_id and student_id in identity.allowed_student_ids:
        return student_id

    # If only one child, use that as default
    if len(identity.allowed_student_ids) == 1:
        return identity.allowed_student_ids[0]

    if student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Parent does not have access to this student's data",
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Parent must specify student_id query parameter",
    )
