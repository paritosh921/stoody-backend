"""Bridge between Stoody authentication and ExamPen roles.

Provides a FastAPI dependency that extracts the current user from the
Stoody JWT (reusing the existing auth pipeline) and maps Stoody roles
to ExamPen roles.  Tutors may receive additional exam-specific roles
(invigilator, evaluator) loaded from the ``exampen_assignments``
collection in the tenant database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.auth import AuthManager
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

security = HTTPBearer()

# ---------------------------------------------------------------------------
# ExamPen user model
# ---------------------------------------------------------------------------

STOODY_TO_EXAMPEN_ROLES: Dict[str, List[str]] = {
    "admin": ["principal"],
    "tutor": ["evaluator"],
    "student": ["student"],
}

NO_ACCESS_ROLE = "no_access"

# Role hierarchy (lowest to highest privilege) — mirrors DCR's rbac.py
ROLE_HIERARCHY: List[str] = [
    "parent",
    "student",
    "invigilator",
    "evaluator",
    "reviewer",
    "hod",
    "tutor",
    "principal",
    "super_admin",
]

_HIERARCHY_INDEX: Dict[str, int] = {
    role: idx for idx, role in enumerate(ROLE_HIERARCHY)
}


@dataclass(frozen=True, slots=True)
class ExamPenUser:
    """Normalized user identity for ExamPen operations."""

    user_id: str
    tenant_id: str
    stoody_role: str
    exampen_roles: List[str] = field(default_factory=list)
    name: str = ""
    email: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _load_exam_assignments(
    db: DatabaseManager,
    db_name: str,
    user_id: str,
) -> List[str]:
    """Load exam-specific roles for a tutor from ``exampen_assignments``.

    Returns a list of additional ExamPen role strings (e.g. ``["invigilator"]``).
    """
    try:
        tenant_db = await db.get_tenant_db(db_name)
        if tenant_db is None:
            return []

        collection = tenant_db["exampen_assignments"]
        cursor = collection.find(
            {"user_id": user_id, "is_active": True},
            {"role": 1},
        )
        docs = await cursor.to_list(length=50)
        roles: List[str] = []
        seen: Set[str] = set()
        for doc in docs:
            role = doc.get("role", "")
            if role and role not in seen:
                seen.add(role)
                roles.append(role)
        return roles
    except Exception as e:
        logger.warning(
            "Failed to load exampen_assignments for user %s: %s",
            user_id,
            e,
        )
        return []


def _map_stoody_role(user_type: str) -> List[str]:
    """Map a Stoody ``user_type`` to base ExamPen roles."""
    return list(STOODY_TO_EXAMPEN_ROLES.get(user_type, [NO_ACCESS_ROLE]))


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def _get_auth_manager(request: Request) -> AuthManager:
    return request.app.state.auth


async def _get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


async def get_exampen_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(_get_auth_manager),
    db: DatabaseManager = Depends(_get_database),
) -> ExamPenUser:
    """FastAPI dependency: extract Stoody JWT and return an ``ExamPenUser``.

    Steps:
    1. Validate Stoody JWT via the existing ``AuthManager``.
    2. Map ``user_type`` to base ExamPen roles.
    3. For tutors, also load exam-specific roles from ``exampen_assignments``.
    """
    token = credentials.credentials

    # Check token revocation (mirrors existing auth_async.get_current_user)
    try:
        from core.token_blacklist import token_blacklist
        if token_blacklist.is_revoked(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ImportError:
        pass  # token_blacklist not available in all environments

    user_data = await auth_manager.verify_token_and_get_user(token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = user_data.get("user_id", "")
    user_type = (user_data.get("user_type") or "").strip().lower()
    tenant_id = user_data.get("tenant_id", "")
    db_name = user_data.get("db_name", "")
    name = user_data.get("full_name") or user_data.get("username") or ""
    email = user_data.get("email") or ""

    # Base role mapping
    exampen_roles = _map_stoody_role(user_type)

    # Tutors may have additional exam-specific roles
    if user_type == "tutor" and db_name:
        extra_roles = await _load_exam_assignments(db, db_name, user_id)
        for role in extra_roles:
            if role not in exampen_roles:
                exampen_roles.append(role)

    return ExamPenUser(
        user_id=user_id,
        tenant_id=tenant_id,
        stoody_role=user_type,
        exampen_roles=exampen_roles,
        name=name,
        email=email,
    )


def require_exampen_role(
    *roles: str,
) -> Callable[..., ExamPenUser]:
    """Dependency factory: require that the user holds at least one of *roles*.

    Usage::

        @router.get("/exams")
        async def list_exams(
            user: ExamPenUser = Depends(require_exampen_role("principal", "evaluator")),
        ):
            ...

    Raises ``HTTPException(403)`` if no matching role is found.
    """
    allowed = frozenset(roles)

    async def _dependency(
        user: ExamPenUser = Depends(get_exampen_user),
    ) -> ExamPenUser:
        if not (set(user.exampen_roles) & allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Forbidden: requires one of {sorted(allowed)}, "
                    f"user has {user.exampen_roles}"
                ),
            )
        return user

    return _dependency
