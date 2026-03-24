"""Role-based access control — ZERO I/O, pure domain logic.

Enforces the Access Control Matrix from STOODY_INTEGRATION_SPEC S6.
Uses the role hierarchy from svc-auth role_mapper.py.

Role hierarchy (lowest to highest privilege):
    parent < student < invigilator < evaluator < reviewer < hod < tutor < principal < super_admin
"""

from __future__ import annotations

from fastapi import HTTPException, status

from exampen.dcr.core.auth_bridge import ExamPenUser

# ---------------------------------------------------------------------------
# Role hierarchy — mirrors svc-auth/src/domain/role_mapper.py
# ---------------------------------------------------------------------------

ROLE_HIERARCHY: list[str] = [
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

_HIERARCHY_INDEX: dict[str, int] = {
    role: idx for idx, role in enumerate(ROLE_HIERARCHY)
}


def role_rank(role: str) -> int:
    """Return numeric rank of *role*.  Higher = more privilege.  Unknown = -1."""
    return _HIERARCHY_INDEX.get(role, -1)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def has_any_role(user: ExamPenUser, roles: set[str] | frozenset[str]) -> bool:
    """Return True if the user holds at least one of *roles*."""
    return bool(set(user.exampen_roles) & roles)


def require_role(
    user: ExamPenUser,
    *allowed_roles: str,
) -> None:
    """Raise 403 unless the user holds one of *allowed_roles*.

    Parameters
    ----------
    user:
        The authenticated user (from ``get_current_user``).
    *allowed_roles:
        One or more role strings that are permitted.

    Raises
    ------
    HTTPException(403) if the user has no matching role.
    """
    allowed = frozenset(allowed_roles)
    if has_any_role(user, allowed):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"Forbidden: requires one of {sorted(allowed)}, "
            f"user has {user.exampen_roles}"
        ),
    )


def require_minimum_role(
    user: ExamPenUser,
    min_role: str,
) -> None:
    """Raise 403 unless the user has at least *min_role* privilege.

    Uses the numeric hierarchy: any exampen_role whose rank >= min_role's
    rank is accepted.

    Raises
    ------
    HTTPException(403) if no user role meets the threshold.
    """
    min_rank = role_rank(min_role)
    if any(role_rank(r) >= min_rank for r in user.exampen_roles):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"Forbidden: requires at least '{min_role}' privilege, "
            f"user has {user.exampen_roles}"
        ),
    )


# ---------------------------------------------------------------------------
# Lifecycle-transition role requirements
# ---------------------------------------------------------------------------

# Transitions that invigilators (assigned to exam) may perform.
_INVIGILATOR_TRANSITIONS: frozenset[str] = frozenset({
    "armed", "timer_running", "sync_pending", "cancelled",
})

# Transitions that require evaluator or above.
_EVALUATOR_PLUS_TRANSITIONS: frozenset[str] = frozenset({
    "scoring", "finalized", "published", "locked",
})


def require_transition_role(
    user: ExamPenUser,
    to_state: str,
    assigned_user_ids: frozenset[str] | None = None,
) -> None:
    """Enforce role requirements for a lifecycle transition.

    Parameters
    ----------
    user:
        The authenticated user.
    to_state:
        The target state of the transition.
    assigned_user_ids:
        Set of user_ids currently assigned as invigilator for this exam.
        Required for invigilator-gated transitions.

    Raises
    ------
    HTTPException(403) if the user lacks the required role.
    """
    if to_state in _INVIGILATOR_TRANSITIONS:
        # super_admin / principal / hod can always do this
        if has_any_role(user, frozenset({"super_admin", "principal", "hod"})):
            return
        # Invigilator must be assigned to this exam
        if has_any_role(user, frozenset({"invigilator"})):
            if assigned_user_ids and user.user_id in assigned_user_ids:
                return
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invigilator not assigned to this exam",
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Forbidden: transition to '{to_state}' requires "
                f"assigned invigilator or above"
            ),
        )

    if to_state in _EVALUATOR_PLUS_TRANSITIONS:
        require_minimum_role(user, "evaluator")
        return

    # Unknown target state — let FSM handle validity; allow any authenticated
    # user through (FSM will reject bad state values).
