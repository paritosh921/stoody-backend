"""Stoody-to-ExamPen role mapping — ZERO I/O, pure domain logic.

Maps Stoody platform roles to ExamPen exam-specific roles using a
configurable mapping table. Provides role hierarchy utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

# Role hierarchy — higher index = more privilege.
# Used for RBAC comparisons: can actor X perform action requiring role Y?
ROLE_HIERARCHY: list[str] = [
    "parent",
    "student",
    "invigilator",
    "evaluator",
    "reviewer",
    "tutor",
    "hod",
    "principal",
    "super_admin",
]

_HIERARCHY_INDEX: dict[str, int] = {
    role: idx for idx, role in enumerate(ROLE_HIERARCHY)
}

# Default mapping when no DB-driven mapping is available.
DEFAULT_ROLE_MAP: dict[str, list[str]] = {
    "admin": ["principal"],
    "super_admin": ["super_admin"],
    "principal": ["principal"],
    "hod": ["hod"],
    "tutor": ["evaluator"],
    "student": ["student"],
    "parent": ["parent"],
}

NO_ACCESS_SENTINEL: str = "no_exampen_access"


@dataclass(frozen=True, slots=True)
class RoleMappingEntry:
    """A single stoody_role -> exampen_roles mapping."""

    stoody_role: str
    exampen_roles: list[str]


def map_roles(
    stoody_role: str,
    overrides: dict[str, list[str]] | None = None,
) -> list[str]:
    """Map a Stoody role to ExamPen roles.

    Parameters
    ----------
    stoody_role:
        The role string from the Stoody JWT (e.g. "tutor", "student").
    overrides:
        Optional DB-loaded mapping table that takes precedence over defaults.

    Returns
    -------
    list[str]
        ExamPen roles. Returns ``["no_exampen_access"]`` for unknown roles.
    """
    effective_map = {**DEFAULT_ROLE_MAP}
    if overrides:
        effective_map.update(overrides)

    roles = effective_map.get(stoody_role)
    if roles is None:
        return [NO_ACCESS_SENTINEL]
    return list(roles)


def role_rank(role: str) -> int:
    """Return the numeric rank of *role* in the hierarchy.

    Higher number = more privilege. Unknown roles return -1.
    """
    return _HIERARCHY_INDEX.get(role, -1)


def has_minimum_role(actor_roles: list[str], required: str) -> bool:
    """Check if any of *actor_roles* meets or exceeds *required* rank."""
    required_rank = role_rank(required)
    return any(role_rank(r) >= required_rank for r in actor_roles)
