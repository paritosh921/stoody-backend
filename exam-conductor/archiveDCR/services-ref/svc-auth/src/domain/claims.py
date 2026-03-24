"""Normalized claims — ZERO I/O, pure domain logic.

Transforms raw Stoody JWT payloads and optional profile data into
the ``NormalizedClaims`` structure consumed by all ExamPen services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.role_mapper import map_roles


@dataclass(frozen=True, slots=True)
class Profile:
    """User profile data (optionally enriched from Stoody API)."""

    display_name: str
    email: str | None = None
    phone: str | None = None
    institute_name: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizedClaims:
    """ExamPen-normalized identity claims.

    Matches the ``NormalizedClaims`` schema in ``auth.openapi.yaml``.
    """

    user_id: str
    tenant_id: str
    stoody_role: str
    exampen_roles: list[str]
    token_source: str = "stoody_jwt"
    token_status: str = "valid"
    profile: Profile = field(default_factory=lambda: Profile(display_name=""))
    subject_ids: list[str] = field(default_factory=list)
    class_ids: list[str] = field(default_factory=list)
    child_student_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict matching the OpenAPI response schema."""
        result: dict[str, Any] = {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "stoody_role": self.stoody_role,
            "exampen_roles": self.exampen_roles,
            "token_source": self.token_source,
            "token_status": self.token_status,
            "profile": {
                "display_name": self.profile.display_name,
            },
        }
        if self.profile.email:
            result["profile"]["email"] = self.profile.email
        if self.profile.phone:
            result["profile"]["phone"] = self.profile.phone
        if self.profile.institute_name:
            result["profile"]["institute_name"] = self.profile.institute_name
        if self.subject_ids:
            result["subject_ids"] = self.subject_ids
        if self.class_ids:
            result["class_ids"] = self.class_ids
        if self.child_student_ids:
            result["child_student_ids"] = self.child_student_ids
        return result


def normalize_claims(
    stoody_jwt_payload: dict[str, Any],
    stoody_profile: dict[str, Any] | None = None,
    role_overrides: dict[str, list[str]] | None = None,
    child_ids: list[str] | None = None,
) -> NormalizedClaims:
    """Build ``NormalizedClaims`` from a decoded Stoody JWT and optional profile.

    Parameters
    ----------
    stoody_jwt_payload:
        Decoded JWT claims dict (must contain ``sub``, ``tenant_id``, ``role``).
    stoody_profile:
        Optional user profile from ``GET /api/users/{user_id}``.
    role_overrides:
        Optional DB-loaded role mapping overrides.
    child_ids:
        Optional list of child student IDs (for parent role).
    """
    user_id = str(
        stoody_jwt_payload.get("sub", stoody_jwt_payload.get("user_id", ""))
    )
    tenant_id = str(stoody_jwt_payload.get("tenant_id", ""))
    stoody_role = stoody_jwt_payload.get(
        "role", stoody_jwt_payload.get("stoody_role", "")
    )

    exampen_roles = map_roles(stoody_role, overrides=role_overrides)

    # Build profile from JWT + optional Stoody API enrichment
    if stoody_profile:
        profile = Profile(
            display_name=stoody_profile.get("name", stoody_profile.get("display_name", user_id)),
            email=stoody_profile.get("email"),
            phone=stoody_profile.get("phone"),
            institute_name=stoody_profile.get("institute_name"),
        )
        subject_ids = stoody_profile.get("subject_ids", [])
        class_ids = stoody_profile.get("class_ids", [])
    else:
        # Graceful degradation: use JWT claims only
        profile = Profile(
            display_name=stoody_jwt_payload.get("name", user_id),
            email=stoody_jwt_payload.get("email"),
        )
        subject_ids = stoody_jwt_payload.get("subject_ids", [])
        class_ids = stoody_jwt_payload.get("class_ids", [])

    return NormalizedClaims(
        user_id=user_id,
        tenant_id=tenant_id,
        stoody_role=stoody_role,
        exampen_roles=exampen_roles,
        profile=profile,
        subject_ids=subject_ids,
        class_ids=class_ids,
        child_student_ids=child_ids or [],
    )
