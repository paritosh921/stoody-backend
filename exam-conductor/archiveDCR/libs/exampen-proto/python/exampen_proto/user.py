"""User identity models: auth claims, profiles, role mappings, revocations."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from .enums import ExamPenRole, StoodyRole, TokenStatus


class Profile(BaseModel):
    """User profile information from Stoody."""

    display_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    institute_name: Optional[str] = None


class NormalizedClaims(BaseModel):
    """Normalized ExamPen claims derived from a Stoody JWT."""

    user_id: str
    tenant_id: str
    stoody_role: StoodyRole
    exampen_roles: list[ExamPenRole]
    token_source: str = "stoody_jwt"
    token_status: TokenStatus
    subject_ids: Optional[list[str]] = None
    class_ids: Optional[list[str]] = None
    child_student_ids: Optional[list[str]] = None
    profile: Profile


class IntrospectRequest(BaseModel):
    """Request to validate and normalize a Stoody JWT."""

    token: str
    expected_role: Optional[ExamPenRole] = None


class RevocationRequest(BaseModel):
    """Request to revoke a token/session within ExamPen."""

    jti: str
    subject_user_id: Optional[str] = None
    reason: str
    expires_at: Optional[datetime] = None


class RevocationStatus(BaseModel):
    """Revocation state for a token JTI."""

    jti: str
    revoked: bool
    revoked_at: Optional[datetime] = None
    reason: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response body."""

    code: str
    message: str
