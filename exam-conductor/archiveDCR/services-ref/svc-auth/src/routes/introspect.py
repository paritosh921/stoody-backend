"""Introspect and /me routes — validate Stoody JWT, return normalized claims.

Endpoints:
  POST /introspect  — accept a raw Stoody JWT, validate, normalize, return claims
  GET  /me          — shorthand: extract bearer token from header, return claims
"""

from __future__ import annotations

from typing import Any

import jwt as pyjwt
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from exampen_common.auth import JWKSManager, validate_token
from exampen_common.logging import get_logger

from src.adapters.stoody_client import StoodyClient
from src.domain.claims import normalize_claims
from src.domain.role_mapper import has_minimum_role

_log = get_logger(__name__)

router = APIRouter()


# -- Request / Response models (match auth.openapi.yaml) -------------------


class IntrospectRequest(BaseModel):
    """POST /introspect request body."""

    token: str
    expected_role: str | None = None


class ProfileResponse(BaseModel):
    """Nested profile in NormalizedClaims response."""

    display_name: str
    email: str | None = None
    phone: str | None = None
    institute_name: str | None = None


class NormalizedClaimsResponse(BaseModel):
    """POST /introspect and GET /me response body."""

    user_id: str
    tenant_id: str
    stoody_role: str
    exampen_roles: list[str]
    token_source: str = "stoody_jwt"
    token_status: str = "valid"
    profile: ProfileResponse
    subject_ids: list[str] | None = None
    class_ids: list[str] | None = None
    child_student_ids: list[str] | None = None


class ErrorResponse(BaseModel):
    """Standard error body."""

    code: str
    message: str


# -- Helpers ---------------------------------------------------------------


async def _build_claims(
    token: str,
    request: Request,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Validate token, enrich with Stoody profile, check revocation."""
    jwks: JWKSManager = request.app.state.jwks_manager
    revocation_repo = request.app.state.revocation_repo
    role_mapping_repo = request.app.state.role_mapping_repo

    # 1. Validate JWT signature and expiry
    exampen_user = await validate_token(token, manager=jwks)

    # 2. Check revocation status
    try:
        unverified = pyjwt.decode(token, options={"verify_signature": False})
        jti = unverified.get("jti", "")
    except Exception:
        jti = ""

    if jti:
        rev_status = await revocation_repo.is_revoked(jti)
        if rev_status.get("revoked"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
            )

    # 3. Load configurable role overrides from DB (tenant-scoped + global defaults)
    role_overrides = await role_mapping_repo.get_all(exampen_user.tenant_id)

    # 4. Fetch Stoody profile for enrichment (graceful degradation)
    stoody_client = StoodyClient()
    profile_data = await stoody_client.get_user_profile(exampen_user.user_id)

    # 5. Fetch parent-child relationships if applicable
    child_ids: list[str] | None = None
    stoody_role = exampen_user.stoody_role
    if stoody_role == "parent":
        child_ids = await stoody_client.get_parent_children(exampen_user.user_id)

    # 6. Build normalized claims
    jwt_payload = {
        "sub": exampen_user.user_id,
        "tenant_id": exampen_user.tenant_id,
        "role": exampen_user.stoody_role,
        "name": exampen_user.name,
        "email": exampen_user.email,
        "jti": jti,
    }
    claims = normalize_claims(
        stoody_jwt_payload=jwt_payload,
        stoody_profile=profile_data,
        role_overrides=role_overrides if role_overrides else None,
        child_ids=child_ids,
    )

    # 7. Optionally verify expected_role
    if expected_role and expected_role not in claims.exampen_roles:
        if not has_minimum_role(claims.exampen_roles, expected_role):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Insufficient role: expected {expected_role}",
            )

    return claims.to_dict()


# -- Endpoints -------------------------------------------------------------


@router.post(
    "/introspect",
    response_model=NormalizedClaimsResponse,
    responses={
        401: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def introspect(body: IntrospectRequest, request: Request) -> dict[str, Any]:
    """Validate a Stoody JWT and return normalized ExamPen claims."""
    return await _build_claims(
        token=body.token,
        request=request,
        expected_role=body.expected_role,
    )


@router.get(
    "/me",
    response_model=NormalizedClaimsResponse,
    responses={401: {"model": ErrorResponse}},
)
async def me(request: Request) -> dict[str, Any]:
    """Return normalized claims for the current bearer token."""
    auth_header: str | None = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )
    token = auth_header.removeprefix("Bearer ").strip()
    return await _build_claims(token=token, request=request)
