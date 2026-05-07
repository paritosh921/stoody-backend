"""
SmartBoard Token Issuance API

Issues scoped JWT tokens for SmartBoard access using the canonical auth stack.
Only authenticated tutors/teachers can access SmartBoard.
Tokens are short-lived (8 hours) and include user identity for audit logging.

This route is retained for backward compatibility. The primary smartboard
pairing flow uses /api/v1/smartboard-pair/* instead.
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from api.v1.auth_async import get_current_user
from core.auth import AuthManager
from core.permissions import has_permission
from core.tenant_features import is_feature_enabled

logger = logging.getLogger(__name__)

router = APIRouter()

SMARTBOARD_TOKEN_EXPIRY_HOURS = 8
SMARTBOARD_BACKEND_URL_ENV = ""


class SmartBoardTokenResponse(BaseModel):
    """Response containing the SmartBoard access token"""
    success: bool = True
    token: str
    expires_at: str
    smartboard_url: str
    user_id: str
    username: str


class SmartBoardTokenError(BaseModel):
    """Error response for token issuance failures"""
    success: bool = False
    detail: str


def _get_smartboard_url(request: Request) -> str:
    import os
    return os.getenv("SMARTBOARD_BACKEND_URL", "")


@router.post(
    "/smartboard/token",
    response_model=SmartBoardTokenResponse,
    responses={
        403: {"model": SmartBoardTokenError, "description": "User not authorized for SmartBoard"},
        500: {"model": SmartBoardTokenError, "description": "Token generation failed"},
    },
    summary="Get SmartBoard Access Token",
    description="""
    Issue a scoped JWT token for SmartBoard access.

    **Authorization:** Requires valid tutor/teacher/admin JWT in Authorization header.

    **Token Properties:**
    - Valid for 8 hours (one teaching session)
    - Contains user identity for audit logging
    - Issued using the canonical auth stack (compatible with decode_access_token)

    **Usage:** Include the returned token in requests to SmartBoard backend:
    - HTTP: `Authorization: Bearer <token>`
    - WebSocket: `wss://smartboard/ws?token=<token>`
    """
)
async def get_smartboard_token(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Issue a scoped JWT token for SmartBoard access.

    Uses the same AuthManager.create_access_token as normal login tokens,
    with the addition of `device: "smartboard"` for audit trails.
    The resulting token is fully compatible with decode_access_token.
    """

    # Check smartboard_cloud_access feature entitlement
    if not is_feature_enabled(
        current_user.get("enabled_features"),
        "smartboard_cloud_access",
        current_user.get("enabled_features_v2"),
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Smartboard cloud access is not enabled for your institution",
        )

    # Get user type/role - handle different field names
    user_type = current_user.get("user_type") or current_user.get("role") or ""
    user_id = str(current_user.get("user_id") or current_user.get("_id") or current_user.get("id") or "")
    username = current_user.get("username") or current_user.get("email") or current_user.get("full_name") or ""

    # Authorization check - only tutors, teachers, and admins can access SmartBoard
    allowed_roles = ["tutor", "teacher", "admin"]
    if user_type.lower() not in allowed_roles:
        logger.warning(f"[SMARTBOARD] Unauthorized access attempt by {username} (role: {user_type})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Only tutors and teachers can access SmartBoard. Your role: {user_type}"
        )
    if user_type.lower() == "admin" and not has_permission(current_user, "manage_smartboard"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    # Generate token using the canonical auth stack
    auth_manager: AuthManager = request.app.state.auth
    expires_delta = timedelta(hours=SMARTBOARD_TOKEN_EXPIRY_HOURS)

    token_payload = {
        "sub": user_id,
        "user_id": user_id,
        "user_type": user_type.lower(),
        "username": username,
        "tutor_id": current_user.get("tutor_id"),
        "admin_id": current_user.get("admin_id"),
        "tenant_id": current_user.get("tenant_id"),
        "db_name": current_user.get("db_name"),
        "institution_id": current_user.get("institution_id"),
        "device": "smartboard",
    }
    # Drop None values so the JWT stays compact.
    token_payload = {k: v for k, v in token_payload.items() if v is not None}

    try:
        access_token = auth_manager.create_access_token(
            token_payload,
            expires_delta=expires_delta,
        )
    except Exception as e:
        logger.error(f"[SMARTBOARD] Token generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate SmartBoard token"
        )

    now = datetime.now(timezone.utc)
    expires_at = now + expires_delta

    logger.info(f"[SMARTBOARD] Token issued for {username} ({user_id}), expires: {expires_at.isoformat()}")

    return SmartBoardTokenResponse(
        success=True,
        token=access_token,
        expires_at=expires_at.isoformat(),
        smartboard_url=_get_smartboard_url(request),
        user_id=user_id,
        username=username
    )


@router.get(
    "/smartboard/status",
    summary="Check SmartBoard Configuration",
    description="Check if SmartBoard integration is properly configured"
)
async def get_smartboard_status(request: Request):
    """Check SmartBoard configuration status (public endpoint for debugging)"""
    import os
    configured = bool(os.getenv("JWT_SECRET_KEY"))
    return {
        "configured": configured,
        "smartboard_url": _get_smartboard_url(request) or "Not configured",
        "token_expiry_hours": SMARTBOARD_TOKEN_EXPIRY_HOURS,
        "note": "Token route now uses canonical auth stack (JWT_SECRET_KEY).",
    }
