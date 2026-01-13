"""
SmartBoard Token Issuance API

Issues scoped JWT tokens for SmartBoard access.
Only authenticated tutors/teachers can access SmartBoard.
Tokens are short-lived (8 hours) and include user identity for audit logging.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
import jwt

from api.v1.auth_async import get_current_user
from core.permissions import has_permission

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
SMARTBOARD_JWT_SECRET = os.getenv("SMARTBOARD_JWT_SECRET")
SMARTBOARD_TOKEN_EXPIRY_HOURS = int(os.getenv("SMARTBOARD_TOKEN_EXPIRY_HOURS", "8"))
SMARTBOARD_BACKEND_URL = os.getenv("SMARTBOARD_BACKEND_URL", "")


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
    - Scoped specifically for SmartBoard backend

    **Usage:** Include the returned token in requests to SmartBoard backend:
    - HTTP: `Authorization: Bearer <token>`
    - WebSocket: `wss://smartboard/ws?token=<token>`
    """
)
async def get_smartboard_token(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Issue a scoped JWT token for SmartBoard access.

    Only tutors/teachers/admins can access SmartBoard.
    Token includes user identity for audit logging on SmartBoard backend.
    """

    # Check if SmartBoard JWT secret is configured
    if not SMARTBOARD_JWT_SECRET:
        logger.error("SMARTBOARD_JWT_SECRET not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SmartBoard authentication not configured"
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

    # Generate token expiration
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=SMARTBOARD_TOKEN_EXPIRY_HOURS)

    # Build JWT payload
    payload = {
        "sub": user_id,
        "username": username,
        "role": user_type,
        "type": "smartboard_access",  # Token type for validation
        "iat": now,
        "exp": expires_at,
    }

    try:
        token = jwt.encode(payload, SMARTBOARD_JWT_SECRET, algorithm="HS256")
    except Exception as e:
        logger.error(f"[SMARTBOARD] Token generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate SmartBoard token"
        )

    logger.info(f"[SMARTBOARD] Token issued for {username} ({user_id}), expires: {expires_at.isoformat()}")

    return SmartBoardTokenResponse(
        success=True,
        token=token,
        expires_at=expires_at.isoformat(),
        smartboard_url=SMARTBOARD_BACKEND_URL,
        user_id=user_id,
        username=username
    )


@router.get(
    "/smartboard/status",
    summary="Check SmartBoard Configuration",
    description="Check if SmartBoard integration is properly configured"
)
async def get_smartboard_status():
    """Check SmartBoard configuration status (public endpoint for debugging)"""
    return {
        "configured": bool(SMARTBOARD_JWT_SECRET),
        "smartboard_url": SMARTBOARD_BACKEND_URL or "Not configured",
        "token_expiry_hours": SMARTBOARD_TOKEN_EXPIRY_HOURS,
    }
