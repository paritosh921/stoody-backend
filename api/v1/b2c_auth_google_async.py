"""
B2C Google OAuth endpoints.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.b2c_auth_dependencies import get_auth_manager, get_database
from api.v1.b2c_auth_schemas import B2CTokenResponse, GoogleLoginRequest
from config_async import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, settings
from core.auth import AuthManager
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


async def verify_google_token(id_token_str: str) -> Optional[Dict[str, Any]]:
    """Verify Google OAuth ID token and return user info."""
    try:
        # Verify the token with clock skew tolerance (5 seconds)
        # This handles slight clock differences between local machine and Google servers
        idinfo = id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=5
        )

        # Verify issuer
        if idinfo["iss"] not in ["accounts.google.com", "https://accounts.google.com"]:
            raise ValueError("Invalid issuer")

        # Token is valid, extract user info
        return {
            "google_id": idinfo["sub"],
            "email": idinfo["email"],
            "email_verified": idinfo.get("email_verified", False),
            "full_name": idinfo.get("name", ""),
            "given_name": idinfo.get("given_name", ""),
            "family_name": idinfo.get("family_name", ""),
            "picture": idinfo.get("picture", ""),
            "locale": idinfo.get("locale", "en"),
        }

    except ValueError as e:
        logger.error(f"Google token verification failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error verifying Google token: {str(e)}")
        return None


async def upsert_b2c_user_from_google(
    db: DatabaseManager,
    google_user: Dict[str, Any]
) -> Dict[str, Any]:
    """Create or update a B2C user from Google profile data."""
    google_id = google_user["google_id"]
    email = google_user["email"]

    existing_user = await db.b2c_find_one("users", {"google_id": google_id})

    if existing_user:
        await db.b2c_update_one(
            "users",
            {"_id": existing_user["_id"]},
            {
                "$set": {
                    "last_login": datetime.utcnow(),
                    "picture": google_user.get("picture", existing_user.get("picture")),
                }
            },
        )
        user_id = str(existing_user["_id"])
        full_name = existing_user.get("full_name", google_user["full_name"])
        logger.info(f"B2C user logged in: {email}")
        return {
            "user_id": user_id,
            "full_name": full_name,
            "email": email,
            "google_id": google_id,
            "is_new": False,
        }

    new_user = {
        "google_id": google_id,
        "email": email,
        "full_name": google_user["full_name"],
        "given_name": google_user.get("given_name", ""),
        "family_name": google_user.get("family_name", ""),
        "picture": google_user.get("picture", ""),
        "locale": google_user.get("locale", "en"),
        "is_active": True,
        "user_type": "b2c_user",
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
        "admin_id": None,
        "subdomain": None,
    }

    user_id = await db.b2c_insert_one("users", new_user)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account",
        )

    full_name = google_user["full_name"]
    logger.info(f"New B2C user created: {email}")
    return {
        "user_id": user_id,
        "full_name": full_name,
        "email": email,
        "google_id": google_id,
        "is_new": True,
    }


def build_session_payload(
    user_id: str,
    email: str,
    full_name: str,
    google_id: str
) -> Dict[str, Any]:
    """Build session payload for B2C users."""
    return {
        "user_id": user_id,
        "user_type": "b2c_user",
        "email": email,
        "full_name": full_name,
        "google_id": google_id,
        "is_b2c": True,
    }


@router.post("/google/login", response_model=B2CTokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def b2c_google_login(
    request: Request,
    login_data: GoogleLoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    B2C user login/signup via Google OAuth.

    - Verifies Google ID token
    - Creates new user in stoody-b2c database if first login
    - Returns JWT token for session
    """
    try:
        google_user = await verify_google_token(login_data.credential)

        if not google_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google token",
            )

        if not google_user.get("email_verified"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not verified with Google",
            )

        user_info = await upsert_b2c_user_from_google(db, google_user)

        session_data = await auth_manager.create_user_session(
            build_session_payload(
                user_info["user_id"],
                user_info["email"],
                user_info["full_name"],
                user_info["google_id"],
            )
        )

        try:
            await db.b2c_insert_one(
                "user_activity_log",
                {
                    "user_id": ObjectId(user_info["user_id"]),
                    "action": "login",
                    "timestamp": datetime.utcnow(),
                    "metadata": {
                        "ip_address": request.client.host if request.client else "unknown",
                        "user_agent": request.headers.get("user-agent", "unknown"),
                        "login_method": "google_oauth",
                    },
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log B2C user activity: {str(e)}")

        return B2CTokenResponse(
            success=True,
            data={
                "access_token": session_data["access_token"],
                "user_type": "b2c_user",
                "user": {
                    "user_id": user_info["user_id"],
                    "email": user_info["email"],
                    "full_name": user_info["full_name"],
                    "picture": google_user.get("picture", ""),
                    "is_b2c": True,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C Google login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        )


@router.get("/google/authorize")
async def b2c_google_authorize(redirect_uri: str):
    """
    Initiates Google OAuth flow for Desktop Agent.
    Redirects user to Google's consent page.
    """
    from fastapi.responses import RedirectResponse
    import urllib.parse

    auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    api_base = os.getenv("PUBLIC_API_URL", "https://api.stoody.in/api/v1")
    callback_url = f"{api_base}/b2c/google/callback"

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": callback_url,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "state": redirect_uri,
    }

    url = f"{auth_url}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url)


@router.get("/google/callback")
async def b2c_google_callback(
    code: str,
    state: str,
    request: Request,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Handles callback from Google, exchanges code for token,
    and redirects back to the Desktop Agent.
    """
    from fastapi.responses import RedirectResponse
    import httpx
    import urllib.parse

    agent_callback = state

    try:
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{os.getenv('PUBLIC_API_URL', 'https://api.stoody.in/api/v1')}/b2c/google/callback",
        }

        async with httpx.AsyncClient() as client:
            token_resp = await client.post(token_url, data=data)
            token_resp.raise_for_status()
            tokens = token_resp.json()

        id_token_str = tokens.get("id_token")

        google_user = await verify_google_token(id_token_str)
        if not google_user:
            return RedirectResponse(f"{agent_callback}?error=invalid_token")

        user_info = await upsert_b2c_user_from_google(db, google_user)

        session_data = await auth_manager.create_user_session(
            build_session_payload(
                user_info["user_id"],
                user_info["email"],
                user_info["full_name"],
                user_info["google_id"],
            )
        )
        access_token = session_data["access_token"]

        encoded_token = urllib.parse.quote(access_token, safe="")
        encoded_email = urllib.parse.quote(user_info["email"], safe="")
        return RedirectResponse(
            f"{agent_callback}?token={encoded_token}&user_id={user_info['user_id']}&email={encoded_email}"
        )

    except Exception as e:
        logger.error(f"Google Callback Error: {e}")
        error_msg = urllib.parse.quote(str(e))
        return RedirectResponse(f"{agent_callback}?error=login_failed&detail={error_msg}")
