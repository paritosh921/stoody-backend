"""
Web Push subscription management for Stoody.

Students subscribe via the browser Push API; their PushSubscription objects
are stored in MongoDB so the notification helpers can send push messages
when new notifications are created.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from config_async import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["push"])
limiter = Limiter(key_func=get_remote_address)


class PushSubscriptionRequest(BaseModel):
    subscription: Dict[str, Any] = Field(
        ..., description="Browser PushSubscription JSON object"
    )


@router.post("/subscribe")
@limiter.limit("30/minute")
async def push_subscribe(
    request: Request,
    body: PushSubscriptionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Store or update a push subscription for the current user."""
    try:
        user_id = (
            current_user.get("user_id")
            or current_user.get("student_id")
            or current_user.get("id")
        )
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found")

        endpoint = body.subscription.get("endpoint", "")
        if not endpoint:
            raise HTTPException(status_code=400, detail="Invalid subscription: missing endpoint")

        # Upsert by endpoint — one device = one subscription
        await db.mongo_update_one(
            "push_subscriptions",
            {"endpoint": endpoint},
            {
                "$set": {
                    "user_id": str(user_id),
                    "subscription": body.subscription,
                    "updated_at": datetime.utcnow(),
                    "user_agent": request.headers.get("user-agent", ""),
                },
                "$setOnInsert": {
                    "created_at": datetime.utcnow(),
                },
            },
            upsert=True,
        )

        return {"success": True, "message": "Push subscription saved"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Push subscribe error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save push subscription",
        )


@router.delete("/unsubscribe")
@limiter.limit("30/minute")
async def push_unsubscribe(
    request: Request,
    body: PushSubscriptionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Remove a push subscription."""
    try:
        endpoint = body.subscription.get("endpoint", "")
        if not endpoint:
            raise HTTPException(status_code=400, detail="Invalid subscription")

        await db.mongo_delete_one("push_subscriptions", {"endpoint": endpoint})
        return {"success": True, "message": "Push subscription removed"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Push unsubscribe error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove push subscription",
        )


@router.get("/vapid-public-key")
async def get_vapid_public_key():
    """Return the VAPID public key so the frontend can subscribe."""
    return {"public_key": settings.VAPID_PUBLIC_KEY}
