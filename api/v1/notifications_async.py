"""
Notification system for Stoody platform.

Provides CRUD endpoints for student notifications and a batch-create helper
used by other modules (pdf_async, tutor_async) to generate notifications
when content is activated or teacher feedback is given.
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from config_async import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Web Push helpers — fire-and-forget, never block notification creation
# ---------------------------------------------------------------------------

def _build_push_url(category: str, metadata: Dict[str, Any]) -> str:
    """Build the URL the browser opens when the user clicks the push notification."""
    doc_id = metadata.get("document_id", "")
    hl = f"highlight={doc_id}" if doc_id else ""

    if category == "practice" or category == "feedback":
        return f"/hustle?{hl}" if hl else "/hustle"
    elif category == "test":
        return f"/mcq?{hl}" if hl else "/mcq"
    elif category == "notes":
        return f"/learning?mode=notes&{hl}" if hl else "/learning?mode=notes"
    return "/dashboard"


async def _send_push_for_recipients(
    db: DatabaseManager,
    recipient_ids: List[str],
    title: str,
    message: str,
    category: str,
    metadata: Dict[str, Any],
) -> None:
    """Look up push subscriptions for recipients and send push messages."""
    try:
        from pywebpush import webpush, WebPushException

        if not settings.VAPID_PRIVATE_KEY or not settings.VAPID_PUBLIC_KEY:
            return

        # Find all subscriptions for these recipients
        subs = await db.mongo_find(
            "push_subscriptions",
            {"user_id": {"$in": [str(r) for r in recipient_ids]}},
            limit=5000,
        )
        if not subs:
            return

        url = _build_push_url(category, metadata)
        payload = json.dumps({
            "title": title,
            "body": message,
            "url": url,
            "tag": f"{category}-{metadata.get('document_id', 'general')}",
        })

        vapid_claims = {"sub": settings.VAPID_CLAIMS_EMAIL}
        stale_endpoints = []

        for sub_doc in subs:
            sub_info = sub_doc.get("subscription")
            if not sub_info or not sub_info.get("endpoint"):
                continue
            try:
                webpush(
                    subscription_info=sub_info,
                    data=payload,
                    vapid_private_key=settings.VAPID_PRIVATE_KEY,
                    vapid_claims=vapid_claims,
                )
            except WebPushException as e:
                # 410 Gone or 404 = subscription expired, mark for cleanup
                if hasattr(e, "response") and e.response is not None:
                    sc = e.response.status_code if hasattr(e.response, "status_code") else 0
                    if sc in (404, 410):
                        stale_endpoints.append(sub_info["endpoint"])
                        continue
                logger.debug(f"Push send failed: {e}")
            except Exception as e:
                logger.debug(f"Push send error: {e}")

        # Clean up stale subscriptions
        if stale_endpoints:
            try:
                await db.mongo_delete_many(
                    "push_subscriptions",
                    {"endpoint": {"$in": stale_endpoints}},
                )
                logger.info(f"Cleaned {len(stale_endpoints)} stale push subscriptions")
            except Exception:
                pass

    except ImportError:
        logger.debug("pywebpush not installed — skipping push")
    except Exception as e:
        logger.warning(f"Push delivery error (non-blocking): {e}")

router = APIRouter(tags=["notifications"])
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Batch helper — called by other modules to create notifications
# ---------------------------------------------------------------------------

async def create_notifications_batch(
    db: DatabaseManager,
    admin_id,
    recipient_ids: List[str],
    notif_type: str,
    category: str,
    title: str,
    message: str,
    metadata: Dict[str, Any],
    created_by: str,
    created_by_name: str,
) -> int:
    """
    Bulk-insert notifications for a list of students.

    Returns the number of notifications inserted.
    """
    if not recipient_ids:
        return 0

    now = datetime.utcnow()

    # Ensure admin_id is an ObjectId for tenant scoping
    if admin_id and not isinstance(admin_id, ObjectId):
        try:
            admin_id = ObjectId(str(admin_id))
        except Exception:
            pass

    docs = [
        {
            "admin_id": admin_id,
            "recipient_id": rid,
            "recipient_type": "student",
            "type": notif_type,
            "category": category,
            "title": title,
            "message": message,
            "metadata": metadata or {},
            "is_read": False,
            "read_at": None,
            "created_at": now,
            "created_by": created_by,
            "created_by_name": created_by_name,
        }
        for rid in recipient_ids
    ]

    try:
        result = await db.mongo_insert_many("notifications", docs)
        count = len(result) if result else 0
        logger.info(
            f"Created {count} notifications | type={notif_type} "
            f"category={category} by={created_by_name}"
        )

        # Fire-and-forget push delivery
        try:
            await _send_push_for_recipients(
                db, recipient_ids, title, message, category, metadata or {},
            )
        except Exception as push_err:
            logger.debug(f"Push side-effect failed: {push_err}")

        return count
    except Exception as e:
        logger.error(f"Failed to batch-insert notifications: {e}", exc_info=True)
        return 0


async def create_single_notification(
    db: DatabaseManager,
    admin_id,
    recipient_id: str,
    notif_type: str,
    category: str,
    title: str,
    message: str,
    metadata: Dict[str, Any],
    created_by: str,
    created_by_name: str,
) -> bool:
    """Insert a single notification. Returns True on success."""
    if not recipient_id:
        return False

    now = datetime.utcnow()

    if admin_id and not isinstance(admin_id, ObjectId):
        try:
            admin_id = ObjectId(str(admin_id))
        except Exception:
            pass

    doc = {
        "admin_id": admin_id,
        "recipient_id": recipient_id,
        "recipient_type": "student",
        "type": notif_type,
        "category": category,
        "title": title,
        "message": message,
        "metadata": metadata or {},
        "is_read": False,
        "read_at": None,
        "created_at": now,
        "created_by": created_by,
        "created_by_name": created_by_name,
    }

    try:
        result = await db.mongo_insert_one("notifications", doc)

        # Fire-and-forget push delivery
        if result:
            try:
                await _send_push_for_recipients(
                    db, [recipient_id], title, message, category, metadata or {},
                )
            except Exception as push_err:
                logger.debug(f"Push side-effect failed: {push_err}")

        return result is not None
    except Exception as e:
        logger.error(f"Failed to insert notification: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Student-facing endpoints
# ---------------------------------------------------------------------------

@router.get("/unread-count")
@limiter.limit("120/minute")
async def get_unread_count(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Lightweight endpoint polled every 30s for the badge count."""
    try:
        user_id = (
            current_user.get("user_id")
            or current_user.get("student_id")
            or current_user.get("id")
        )
        if not user_id:
            return {"count": 0}

        count = await db.mongo_count(
            "notifications",
            {"recipient_id": str(user_id), "is_read": False},
        )
        return {"count": count}

    except Exception as e:
        logger.error(f"Unread count error: {e}", exc_info=True)
        return {"count": 0}


@router.get("")
@limiter.limit("60/minute")
async def get_notifications(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Fetch paginated notifications for the current student, newest first."""
    try:
        user_id = (
            current_user.get("user_id")
            or current_user.get("student_id")
            or current_user.get("id")
        )
        if not user_id:
            return {"notifications": [], "total": 0, "page": page, "limit": limit}

        filt = {"recipient_id": str(user_id)}
        skip_n = (page - 1) * limit

        total = await db.mongo_count("notifications", filt)

        docs = await db.mongo_find(
            "notifications",
            filt,
            sort=[("created_at", -1)],
            skip=skip_n,
            limit=limit,
        )

        notifications = []
        for d in docs:
            created_at = d.get("created_at")
            read_at = d.get("read_at")
            notifications.append({
                "id": str(d["_id"]),
                "type": d.get("type", ""),
                "category": d.get("category", ""),
                "title": d.get("title", ""),
                "message": d.get("message", ""),
                "metadata": d.get("metadata", {}),
                "is_read": d.get("is_read", False),
                "read_at": read_at.isoformat() if hasattr(read_at, "isoformat") else None,
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
                "created_by_name": d.get("created_by_name", ""),
            })

        return {
            "notifications": notifications,
            "total": total,
            "page": page,
            "limit": limit,
        }

    except Exception as e:
        logger.error(f"Get notifications error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch notifications",
        )


@router.patch("/{notification_id}/read")
@limiter.limit("120/minute")
async def mark_notification_read(
    request: Request,
    notification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Mark a single notification as read."""
    try:
        user_id = (
            current_user.get("user_id")
            or current_user.get("student_id")
            or current_user.get("id")
        )

        try:
            oid = ObjectId(notification_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid notification ID")

        updated = await db.mongo_update_one(
            "notifications",
            {"_id": oid, "recipient_id": str(user_id)},
            {"$set": {"is_read": True, "read_at": datetime.utcnow()}},
        )

        if not updated:
            raise HTTPException(status_code=404, detail="Notification not found")

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mark read error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark notification as read",
        )


@router.patch("/read-all")
@limiter.limit("30/minute")
async def mark_all_read(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Mark all notifications as read for the current user."""
    try:
        user_id = (
            current_user.get("user_id")
            or current_user.get("student_id")
            or current_user.get("id")
        )
        if not user_id:
            return {"success": True, "updated": 0}

        now = datetime.utcnow()
        updated_count = await db.mongo_update_many(
            "notifications",
            {"recipient_id": str(user_id), "is_read": False},
            {"$set": {"is_read": True, "read_at": now}},
        )

        return {"success": True, "updated": updated_count}

    except Exception as e:
        logger.error(f"Mark all read error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark all notifications as read",
        )
