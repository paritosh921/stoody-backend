"""
Parent-facing GDPR endpoints for B2C accounts.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.b2c_auth_dependencies import (
    get_auth_manager,
    get_current_b2c_user,
    get_database,
)
from core.auth import AuthManager
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/parent/dashboard")
async def get_parent_dashboard(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get parent dashboard data for minor accounts.

    Returns:
    - Parent/guardian info
    - Child's data summary
    - Consent record details
    """
    try:
        user_id = current_user.get("user_id")

        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        if not user.get("is_minor") or not user.get("has_parental_consent"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is only for accounts with parental consent",
            )

        learning_sessions = await db.b2c_find(
            "user_activity_log", {"user_id": ObjectId(user_id)}
        )

        data_summary = {
            "learning_sessions": len(
                [l for l in learning_sessions if l.get("action") == "learning_session"]
            ),
            "handwriting_samples": 0,
            "audio_recordings": 0,
            "practice_tests": len(
                [
                    l
                    for l in learning_sessions
                    if l.get("action") in ["test_submitted", "practice_submitted"]
                ]
            ),
            "total_study_hours": 0,
            "last_active": user.get("last_login", datetime.utcnow()).isoformat()
            if user.get("last_login")
            else "Never",
        }

        consent_record = None
        consent_log = await db.b2c_find_one(
            "consent_audit_log",
            {"user_id": ObjectId(user_id), "action": "consent_granted"},
            sort=[("timestamp", -1)],
        )
        if consent_log:
            consent_record = {
                "consent_timestamp": consent_log.get(
                    "timestamp", datetime.utcnow()
                ).isoformat(),
                "consent_version": consent_log.get("consent_version", "1.0"),
                "scc_version": consent_log.get("scc_version", "2021/914"),
                "ip_address": "Logged securely",
            }

        return {
            "success": True,
            "data": {
                "parent_info": user.get("parent_info"),
                "data_summary": data_summary,
                "consent_record": consent_record,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parent dashboard error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch parent dashboard",
        )


@router.post("/parent/export-data")
@limiter.limit("3/hour")
async def request_data_export(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """
    Request data export (GDPR data portability).

    Creates an export request that will be processed and emailed to the parent.
    """
    try:
        user_id = current_user.get("user_id")

        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})

        if not user or not user.get("has_parental_consent"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Data export requires parental consent",
            )

        await db.b2c_insert_one(
            "data_export_requests",
            {
                "user_id": ObjectId(user_id),
                "parent_email": user.get("parent_info", {}).get("email"),
                "status": "pending",
                "requested_at": datetime.utcnow(),
                "ip_address": request.client.host if request.client else "unknown",
            },
        )

        await db.b2c_insert_one(
            "consent_audit_log",
            {
                "user_id": ObjectId(user_id),
                "consent_type": "parental",
                "action": "data_export_requested",
                "timestamp": datetime.utcnow(),
                "ip_address": request.client.host if request.client else "unknown",
            },
        )

        logger.info(f"Data export requested for user: {user_id}")

        return {
            "success": True,
            "message": "Data export request submitted. You will receive an email within 24 hours.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data export request error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit data export request",
        )


@router.post("/parent/withdraw-consent")
@limiter.limit("3/hour")
async def withdraw_consent(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Withdraw parental consent.

    - Suspends the child's account
    - Queues data for deletion
    - Sends confirmation email
    """
    try:
        user_id = current_user.get("user_id")

        await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "is_active": False,
                    "consent_withdrawn": True,
                    "consent_withdrawn_at": datetime.utcnow(),
                    "deletion_scheduled": True,
                    "deletion_scheduled_at": datetime.utcnow(),
                }
            },
        )

        await db.b2c_insert_one(
            "consent_audit_log",
            {
                "user_id": ObjectId(user_id),
                "consent_type": "parental",
                "action": "consent_withdrawn",
                "timestamp": datetime.utcnow(),
                "ip_address": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            },
        )

        await auth_manager.invalidate_user_session(user_id)

        logger.info(f"Parental consent withdrawn for user: {user_id}")

        return {
            "success": True,
            "message": "Consent withdrawn. Account suspended and data queued for deletion.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Consent withdrawal error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to withdraw consent",
        )


@router.delete("/parent/delete-data")
@limiter.limit("1/hour")
async def delete_all_data(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Delete all user data (GDPR Right to be Forgotten).

    - Queues all data for permanent deletion
    - Keeps only legal compliance logs
    - Sends confirmation email
    """
    try:
        user_id = current_user.get("user_id")

        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})

        if not user or not user.get("has_parental_consent"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This action requires parental consent verification",
            )

        await db.b2c_insert_one(
            "data_deletion_requests",
            {
                "user_id": ObjectId(user_id),
                "parent_email": user.get("parent_info", {}).get("email"),
                "status": "pending",
                "requested_at": datetime.utcnow(),
                "ip_address": request.client.host if request.client else "unknown",
            },
        )

        await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "is_active": False,
                    "deletion_requested": True,
                    "deletion_requested_at": datetime.utcnow(),
                }
            },
        )

        await db.b2c_insert_one(
            "consent_audit_log",
            {
                "user_id": ObjectId(user_id),
                "consent_type": "parental",
                "action": "data_deletion_requested",
                "timestamp": datetime.utcnow(),
                "ip_address": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "note": "Compliance log - retained for legal purposes",
            },
        )

        await auth_manager.invalidate_user_session(user_id)

        logger.info(f"Data deletion requested for user: {user_id}")

        return {
            "success": True,
            "message": "Data deletion request submitted. All data will be permanently deleted within 30 days.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit data deletion request",
        )
