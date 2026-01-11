"""
B2C user onboarding and profile endpoints.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.b2c_auth_dependencies import get_current_b2c_user, get_database
from api.v1.b2c_auth_schemas import B2COnboardingRequest, B2CProfileUpdateRequest
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/profile/onboarding")
@limiter.limit("10/minute")
async def b2c_onboarding(
    request: Request,
    onboarding_data: B2COnboardingRequest,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """
    Complete B2C user onboarding with plan selection and personal details.

    - Updates user profile with exam type (JEE/NEET) and class level
    - Stores personal details (phone, school, city)
    - Marks onboarding as complete
    """
    try:
        user_id = current_user.get("user_id")

        if onboarding_data.exam_type not in ["JEE", "NEET"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid exam type. Must be 'JEE' or 'NEET'",
            )

        valid_classes = ["9", "10", "11", "12", "Dropper"]
        if onboarding_data.class_level not in valid_classes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid class level. Must be one of: {', '.join(valid_classes)}",
            )

        update_data = {
            "full_name": onboarding_data.full_name,
            "phone": onboarding_data.phone,
            "exam_type": onboarding_data.exam_type,
            "class_level": onboarding_data.class_level,
            "onboarding_complete": True,
            "onboarding_completed_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        if onboarding_data.school_name:
            update_data["school_name"] = onboarding_data.school_name
        if onboarding_data.city:
            update_data["city"] = onboarding_data.city

        if onboarding_data.exam_type == "JEE":
            update_data["subjects"] = ["Physics", "Chemistry", "Mathematics"]
        else:
            update_data["subjects"] = ["Physics", "Chemistry", "Biology"]

        if onboarding_data.class_level == "Dropper":
            update_data["standard"] = "12"
            update_data["is_dropper"] = True
        else:
            update_data["standard"] = onboarding_data.class_level
            update_data["is_dropper"] = False

        update_data["plan_types"] = [onboarding_data.exam_type]

        result = await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": update_data},
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update profile",
            )

        try:
            await db.b2c_insert_one(
                "user_activity_log",
                {
                    "user_id": ObjectId(user_id),
                    "action": "onboarding_complete",
                    "timestamp": datetime.utcnow(),
                    "metadata": {
                        "exam_type": onboarding_data.exam_type,
                        "class_level": onboarding_data.class_level,
                    },
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log onboarding activity: {str(e)}")

        logger.info(
            "B2C user onboarding complete: %s - %s/%s",
            user_id,
            onboarding_data.exam_type,
            onboarding_data.class_level,
        )

        return {
            "success": True,
            "message": "Onboarding completed successfully",
            "data": {
                "user_id": user_id,
                "exam_type": onboarding_data.exam_type,
                "class_level": onboarding_data.class_level,
                "subjects": update_data["subjects"],
                "standard": update_data["standard"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"B2C onboarding error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Onboarding failed",
        )


@router.get("/profile")
async def get_b2c_user_full_profile(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get full B2C user profile including plan and personal details."""
    try:
        user_id = current_user.get("user_id")

        user = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        return {
            "success": True,
            "data": {
                "user_id": str(user["_id"]),
                "email": user.get("email", ""),
                "full_name": user.get("full_name", ""),
                "picture": user.get("picture"),
                "phone": user.get("phone"),
                "school_name": user.get("school_name"),
                "city": user.get("city"),
                "exam_type": user.get("exam_type"),
                "class_level": user.get("class_level"),
                "standard": user.get("standard"),
                "subjects": user.get("subjects", []),
                "plan_types": user.get("plan_types", []),
                "is_dropper": user.get("is_dropper", False),
                "onboarding_complete": user.get("onboarding_complete", False),
                "consent_completed": user.get("consent_completed", False),
                "is_minor": user.get("is_minor", False),
                "has_parental_consent": user.get("has_parental_consent", False),
                "parent_info": user.get("parent_info"),
                "gdpr_consent": user.get("gdpr_consent"),
                "ai_personalization_consent": user.get("ai_personalization_consent"),
                "marketing_consent": user.get("marketing_consent"),
                "consent_timestamp": user.get("consent_timestamp"),
                "created_at": user.get("created_at", datetime.utcnow()).isoformat(),
                "last_login": user.get("last_login", datetime.utcnow()).isoformat()
                if user.get("last_login")
                else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get B2C profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch profile",
        )


@router.put("/profile")
@limiter.limit("10/minute")
async def update_b2c_user_profile(
    request: Request,
    profile_data: B2CProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """Update B2C user profile."""
    try:
        user_id = current_user.get("user_id")

        update_data = {"updated_at": datetime.utcnow()}

        if profile_data.full_name:
            update_data["full_name"] = profile_data.full_name
        if profile_data.phone:
            update_data["phone"] = profile_data.phone
        if profile_data.school_name is not None:
            update_data["school_name"] = profile_data.school_name
        if profile_data.city is not None:
            update_data["city"] = profile_data.city

        if profile_data.exam_type:
            if profile_data.exam_type not in ["JEE", "NEET"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid exam type. Must be 'JEE' or 'NEET'",
                )
            update_data["exam_type"] = profile_data.exam_type
            update_data["plan_types"] = [profile_data.exam_type]

            if profile_data.exam_type == "JEE":
                update_data["subjects"] = ["Physics", "Chemistry", "Mathematics"]
            else:
                update_data["subjects"] = ["Physics", "Chemistry", "Biology"]

        if profile_data.class_level:
            valid_classes = ["9", "10", "11", "12", "Dropper"]
            if profile_data.class_level not in valid_classes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid class level. Must be one of: {', '.join(valid_classes)}",
                )
            update_data["class_level"] = profile_data.class_level

            if profile_data.class_level == "Dropper":
                update_data["standard"] = "12"
                update_data["is_dropper"] = True
            else:
                update_data["standard"] = profile_data.class_level
                update_data["is_dropper"] = False

        result = await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": update_data},
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update profile",
            )

        logger.info(f"B2C user profile updated: {user_id}")

        return {
            "success": True,
            "message": "Profile updated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update B2C profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile",
        )


@router.get("/profile/check-onboarding")
async def check_b2c_onboarding_status(
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """Check if B2C user has completed onboarding."""
    try:
        user_id = current_user.get("user_id")

        user = await db.b2c_find_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"onboarding_complete": 1, "exam_type": 1, "class_level": 1},
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        is_complete = user.get("onboarding_complete", False)

        return {
            "success": True,
            "data": {
                "onboarding_complete": is_complete,
                "exam_type": user.get("exam_type") if is_complete else None,
                "class_level": user.get("class_level") if is_complete else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Check onboarding status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check onboarding status",
        )
