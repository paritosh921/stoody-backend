"""
B2C admin leaderboard and progress endpoints.
"""

import logging
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.b2c_auth_dependencies import get_current_b2c_admin, get_database
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/admin/leaderboard/progress")
@limiter.limit("30/minute")
async def get_b2c_student_progress(
    request: Request,
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get B2C student progress for leaderboard.
    Uses stoody-b2c database only.
    """
    try:
        users = await db.b2c_find(
            "users",
            {"is_active": True},
            projection={"google_id": 0},
        )

        progress_data = []
        for user in users:
            user_id = user.get("_id")
            user_id_str = str(user_id)

            attempts = await db.b2c_find(
                "student_test_attempts",
                {"student_id": user_id_str},
            )

            total_attempts = len(attempts)
            total_score = sum(a.get("score", 0) for a in attempts)
            total_points = sum(a.get("total_points", 0) for a in attempts)
            avg_score = (total_score / total_points * 100) if total_points > 0 else 0

            question_progress = await db.b2c_find(
                "question_progress",
                {"user_id": user_id_str, "is_correct": True},
            )
            problems_solved = len(question_progress)

            total_time = sum(a.get("time_taken", 0) for a in attempts) / 60

            streak_days = user.get("streak_days", 0)
            level = user.get("level", 1)
            xp = user.get("xp", 0)

            progress_data.append(
                {
                    "student_id": user_id_str,
                    "student_name": user.get(
                        "full_name", user.get("given_name", "Unknown")
                    ),
                    "email": user.get("email", ""),
                    "grade": user.get("class_level", "Unknown"),
                    "section": user.get("exam_type", "Unknown"),
                    "total_sessions": total_attempts,
                    "total_time_spent": int(total_time),
                    "problems_solved": problems_solved,
                    "average_score": round(avg_score, 1),
                    "last_active_at": user.get("last_login"),
                    "streak_days": streak_days,
                    "level": level,
                    "xp": xp,
                    "is_online": user.get("is_online", False),
                }
            )

        progress_data.sort(key=lambda x: x["average_score"], reverse=True)

        return {"success": True, "data": progress_data}

    except Exception as e:
        logger.error(f"Get B2C student progress error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get student progress",
        )


@router.get("/admin/leaderboard/test-attempts")
@limiter.limit("30/minute")
async def get_b2c_test_attempts(
    request: Request,
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get all B2C test attempts for admin leaderboard.
    Uses stoody-b2c database only.
    """
    try:
        attempts = await db.b2c_find(
            "student_test_attempts",
            {},
            sort=[("submitted_at", -1)],
        )

        formatted_attempts = []
        for attempt in attempts:
            student_id = attempt.get("student_id")
            student = (
                await db.b2c_find_one("users", {"_id": ObjectId(student_id)})
                if student_id
                else None
            )

            formatted_attempts.append(
                {
                    "attempt_id": str(attempt.get("_id")),
                    "student_id": student_id,
                    "student_name": student.get("full_name", "Unknown")
                    if student
                    else attempt.get("student_name", "Unknown"),
                    "student_grade": student.get("class_level", "Unknown")
                    if student
                    else attempt.get("student_grade", "Unknown"),
                    "document_id": attempt.get("document_id"),
                    "document_title": attempt.get("document_title", "Unknown Test"),
                    "subject": attempt.get("subject", "Unknown"),
                    "score": attempt.get("score", 0),
                    "total_points": attempt.get("total_points", 0),
                    "percentage": attempt.get("percentage", 0),
                    "total_questions": attempt.get("total_questions", 0),
                    "correct_count": attempt.get("correct_count", 0),
                    "incorrect_count": attempt.get("incorrect_count", 0),
                    "unanswered_count": attempt.get("unanswered_count", 0),
                    "time_taken": attempt.get("time_taken", 0),
                    "total_minutes": attempt.get("total_minutes", 0),
                    "can_reattempt": attempt.get("can_reattempt", False),
                    "submitted_at": attempt.get("submitted_at").isoformat()
                    if attempt.get("submitted_at")
                    else None,
                }
            )

        logger.info(f"Retrieved {len(formatted_attempts)} B2C test attempts")

        return {
            "success": True,
            "data": {"attempts": formatted_attempts, "total": len(formatted_attempts)},
        }

    except Exception as e:
        logger.error(f"Failed to get B2C test attempts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/leaderboard/test-attempts/{attempt_id}/toggle-reattempt")
@limiter.limit("30/minute")
async def toggle_b2c_reattempt(
    request: Request,
    attempt_id: str,
    current_admin: Dict[str, Any] = Depends(get_current_b2c_admin),
    db: DatabaseManager = Depends(get_database),
):
    """
    Toggle the can_reattempt flag for a B2C test attempt.
    Uses stoody-b2c database only.
    """
    try:
        attempt = await db.b2c_find_one(
            "student_test_attempts", {"_id": ObjectId(attempt_id)}
        )
        if not attempt:
            raise HTTPException(status_code=404, detail="Test attempt not found")

        new_value = not attempt.get("can_reattempt", False)

        await db.b2c_update_one(
            "student_test_attempts",
            {"_id": ObjectId(attempt_id)},
            {"$set": {"can_reattempt": new_value}},
        )

        logger.info(f"Toggled B2C re-attempt for attempt {attempt_id} to {new_value}")

        return {
            "success": True,
            "message": f"Re-attempt {'enabled' if new_value else 'disabled'} successfully",
            "data": {"can_reattempt": new_value},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle B2C re-attempt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
