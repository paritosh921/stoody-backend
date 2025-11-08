"""
Async Student API for SkillBot
Student-specific endpoints with authentication
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user, get_database, get_cache
from config_async import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
class StudentProfile(BaseModel):
    user_id: str
    username: str
    full_name: str
    email: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None

class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=6)
    new_password: str = Field(..., min_length=6)

class AttemptSubmission(BaseModel):
    question_id: str
    answer: str
    mode: str = Field(default="practice")
    canvas_data: Optional[str] = None
    time_spent: int = Field(default=0, ge=0)

class AttemptResponse(BaseModel):
    id: str
    question_id: str
    answer: str
    is_correct: bool
    score: float
    feedback: Optional[str] = None
    submitted_at: datetime

class StudentDashboardStats(BaseModel):
    total_attempts: int
    correct_attempts: int
    accuracy_rate: float
    total_time_spent: int
    subjects_practiced: List[str]
    recent_activity: int

def require_student_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require student or admin access"""
    if current_user.get("user_type") not in ["student", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student or admin access required"
        )
    return current_user

def require_student(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require student access"""
    if current_user.get("user_type") != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student access required"
        )
    return current_user

@router.get("/profile", response_model=StudentProfile)
@limiter.limit("60/minute")
async def get_student_profile(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get student profile"""
    try:
        user_id = current_user["user_id"]
        user_type = current_user["user_type"]

        # Admin can view any profile, students can only view their own
        if user_type == "student":
            student_data = await db.mongo_find_one(
                "students",
                {"_id": user_id},
                projection={"password_hash": 0}
            )
        else:
            # For admin access (if needed for debugging)
            student_data = await db.mongo_find_one(
                "students",
                {"_id": user_id},
                projection={"password_hash": 0}
            )

        if not student_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student profile not found"
            )

        return StudentProfile(
            user_id=str(student_data["_id"]),
            username=student_data["username"],
            full_name=student_data["full_name"],
            email=student_data.get("email"),
            created_at=student_data["created_at"],
            last_login=student_data.get("last_login")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get student profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get student profile"
        )

@router.post("/attempts", response_model=AttemptResponse)
@limiter.limit("100/minute")
async def submit_attempt(
    request: Request,
    attempt_data: AttemptSubmission,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database)
):
    """Submit a question attempt"""
    try:
        user_id = current_user["user_id"]

        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Get the question from admin's ChromaDB collection
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)
        question = question_service.get_question(attempt_data.question_id)

        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found in your admin's collection"
            )

        # For now, we'll create a simple scoring system
        # In production, you'd want more sophisticated answer evaluation
        is_correct = False
        score = 0.0
        feedback = "Answer submitted successfully"

        # Create attempt record
        attempt_record = {
            "student_id": user_id,
            "question_id": attempt_data.question_id,
            "answer": attempt_data.answer,
            "canvas_data": attempt_data.canvas_data,
            "mode": attempt_data.mode,
            "is_correct": is_correct,
            "score": score,
            "feedback": feedback,
            "time_spent": attempt_data.time_spent,
            "submitted_at": datetime.utcnow()
        }

        attempt_id = await db.mongo_insert_one("attempts", attempt_record)

        if not attempt_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to submit attempt"
            )

        return AttemptResponse(
            id=attempt_id,
            question_id=attempt_data.question_id,
            answer=attempt_data.answer,
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            submitted_at=attempt_record["submitted_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit attempt error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit attempt"
        )

@router.get("/attempts")
@limiter.limit("60/minute")
async def get_my_attempts(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    mode: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database)
):
    """Get student's attempts"""
    try:
        user_id = current_user["user_id"]

        # Build filter
        filter_dict = {"student_id": user_id}
        if mode:
            filter_dict["mode"] = mode

        # Get paginated attempts
        skip = (page - 1) * limit
        attempts = await db.mongo_find(
            "attempts",
            filter_dict,
            sort=[("submitted_at", -1)],
            skip=skip,
            limit=limit
        )

        # Get total count
        total_attempts = len(await db.mongo_find("attempts", filter_dict))

        attempts_response = [
            AttemptResponse(
                id=str(attempt["_id"]),
                question_id=attempt["question_id"],
                answer=attempt["answer"],
                is_correct=attempt.get("is_correct", False),
                score=attempt.get("score", 0.0),
                feedback=attempt.get("feedback"),
                submitted_at=attempt["submitted_at"]
            )
            for attempt in attempts
        ]

        return {
            "attempts": [a.dict() for a in attempts_response],
            "total": total_attempts,
            "page": page,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Get attempts error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get attempts"
        )

@router.get("/dashboard/stats", response_model=StudentDashboardStats)
@limiter.limit("30/minute")
async def get_student_dashboard_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get student dashboard statistics"""
    try:
        user_id = current_user["user_id"]

        # Check cache first
        cache_key = f"student_stats:{user_id}"
        cached_stats = await cache.get(cache_key, "student")
        if cached_stats:
            return StudentDashboardStats(**cached_stats)

        # Get all attempts for this student
        all_attempts = await db.mongo_find("attempts", {"student_id": user_id})

        total_attempts = len(all_attempts)
        correct_attempts = len([a for a in all_attempts if a.get("is_correct", False)])
        accuracy_rate = (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0
        total_time_spent = sum(a.get("time_spent", 0) for a in all_attempts)

        # Get unique subjects practiced (would need to be enhanced based on question metadata)
        subjects_practiced = ["Math", "Physics", "Chemistry"]  # Mock data

        # Recent activity (last 7 days)
        from datetime import timedelta
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_attempts = [a for a in all_attempts if a["submitted_at"] >= recent_cutoff]

        stats_data = {
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "accuracy_rate": round(accuracy_rate, 1),
            "total_time_spent": total_time_spent,
            "subjects_practiced": subjects_practiced,
            "recent_activity": len(recent_attempts)
        }

        # Cache for 10 minutes
        await cache.set(cache_key, stats_data, 600, "student")

        return StudentDashboardStats(**stats_data)

    except Exception as e:
        logger.error(f"Student dashboard stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dashboard statistics"
        )

@router.post("/change-password")
@limiter.limit("5/minute")
async def change_password(
    request: Request,
    password_data: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database)
):
    """Change student password"""
    try:
        from core.auth import AuthManager
        auth_manager = AuthManager()

        user_id = current_user["user_id"]

        # Change password
        success = await auth_manager.change_password(
            user_id,
            password_data.old_password,
            password_data.new_password,
            "student",
            db
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid old password or password change failed"
            )

        return {"message": "Password changed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )