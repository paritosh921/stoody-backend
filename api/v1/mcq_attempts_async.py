"""
Async MCQ attempt endpoints.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_database, get_cache
from api.v1.mcq_dependencies import require_student_or_admin
from api.v1.mcq_schemas import MCQAttempt, MCQAttemptResponse, MCQStats
from core.database import DatabaseManager
from core.cache import CacheManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.post("/attempt", response_model=MCQAttemptResponse)
@limiter.limit("200/minute")
async def attempt_mcq_question(
    request: Request,
    attempt_data: MCQAttempt,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Submit an attempt for an MCQ question"""
    try:
        user_id = current_user["user_id"]

        # Get the question
        question = await db.mongo_find_one("mcq_questions", {"_id": attempt_data.question_id, "is_active": True})

        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCQ question not found"
            )

        # Find the correct option
        correct_option = None
        selected_option = None

        for option in question["options"]:
            if option["is_correct"]:
                correct_option = option
            if option["id"] == attempt_data.selected_option_id:
                selected_option = option

        if not correct_option:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Question has no correct answer marked"
            )

        if not selected_option:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid option selected"
            )

        is_correct = selected_option["is_correct"]

        # Create attempt record
        attempt_record = {
            "student_id": user_id,
            "question_id": attempt_data.question_id,
            "selected_option_id": attempt_data.selected_option_id,
            "correct_option_id": correct_option["id"],
            "is_correct": is_correct,
            "time_spent": attempt_data.time_spent,
            "submitted_at": datetime.utcnow()
        }

        attempt_id = await db.mongo_insert_one("mcq_attempts", attempt_record)

        if not attempt_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to submit attempt"
            )

        return MCQAttemptResponse(
            id=attempt_id,
            question_id=attempt_data.question_id,
            selected_option_id=attempt_data.selected_option_id,
            correct_option_id=correct_option["id"],
            is_correct=is_correct,
            time_spent=attempt_data.time_spent,
            submitted_at=attempt_record["submitted_at"],
            explanation=question.get("explanation")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCQ attempt error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit MCQ attempt"
        )


@router.get("/attempts/my")
@limiter.limit("60/minute")
async def get_my_mcq_attempts(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    question_id: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get student's MCQ attempts"""
    try:
        user_id = current_user["user_id"]

        # Build filter
        filter_dict = {"student_id": user_id}
        if question_id:
            filter_dict["question_id"] = question_id

        # Get total count
        all_attempts = await db.mongo_find("mcq_attempts", filter_dict)
        total_attempts = len(all_attempts)

        # Get paginated results
        skip = (page - 1) * limit
        attempts_data = await db.mongo_find(
            "mcq_attempts",
            filter_dict,
            sort=[("submitted_at", -1)],
            skip=skip,
            limit=limit
        )

        attempts = [
            MCQAttemptResponse(
                id=str(attempt["_id"]),
                question_id=attempt["question_id"],
                selected_option_id=attempt["selected_option_id"],
                correct_option_id=attempt["correct_option_id"],
                is_correct=attempt["is_correct"],
                time_spent=attempt["time_spent"],
                submitted_at=attempt["submitted_at"],
                explanation=None  # Would need to join with question data
            )
            for attempt in attempts_data
        ]

        return {
            "attempts": [a.dict() for a in attempts],
            "total": total_attempts,
            "page": page,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Get MCQ attempts error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCQ attempts"
        )


@router.get("/stats", response_model=MCQStats)
@limiter.limit("30/minute")
async def get_mcq_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get MCQ statistics"""
    try:
        user_id = current_user["user_id"]
        user_type = current_user["user_type"]

        # Check cache first
        cache_key = f"mcq_stats:{user_id}" if user_type == "student" else "mcq_stats:admin"
        cached_stats = await cache.get(cache_key, "mcq")
        if cached_stats:
            return MCQStats(**cached_stats)

        # Get questions count
        questions_filter = {"is_active": True}
        total_questions = len(await db.mongo_find("mcq_questions", questions_filter))

        # Get attempts (filtered by student if needed)
        attempts_filter = {}
        if user_type == "student":
            attempts_filter["student_id"] = user_id

        all_attempts = await db.mongo_find("mcq_attempts", attempts_filter)
        total_attempts = len(all_attempts)
        correct_attempts = len([a for a in all_attempts if a["is_correct"]])

        accuracy_rate = (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0

        # Get questions for breakdown analysis
        all_questions = await db.mongo_find("mcq_questions", questions_filter)

        # Subject and difficulty breakdown
        subject_breakdown = {}
        difficulty_breakdown = {}

        for attempt in all_attempts:
            # Find the question for this attempt
            question = next((q for q in all_questions if str(q["_id"]) == attempt["question_id"]), None)
            if question:
                subject = question["subject"]
                difficulty = question["difficulty"]

                # Subject breakdown
                if subject not in subject_breakdown:
                    subject_breakdown[subject] = {"total": 0, "correct": 0}
                subject_breakdown[subject]["total"] += 1
                if attempt["is_correct"]:
                    subject_breakdown[subject]["correct"] += 1

                # Difficulty breakdown
                if difficulty not in difficulty_breakdown:
                    difficulty_breakdown[difficulty] = {"total": 0, "correct": 0}
                difficulty_breakdown[difficulty]["total"] += 1
                if attempt["is_correct"]:
                    difficulty_breakdown[difficulty]["correct"] += 1

        stats_data = {
            "total_questions": total_questions,
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "accuracy_rate": round(accuracy_rate, 1),
            "subject_breakdown": subject_breakdown,
            "difficulty_breakdown": difficulty_breakdown
        }

        # Cache for 10 minutes
        await cache.set(cache_key, stats_data, 600, "mcq")

        return MCQStats(**stats_data)

    except Exception as e:
        logger.error(f"MCQ stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCQ statistics"
        )


