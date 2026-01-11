"""
Practice Module - Session Service
Logic for managing practice sessions
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from bson import ObjectId

from core.database import DatabaseManager
from core.cache import CacheManager
from ..models import (
    StartSessionRequest,
    SessionResponse,
    SessionAnswer,
    SessionsListResponse,
    PracticeStats
)

logger = logging.getLogger(__name__)


async def create_session(
    session_data: StartSessionRequest,
    current_user: Dict[str, Any],
    db: DatabaseManager
) -> SessionResponse:
    """Create a new practice session.

    Args:
        session_data: Session configuration
        current_user: Current authenticated user
        db: Database manager

    Returns:
        SessionResponse with created session data
    """
    user_id = current_user["user_id"]

    # Detect if user is B2C
    user_type = current_user.get("user_type", "")
    is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

    # Create session record
    session_record = {
        "student_id": user_id,
        "mode": session_data.mode,
        "subject": session_data.subject,
        "difficulty": session_data.difficulty,
        "time_limit": session_data.time_limit,
        "document_id": session_data.document_id,
        "questions_attempted": 0,
        "correct_answers": 0,
        "total_time_spent": 0,
        "started_at": datetime.utcnow(),
        "is_completed": False,
        "questions": []
    }

    # Use B2C database for B2C users
    if is_b2c:
        session_id = await db.b2c_insert_one("practice_sessions", session_record)
    else:
        session_id = await db.mongo_insert_one("practice_sessions", session_record)

    if not session_id:
        return None

    return SessionResponse(
        id=session_id,
        mode=session_data.mode,
        subject=session_data.subject,
        difficulty=session_data.difficulty,
        questions_attempted=0,
        correct_answers=0,
        accuracy_rate=0.0,
        total_time_spent=0,
        started_at=session_record["started_at"],
        is_completed=False
    )


async def submit_answer(
    session_id: str,
    answer_data: SessionAnswer,
    current_user: Dict[str, Any],
    db: DatabaseManager
) -> Dict[str, Any]:
    """Submit an answer for a question in a practice session.

    Args:
        session_id: Session ID
        answer_data: Answer submission data
        current_user: Current authenticated user
        db: Database manager

    Returns:
        Dict with submission result
    """
    user_id = current_user["user_id"]

    # Get session
    session = await db.mongo_find_one("practice_sessions", {"_id": session_id})

    if not session:
        return {"error": "Practice session not found", "status_code": 404}

    # Check ownership
    if (current_user["user_type"] == "student" and
        session["student_id"] != user_id):
        return {"error": "Access denied", "status_code": 403}

    if session["is_completed"]:
        return {"error": "Session is already completed", "status_code": 400}

    # Get question to validate answer
    question = await db.mongo_find_one("questions", {"question_id": answer_data.question_id})

    # Validate answer
    is_correct = False
    score = 0
    if question:
        correct_answer = question.get("correct_answer", "")
        is_correct = (answer_data.answer.strip().lower() == correct_answer.strip().lower())
        if is_correct:
            score = question.get("points", 1.0)

    # Create question attempt record
    question_attempt = {
        "question_id": answer_data.question_id,
        "answer": answer_data.answer,
        "is_correct": is_correct,
        "time_spent": answer_data.time_spent,
        "answered_at": datetime.utcnow()
    }

    # Update session
    update_data = {
        "$push": {"questions": question_attempt},
        "$inc": {
            "questions_attempted": 1,
            "total_time_spent": answer_data.time_spent
        }
    }

    if is_correct:
        update_data["$inc"]["correct_answers"] = 1

    await db.mongo_update_one(
        "practice_sessions",
        {"_id": session_id},
        update_data
    )

    # Track in question_attempts collection for student monitoring
    if current_user["user_type"] == "student":
        await _track_question_attempt(
            user_id=user_id,
            session_id=session_id,
            answer_data=answer_data,
            is_correct=is_correct,
            score=score,
            question=question,
            current_user=current_user,
            db=db
        )

    return {
        "message": "Answer submitted successfully",
        "is_correct": is_correct,
        "question_id": answer_data.question_id,
        "score": score
    }


async def _track_question_attempt(
    user_id: str,
    session_id: str,
    answer_data: SessionAnswer,
    is_correct: bool,
    score: float,
    question: Optional[Dict[str, Any]],
    current_user: Dict[str, Any],
    db: DatabaseManager
) -> None:
    """Track question attempt for analytics."""
    try:
        student_oid = ObjectId(user_id)

        admin_id = current_user.get("admin_id")
        if not admin_id:
            logger.warning(f"Student {user_id} has no admin_id in JWT token")

        # Insert into question_attempts collection
        attempt_doc = {
            "student_id": student_oid,
            "question_id": answer_data.question_id,
            "session_id": session_id,
            "answer": answer_data.answer,
            "is_correct": is_correct,
            "score": score,
            "time_spent": answer_data.time_spent,
            "created_at": datetime.utcnow(),
            "metadata": {
                "subject": question.get("subject") if question else None,
                "difficulty": question.get("difficulty") if question else None
            }
        }

        if admin_id:
            attempt_doc["admin_id"] = admin_id

        await db.mongo_insert_one("question_attempts", attempt_doc)

        # Log activity
        activity_doc = {
            "student_id": student_oid,
            "action": "question_attempted",
            "timestamp": datetime.utcnow(),
            "metadata": {
                "question_id": answer_data.question_id,
                "session_id": session_id,
                "is_correct": is_correct,
                "score": score,
                "time_spent": answer_data.time_spent
            }
        }

        if admin_id:
            activity_doc["admin_id"] = admin_id

        await db.mongo_insert_one("student_activity_log", activity_doc)
    except Exception as e:
        logger.warning(f"Failed to track question attempt: {str(e)}")


async def complete_session(
    session_id: str,
    current_user: Dict[str, Any],
    db: DatabaseManager
) -> SessionResponse:
    """Complete a practice session.

    Args:
        session_id: Session ID
        current_user: Current authenticated user
        db: Database manager

    Returns:
        SessionResponse with completed session data
    """
    user_id = current_user["user_id"]

    # Get session
    session = await db.mongo_find_one("practice_sessions", {"_id": session_id})

    if not session:
        return {"error": "Practice session not found", "status_code": 404}

    # Check ownership
    if (current_user["user_type"] == "student" and
        session["student_id"] != user_id):
        return {"error": "Access denied", "status_code": 403}

    if session["is_completed"]:
        return {"error": "Session is already completed", "status_code": 400}

    # Mark session as completed
    await db.mongo_update_one(
        "practice_sessions",
        {"_id": session_id},
        {
            "$set": {
                "is_completed": True,
                "completed_at": datetime.utcnow()
            }
        }
    )

    # Get updated session
    updated_session = await db.mongo_find_one("practice_sessions", {"_id": session_id})

    accuracy_rate = 0.0
    if updated_session["questions_attempted"] > 0:
        accuracy_rate = (updated_session["correct_answers"] / updated_session["questions_attempted"]) * 100

    return SessionResponse(
        id=session_id,
        mode=updated_session["mode"],
        subject=updated_session.get("subject"),
        difficulty=updated_session.get("difficulty"),
        questions_attempted=updated_session["questions_attempted"],
        correct_answers=updated_session["correct_answers"],
        accuracy_rate=round(accuracy_rate, 1),
        total_time_spent=updated_session["total_time_spent"],
        started_at=updated_session["started_at"],
        completed_at=updated_session["completed_at"],
        is_completed=True
    )


async def get_sessions(
    page: int,
    limit: int,
    mode: Optional[str],
    is_completed: Optional[bool],
    current_user: Dict[str, Any],
    db: DatabaseManager
) -> SessionsListResponse:
    """Get practice sessions with pagination.

    Args:
        page: Page number
        limit: Items per page
        mode: Filter by mode
        is_completed: Filter by completion status
        current_user: Current authenticated user
        db: Database manager

    Returns:
        SessionsListResponse with paginated sessions
    """
    user_id = current_user["user_id"]
    user_type = current_user["user_type"]

    # Build filter
    filter_dict = {}
    if user_type == "student":
        filter_dict["student_id"] = user_id

    if mode:
        filter_dict["mode"] = mode
    if is_completed is not None:
        filter_dict["is_completed"] = is_completed

    # Get total count
    all_sessions = await db.mongo_find("practice_sessions", filter_dict)
    total_sessions = len(all_sessions)

    # Get paginated results
    skip = (page - 1) * limit
    sessions_data = await db.mongo_find(
        "practice_sessions",
        filter_dict,
        sort=[("started_at", -1)],
        skip=skip,
        limit=limit
    )

    sessions = []
    for session in sessions_data:
        accuracy_rate = 0.0
        if session["questions_attempted"] > 0:
            accuracy_rate = (session["correct_answers"] / session["questions_attempted"]) * 100

        sessions.append(SessionResponse(
            id=str(session["_id"]),
            mode=session["mode"],
            subject=session.get("subject"),
            difficulty=session.get("difficulty"),
            questions_attempted=session["questions_attempted"],
            correct_answers=session["correct_answers"],
            accuracy_rate=round(accuracy_rate, 1),
            total_time_spent=session["total_time_spent"],
            started_at=session["started_at"],
            completed_at=session.get("completed_at"),
            is_completed=session["is_completed"]
        ))

    return SessionsListResponse(
        sessions=sessions,
        total=total_sessions,
        page=page,
        limit=limit
    )


async def get_stats(
    current_user: Dict[str, Any],
    db: DatabaseManager,
    cache: CacheManager
) -> PracticeStats:
    """Get practice statistics.

    Args:
        current_user: Current authenticated user
        db: Database manager
        cache: Cache manager

    Returns:
        PracticeStats with aggregated statistics
    """
    user_id = current_user["user_id"]
    user_type = current_user["user_type"]

    # Check cache first
    cache_key = f"practice_stats:{user_id}" if user_type == "student" else "practice_stats:admin"
    cached_stats = await cache.get(cache_key, "practice")
    if cached_stats:
        return PracticeStats(**cached_stats)

    # Build filter
    filter_dict = {}
    if user_type == "student":
        filter_dict["student_id"] = user_id

    # Get all sessions
    all_sessions = await db.mongo_find("practice_sessions", filter_dict)

    total_sessions = len(all_sessions)
    total_time_spent = sum(s.get("total_time_spent", 0) for s in all_sessions)

    # Calculate average accuracy
    completed_sessions = [s for s in all_sessions if s.get("is_completed", False)]
    total_accuracy = 0
    if completed_sessions:
        for session in completed_sessions:
            if session["questions_attempted"] > 0:
                accuracy = (session["correct_answers"] / session["questions_attempted"]) * 100
                total_accuracy += accuracy
        average_accuracy = total_accuracy / len(completed_sessions)
    else:
        average_accuracy = 0.0

    # Sessions by mode
    sessions_by_mode = {}
    for session in all_sessions:
        mode = session.get("mode", "unknown")
        sessions_by_mode[mode] = sessions_by_mode.get(mode, 0) + 1

    # Recent activity (last 7 days)
    recent_cutoff = datetime.utcnow() - timedelta(days=7)
    recent_sessions = [s for s in all_sessions if s["started_at"] >= recent_cutoff]
    recent_activity = [
        {
            "date": session["started_at"].date().isoformat(),
            "mode": session["mode"],
            "questions_attempted": session["questions_attempted"],
            "accuracy": round((session["correct_answers"] / session["questions_attempted"]) * 100, 1) if session["questions_attempted"] > 0 else 0
        }
        for session in recent_sessions[-10:]
    ]

    stats_data = {
        "total_sessions": total_sessions,
        "total_time_spent": total_time_spent,
        "average_accuracy": round(average_accuracy, 1),
        "sessions_by_mode": sessions_by_mode,
        "recent_activity": recent_activity
    }

    # Cache for 10 minutes
    await cache.set(cache_key, stats_data, 600, "practice")

    return PracticeStats(**stats_data)
