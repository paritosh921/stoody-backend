import logging
import traceback
from typing import Dict, Any, List
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, HTTPException, Depends, status, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from api.v1.auth_async import get_database
from api.v1.mcq_dependencies import require_student_or_admin

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/test-series/{document_id}/check-attempt")
@limiter.limit("60/minute")
async def check_test_attempt(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Check if student has already attempted this test series
    Returns attempt status and whether re-attempt is allowed
    """
    try:
        user_id = current_user["user_id"]
        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

        # Admins can always access
        if user_type == "admin":
            return {
                "success": True,
                "has_attempted": False,
                "can_attempt": True,
                "attempt_count": 0
            }

        # B2C users query B2C database
        if is_b2c:
            attempts = await db.b2c_find(
                "student_test_attempts",
                {
                    "student_id": user_id,
                    "document_id": document_id
                },
                sort=[("submitted_at", -1)]
            )
        else:
            # Regular B2B students query main database
            attempts = await db.mongo_find(
                "student_test_attempts",
                {
                    "student_id": user_id,
                    "document_id": document_id
                },
                sort=[("submitted_at", -1)]
            )

        has_attempted = len(attempts) > 0
        attempt_count = len(attempts)

        # Check if re-attempt is allowed
        can_attempt = True
        if has_attempted:
            # Check if admin has enabled re-attempt for this student
            # In this current logic, attempts are always allowed
            can_attempt = True 

        return {
            "success": True,
            "has_attempted": has_attempted,
            "can_attempt": can_attempt,
            "attempt_count": attempt_count,
            "latest_attempt": {
                "attempt_id": str(attempts[0]["_id"]),
                "score": attempts[0].get("score", 0),
                "total_points": attempts[0].get("total_points", 0),
                "submitted_at": attempts[0].get("submitted_at").isoformat() if attempts[0].get("submitted_at") else None
            } if has_attempted else None
        }

    except Exception as e:
        logger.error(f"Check test attempt error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check test attempt: {str(e)}"
        )


@router.post("/test-series/{document_id}/submit")
@limiter.limit("10/minute")
async def submit_test_series(
    request: Request,
    document_id: str,
    submission_data: dict,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Submit test series attempt with answers
    Calculate score with positive and negative marking
    Store in student_test_attempts collection
    """
    try:
        user_id = current_user["user_id"]
        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

        logger.info(f"Test submission: user={user_id}, is_b2c={is_b2c}, document={document_id}")

        # Get document from appropriate database
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
            if not document:
                # Try by _id
                try:
                    document = await db.b2c_find_one("documents", {"_id": ObjectId(document_id)})
                except:
                    pass
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})
        
        if not document:
            raise HTTPException(status_code=404, detail="Test series not found")

        if document.get("document_type") != "Test Series":
            raise HTTPException(status_code=400, detail="Document is not a Test Series")

        # For students/B2C users, check if they can attempt
        if user_type in ["student", "b2c_user"]:
            # Check existing attempts from appropriate database
            if is_b2c:
                attempts = await db.b2c_find(
                    "student_test_attempts",
                    {
                        "student_id": user_id,
                        "document_id": document_id
                    }
                )
            else:
                attempts = await db.mongo_find(
                    "student_test_attempts",
                    {
                        "student_id": user_id,
                        "document_id": document_id
                    }
                )

            if len(attempts) > 0:
                latest_attempt = attempts[-1]
                if not latest_attempt.get("can_reattempt", True):
                    raise HTTPException(
                        status_code=403,
                        detail="You have already attempted this test. Re-attempt not allowed."
                    )

        # Get questions from appropriate database
        if is_b2c:
            questions = await db.b2c_find("questions", {"document_id": document_id})
            if not questions:
                # Try by pdf_source
                questions = await db.b2c_find("questions", {"pdf_source": document.get("filename")})
        else:
            questions = await db.mongo_find("questions", {"document_id": document_id})
        
        if not questions:
            raise HTTPException(status_code=404, detail="No questions found for this test")

        # Get student answers from submission
        student_answers = submission_data.get("answers", {})  # {question_id: selected_answer}
        time_taken = submission_data.get("time_taken", 0)  # in seconds

        # Evaluate answers
        total_questions = len(questions)
        correct_count = 0
        incorrect_count = 0
        unanswered_count = 0
        score = 0
        
        # Calculate total_points from document, or sum from questions if not set
        total_points = document.get("total_points", 0)
        if total_points == 0:
            # Calculate from actual questions
            total_points = sum(q.get("points", 4) for q in questions)

        question_results = []

        for question in questions:
            question_id = question.get("id") or str(question.get("_id"))
            correct_answer = question.get("correct_answer")
            if correct_answer is not None:
                correct_answer = str(correct_answer).strip()
            else:
                correct_answer = ""
            student_answer = str(student_answers.get(question_id, "")).strip()
            question_points = question.get("points", 4)
            penalty_marks = question.get("penalty", question.get("penalty_marks", 1))

            is_correct = False
            # Check if question was skipped or not attempted
            is_attempted = bool(student_answer) and student_answer.upper() != "SKIPPED"

            # Skipped questions get 0 points (no penalty, no positive marks)
            if not is_attempted or student_answer.upper() == "SKIPPED":
                unanswered_count += 1
                points_earned = 0
            elif student_answer == correct_answer:
                is_correct = True
                correct_count += 1
                points_earned = question_points
                score += question_points
            else:
                incorrect_count += 1
                # Use penalty_marks from the question itself
                points_earned = -penalty_marks
                score -= penalty_marks

            question_results.append({
                "question_id": question_id,
                "student_answer": student_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "is_attempted": is_attempted,
                "points": question_points,
                "penalty_marks": penalty_marks,
                "points_earned": points_earned
            })

        # Calculate percentage
        percentage = (score / total_points * 100) if total_points > 0 else 0

        # Get student info from appropriate database
        if is_b2c:
            student = await db.b2c_find_one("users", {"_id": ObjectId(user_id)})
            student_name = student.get("full_name", student.get("name", "B2C User")) if student else "B2C User"
            student_grade = student.get("standard", student.get("class_level", "")) if student else ""
        elif user_type == "student":
            student = await db.mongo_find_one("students", {"_id": ObjectId(user_id)})
            student_name = student.get("name", "Student") if student else "Student"
            student_grade = student.get("grade", "") if student else ""
        else:
            student_name = "Admin"
            student_grade = ""

        # Create attempt record
        attempt_record = {
            "student_id": user_id,
            "student_name": student_name,
            "student_grade": student_grade,
            "document_id": document_id,
            "document_title": document.get("title", ""),
            "subject": document.get("subject", ""),
            "total_questions": total_questions,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "unanswered_count": unanswered_count,
            "score": round(score, 2),
            "total_points": total_points,
            "percentage": round(percentage, 2),
            "time_taken": time_taken,
            "total_minutes": document.get("total_minutes", 0),
            "answers": student_answers,
            "question_results": question_results,
            "can_reattempt": True,  # Allow unlimited practice
            "submitted_at": datetime.utcnow(),
            "is_b2c": is_b2c  # Mark as B2C attempt for analytics
        }

        # Insert into appropriate database
        if is_b2c:
            attempt_id = await db.b2c_insert_one("student_test_attempts", attempt_record)
        else:
            attempt_id = await db.mongo_insert_one("student_test_attempts", attempt_record)

        logger.info(f"Test series submitted: {document_id} by {student_name} (B2C={is_b2c}) - Score: {score}/{total_points}")

        return {
            "success": True,
            "message": "Test submitted successfully",
            "data": {
                "attempt_id": attempt_id,
                "score": round(score, 2),
                "total_points": total_points,
                "percentage": round(percentage, 2),
                "total_questions": total_questions,
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "unanswered_count": unanswered_count,
                "time_taken": time_taken,
                "question_results": question_results
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit test series error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit test: {str(e)}"
        )
