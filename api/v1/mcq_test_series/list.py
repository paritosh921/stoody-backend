import logging
from typing import Dict, Any, List, Optional
from bson import ObjectId

from fastapi import APIRouter, HTTPException, Depends, status, Query, Request

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from services.question_service import QuestionService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/test-series/list")
async def get_test_series_list(
    request: Request,
    subject: Optional[str] = Query(None, description="Filter by subject"),
    course_plan: Optional[str] = Query(None, description="Filter by course plan"),
    standard: Optional[str] = Query(None, description="Filter by grade/standard"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get list of available Test Series documents
    Works with ChromaDB data when MongoDB documents are missing
    """
    try:
        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"
        
        # Handle B2C users - query from B2C database
        if is_b2c:
            # Get B2C user profile from B2C database
            b2c_user = await db.b2c_find_one("users", {"_id": ObjectId(current_user["user_id"])})
            
            if not b2c_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="B2C user profile not found"
                )
            
            # Check if onboarding is complete
            if not b2c_user.get("onboarding_complete"):
                return {
                    "success": True,
                    "data": {
                        "test_series": [],
                        "total": 0,
                        "onboarding_required": True
                    }
                }
            
            # Get user's plan details
            user_exam_type = b2c_user.get("exam_type")
            user_class_level = b2c_user.get("class_level")
            user_standard = b2c_user.get("standard")
            user_subjects = b2c_user.get("subjects", [])
            user_plan_types = b2c_user.get("plan_types", [])
            
            # Get B2C admin ID for content filtering
            b2c_admin = await db.b2c_find_one("admins", {}, {"_id": 1})
            b2c_admin_id = b2c_admin["_id"] if b2c_admin else None
            
            # Build filter for B2C test series
            filter_query = {
                "document_type": "Test Series",
                "ocr_status": "completed",
                "is_active": {"$ne": False}
            }
            
            if b2c_admin_id:
                try:
                    filter_query["admin_id"] = ObjectId(b2c_admin_id)
                except:
                    filter_query["admin_id"] = b2c_admin_id
            
            # Apply plan type filter
            if course_plan:
                filter_query["course_plan"] = course_plan
            elif user_plan_types:
                filter_query["course_plan"] = {"$in": user_plan_types}
            elif user_exam_type:
                filter_query["course_plan"] = user_exam_type
            
            # Apply subject filter
            if subject:
                filter_query["subject"] = subject
            elif user_subjects:
                filter_query["subject"] = {"$in": user_subjects}
            
            # Apply standard filter
            if standard:
                filter_query["standard"] = standard
            elif user_standard:
                filter_query["standard"] = user_standard
            
            logger.info(f"B2C user {current_user['user_id']} test series query: {filter_query}")
            
            # Get test series from B2C database
            documents = await db.b2c_find(
                "documents",
                filter_query,
                sort=[("title", 1)]
            )
            
            logger.info(f"B2C test series found: {len(documents)}")
            
            # Format response
            test_series_list = []
            user_id = current_user["user_id"]
            
            for doc in documents:
                doc_id = doc.get("document_id") or str(doc.get("_id"))
                
                # Check if B2C user has attempted this test
                attempts = await db.b2c_find(
                    "student_test_attempts",
                    {
                        "student_id": user_id,
                        "document_id": doc_id
                    },
                    sort=[("submitted_at", -1)]
                )
                
                has_attempted = len(attempts) > 0
                attempt_count = len(attempts)
                latest_attempt = None
                
                if has_attempted:
                    latest_attempt = {
                        "attempt_id": str(attempts[0]["_id"]),
                        "score": attempts[0].get("score", 0),
                        "total_points": attempts[0].get("total_points", 0),
                        "percentage": attempts[0].get("percentage", 0),
                        "submitted_at": attempts[0].get("submitted_at").isoformat() if attempts[0].get("submitted_at") else None
                    }
                
                test_series_list.append({
                    "document_id": doc_id,
                    "title": doc.get("title"),
                    "subject": doc.get("subject"),
                    "standard": doc.get("standard"),
                    "course_plan": doc.get("course_plan"),
                    "difficulty": doc.get("difficulty"),
                    "questions_count": doc.get("extracted_questions_count", 0),
                    "total_points": doc.get("total_points", 0),
                    "total_minutes": doc.get("total_minutes", 0),
                    "is_validated": doc.get("is_validated", False),
                    "file_exists": True,
                    "attempted": has_attempted,
                    "attempt_count": attempt_count,
                    "latest_attempt": latest_attempt
                })
            
            return {
                "success": True,
                "data": {
                    "test_series": test_series_list,
                    "total": len(test_series_list)
                }
            }

        # Get admin_id for data isolation
        try:
            from api.v1.questions_async import get_admin_id_from_user
            admin_id = get_admin_id_from_user(current_user)
        except ImportError:
            # Fallback if circular import or missing function (should not happen based on existing code)
            admin_id = current_user.get("user_id") if current_user.get("user_type") == "admin" else None


        # Try to get from MongoDB first (normal case)
        # Build filter for test series
        filter_query = {"document_type": "Test Series"}
        try:
            filter_query["admin_id"] = ObjectId(admin_id)
        except Exception:
            filter_query["admin_id"] = admin_id

        # Optional filters from query params
        if subject:
            filter_query["subject"] = subject
        if course_plan:
            filter_query["course_plan"] = course_plan
        if standard:
            filter_query["standard"] = standard

        # If user is a regular student, apply profile-based filtering
        if current_user.get("user_type") == "student":
            filter_query["ocr_status"] = "completed"
            # is_active: {$ne: False} matches True, None, or missing field (default active)
            filter_query["is_active"] = {"$ne": False}
            
            # Get student profile for filtering
            student_profile = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})
            
            if student_profile:
                student_grade = student_profile.get("grade")
                student_subjects = student_profile.get("subjects", [])
                student_plan_types = student_profile.get("plan_types", [])
                
                # Filter by student's grade if available - EXACT match
                if student_grade and not standard:  # Only if not already filtered by query param
                    filter_query["standard"] = student_grade
                
                # Filter by student's subjects if available
                if student_subjects and not subject:  # Only if not already filtered by query param
                    filter_query["subject"] = {"$in": student_subjects}
                
                # Filter by student's plan types if available
                if student_plan_types and not course_plan:  # Only if not already filtered by query param
                    filter_query["course_plan"] = {"$in": student_plan_types}

        documents = await db.mongo_find("documents", filter_query, sort=[("title", 1)])

        if documents:
            test_series_list = []
            user_id = current_user["user_id"]

            for doc in documents:
                doc_id = doc.get("document_id")

                # Check if student has attempted this test
                attempts = await db.mongo_find(
                    "student_test_attempts",
                    {
                        "student_id": user_id,
                        "document_id": doc_id
                    },
                    sort=[("submitted_at", -1)]
                )

                has_attempted = len(attempts) > 0
                attempt_count = len(attempts)
                latest_attempt = None

                if has_attempted:
                    latest_attempt = {
                        "attempt_id": str(attempts[0]["_id"]),
                        "score": attempts[0].get("score", 0),
                        "total_points": attempts[0].get("total_points", 0),
                        "percentage": attempts[0].get("percentage", 0),
                        "submitted_at": attempts[0].get("submitted_at").isoformat() if attempts[0].get("submitted_at") else None
                    }

                test_series_list.append({
                    "document_id": doc_id,
                    "title": doc.get("title"),
                    "subject": doc.get("subject"),
                    "standard": doc.get("standard"),
                    "course_plan": doc.get("course_plan"),
                    "difficulty": doc.get("difficulty"),
                    "questions_count": doc.get("extracted_questions_count", 0),
                    "total_points": doc.get("total_points", 0),
                    "total_minutes": doc.get("total_minutes", 0),
                    "is_validated": doc.get("is_validated", False),
                    "file_exists": True,  # assume available if listed
                    "attempted": has_attempted,
                    "attempt_count": attempt_count,
                    "latest_attempt": latest_attempt
                })

            return {
                "success": True,
                "data": {
                    "test_series": test_series_list,
                    "total": len(test_series_list)
                }
            }

        # Fallback to ChromaDB when MongoDB data is missing
        logger.info("Using ChromaDB fallback for test series list")

        # Use QuestionService for admin-specific ChromaDB access
        question_service = QuestionService(admin_id)

        # Get test series questions from admin's collection
        test_series_questions = question_service.search_questions(
            query=None,
            document_type="Test Series",
            limit=1000
        )

        # Extract unique document info from questions
        unique_docs: Dict[str, Dict[str, Any]] = {}
        for question in test_series_questions:
            # question is a dict (QuestionService returns dicts)
            qdict = question if isinstance(question, dict) else getattr(question, '__dict__', {})
            meta = qdict.get('metadata') or {}
            # Prefer explicit pdfSource; fallback to document_id in dict or metadata
            doc_id = qdict.get('pdfSource') or qdict.get('document_id') or meta.get('document_id') or 'unknown'

            if doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    "document_id": doc_id,
                    "title": f"Test Series - {doc_id}",
                    "subject": qdict.get('subject', meta.get('subject', 'General')),
                    "standard": meta.get('standard', "Unknown"),
                    "course_plan": meta.get('course_plan', "Unknown"),
                    "difficulty": qdict.get('difficulty', meta.get('difficulty', 'medium')),
                    "questions_count": 0,
                    "total_points": 0,
                    "total_minutes": 0,
                    "is_validated": False,
                    "file_exists": False,
                    "attempted": False,
                    "attempt_count": 0,
                    "latest_attempt": None
                }
            unique_docs[doc_id]["questions_count"] += 1

        # Check attempt status for each test (ChromaDB fallback)
        user_id = current_user["user_id"]
        for doc_id in unique_docs.keys():
            attempts = await db.mongo_find(
                "student_test_attempts",
                {
                    "student_id": user_id,
                    "document_id": doc_id
                },
                sort=[("submitted_at", -1)]
            )

            has_attempted = len(attempts) > 0
            attempt_count = len(attempts)

            if has_attempted:
                unique_docs[doc_id]["attempted"] = True
                unique_docs[doc_id]["attempt_count"] = attempt_count
                unique_docs[doc_id]["latest_attempt"] = {
                    "attempt_id": str(attempts[0]["_id"]),
                    "score": attempts[0].get("score", 0),
                    "total_points": attempts[0].get("total_points", 0),
                    "percentage": attempts[0].get("percentage", 0),
                    "submitted_at": attempts[0].get("submitted_at").isoformat() if attempts[0].get("submitted_at") else None
                }

        test_series_list = list(unique_docs.values())
        if not test_series_list or (len(test_series_list) == 1 and test_series_list[0].get("document_id") in (None, "", "unknown")):
            test_series_list = [{
                "document_id": "legacy_all",
                "title": "All Test Series (Legacy)",
                "subject": subject or "General",
                "standard": "Unknown",
                "course_plan": course_plan or "Unknown",
                "difficulty": "medium",
                "questions_count": len(test_series_questions),
                "total_points": 0,
                "total_minutes": 0,
                "is_validated": False,
                "file_exists": False,
                "attempted": False,
                "attempt_count": 0,
                "latest_attempt": None
            }]

        return {
            "success": True,
            "data": {
                "test_series": test_series_list,
                "total": len(test_series_list)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Failed to get test series list: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve test series list: {str(e)}"
        )
