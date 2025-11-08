"""
Async MCQ API for SkillBot
Multiple Choice Question management endpoints with analytics
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from bson import ObjectId

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user, get_database, get_cache
from config_async import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
class MCQOption(BaseModel):
    id: str
    text: str
    is_correct: bool = False

class MCQQuestion(BaseModel):
    id: Optional[str] = None
    question_text: str
    subject: str
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$")
    options: List[MCQOption] = Field(..., min_items=2, max_items=6)
    explanation: Optional[str] = None
    tags: List[str] = []
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    is_active: bool = True

class MCQResponse(BaseModel):
    id: str
    question_text: str
    subject: str
    difficulty: str
    options: List[MCQOption]
    explanation: Optional[str] = None
    tags: List[str] = []
    created_at: datetime

class MCQListResponse(BaseModel):
    questions: List[MCQResponse]
    total: int
    page: int
    limit: int

class MCQAttempt(BaseModel):
    question_id: str
    selected_option_id: str
    time_spent: int = Field(default=0, ge=0)

class MCQAttemptResponse(BaseModel):
    id: str
    question_id: str
    selected_option_id: str
    correct_option_id: str
    is_correct: bool
    time_spent: int
    submitted_at: datetime
    explanation: Optional[str] = None

class MCQStats(BaseModel):
    total_questions: int
    total_attempts: int
    correct_attempts: int
    accuracy_rate: float
    subject_breakdown: Dict[str, Dict[str, int]]  # subject -> {total, correct}
    difficulty_breakdown: Dict[str, Dict[str, int]]  # difficulty -> {total, correct}

def require_student_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require student or admin access"""
    if current_user.get("user_type") not in ["student", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student or admin access required"
        )
    return current_user

async def get_current_user_optional(db: DatabaseManager = Depends(get_database)):
    """Optional authentication - returns user if authenticated, None if not"""
    try:
        # Try to get JWT token from request
        from fastapi import Request
        from core.auth import AuthManager
        import jwt
        from config_async import settings

        request = Request.__new__(Request)
        # This is a simplified version - in practice we'd need the request object
        # For now, return a default user that allows basic access
        return {"user_type": "student", "user_id": "anonymous"}
    except Exception:
        # If authentication fails, return anonymous user
        return {"user_type": "student", "user_id": "anonymous"}

def require_admin_for_write(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access for write operations"""
    if current_user.get("user_type") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

@router.get("/", response_model=MCQListResponse)
@limiter.limit("60/minute")
async def get_mcq_questions(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    subject: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    search: Optional[str] = Query(None, max_length=100),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get paginated list of MCQ questions"""
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)
        try:
            admin_oid = ObjectId(admin_id)
            admin_filter = admin_oid
        except Exception:
            admin_filter = admin_id

        # Build admin filter supporting ObjectId or string
        try:
            admin_oid = ObjectId(admin_id)
            admin_filter = admin_oid
        except Exception:
            admin_filter = admin_id
        # Build admin filter supporting ObjectId or string
        try:
            admin_oid = ObjectId(admin_id)
            admin_filter = admin_oid
        except Exception:
            admin_filter = admin_id
        # Documents.admin_id is stored as ObjectId; tokens carry strings. Build a robust filter.
        try:
            admin_oid = ObjectId(admin_id)
            admin_filter = {"$in": [admin_oid, admin_id]}
        except Exception:
            admin_oid = None
            admin_filter = admin_id

        # Build cache key with admin_id
        cache_key = f"mcq:{admin_id}:{page}:{limit}:{subject}:{difficulty}:{search}"
        cached_result = await cache.get_cached_question_results(cache_key)

        if cached_result:
            return MCQListResponse(**cached_result)

        # Build filter with admin_id
        filter_dict = {"is_active": True, "admin_id": admin_id}
        if subject:
            filter_dict["subject"] = subject
        if difficulty:
            filter_dict["difficulty"] = difficulty
        if search:
            filter_dict["question_text"] = {"$regex": search, "$options": "i"}

        # Get total count
        all_questions = await db.mongo_find("mcq_questions", filter_dict)
        total_questions = len(all_questions)

        # Get paginated results
        skip = (page - 1) * limit
        questions_data = await db.mongo_find(
            "mcq_questions",
            filter_dict,
            sort=[("created_at", -1)],
            skip=skip,
            limit=limit
        )

        questions = [
            MCQResponse(
                id=str(q["_id"]),
                question_text=q["question_text"],
                subject=q["subject"],
                difficulty=q["difficulty"],
                options=q["options"],
                explanation=q.get("explanation"),
                tags=q.get("tags", []),
                created_at=q["created_at"]
            )
            for q in questions_data
        ]

        response_data = {
            "questions": [q.dict() for q in questions],
            "total": total_questions,
            "page": page,
            "limit": limit
        }

        # Cache the result
        await cache.cache_question_results(cache_key, response_data, 1800)  # 30 minutes

        return MCQListResponse(**response_data)

    except Exception as e:
        logger.error(f"Get MCQ questions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCQ questions"
        )

async def _get_mcq_questions_with_images(db: DatabaseManager, where_filter: dict, limit: int = 10000):
    """Helper function to fetch MCQ questions with images from MongoDB"""
    # Get questions directly from MongoDB instead of ChromaDB
    # ChromaDB is optional and may not be available
    question_docs = await db.mongo_find("questions", where_filter, sort=[("page_number", 1)], limit=limit)

    if not question_docs:
        return []

    # Parse questions from MongoDB results
    questions = []
    for question_doc in question_docs:
        if question_doc and (question_doc.get("options") or question_doc.get("enhanced_options")):
            question_id = question_doc.get("id")

            # Build image URLs for QUESTION FIGURES (diagrams that are part of the question)
            # ROBUST APPROACH: Try multiple sources for image data
            question_figures_with_urls = []
            for fig_idx, fig_ref in enumerate(question_doc.get("question_figures", [])):
                try:
                    # Extract figure ID
                    fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref

                    # Strategy 1: Use base64Data if already in question document (most common)
                    base64_data = None
                    if isinstance(fig_ref, dict) and "base64Data" in fig_ref and fig_ref["base64Data"]:
                        base64_data = fig_ref["base64Data"]
                        logger.debug(f"Question {question_id} figure {fig_idx}: Using embedded base64Data ({len(base64_data)} chars)")

                    # Strategy 2: Fetch from images collection if not embedded
                    img_doc = None
                    if not base64_data:
                        img_doc = await db.mongo_find_one("images", {"_id": fig_id})
                        if img_doc:
                            if "base64Data" in img_doc and img_doc["base64Data"]:
                                base64_data = img_doc["base64Data"]
                                logger.debug(f"Question {question_id} figure {fig_idx}: Fetched base64Data from images collection")
                            # Strategy 3: Read from file_path and convert to base64
                            elif img_doc.get("file_path"):
                                import os
                                import base64 as b64
                                file_path = img_doc["file_path"]
                                if os.path.exists(file_path):
                                    try:
                                        with open(file_path, "rb") as f:
                                            image_bytes = f.read()
                                            base64_encoded = b64.b64encode(image_bytes).decode('utf-8')
                                            content_type = img_doc.get("content_type", "image/jpeg")
                                            if not content_type.startswith("image/"):
                                                content_type = "image/jpeg"
                                            base64_data = f"data:{content_type};base64,{base64_encoded}"
                                            logger.info(f"✅ Loaded MCQ figure {fig_id} from file: {len(base64_data)} bytes")
                                    except Exception as file_err:
                                        logger.error(f"❌ Failed to read MCQ figure file {file_path}: {file_err}")
                                else:
                                    logger.warning(f"⚠️ MCQ figure file not found: {file_path}")
                            else:
                                logger.warning(f"Question {question_id} figure {fig_idx}: Image doc found but no base64Data or file_path")

                    # Always add the figure, even if we don't have base64 (frontend will use URL fallback)
                    figure_data = {
                        "id": fig_id,
                        "url": f"/api/v1/images/{fig_id}",  # Fallback URL
                        "contentType": (img_doc.get("content_type", "image/jpeg") if img_doc else "image/jpeg"),
                        "filename": (img_doc.get("original_filename", fig_id) if img_doc else fig_id),
                        "base64Data": base64_data,  # May be None - frontend will handle
                        "description": (fig_ref.get("description", "") if isinstance(fig_ref, dict) else "")
                    }

                    question_figures_with_urls.append(figure_data)

                    if not base64_data:
                        logger.warning(f"Question {question_id} figure {fig_idx}: No base64Data available, frontend will try URL")

                except Exception as e:
                    logger.error(f"Error processing figure {fig_idx} for question {question_id}: {str(e)}")
                    # Still add a placeholder so students know there should be an image
                    question_figures_with_urls.append({
                        "id": f"error_{fig_idx}",
                        "url": "",
                        "contentType": "image/jpeg",
                        "filename": "image_error",
                        "base64Data": None,
                        "description": "Image loading error - please report to admin"
                    })

            # Get enhanced options
            enhanced_options = question_doc.get("enhanced_options", [])

            # Debug log for image options
            for opt in enhanced_options:
                if opt.get('type') == 'image':
                    content_preview = opt.get('content', '')[:50] if opt.get('content') else 'NO CONTENT'
                    logger.info(f"Question {question_id} option {opt.get('label')}: type=image, content_length={len(opt.get('content', ''))}, preview={content_preview}")

            question_data = {
                "id": question_id,
                "text": question_doc.get("text", ""),
                "subject": question_doc.get("subject", ""),
                "difficulty": question_doc.get("difficulty", "medium"),
                "questionType": question_doc.get("question_type", "mcq"),
                "options": question_doc.get("options", []),
                "enhancedOptions": enhanced_options,
                "questionFigures": question_figures_with_urls,
                "correctAnswer": question_doc.get("correct_answer", ""),
                "metadata": question_doc.get("metadata", {})
            }

            questions.append(question_data)

    return questions

@router.get("/all-questions")
@limiter.limit("30/minute")
async def get_all_mcq_questions(
    request: Request,
    subject: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    document_type: Optional[str] = Query(None, description="Filter by document type (Test Series, Practice Sets)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Get ALL MCQ questions from Test Series and Practice Sets (JEE-style exam mode)"""
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Build filters - get both Test Series and Practice Sets if no specific type requested
        if not document_type:
            document_types = ["Test Series", "Practice Sets"]
        else:
            document_types = [document_type]

        questions = []
        for doc_type in document_types:
            # Search questions from admin's collection
            type_questions = question_service.search_questions(
                query=None,
                subject=subject,
                difficulty=difficulty,
                document_type=doc_type,
                limit=1000
            )
            questions.extend(type_questions)

        logger.info(f"Fetched {len(questions)} MCQ questions from admin {admin_id} (subject={subject}, difficulty={difficulty}, document_type={document_type or 'all'})")

        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No MCQ questions found. Please upload documents and process them, or create questions manually."
            )

        # Normalize to list of dicts
        normalized = [q if isinstance(q, dict) else q.to_dict() for q in questions]
        return {
            "success": True,
            "questions": normalized,
            "count": len(normalized)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching all MCQ questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch MCQ questions: {str(e)}"
        )

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
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

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

        # If user is a student, only show completed OCR documents
        if current_user.get("user_type") == "student":
            filter_query["ocr_status"] = "completed"

        documents = await db.mongo_find("documents", filter_query, sort=[("title", 1)])
        if documents:
            test_series_list = [{
                "document_id": doc.get("document_id"),
                "title": doc.get("title"),
                "subject": doc.get("subject"),
                "standard": doc.get("standard"),
                "course_plan": doc.get("course_plan"),
                "difficulty": doc.get("difficulty"),
                "questions_count": doc.get("extracted_questions_count", 0),
                "total_points": doc.get("total_points", 0),
                "total_minutes": doc.get("total_minutes", 0),
                "is_validated": doc.get("is_validated", False),
                "file_exists": True  # assume available if listed
            } for doc in documents]

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
        from services.question_service import QuestionService
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
                    "file_exists": False
                }
            unique_docs[doc_id]["questions_count"] += 1

        test_series_list = list(unique_docs.values())
        if not test_series_list or (len(test_series_list) == 1 and test_series_list[0].get("document_id") in (None, "", "unknown")):
            test_series_list = [{
                "document_id": "legacy_all",
                "title": "All Test Series (Legacy)",
                "subject": subject or "General",
                "standard": "Unknown",
                "course_plan": course_plan or "Unknown",
                "difficulty": difficulty or "medium" if 'difficulty' in locals() else "medium",
                "questions_count": len(test_series_questions),
                "total_points": 0,
                "total_minutes": 0,
                "is_validated": False,
                "file_exists": False
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

@router.get("/available-options")
@limiter.limit("30/minute")
async def get_mcq_available_options(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Get available course plans, subjects for MCQ test series based on admin's uploaded content"""
    try:
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)
        try:
            admin_oid = ObjectId(admin_id)
            admin_filter = {"$in": [admin_oid, admin_id]}
        except Exception:
            admin_filter = admin_id

        # Build filter for test series
        filter_query = {
            "document_type": "Test Series",
            "admin_id": admin_filter
        }

        # If user is a student, only show completed OCR documents
        if current_user.get("user_type") == "student":
            filter_query["ocr_status"] = "completed"

        # Get all test series documents for this admin
        documents = await db.mongo_find("documents", filter_query)

        # Extract unique values for each field
        course_plans = set()
        subjects = set()
        standards = set()

        for doc in documents:
            if doc.get("course_plan"):
                course_plans.add(doc["course_plan"])
            if doc.get("subject"):
                subjects.add(doc["subject"])
            if doc.get("standard"):
                standards.add(doc["standard"])

        return {
            "success": True,
            "data": {
                "course_plans": sorted(list(course_plans)),
                "subjects": sorted(list(subjects)),
                "standards": sorted(list(standards))
            }
        }

    except Exception as e:
        logger.error(f"Get MCQ available options error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available options"
        )

@router.get("/test-series/{document_id}/questions")
async def get_test_series_questions(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all questions from a specific Test Series document
    Works with ChromaDB data when MongoDB documents are missing
    No authentication required for basic access
    """
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)
        try:
            admin_oid = ObjectId(admin_id)
            admin_filter = {"$in": [admin_oid, admin_id]}
        except Exception:
            admin_filter = admin_id

        # Try to get document from MongoDB first (filtered by admin_id)
        document = await db.mongo_find_one("documents", {"document_id": document_id, "admin_id": admin_filter})

        if document:
            # Verify document type
            if document.get("document_type") != "Test Series":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This document is not a Test Series"
                )

            # 1) Preferred: read directly from Mongo 'questions' by document_id (populated during OCR)
            mongo_questions = await db.mongo_find(
                "questions",
                {"document_id": document_id},
                sort=[("metadata.page", 1)]
            )

            questions_with_images = []
            if mongo_questions:
                # Normalize to JSON-serializable payloads expected by frontend
                from datetime import datetime
                def to_jsonable(value):
                    if isinstance(value, datetime):
                        return value.isoformat()
                    try:
                        if isinstance(value, ObjectId):
                            return str(value)
                    except Exception:
                        pass
                    if isinstance(value, list):
                        return [to_jsonable(v) for v in value]
                    if isinstance(value, dict):
                        return {k: to_jsonable(v) for k, v in value.items()}
                    return value

                for q in mongo_questions:
                    payload = {
                        "id": str(q.get("id") or q.get("_id")),
                        "text": q.get("text", q.get("question_text", "")),
                        "question_text": q.get("question_text", q.get("text", "")),
                        "subject": q.get("subject", document.get("subject")),
                        "difficulty": q.get("difficulty", "medium"),
                        "document_type": "Test Series",
                        "document_id": document_id,
                        "pdf_source": q.get("pdf_source", document.get("filename")),
                        "images": q.get("images", []),
                        "question_figures": q.get("question_figures", []),
                        "options": q.get("options", []),
                        "enhanced_options": q.get("enhanced_options", []),
                        "correct_answer": q.get("correct_answer"),
                        "metadata": q.get("metadata", {}),
                        "points": q.get("points", 1),
                        "penalty": q.get("penalty", 0),
                        "created_at": q.get("created_at"),
                        "extracted_at": q.get("extracted_at"),
                    }

                    # Enrich figures with base64 (for UI that displays diagrams)
                    try:
                        figures: List[Dict[str, Any]] = []
                        for fig_ref in (q.get("question_figures", []) or []):
                            fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref
                            base64_data = None
                            if isinstance(fig_ref, dict) and fig_ref.get("base64Data"):
                                base64_data = fig_ref["base64Data"]
                            else:
                                img_doc = await db.mongo_find_one("images", {"_id": fig_id})
                                if img_doc:
                                    if img_doc.get("base64Data"):
                                        base64_data = img_doc["base64Data"]
                                    elif img_doc.get("file_path"):
                                        import os, base64
                                        fp = img_doc["file_path"]
                                        if os.path.exists(fp):
                                            with open(fp, "rb") as f:
                                                enc = base64.b64encode(f.read()).decode("utf-8")
                                                ct = img_doc.get("content_type", "image/jpeg")
                                                if not ct.startswith("image/"):
                                                    ct = "image/jpeg"
                                                base64_data = f"data:{ct};base64,{enc}"
                            figures.append({
                                "id": fig_id,
                                "url": f"/api/v1/images/{fig_id}",
                                "base64Data": base64_data,
                                "contentType": "image/jpeg",
                                "filename": (fig_ref.get("filename") if isinstance(fig_ref, dict) else str(fig_id)),
                                "type": "diagram"
                            })
                        payload["questionFigures"] = figures
                    except Exception:
                        payload["questionFigures"] = []

                    # Inline base64 for image-type enhanced options when content is an image id
                    try:
                        eos = payload.get("enhanced_options") or []
                        for i, opt in enumerate(list(eos)):
                            if isinstance(opt, dict) and opt.get("type") == "image":
                                content = opt.get("content")
                                if isinstance(content, str) and content and not content.startswith("data:image"):
                                    img_doc = await db.mongo_find_one("images", {"_id": content})
                                    if img_doc:
                                        b64 = img_doc.get("base64Data")
                                        if not b64 and img_doc.get("file_path"):
                                            import os, base64
                                            fp = img_doc["file_path"]
                                            if os.path.exists(fp):
                                                with open(fp, "rb") as f:
                                                    enc = base64.b64encode(f.read()).decode("utf-8")
                                                    ct = img_doc.get("content_type", "image/jpeg")
                                                    if not ct.startswith("image/"):
                                                        ct = "image/jpeg"
                                                    b64 = f"data:{ct};base64,{enc}"
                                        if b64:
                                            payload["enhanced_options"][i]["content"] = b64
                    except Exception:
                        pass
                    questions_with_images.append(to_jsonable(payload))
            else:
                # 2) Fallback: use Chroma via QuestionService
                from services.question_service import QuestionService
                question_service = QuestionService(admin_id)
                questions = question_service.search_questions(
                    query=None,
                    document_type="Test Series",
                    limit=1000
                )

                if document_id in ("legacy_all", "all", "ALL"):
                    questions_with_images = questions
                else:
                    questions_with_images = [q for q in questions if q.get('pdfSource', '') == document_id or q.get('document_id', '') == document_id]

            return {
                "success": True,
                "data": {
                    "document_id": document.get("document_id"),
                    "title": document.get("title"),
                    "subject": document.get("subject"),
                    "total_points": document.get("total_points", 0),
                    "total_minutes": document.get("total_minutes", 0),
                    "questions": questions_with_images,
                    "total": len(questions_with_images)
                }
            }
        else:
            # Fallback: Get questions directly from admin's collection
            logger.info(f"Document {document_id} not found in MongoDB, searching admin's collection")

            # Get questions from admin's collection
            from services.question_service import QuestionService
            question_service = QuestionService(admin_id)
            questions = question_service.search_questions(
                query=None,
                document_type="Test Series",
                limit=1000
            )

            # Filter by document_id (questions is already a list of dicts)
            questions_with_images = [q for q in questions if q.get('pdfSource', '') == document_id or q.get('document_id', '') == document_id]

            if not questions_with_images:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No questions found for document: {document_id}"
                )

            # Use first question's metadata for document info
            first_question = questions_with_images[0]
            doc_title = first_question.get("metadata", {}).get("document_title", f"Test Series {document_id}")
            doc_subject = first_question.get("subject", "Unknown")

            return {
                "success": True,
                "data": {
                    "document_id": document_id,
                    "title": doc_title,
                    "subject": doc_subject,
                    "total_points": len(questions_with_images) * 4,  # Estimate 4 points per question
                    "total_minutes": len(questions_with_images) * 2,  # Estimate 2 minutes per question
                    "questions": questions_with_images,
                    "total": len(questions_with_images)
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test series questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve questions: {str(e)}"
        )

@router.get("/random-question")
@limiter.limit("60/minute")
async def get_random_mcq_question(
    request: Request,
    subject: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    document_type: Optional[str] = Query(None, description="Filter by document type (Test Series, Practice Sets)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Get a random MCQ question from Test Series and Practice Sets"""
    try:
        import random

        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Build filters
        if not document_type:
            # Get both Test Series and Practice Sets
            document_types = ["Test Series", "Practice Sets"]
        else:
            document_types = [document_type]

        questions = []
        for doc_type in document_types:
            # Search questions from admin's collection
            type_questions = question_service.search_questions(
                query=None,
                subject=subject,
                difficulty=difficulty,
                document_type=doc_type,
                limit=1000
            )
            questions.extend(type_questions)

        logger.info(f"Fetched {len(questions)} MCQ questions from admin {admin_id} collection")

        logger.info(f"Fetched {len(questions)} MCQ questions for random selection (subject={subject}, difficulty={difficulty}, document_type={document_type or 'all'})")

        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No MCQ questions found. Please upload documents and process them, or create questions manually."
            )

        # Select random question
        random_question = random.choice(questions)

        # Convert to dict for response
        question_dict = random_question if isinstance(random_question, dict) else random_question.to_dict()

        return {
            "success": True,
            "question": question_dict
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get random MCQ question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get random question: {str(e)}"
        )

@router.get("/{question_id}", response_model=MCQResponse)
@limiter.limit("120/minute")
async def get_mcq_question(
    request: Request,
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get MCQ question by ID"""
    try:
        # Check cache first
        cached_question = await cache.get(f"mcq:{question_id}", "questions")
        if cached_question:
            return MCQResponse(**cached_question)

        # Get from database
        question = await db.mongo_find_one("mcq_questions", {"_id": question_id, "is_active": True})

        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCQ question not found"
            )

        question_response = MCQResponse(
            id=str(question["_id"]),
            question_text=question["question_text"],
            subject=question["subject"],
            difficulty=question["difficulty"],
            options=question["options"],
            explanation=question.get("explanation"),
            tags=question.get("tags", []),
            created_at=question["created_at"]
        )

        # Cache the result
        await cache.set(f"mcq:{question_id}", question_response.dict(), 3600, "questions")

        return question_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get MCQ question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCQ question"
        )

@router.post("/", response_model=MCQResponse)
@limiter.limit("20/minute")
async def create_mcq_question(
    request: Request,
    question_data: MCQQuestion,
    current_user: Dict[str, Any] = Depends(require_admin_for_write),
    db: DatabaseManager = Depends(get_database)
):
    """Create a new MCQ question"""
    try:
        # Validate that at least one option is correct
        correct_options = [opt for opt in question_data.options if opt.is_correct]
        if not correct_options:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one option must be marked as correct"
            )

        # Get admin_id for data isolation
        admin_id = current_user.get("admin_id", current_user["user_id"])

        # Create question document
        question_doc = {
            "question_text": question_data.question_text,
            "subject": question_data.subject,
            "difficulty": question_data.difficulty,
            "options": [opt.dict() for opt in question_data.options],
            "explanation": question_data.explanation,
            "tags": question_data.tags,
            "created_by": current_user["user_id"],
            "admin_id": admin_id,  # Add admin_id for data isolation
            "created_at": datetime.utcnow(),
            "is_active": True
        }

        question_id = await db.mongo_insert_one("mcq_questions", question_doc)

        if not question_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create MCQ question"
            )

        return MCQResponse(
            id=question_id,
            question_text=question_data.question_text,
            subject=question_data.subject,
            difficulty=question_data.difficulty,
            options=question_data.options,
            explanation=question_data.explanation,
            tags=question_data.tags,
            created_at=question_doc["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create MCQ question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create MCQ question"
        )

@router.post("/check")
@limiter.limit("60/minute")
async def check_mcq_answer(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Check answer.

    Behavior by question type:
    - integer: Validate directly against stored `correct_answer` (no LLM, no cache).
    - mcq: Prefer stored `correct_answer` if present, else use cached solution, else LLM.
    """
    try:
        from services.async_openai_service import AsyncOpenAIService
        import json as json_module
        from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
        
        # Parse request body
        body = await request.json()
        question_id = body.get("question_id")
        selected_answer = body.get("selected_answer")
        
        if not question_id or not selected_answer:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required fields: question_id and selected_answer"
            )
        
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Get question from admin's collection
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)
        question_obj = question_service.get_question(question_id)

        if not question_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found in your admin's collection"
            )

        # Convert to dict format
        question_doc = question_obj.to_dict()
        
        # Determine question type (default to 'mcq' for backward compatibility)
        question_type = str(question_doc.get("question_type", "")).lower() or "mcq"
        options_list = question_doc.get("options") or []
        stored_answer_raw = str(question_doc.get("correct_answer") or "").strip()

        # Heuristic detection for integer-type even if question_type is missing:
        def _is_numeric_value(val: str) -> bool:
            try:
                _ = Decimal(val)
                return True
            except Exception:
                return False

        is_integer_like = (
            question_type == "integer"
            or (not options_list)  # No options => likely integer/numerical
            or (_is_numeric_value(stored_answer_raw) and stored_answer_raw.upper() not in ["A","B","C","D","E","F"])
        )

        # 1) INTEGER/Numerical type: validate directly, no LLM
        if is_integer_like:
            stored_answer = stored_answer_raw

            # Robust numerical comparison: accept exact string match OR numerically equal values
            def normalize_numeric(s: str) -> str:
                s = s.strip().replace(" ", "")
                # Normalize leading + sign
                if s.startswith("+"):
                    s = s[1:]
                return s

            user_raw = normalize_numeric(str(selected_answer))
            stored_raw = normalize_numeric(stored_answer)

            is_correct = False
            # Try numeric equality with Decimal for precision-safe comparison
            try:
                user_num = Decimal(user_raw)
                stored_num = Decimal(stored_raw)
                is_correct = (user_num == stored_num)
            except (InvalidOperation, TypeError):
                # Fallback to plain string equality if parsing fails
                is_correct = user_raw == stored_raw

            result = {
                "question_id": question_id,
                "selected_answer": str(selected_answer),
                "correct_answer": stored_answer,
                "is_correct": bool(is_correct),
                "explanation": "Validated against answer key.",
                "solution_source": "answer_key",
                "confidence_score": 1.0
            }

            return {"success": True, "result": result}

        # 2) MCQ type: prefer stored answer key if available
        stored_correct_answer = stored_answer_raw.upper()
        if stored_correct_answer in ["A", "B", "C", "D", "E", "F"]:
            is_correct = str(selected_answer).strip().upper() == stored_correct_answer

            result = {
                "question_id": question_id,
                "selected_answer": selected_answer,
                "correct_answer": stored_correct_answer,
                "is_correct": is_correct,
                "explanation": "Validated against answer key.",
                "solution_source": "answer_key",
                "confidence_score": 1.0
            }
            return {"success": True, "result": result}

        # 3) Check cached LLM/database solution next
        solution_doc = await db.mongo_find_one("mcq_solutions", {"question_id": question_id})
        if solution_doc:
            logger.info(f"Using cached solution for question {question_id}")
            correct_answer = solution_doc.get("correct_answer", "")
            is_correct = str(selected_answer).strip().upper() == str(correct_answer).strip().upper()

            result = {
                "question_id": question_id,
                "selected_answer": selected_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "explanation": solution_doc.get("explanation", ""),
                "solution_source": "database",
                "confidence_score": solution_doc.get("confidence_score", 1.0)
            }
            return {"success": True, "result": result}
        
        # 4) No answer key and no cache - use LLM
        logger.info(f"Generating solution with LLM for question {question_id}")
        openai_service = AsyncOpenAIService()
        
        # Prepare prompt
        question_text = question_doc.get("text", "")
        options = question_doc.get("options", [])
        
        options_text = ""
        for i, option in enumerate(options):
            clean_option = option.strip()
            # Check if option is just an image reference
            if clean_option.startswith("img-") and clean_option.endswith((".jpeg", ".jpg", ".png")):
                options_text += f"{chr(65+i)}. [Image: {clean_option}]\n"
            else:
                options_text += f"{chr(65+i)}. {clean_option}\n"
        
        # Add note about images if question has them
        images_note = ""
        question_images = question_doc.get("images", [])
        if question_images:
            images_note = f"\n\nNote: This question includes {len(question_images)} image(s). Analyze the question carefully based on the text and options provided."
        
        prompt = f"""You are an expert tutor specializing in physics and mathematics. Analyze this multiple choice question and identify the correct answer.

Question: {question_text}

Options:
{options_text}

Student's selected answer: {selected_answer}{images_note}

CRITICAL INSTRUCTIONS:
1. Carefully read the question and ALL options
2. Identify which option (A, B, C, or D) is scientifically/mathematically correct
3. If options are images or formulas, analyze based on the question context
4. Do NOT default to "A" - analyze each option thoroughly

Respond in this EXACT JSON format (no markdown, no code blocks):
{{
    "correct_answer": "B",
    "is_correct": false,
    "explanation": "Detailed explanation of the correct answer and why other options are wrong",
    "confidence_score": 0.9
}}

Requirements:
- correct_answer: MUST be the letter (A, B, C, or D) of the truly correct option
- is_correct: true if student's answer matches correct_answer, false otherwise
- explanation: Clear, educational explanation (2-3 sentences minimum)
- confidence_score: 0.0 to 1.0 based on certainty
- Output ONLY valid JSON, no additional text or markdown"""
        
        # Call LLM
        llm_response = await openai_service.chat_completion_async(
            messages=[
                {"role": "system", "content": "You are an expert physics and mathematics tutor. Always analyze questions thoroughly and identify the truly correct answer, not just default to option A."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent analysis
            max_tokens=1500
        )
        
        if not llm_response.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM evaluation failed: {llm_response.get('error', 'Unknown error')}"
            )
        
        # Parse LLM response
        try:
            llm_text = llm_response.get("response", "").strip()
            
            # Try to extract JSON from markdown code blocks if present
            if "```json" in llm_text:
                llm_text = llm_text.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_text:
                llm_text = llm_text.split("```")[1].split("```")[0].strip()
            
            parsed = json_module.loads(llm_text)
            correct_answer = parsed.get("correct_answer", "").strip().upper()
            explanation = parsed.get("explanation", "")
            confidence = parsed.get("confidence_score", 0.8)
            
            # Validate correct_answer is a valid option letter
            if not correct_answer or correct_answer not in ["A", "B", "C", "D", "E", "F"]:
                logger.error(f"Invalid correct_answer from LLM: '{correct_answer}'. LLM response: {llm_text[:500]}")
                # Try to find answer in the text
                import re
                match = re.search(r'"correct_answer"\s*:\s*"([A-F])"', llm_text, re.IGNORECASE)
                if match:
                    correct_answer = match.group(1).upper()
                    logger.info(f"Extracted correct answer via regex: {correct_answer}")
                else:
                    raise ValueError(f"Could not extract valid answer from LLM response")
                    
        except Exception as e:
            # Fallback parsing - log the error
            logger.error(f"Failed to parse LLM JSON response: {str(e)}")
            logger.error(f"LLM raw response (first 1000 chars): {llm_response.get('response', '')[:1000]}")
            
            # Try regex extraction as last resort
            llm_text = llm_response.get("response", "")
            import re
            match = re.search(r'"correct_answer"\s*:\s*"([A-F])"', llm_text, re.IGNORECASE)
            if match:
                correct_answer = match.group(1).upper()
                logger.info(f"Fallback: extracted answer via regex: {correct_answer}")
            else:
                # Absolute fallback - try to find letter in first line
                first_line = llm_text.split('\n')[0] if llm_text else ""
                letter_match = re.search(r'\b([A-F])\b', first_line)
                if letter_match:
                    correct_answer = letter_match.group(1).upper()
                    logger.warning(f"Last resort: using first letter found: {correct_answer}")
                else:
                    logger.error(f"CRITICAL: Could not extract any answer. Defaulting to 'A' as absolute fallback.")
                    correct_answer = "A"
            
            explanation = llm_text if llm_text else "Unable to generate explanation"
            confidence = 0.4  # Low confidence for fallback
        
        is_correct = selected_answer.strip().upper() == correct_answer.strip().upper()
        
        # Save solution to database for future use (answer key cache)
        solution_to_save = {
            "id": f"sol_{question_id}",
            "question_id": question_id,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "generated_by": "llm",
            "generated_at": datetime.utcnow(),
            "llm_model": llm_response.get("model", "gpt-3.5-turbo"),
            "confidence_score": confidence,
            "validated": False
        }
        
        try:
            await db.mongo_insert_one("mcq_solutions", solution_to_save)
            logger.info(f"Saved new MCQ solution for question {question_id}")
        except Exception as e:
            logger.warning(f"Failed to save MCQ solution: {str(e)}")
        
        result = {
            "question_id": question_id,
            "selected_answer": selected_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": explanation,
            "solution_source": "llm_generated",
            "confidence_score": confidence
        }
        
        return {
            "success": True,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Check MCQ answer error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check answer: {str(e)}"
        )

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

        # Admins can always access
        if user_type == "admin":
            return {
                "success": True,
                "has_attempted": False,
                "can_attempt": True,
                "attempt_count": 0
            }

        # Check for existing attempts
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
            latest_attempt = attempts[0]
            can_attempt = latest_attempt.get("can_reattempt", False)

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

        # Get document
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(status_code=404, detail="Test series not found")

        if document.get("document_type") != "Test Series":
            raise HTTPException(status_code=400, detail="Document is not a Test Series")

        # For students, check if they can attempt
        if user_type == "student":
            # Check existing attempts
            attempts = await db.mongo_find(
                "student_test_attempts",
                {
                    "student_id": user_id,
                    "document_id": document_id
                }
            )

            if len(attempts) > 0:
                latest_attempt = attempts[-1]
                if not latest_attempt.get("can_reattempt", False):
                    raise HTTPException(
                        status_code=403,
                        detail="You have already attempted this test. Re-attempt not allowed."
                    )

        # Get questions
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
        total_points = document.get("total_points", 0)

        question_results = []

        for question in questions:
            question_id = question.get("id")
            correct_answer = question.get("correct_answer")
            if correct_answer is not None:
                correct_answer = correct_answer.strip()
            else:
                correct_answer = ""
            student_answer = student_answers.get(question_id, "").strip()
            question_points = question.get("points", 1)
            penalty_marks = question.get("penalty_marks", 0)  # Get penalty from question

            is_correct = False
            is_attempted = bool(student_answer)

            if not is_attempted:
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

        # Get student info
        student = await db.mongo_find_one("students", {"_id": ObjectId(user_id)}) if user_type == "student" else None
        student_name = student.get("name", "Admin") if student else "Admin"
        student_grade = student.get("grade", "") if student else ""

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
            "can_reattempt": False,  # Admin can enable this later
            "submitted_at": datetime.utcnow()
        }

        # Insert into database
        attempt_id = await db.mongo_insert_one("student_test_attempts", attempt_record)

        logger.info(f"Test series submitted: {document_id} by {student_name} - Score: {score}/{total_points}")

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