"""
Async Questions API for SkillBot
Question management endpoints with ChromaDB integration
"""

import json
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
class QuestionImage(BaseModel):
    id: str
    filename: Optional[str] = None
    path: Optional[str] = None
    base64Data: Optional[str] = None

class QuestionOption(BaseModel):
    id: str
    type: str = "text"
    content: str
    label: Optional[str] = None
    description: Optional[str] = None

class Question(BaseModel):
    id: str
    text: str
    subject: str
    difficulty: str
    extractedAt: str
    pdfSource: str = ""
    images: List[QuestionImage] = []
    options: List[str] = []
    enhancedOptions: Optional[List[QuestionOption]] = None
    correctAnswer: str = ""
    explanation: str = ""
    tags: List[str] = []
    isActive: bool = True

class QuestionResponse(BaseModel):
    id: str
    text: str
    subject: str
    difficulty: str
    images: List[QuestionImage] = []
    options: List[str] = []
    enhancedOptions: Optional[List[QuestionOption]] = None

class QuestionsListResponse(BaseModel):
    success: bool = True
    questions: List[QuestionResponse]
    count: int
    total: int
    page: int
    limit: int

class QuestionStats(BaseModel):
    total_questions: int
    subjects: Dict[str, int]
    difficulties: Dict[str, int]
    recent_additions: int

def require_admin_for_write(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access for write operations"""
    if current_user.get("user_type") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def get_admin_id_from_user(current_user: Dict[str, Any]) -> str:
    """Extract admin_id from user token (works for admin, tutor, and student users)"""
    user_type = current_user.get("user_type")
    if user_type == "admin":
        return current_user.get("user_id")
    if user_type == "tutor":
        admin_id = current_user.get("admin_id")
        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tutor account is not properly linked to an admin"
            )
        return admin_id
    if user_type == "student":
        admin_id = current_user.get("admin_id")
        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Student account not properly configured"
            )
        return admin_id
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Unknown user type"
    )

@router.get("/search")
@limiter.limit("60/minute")
async def search_questions(
    request: Request,
    query: Optional[str] = Query(None, max_length=200),
    limit: int = Query(20, ge=1, le=100),
    subject: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    document_type: Optional[str] = Query(None),
    include_images: bool = Query(False),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Search questions from admin-specific collection"""
    try:
        # Get admin_id for data isolation
        admin_id = get_admin_id_from_user(current_user)

        # Build cache key with admin_id for isolation
        cache_key = f"search:{admin_id}:{query}:{limit}:{subject}:{difficulty}:{document_type}:{include_images}"
        cached_result = None
        if cache:
            cached_result = await cache.get_cached_question_results(cache_key)

        if cached_result:
            # Return cached dict as-is to preserve response shape (incl. `success`)
            return cached_result

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Search questions using admin-specific collection
        questions = question_service.search_questions(
            query=query,
            subject=subject,
            difficulty=difficulty,
            document_type=document_type,
            limit=limit
        )

        # Convert questions to response format
        question_responses = []
        for question in questions:
            question_response = QuestionResponse(
                id=question.id,
                text=question.text,
                subject=question.subject,
                difficulty=question.difficulty,
                images=[QuestionImage(id=img.id, filename=img.filename, path=img.path) for img in question.images],
                options=question.options,
                enhancedOptions=[QuestionOption(id=opt.id, type=opt.type, content=opt.content, label=opt.label, description=opt.description) for opt in (question.enhancedOptions or [])]
            )
            question_responses.append(question_response)

        response_data = {
            "success": True,
            "questions": [q.dict() for q in question_responses],
            "count": len(question_responses),
            "total": len(question_responses),
            "page": 1,
            "limit": limit
        }

        # Cache search results for 30 minutes (only if cache is available)
        if cache:
            await cache.cache_question_results(cache_key, response_data, 1800)

        # Return dict instead of Pydantic model for frontend compatibility
        return response_data

    except Exception as e:
        logger.error(f"Search questions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search questions"
        )

@router.get("/", response_model=QuestionsListResponse)
@limiter.limit("60/minute")
async def get_questions(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    subject: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    document_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None, max_length=100),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get paginated list of questions with optional document_type filter"""
    try:
        # Get admin_id for data isolation
        admin_id = get_admin_id_from_user(current_user)

        # Build cache key with admin_id
        cache_key = f"questions:{admin_id}:{page}:{limit}:{subject}:{difficulty}:{document_type}:{search}"
        cached_result = None
        if cache:
            cached_result = await cache.get_cached_question_results(cache_key)

        if cached_result:
            # Return cached dict as-is to preserve response shape (incl. `success`)
            return cached_result

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Search questions from admin's collection
        if search:
            all_questions = question_service.search_questions(
                query=search,
                subject=subject,
                difficulty=difficulty,
                document_type=document_type,
                limit=1000  # Get more for pagination
            )
        else:
            all_questions = question_service.search_questions(
                query=None,
                subject=subject,
                difficulty=difficulty,
                document_type=document_type,
                limit=1000  # Get more for pagination
            )

        # Apply pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_questions = all_questions[start_idx:end_idx]

        # Convert to response format
        question_responses = []
        for question in paginated_questions:
            question_response = QuestionResponse(
                id=question.id,
                text=question.text,
                subject=question.subject,
                difficulty=question.difficulty,
                images=[QuestionImage(id=img.id, filename=img.filename, path=img.path) for img in question.images],
                options=question.options,
                enhancedOptions=[QuestionOption(id=opt.id, type=opt.type, content=opt.content, label=opt.label, description=opt.description) for opt in (question.enhanced_options or [])]
            )
            question_responses.append(question_response)

        response_data = {
            "success": True,
            "questions": [q.dict() for q in question_responses],
            "count": len(question_responses),
            "total": len(all_questions),
            "page": page,
            "limit": limit
        }

        # Cache the result (only if cache is available)
        if cache:
            await cache.cache_question_results(cache_key, response_data, 3600)  # 1 hour cache

        # Return dict for frontend compatibility
        return response_data

    except Exception as e:
        logger.error(f"Get questions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get questions"
        )

@router.get("/stats", response_model=QuestionStats)
@limiter.limit("30/minute")
async def get_question_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get question statistics"""
    try:
        # Check cache first (only if cache is available)
        cached_stats = None
        if cache:
            cached_stats = await cache.get("question_stats", "admin")
            if cached_stats:
                return QuestionStats(**cached_stats)

        # Get admin-specific question count
        admin_id = get_admin_id_from_user(current_user)
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)
        total_questions = len(question_service.search_questions(limit=10000))  # Approximate count

        # For now, return mock statistics
        # In production, you'd aggregate this data from ChromaDB metadata
        stats_data = {
            "total_questions": total_questions,
            "subjects": {
                "Math": total_questions // 3,
                "Physics": total_questions // 3,
                "Chemistry": total_questions // 3
            },
            "difficulties": {
                "Easy": total_questions // 3,
                "Medium": total_questions // 3,
                "Hard": total_questions // 3
            },
            "recent_additions": 5  # Mock data
        }

        # Cache for 30 minutes (only if cache is available)
        if cache:
            await cache.set("question_stats", stats_data, 1800, "admin")

        return QuestionStats(**stats_data)

    except Exception as e:
        logger.error(f"Question stats error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get question statistics"
        )

@router.get("/{question_id}", response_model=QuestionResponse)
@limiter.limit("120/minute")
async def get_question(
    request: Request,
    question_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get question by ID"""
    try:
        # Get admin_id for data isolation
        admin_id = get_admin_id_from_user(current_user)

        # Check cache first with admin_id (only if cache is available)
        cached_question = None
        if cache:
            cached_question = await cache.get(f"question:{admin_id}:{question_id}", "questions")
        if cached_question:
            return QuestionResponse(**cached_question)

        # Get from admin-specific ChromaDB collection
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Get question from admin's collection
        question = question_service.get_question(question_id)

        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found in your collection"
            )

        # Convert Question object to response
        question_response = QuestionResponse(
            id=question.id,
            text=question.text,
            subject=question.subject,
            difficulty=question.difficulty,
            images=[img.to_dict() for img in question.images] if question.images else [],
            options=[opt.to_dict() for opt in question.options] if question.options else [],
            enhancedOptions=[opt.to_dict() for opt in question.enhanced_options] if question.enhanced_options else None
        )

        # Cache the result with admin_id (only if cache is available)
        if cache:
            await cache.set(f"question:{admin_id}:{question_id}", question_response.dict(), 3600, "questions")

        return question_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get question"
        )

@router.post("/save")
@limiter.limit("20/minute")
async def save_question(
    request: Request,
    question_data: Question,
    current_user: Dict[str, Any] = Depends(require_admin_for_write),
    db: DatabaseManager = Depends(get_database)
):
    """Save a single question to admin-specific collection"""
    try:
        # Get admin_id for data isolation
        admin_id = get_admin_id_from_user(current_user)

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Convert Pydantic model to dict for question service
        question_dict = question_data.dict()

        # Save question using admin-specific service
        success, question_id, error = question_service.save_question(question_dict)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error or "Failed to save question"
            )

        return {
            "success": True,
            "question_id": question_data.id,
            "message": "Question saved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Save question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save question: {str(e)}"
        )

@router.post("/batch-save")
@limiter.limit("10/minute")
async def batch_save_questions(
    request: Request,
    questions_data: List[Question],
    current_user: Dict[str, Any] = Depends(require_admin_for_write),
    db: DatabaseManager = Depends(get_database)
):
    """Save multiple questions to admin-specific collection"""
    try:
        if not questions_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No questions provided"
            )

        # Get admin_id for data isolation
        admin_id = get_admin_id_from_user(current_user)

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Convert Pydantic models to dicts
        question_dicts = [q.dict() for q in questions_data]

        # Save questions using admin-specific service
        success_count, total_count = question_service.save_questions_batch(question_dicts)

        if success_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save questions batch"
            )

        return {
            "success": True,
            "success_count": success_count,
            "total_count": total_count,
            "message": f"Successfully saved {success_count}/{total_count} questions"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch save questions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save questions batch: {str(e)}"
        )

@router.post("/", response_model=QuestionResponse)
@limiter.limit("20/minute")
async def create_question(
    request: Request,
    question_data: Question,
    current_user: Dict[str, Any] = Depends(require_admin_for_write),
    db: DatabaseManager = Depends(get_database)
):
    """Create a new question"""
    try:
        # Prepare document for ChromaDB
        document = question_data.text
        metadata = {
            "text": question_data.text,
            "subject": question_data.subject,
            "difficulty": question_data.difficulty,
            "extractedAt": question_data.extractedAt,
            "pdfSource": question_data.pdfSource,
            "correctAnswer": question_data.correctAnswer,
            "explanation": question_data.explanation,
            "tags": question_data.tags,
            "isActive": question_data.isActive,
            "createdBy": current_user["user_id"],
            "createdAt": datetime.utcnow().isoformat()
        }

        # Add to ChromaDB
        success = await db.chroma_add(
            ids=[question_data.id],
            documents=[document],
            metadatas=[metadata]
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save question"
            )

        return QuestionResponse(
            id=question_data.id,
            text=question_data.text,
            subject=question_data.subject,
            difficulty=question_data.difficulty,
            images=question_data.images,
            options=question_data.options,
            enhancedOptions=question_data.enhancedOptions
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create question"
        )

@router.delete("/{question_id}")
@limiter.limit("10/minute")
async def delete_question(
    request: Request,
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_for_write),
    db: DatabaseManager = Depends(get_database)
):
    """Delete question by ID from admin-specific collection"""
    try:
        # Get admin_id for data isolation
        admin_id = get_admin_id_from_user(current_user)

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Delete question from admin's collection
        success = question_service.delete_question(question_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found in your collection"
            )

        return {"success": True, "message": "Question deleted successfully", "question_id": question_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete question"
        )

@router.delete("/chromadb/clear")
@limiter.limit("5/hour")
async def clear_chromadb(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_for_write),
    db: DatabaseManager = Depends(get_database)
):
    """Clear questions from admin's ChromaDB collection"""
    try:
        # Get admin_id for data isolation
        admin_id = get_admin_id_from_user(current_user)

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Get count before clearing (approximate)
        search_results = question_service.search_questions(limit=10000)
        count_before = len(search_results)

        # Clear admin's collection by recreating it
        chromadb_client = question_service.chromadb_client
        collection_name = chromadb_client.collection_name

        # Delete and recreate the collection
        try:
            chromadb_client.client.delete_collection(collection_name)
        except:
            pass  # Collection might not exist

        # Recreate empty collection
        chromadb_client.collection = chromadb_client.client.create_collection(
            name=collection_name,
            metadata={"description": f"Questions for admin {admin_id}"}
        )

        logger.info(f"Admin {admin_id} ChromaDB collection cleared - {count_before} questions removed")

        return {
            "success": True,
            "message": f"Your question collection cleared successfully",
            "questions_removed": count_before
        }

    except Exception as e:
        logger.error(f"Clear ChromaDB error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear ChromaDB: {str(e)}"
        )
