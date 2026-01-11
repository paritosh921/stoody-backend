"""
Practice Module - Router
FastAPI router with practice API endpoints

Original: 1,863 lines
Refactored: ~350 lines (endpoints only, logic in services)
"""

import json
import random
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from slowapi import Limiter
from slowapi.util import get_remote_address
from bson import ObjectId

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user, get_database, get_cache

from .models import (
    StartSessionRequest,
    SessionResponse,
    SessionAnswer,
    SessionsListResponse,
    PracticeStats,
    EvaluateRequest,
    EvaluateResponse,
)
from .dependencies import require_student_or_admin
from .services import (
    evaluate_student_submission,
    grade_student_submission,
    create_session,
    submit_answer,
    complete_session,
    get_sessions,
    get_stats,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@router.post("/next")
@limiter.limit("60/minute")
async def get_next_practice_question(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Return a random next question using ChromaDB metadata first, with robust fallbacks."""
    try:
        from pydantic import BaseModel

        class NextQuestionRequest(BaseModel):
            subject: Optional[str] = None
            difficulty: Optional[str] = None
            excludeIds: Optional[List[str]] = None

        # Safely parse request body
        subject: Optional[str] = None
        difficulty: Optional[str] = None
        exclude_ids: List[str] = []
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.json()
                if isinstance(body, dict):
                    req_data = NextQuestionRequest(**body)
                    subject = req_data.subject
                    difficulty = req_data.difficulty
                    if req_data.excludeIds:
                        exclude_ids = list(req_data.excludeIds)
        except Exception:
            exclude_ids = []

        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Search for Practice Sets questions
        practice_questions = question_service.search_questions(
            query=None,
            subject=subject,
            difficulty=difficulty,
            document_type="Practice Sets",
            limit=1000
        )

        logger.info(f"Fetched {len(practice_questions)} Practice Sets questions from admin {admin_id}")

        # Convert to expected format
        fetched_ids = [q.id for q in practice_questions]
        metadatas = []
        for q in practice_questions:
            metadata = {
                "fullData": json.dumps(q.to_dict()),
                "subject": q.subject,
                "difficulty": q.difficulty,
                "document_type": getattr(q, 'document_type', 'Chapter Notes')
            }
            metadatas.append(metadata)

        # Fallback to MongoDB
        if not fetched_ids:
            mongo_filter = {"metadata.document_type": "Practice Sets"}
            try:
                mongo_filter["admin_id"] = ObjectId(admin_id)
            except Exception:
                mongo_filter["admin_id"] = admin_id
            if subject:
                mongo_filter["subject"] = subject
            if difficulty:
                mongo_filter["difficulty"] = difficulty

            mongo_questions = await db.mongo_find("questions", mongo_filter, projection={"id": 1})
            fetched_ids = [q.get("id") for q in mongo_questions if q.get("id")]
            logger.info(f"MongoDB fallback fetched {len(fetched_ids)} question ids")

        if not fetched_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No practice questions found. Please upload Practice Sets documents."
            )

        # Refine via fullData if available
        if metadatas and fetched_ids:
            refined: List[str] = []
            for qid, md in zip(fetched_ids, metadatas):
                full_json = md.get('fullData')
                if not full_json:
                    refined.append(qid)
                    continue
                try:
                    fd = json.loads(full_json)
                    doc_type = (fd.get('metadata', {}) or {}).get('document_type')
                    if doc_type and doc_type != 'Practice Sets':
                        continue
                    if subject and fd.get('subject') != subject:
                        continue
                    if difficulty and fd.get('difficulty') != difficulty:
                        continue
                    refined.append(qid)
                except Exception:
                    refined.append(qid)
            fetched_ids = refined

        # Filter out excluded IDs
        available_ids = [qid for qid in fetched_ids if qid not in exclude_ids]

        if not available_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No new questions available. All questions have been attempted."
            )

        # Select random question
        question_id = random.choice(available_ids)

        # Load question document
        question_doc: Dict[str, Any] = {}
        try:
            chroma_one = await db.chroma_get(ids=[question_id])
            md_list = chroma_one.get('metadatas') or []
            if md_list and md_list[0].get('fullData'):
                question_doc = json.loads(md_list[0]['fullData']) or {}
        except Exception as _e:
            logger.warning(f"Failed to load fullData for {question_id}: {_e}")

        if not question_doc:
            question_doc = await db.mongo_find_one("questions", {"id": question_id}) or {}

        if not question_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question data not found"
            )

        # Build image URLs
        images_with_urls = await _build_image_urls(question_doc, question_id, db)
        figures_with_urls = await _build_figure_urls(question_doc, db)

        merged_images = images_with_urls + figures_with_urls

        # Format LaTeX
        from utils.latex_formatter import format_question_latex

        question = {
            "id": question_id,
            "text": question_doc.get("text", ""),
            "subject": question_doc.get("subject", ""),
            "difficulty": question_doc.get("difficulty", "medium"),
            "options": question_doc.get("options", []),
            "images": merged_images,
            "questionFigures": figures_with_urls,
            "enhancedOptions": question_doc.get("enhancedOptions"),
            "correctAnswer": question_doc.get("correctAnswer") or question_doc.get("correct_answer"),
            "metadata": question_doc.get("metadata", {})
        }

        question = format_question_latex(question)

        logger.info(f"Returning question {question_id}: {len(images_with_urls)} option images, {len(figures_with_urls)} figures")

        return {
            "success": True,
            "question": question
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get next practice question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get next question: {str(e)}"
        )


async def _build_image_urls(question_doc: Dict[str, Any], question_id: str, db: DatabaseManager) -> List[Dict]:
    """Build image URLs for question options."""
    images_with_urls = []
    for img_ref in question_doc.get("images", []) or []:
        img_id = img_ref.get("id") if isinstance(img_ref, dict) else img_ref
        if not img_id:
            continue
        img_doc = await db.mongo_find_one("images", {"_id": img_id})
        if img_doc:
            images_with_urls.append({
                "id": img_id,
                "url": f"/api/v1/images/{img_id}",
                "contentType": img_doc.get("content_type", "image/jpeg"),
                "filename": img_doc.get("original_filename", str(img_id))
            })
        else:
            logger.warning(f"Image {img_id} not found for question {question_id}")
    return images_with_urls


async def _build_figure_urls(question_doc: Dict[str, Any], db: DatabaseManager) -> List[Dict]:
    """Build figure URLs for question diagrams."""
    import os
    import base64

    figures_with_urls = []
    for fig_ref in question_doc.get("question_figures", []):
        try:
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
                        file_path = img_doc["file_path"]
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, "rb") as f:
                                    image_bytes = f.read()
                                    base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
                                    content_type = img_doc.get("content_type", "image/jpeg")
                                    if not content_type.startswith("image/"):
                                        content_type = "image/jpeg"
                                    base64_data = f"data:{content_type};base64,{base64_encoded}"
                            except Exception as file_err:
                                logger.error(f"Failed to read image file {file_path}: {file_err}")

            figures_with_urls.append({
                "id": fig_id,
                "url": f"/api/v1/images/{fig_id}",
                "contentType": "image/jpeg",
                "filename": (fig_ref.get("filename") if isinstance(fig_ref, dict) else str(fig_id)),
                "base64Data": base64_data,
                "description": (fig_ref.get("description", "") if isinstance(fig_ref, dict) else ""),
                "type": "diagram"
            })
        except Exception as _e:
            logger.error(f"Practice figures processing error: {_e}", exc_info=True)

    return figures_with_urls


@router.post("/evaluate", response_model=EvaluateResponse)
@limiter.limit("120/minute")
async def evaluate_submission(
    request: Request,
    payload: EvaluateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Evaluate student's submission with AI tutor feedback."""
    try:
        result = await evaluate_student_submission(payload, current_user, db)

        if "error" in result:
            raise HTTPException(
                status_code=result.get("status_code", 500),
                detail=result["error"]
            )

        return EvaluateResponse(success=True, evaluation=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate submission: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate submission"
        )


@router.post("/sessions", response_model=SessionResponse)
@limiter.limit("30/minute")
async def start_practice_session(
    request: Request,
    session_data: StartSessionRequest,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Start a new practice session."""
    try:
        result = await create_session(session_data, current_user, db)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start practice session"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Start practice session error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start practice session"
        )


@router.post("/sessions/{session_id}/answer")
@limiter.limit("200/minute")
async def submit_session_answer(
    request: Request,
    session_id: str,
    answer_data: SessionAnswer,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Submit answer for a question in a practice session."""
    try:
        result = await submit_answer(session_id, answer_data, current_user, db)

        if "error" in result:
            raise HTTPException(
                status_code=result.get("status_code", 500),
                detail=result["error"]
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit session answer error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit answer"
        )


@router.post("/sessions/{session_id}/complete")
@limiter.limit("30/minute")
async def complete_practice_session(
    request: Request,
    session_id: str,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Complete a practice session."""
    try:
        result = await complete_session(session_id, current_user, db)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=result.get("status_code", 500),
                detail=result["error"]
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete practice session error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete practice session"
        )


@router.get("/sessions", response_model=SessionsListResponse)
@limiter.limit("60/minute")
async def get_practice_sessions(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    mode: Optional[str] = Query(None),
    is_completed: Optional[bool] = Query(None),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get practice sessions."""
    try:
        return await get_sessions(page, limit, mode, is_completed, current_user, db)

    except Exception as e:
        logger.error(f"Get practice sessions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get practice sessions"
        )


@router.get("/stats", response_model=PracticeStats)
@limiter.limit("30/minute")
async def get_practice_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get practice statistics."""
    try:
        return await get_stats(current_user, db, cache)

    except Exception as e:
        logger.error(f"Practice stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get practice statistics"
        )


@router.post("/grade", response_model=EvaluateResponse)
@limiter.limit("120/minute")
async def grade_submission(
    request: Request,
    payload: EvaluateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Comprehensive evaluation of student submissions using LLM analysis."""
    try:
        result = await grade_student_submission(payload, current_user, db)

        if "error" in result:
            raise HTTPException(
                status_code=result.get("status_code", 500),
                detail=result["error"]
            )

        return EvaluateResponse(success=True, evaluation=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grade submission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to grade submission"
        )
