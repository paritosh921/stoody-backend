"""
Async MCQ question endpoints.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from bson import ObjectId
from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database, get_cache
from api.v1.mcq_dependencies import require_student_or_admin, require_admin_for_write
from api.v1.mcq_schemas import MCQQuestion, MCQResponse, MCQListResponse
from core.database import DatabaseManager
from core.cache import CacheManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

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

        # If user is a student, only show completed OCR documents that are active
        if current_user.get("user_type") == "student":
            filter_query["ocr_status"] = "completed"
            # is_active: {$ne: False} matches True, None, or missing field (default active)
            filter_query["is_active"] = {"$ne": False}

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


