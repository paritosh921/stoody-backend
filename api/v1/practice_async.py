"""
Async Practice API for SkillBot
Practice session management endpoints with analytics
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field, validator, root_validator
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
class PracticeSession(BaseModel):
    id: Optional[str] = None
    student_id: str
    mode: str = Field(..., pattern="^(practice|exam|timed)$")
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    questions_attempted: int = Field(default=0, ge=0)
    correct_answers: int = Field(default=0, ge=0)
    total_time_spent: int = Field(default=0, ge=0)  # in seconds
    started_at: datetime
    completed_at: Optional[datetime] = None
    is_completed: bool = False

class SessionQuestion(BaseModel):
    question_id: str
    answer: str
    is_correct: bool
    time_spent: int = Field(ge=0)  # in seconds
    answered_at: datetime

class SessionAnswer(BaseModel):
    question_id: str
    answer: str
    time_spent: int = Field(default=0, ge=0)

class StartSessionRequest(BaseModel):
    mode: str = Field(..., pattern="^(practice|exam|timed)$")
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    time_limit: Optional[int] = Field(None, ge=1)  # in minutes
    document_id: Optional[str] = None  # Practice set document ID

class SessionResponse(BaseModel):
    id: str
    mode: str
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    questions_attempted: int
    correct_answers: int
    accuracy_rate: float
    total_time_spent: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    is_completed: bool

class SessionsListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int
    page: int
    limit: int

class PracticeStats(BaseModel):
    total_sessions: int
    total_time_spent: int
    average_accuracy: float
    sessions_by_mode: Dict[str, int]
    recent_activity: List[Dict[str, Any]]

# ----------------------
# Helper utilities (local)
# ----------------------
async def _load_question_doc(db: DatabaseManager, qid: str) -> Dict[str, Any]:
    """Fetch question from Chroma (fullData) with Mongo fallback."""
    try:
        chroma = await db.chroma_get(ids=[qid])
        metas = chroma.get("metadatas") or []
        if metas and metas[0].get("fullData"):
            import json as _json
            return _json.loads(metas[0]["fullData"]) or {}
    except Exception:
        pass
    return await db.mongo_find_one("questions", {"id": qid}) or {}

def _options_text_from_question(q: Dict[str, Any]) -> str:
    opts = q.get("options", []) or []
    if opts:
        return "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts)])
    enh = q.get("enhancedOptions") or []
    if enh:
        parts = []
        for i, opt in enumerate(enh):
            label = chr(65 + i)
            if isinstance(opt, dict):
                if opt.get("type") == "text" and opt.get("content"):
                    parts.append(f"{label}. {opt.get('content')}")
                elif opt.get("type") == "image":
                    desc = opt.get("description") or "image option"
                    parts.append(f"{label}. [IMAGE] {desc}")
            else:
                parts.append(f"{label}. {str(opt)}")
        return "\n".join(parts)
    return ""

def _figure_images_base64(q: Dict[str, Any]) -> List[str]:
    imgs: List[str] = []
    for fig_ref in (q.get("question_figures", []) or []):
        try:
            b64 = None
            if isinstance(fig_ref, dict) and fig_ref.get("base64Data"):
                b64 = fig_ref["base64Data"]
            if b64 and not b64.startswith("data:image"):
                b64 = f"data:image/png;base64,{b64}"
            if b64:
                imgs.append(b64)
        except Exception:
            pass
    return imgs

def _normalize_choice_text(s: str) -> str:
    import re as _re
    t = (s or '').upper().strip()
    # Support any single letter A-Z for MCQ options (not just A-D)
    m = _re.search(r"\b([A-Z])\b", t)
    return m.group(1) if m else t

def _normalize_numeric_text(s: str) -> str:
    t = (s or '').strip().replace(' ', '').replace(',', '.')
    if ':' in t and t.count(':') == 1 and all(part.isdigit() for part in t.split(':')):
        t = t.replace(':', '.')
    return t

def require_student_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require student or admin access"""
    if current_user.get("user_type") not in ["student", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student or admin access required"
        )
    return current_user

@router.post("/next")
@limiter.limit("60/minute")
async def get_next_practice_question(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Return a random next question using ChromaDB metadata first, with robust fallbacks.

    Behavior:
    - Prefer questions tagged as Practice Sets in metadata when available
    - Fall back gracefully to all questions if tag is missing
    - Avoid hard dependency on MongoDB `questions` collection; use Chroma `fullData`
    - Build figure/image payloads with base64 when available; else serve via images API
    """
    try:
        import random
        from pydantic import BaseModel
        
        class NextQuestionRequest(BaseModel):
            subject: Optional[str] = None
            difficulty: Optional[str] = None
            excludeIds: Optional[List[str]] = None

        # Safely parse request body (optional)
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
        except Exception as _e:
            # Fall back to defaults if parsing fails; do not reject
            exclude_ids = []
        
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Initialize admin-specific question service
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)

        # Search for Practice Sets questions from admin's collection
        practice_questions = question_service.search_questions(
            query=None,
            subject=subject,
            difficulty=difficulty,
            document_type="Practice Sets",
            limit=1000  # High limit for randomization
        )

        logger.info(f"Fetched {len(practice_questions)} Practice Sets questions from admin {admin_id} collection (subject={subject}, difficulty={difficulty})")

        # If no Practice Sets found, do NOT mix in Test Series; keep Hustle strictly Practice Sets
        # We'll rely on Mongo fallback below which filters by document_type.

        # Convert to the expected format
        fetched_ids = [q.id for q in practice_questions]
        # For metadata, we'll need to reconstruct it from the question objects
        metadatas = []
        for q in practice_questions:
            metadata = {
                "fullData": json.dumps(q.to_dict()),
                "subject": q.subject,
                "difficulty": q.difficulty,
                "document_type": getattr(q, 'document_type', 'Chapter Notes')
            }
            metadatas.append(metadata)

        # Fallback to MongoDB if ChromaDB has no entries (strictly Practice Sets)
        if not fetched_ids:
            mongo_filter = {"metadata.document_type": "Practice Sets"}
            # Scope by admin
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
                detail="No practice questions found. Please upload Practice Sets documents and process them."
            )

        # If we have metadatas (from Chroma), prefer to refine the pool via fullData
        if metadatas and fetched_ids:
            refined: List[str] = []
            for qid, md in zip(fetched_ids, metadatas):
                full_json = md.get('fullData')
                if not full_json:
                    refined.append(qid)
                    continue
                try:
                    import json as _json
                    fd = _json.loads(full_json)
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
        
        # Select random question from available
        question_id = random.choice(available_ids)
        
        # Try to reconstruct question from Chroma fullData first; fallback to Mongo if needed
        question_doc: Dict[str, Any] = {}
        try:
            chroma_one = await db.chroma_get(ids=[question_id])
            md_list = chroma_one.get('metadatas') or []
            if md_list and md_list[0].get('fullData'):
                import json as _json
                question_doc = _json.loads(md_list[0]['fullData']) or {}
        except Exception as _e:
            logger.warning(f"Failed to load fullData for {question_id}: {_e}")

        if not question_doc:
            question_doc = await db.mongo_find_one("questions", {"id": question_id}) or {}
        
        if not question_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question data not found"
            )
        
        # Build image URLs for frontend - images are served from local disk via /api/v1/images/{id}
        images_with_urls = []
        for img_ref in question_doc.get("images", []) or []:
            # Support both string ID and dict refs having id
            img_id = img_ref.get("id") if isinstance(img_ref, dict) else img_ref
            if not img_id:
                continue
            # Check if image exists in MongoDB using _id field (MongoDB primary key)
            img_doc = await db.mongo_find_one("images", {"_id": img_id})
            if img_doc:
                images_with_urls.append({
                    "id": img_id,
                    "url": f"/api/v1/images/{img_id}",  # Serve from local disk
                    "contentType": img_doc.get("content_type", "image/jpeg"),
                    "filename": img_doc.get("original_filename", str(img_id))
                })
            else:
                logger.warning(f"Image {img_id} referenced in question {question_id} but not found in database")
        
        # Also include QUESTION FIGURES (diagrams)
        figures_with_urls = []
        for fig_ref in question_doc.get("question_figures", []):
            try:
                fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref
                base64_data = None

                # First check if base64Data is embedded in the figure reference
                if isinstance(fig_ref, dict) and fig_ref.get("base64Data"):
                    base64_data = fig_ref["base64Data"]
                    logger.info(f"Using embedded base64Data for figure {fig_id}")
                else:
                    # Try to get base64Data from images collection
                    img_doc = await db.mongo_find_one("images", {"_id": fig_id})
                    if img_doc:
                        # Check if base64Data is stored in the document
                        if img_doc.get("base64Data"):
                            base64_data = img_doc["base64Data"]
                            logger.info(f"Using stored base64Data for figure {fig_id}")
                        # If not, read from file_path and convert to base64
                        elif img_doc.get("file_path"):
                            import os
                            import base64
                            file_path = img_doc["file_path"]
                            if os.path.exists(file_path):
                                try:
                                    with open(file_path, "rb") as f:
                                        image_bytes = f.read()
                                        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
                                        # Determine content type from file extension or stored content_type
                                        content_type = img_doc.get("content_type", "image/jpeg")
                                        if not content_type.startswith("image/"):
                                            # Default to jpeg if content type is not an image
                                            content_type = "image/jpeg"
                                        base64_data = f"data:{content_type};base64,{base64_encoded}"
                                        logger.info(f"✅ Loaded and converted image {fig_id} from file: {len(base64_data)} bytes")
                                except Exception as file_err:
                                    logger.error(f"❌ Failed to read image file {file_path}: {file_err}")
                            else:
                                logger.warning(f"⚠️ Image file not found: {file_path}")
                        else:
                            logger.warning(f"⚠️ No base64Data or file_path for image {fig_id}")
                    else:
                        logger.warning(f"⚠️ Image document not found: {fig_id}")

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
                logger.error(f"❌ Practice figures processing error: {_e}", exc_info=True)

        merged_images = images_with_urls + figures_with_urls

        # Format LaTeX in question text and options
        from utils.latex_formatter import format_question_latex

        question = {
            "id": question_id,
            "text": question_doc.get("text", ""),
            "subject": question_doc.get("subject", ""),
            "difficulty": question_doc.get("difficulty", "medium"),
            "options": question_doc.get("options", []),
            "images": merged_images,  # Include both option images and figures
            "questionFigures": figures_with_urls,  # Separate field for diagrams/figures
            "enhancedOptions": question_doc.get("enhancedOptions"),
            "correctAnswer": question_doc.get("correctAnswer") or question_doc.get("correct_answer"),  # Include answer for debugging
            "metadata": question_doc.get("metadata", {})
        }

        # Format LaTeX expressions in question text and options
        question = format_question_latex(question)
        
        logger.info(f"Returning question {question_id}: {len(images_with_urls)} option images, {len(figures_with_urls)} figures")
        if figures_with_urls:
            for idx, fig in enumerate(figures_with_urls):
                logger.info(f"  Figure {idx + 1}: ID={fig.get('id')}, has_base64={bool(fig.get('base64Data'))}, base64_len={len(fig.get('base64Data', ''))}")
        
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


class EvaluateRequest(BaseModel):
    questionId: str
    answerText: Optional[str] = None
    canvasData: Optional[str] = None
    canvasPages: Optional[List[str]] = None

    # Be flexible: accept pages as strings or objects with common keys; normalize to data URLs
    @validator('canvasData', pre=True)
    def _normalize_canvas_data(cls, v):
        try:
            if v and isinstance(v, str) and not v.startswith('data:image'):
                return f"data:image/png;base64,{v}"
        except Exception:
            pass
        return v

    @validator('canvasPages', pre=True)
    def _normalize_canvas_pages(cls, v):
        if v is None:
            return v
        try:
            if isinstance(v, list):
                out: List[str] = []
                for item in v:
                    s = None
                    if isinstance(item, str):
                        s = item
                    elif isinstance(item, dict):
                        s = (
                            item.get('dataUrl')
                            or item.get('url')
                            or item.get('data')
                            or item.get('image')
                            or item.get('src')
                        )
                    if s:
                        if not s.startswith('data:image'):
                            s = f"data:image/png;base64,{s}"
                        out.append(s)
                return out
            # If a single string is provided, wrap as list
            if isinstance(v, str):
                s = v
                if not s.startswith('data:image'):
                    s = f"data:image/png;base64,{s}"
                return [s]
        except Exception:
            return v
        return v

    @root_validator(pre=True)
    def _coerce_aliases(cls, values):
        # Accept snake_case aliases from frontend
        mapping = {
            'question_id': 'questionId',
            'answer_text': 'answerText',
            'canvas_data': 'canvasData',
            'canvas_pages': 'canvasPages'
        }
        for src, dst in mapping.items():
            if src in values and dst not in values:
                values[dst] = values[src]
        # If canvasPages provided as single object/string elsewhere, normalize to list
        cp = values.get('canvasPages')
        if isinstance(cp, str):
            values['canvasPages'] = [cp]
        return values

class EvaluateResponse(BaseModel):
    success: bool = True
    evaluation: Dict[str, Any]


@router.post("/evaluate", response_model=EvaluateResponse)
@limiter.limit("120/minute")
async def evaluate_submission(
    request: Request,
    payload: EvaluateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Evaluate student's submission (canvas image and/or text) for a question with AI tutor feedback.

    Returns: { success, evaluation: { correct, score, extractedAnswer, feedback, reasoning } }
    """
    try:
        qid = payload.questionId
        answer_text = (payload.answerText or "").strip()
        canvas_data = payload.canvasData
        
        # Normalize canvas data header if client sent raw base64
        if canvas_data and not canvas_data.startswith("data:image"):
            canvas_data = f"data:image/png;base64,{canvas_data}"

        # Fetch question from Chroma (fullData) first; fallback to MongoDB
        question_doc = await _load_question_doc(db, qid)
        if not question_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")

        # Pull correct answer
        ca_primary = question_doc.get("correctAnswer")
        ca_alt = question_doc.get("correct_answer")
        correct_answer = str((ca_primary if ca_primary is not None else (ca_alt if ca_alt is not None else ""))).strip()

        # Extract question text and options
        question_text = str(question_doc.get("text", ""))
        options_text = _options_text_from_question(question_doc)

        # Initialize AI service
        from services.async_openai_service import AsyncOpenAIService
        ai = AsyncOpenAIService()

        # Prepare images: Question Figures + Student Canvas
        # 1. Question Figures
        question_images = _figure_images_base64(question_doc)
        
        # 2. Student Canvas Images
        student_images = []
        if payload.canvasPages and len(payload.canvasPages) > 0:
            student_images = payload.canvasPages
        elif canvas_data:
            student_images = [canvas_data]
            
        # Combine all images for the LLM
        all_images = question_images + student_images
        num_q_images = len(question_images)
        
        # Construct the Prompt
        prompt = (
            "You are an expert personal tutor. You are evaluating a student's answer to a question.\n\n"
            f"QUESTION:\n{question_text}\n\n"
        )
        
        if options_text:
            prompt += f"OPTIONS:\n{options_text}\n\n"
            
        if correct_answer:
            prompt += f"CORRECT ANSWER: {correct_answer}\n\n"
        else:
            prompt += "CORRECT ANSWER: (Not provided, please solve it yourself to verify)\n\n"
            
        prompt += "STUDENT INPUT:\n"
        if answer_text:
            prompt += f"Typed Answer: {answer_text}\n"
        else:
            prompt += "Typed Answer: (None)\n"
            
        if student_images:
            prompt += f"Canvas Work: The student has submitted {len(student_images)} pages of handwritten work (images attached).\n"
            if num_q_images > 0:
                prompt += f"Note: The first {num_q_images} images are diagrams belonging to the question. The remaining images are the student's work.\n"
        else:
            prompt += "Canvas Work: (None)\n"
            
        prompt += (
            "\nTASK:\n"
            "1. Analyze the student's input (typed text and/or handwritten work).\n"
            "2. Determine their final answer.\n"
            "3. Compare it with the CORRECT ANSWER.\n"
            "4. Provide high-quality, encouraging, and subject-specific feedback.\n"
            "   - If correct: Confirm it and briefly explain why it's correct (reinforce the concept).\n"
            "   - If incorrect: Point out the mistake kindly, explain the correct concept/method, and guide them to the right answer. Do NOT just say 'Wrong'. Be a tutor.\n"
            "   - If the student's work is unclear, ask them to clarify.\n"
            "5. Return a JSON object with the following fields:\n"
            "   - 'extracted_answer': The answer you extracted from the student's work.\n"
            "   - 'is_correct': boolean (true/false).\n"
            "   - 'feedback': The message to show to the student.\n"
            "   - 'reasoning': Internal reasoning for your evaluation.\n"
        )
        
        logger.info(f"📤 Sending evaluation request to LLM for Q:{qid}. Images: {len(all_images)} ({len(question_images)} Q + {len(student_images)} S)")
        
        # Call LLM
        system_prompt = "You are a helpful, encouraging, and highly knowledgeable tutor. You always output valid JSON."
        
        if all_images:
            response = await ai.analyze_images_and_text_async(
                all_images,
                prompt,
                max_tokens=1000,
                system_prompt=system_prompt
            )
        else:
            response = await ai.chat_completion_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
        raw_response = (response.get("response") or "").strip()
        logger.info(f"LLM Evaluation Response: {raw_response[:200]}...")
        
        # Parse JSON Response
        import json as _json
        import re as _re
        
        evaluation_data = {
            "correct": False,
            "score": 0.0,
            "extractedAnswer": "",
            "feedback": "",
            "reasoning": "",
            "answerSource": "ai_eval"
        }
        
        try:
            # Extract JSON block
            m = _re.search(r"\{.*\}", raw_response, _re.DOTALL)
            if m:
                json_str = m.group(0)
                parsed = _json.loads(json_str)
                
                evaluation_data["correct"] = bool(parsed.get("is_correct", False))
                evaluation_data["score"] = 1.0 if evaluation_data["correct"] else 0.0
                evaluation_data["extractedAnswer"] = str(parsed.get("extracted_answer", "")).strip()
                evaluation_data["feedback"] = str(parsed.get("feedback", "")).strip()
                evaluation_data["reasoning"] = str(parsed.get("reasoning", "")).strip()
            else:
                # Fallback if no JSON found
                evaluation_data["feedback"] = raw_response
                evaluation_data["reasoning"] = "Could not parse JSON from LLM response."
                
        except Exception as parse_err:
            logger.error(f"Failed to parse LLM evaluation JSON: {parse_err}")
            evaluation_data["feedback"] = raw_response
            evaluation_data["reasoning"] = f"JSON parse error: {parse_err}"

        # If feedback is empty (parsing failed completely), use raw response
        if not evaluation_data["feedback"]:
             evaluation_data["feedback"] = raw_response

        return EvaluateResponse(success=True, evaluation=evaluation_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate submission: {str(e)}")
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
    """Start a new practice session"""
    try:
        user_id = current_user["user_id"]

        # Create session record
        session_record = {
            "student_id": user_id,
            "mode": session_data.mode,
            "subject": session_data.subject,
            "difficulty": session_data.difficulty,
            "time_limit": session_data.time_limit,
            "document_id": session_data.document_id,  # Track which practice set
            "questions_attempted": 0,
            "correct_answers": 0,
            "total_time_spent": 0,
            "started_at": datetime.utcnow(),
            "is_completed": False,
            "questions": []  # Will store question attempts
        }

        session_id = await db.mongo_insert_one("practice_sessions", session_record)

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start practice session"
            )

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
    """Submit answer for a question in a practice session"""
    try:
        user_id = current_user["user_id"]

        # Get session
        session = await db.mongo_find_one("practice_sessions", {"_id": session_id})

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Practice session not found"
            )

        # Check ownership (students can only access their own sessions)
        if (current_user["user_type"] == "student" and
            session["student_id"] != user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        if session["is_completed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session is already completed"
            )

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

        # Create question attempt record in session
        question_attempt = {
            "question_id": answer_data.question_id,
            "answer": answer_data.answer,
            "is_correct": is_correct,
            "time_spent": answer_data.time_spent,
            "answered_at": datetime.utcnow()
        }

        # Update session with new attempt
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
            try:
                student_oid = ObjectId(user_id)

                # Get admin_id from JWT token for data isolation
                admin_id = current_user.get("admin_id")
                if not admin_id:
                    logger.warning(f"Student {user_id} has no admin_id in JWT token")
                    admin_id = None

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

                # Add admin_id for data isolation if available
                if admin_id:
                    attempt_doc["admin_id"] = admin_id

                await db.mongo_insert_one("question_attempts", attempt_doc)

                # Log activity in student_activity_log
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

                # Add admin_id for data isolation if available
                if admin_id:
                    activity_doc["admin_id"] = admin_id

                await db.mongo_insert_one("student_activity_log", activity_doc)
            except Exception as e:
                logger.warning(f"Failed to track question attempt: {str(e)}")

        return {
            "message": "Answer submitted successfully",
            "is_correct": is_correct,
            "question_id": answer_data.question_id,
            "score": score
        }

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
    """Complete a practice session"""
    try:
        user_id = current_user["user_id"]

        # Get session
        session = await db.mongo_find_one("practice_sessions", {"_id": session_id})

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Practice session not found"
            )

        # Check ownership
        if (current_user["user_type"] == "student" and
            session["student_id"] != user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        if session["is_completed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session is already completed"
            )

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
    """Get practice sessions"""
    try:
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
    """Get practice statistics"""
    try:
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
            for session in recent_sessions[-10:]  # Last 10 recent sessions
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
    db: DatabaseManager = Depends(get_database)
):
    """Comprehensive evaluation of student submissions using LLM analysis.

    Supports:
    - Multiple choice questions
    - Written solutions and explanations
    - Mathematical derivations
    - Diagrams and visual solutions
    - Definitions and conceptual answers

    Returns detailed feedback comparing student work with expected solution.
    """
    try:
        from services.async_openai_service import AsyncOpenAIService
        ai = AsyncOpenAIService()

        qid = payload.questionId
        answer_text = (payload.answerText or "").strip()
        canvas_data = payload.canvasData
        if canvas_data and not canvas_data.startswith("data:image"):
            canvas_data = f"data:image/png;base64,{canvas_data}"

        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Get question from admin's collection
        from services.question_service import QuestionService
        question_service = QuestionService(admin_id)
        question_obj = question_service.get_question(qid)

        if not question_obj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found in your admin's collection")

        # Convert to dict format expected by the evaluation logic
        q = question_obj.to_dict()

        ca_primary = q.get("correctAnswer")
        ca_alt = q.get("correct_answer")
        correct_answer = str((ca_primary if ca_primary is not None else (ca_alt if ca_alt is not None else ""))).strip()

        # Comprehensive evaluation system prompt
        system_prompt = """You are an expert academic evaluator. Your job is to comprehensively analyze student solutions.

CRITICAL: Return ONLY a single line of valid JSON with NO extra text, NO newlines, NO formatting.

ANALYSIS CAPABILITIES:
- Understand handwritten equations, formulas, and mathematical expressions
- Read handwritten text, definitions, and explanations (including "I don't know", "not sure", etc.)
- Analyze diagrams, graphs, and visual problem-solving steps
- Recognize scientific notation, chemical formulas, and technical symbols
- Understand multi-step solutions and problem-solving approaches
- Detect when students are asking for help or indicating uncertainty

EVALUATION PROCESS:
1. First, solve the given question yourself to determine the correct answer
2. Extract and interpret ALL content from the student's canvas submission
3. Distinguish between:
   - MCQ answers: Single letter choices (A, B, C, D, etc.)
   - Written explanations: Phrases like "I don't know", "not sure", "help me"
   - Solution attempts: Diagrams, equations, calculations, definitions
4. Evaluate the student's approach, calculations, and final answer
5. Provide detailed feedback comparing what they wrote vs. what they should have written
6. Give constructive guidance on areas for improvement

HANDWRITING RECOGNITION:
- Be generous in interpreting unclear handwriting
- Look for mathematical symbols: +, -, ×, ÷, =, ≠, ≈, ∫, Σ, √, π, ∞, etc.
- Recognize scientific notation: 2.5 × 10³, 6.02 × 10²³, etc.
- Identify equation structures: variables, constants, operations
- Understand chemical formulas: H₂O, CO₂, CH₄, etc.
- Read definitions and explanatory text (word-for-word)
- Interpret diagrams and their labels
- Detect help requests: "I don't know", "help", "not sure", "unclear", "confused", "stuck"

FEEDBACK FORMAT:
- "What you wrote": Describe student's work clearly and accurately
- "What is expected": Explain the correct approach/answer
- "Suggestions": Provide specific improvement guidance

REQUIRED JSON FORMAT (single line):
{"correct":false,"score":0.0,"extractedAnswer":"I don't know the answer","feedback":"What you wrote: 'I don't know the answer'. This is okay - it's better to ask for help than guess! What is expected: For this magnetic field question, option C shows the correct field lines. Suggestions: Try to recall that magnetic field lines always form closed loops from North to South pole.","reasoning":"Student expressed uncertainty rather than attempting to answer"}

IMPORTANT DISTINCTION:
- If student writes a SINGLE LETTER (A/B/C/D): extractedAnswer should be just that letter
- If student writes TEXT/PHRASE: extractedAnswer should be the FULL TEXT (e.g., "I don't know the answer")
- If student draws DIAGRAM: extractedAnswer should describe what they drew

For multiple choice:
  - If single letter written: {"extractedAnswer": "A"}
  - If explanation written: {"extractedAnswer": "I don't know"} or {"extractedAnswer": "The answer should be C because..."}

Return ONLY the JSON line. No other text."""

        # Build comprehensive evaluation context
        question_text = str(q.get("text", ""))
        subject = q.get("subject", "Unknown")
        difficulty = q.get("difficulty", "medium")

        context_parts = [
            f"Question: {question_text}",
            f"Subject: {subject}",
            f"Difficulty: {difficulty}",
        ]

        # Add options if available
        options_list = _options_text_from_question(q)
        if options_list:
            context_parts.append(f"\nOptions:\n{options_list}")

        # Add student submission info
        context_parts.append("\n=== STUDENT SUBMISSION ===")
        if answer_text and canvas_data:
            context_parts.append(f"TYPED TEXT: {answer_text}")
            context_parts.append("CANVAS: See image below - contains student's handwritten work")
        elif answer_text:
            context_parts.append(f"TYPED TEXT: {answer_text}")
            context_parts.append("CANVAS: None provided")
        elif canvas_data:
            context_parts.append("TYPED TEXT: None")
            context_parts.append("CANVAS: See image below - contains all student work")
        else:
            context_parts.append("TYPED TEXT: None")
            context_parts.append("CANVAS: None")

        context_parts.append("\n=== EVALUATION TASK ===")
        context_parts.append("1. Solve the question yourself to determine the correct answer")
        context_parts.append("2. Analyze ALL student content (typed + canvas)")
        context_parts.append("3. Extract equations, formulas, calculations, diagrams, and explanations")
        context_parts.append("4. Evaluate correctness of approach, calculations, and final answer")
        context_parts.append("5. Provide detailed feedback: 'What you wrote: X. What is expected: Y. Suggestions: Z'")

        prompt_text = "\n".join(context_parts)

        # Collect images for context (question figures first, then all student pages)
        images_for_eval: list[str] = []
        figures = _figure_images_base64(q)
        for fig in figures[:2]:
            if fig:
                images_for_eval.append(fig)
        student_pages = payload.canvasPages or ([] if not canvas_data else [canvas_data])
        images_for_eval.extend(student_pages)

        logger.info(f"📤 Comprehensive evaluation request:")
        logger.info(f"   - Question ID: {qid}")
        logger.info(f"   - Subject: {subject}")
        logger.info(f"   - Total images: {len(images_for_eval)} (question figures + student canvas)")
        logger.info(f"   - Has typed text: {bool(answer_text)}")
        logger.info(f"   - Has canvas: {bool(canvas_data)}")

        # Call LLM for comprehensive evaluation
        if images_for_eval:
            res = await ai.analyze_images_and_text_async(
                images_for_eval,
                prompt_text,
                max_tokens=800,
                system_prompt=system_prompt
            )
        else:
            # Text-only evaluation (no images)
            res = await ai.evaluate_answer_async(
                question=question_text,
                student_answer=answer_text,
                correct_answer=correct_answer
            )

        raw_response = (res.get("response") or "").strip()
        logger.info(f"📥 LLM evaluation response: {raw_response[:500]}...")

        # Parse JSON response
        import re as _re, json as _json

        evaluation = None
        # Try to extract JSON from response
        json_match = _re.search(r'\{.*\}', raw_response, _re.DOTALL)
        if json_match:
            try:
                parsed = _json.loads(json_match.group(0))
                evaluation = {
                    "correct": bool(parsed.get("correct", False)),
                    "score": float(parsed.get("score", 0.0)),
                    "extractedAnswer": str(parsed.get("extractedAnswer", "Not found")),
                    "feedback": str(parsed.get("feedback", "No feedback provided")),
                    "reasoning": str(parsed.get("reasoning", "No reasoning provided"))
                }
                logger.info(f"✅ Successfully parsed LLM evaluation: correct={evaluation['correct']}, score={evaluation['score']}")
            except Exception as parse_error:
                logger.warning(f"⚠️ JSON parse failed: {parse_error}")

        # Fallback: construct evaluation from raw response if JSON parsing failed
        if not evaluation:
            logger.warning("Using fallback evaluation construction")
            evaluation = {
                "correct": False,
                "score": 0.5,  # Partial credit for attempting
                "extractedAnswer": answer_text or "See canvas",
                "feedback": raw_response[:500] if raw_response else "Unable to evaluate submission. Please try again.",
                "reasoning": "Comprehensive analysis attempted but response format needs review."
            }

        # Validate against known correct answer ONLY for clear MCQ submissions
        # Trust LLM evaluation for all other cases (written explanations, diagrams, etc.)
        if correct_answer:
            extracted = evaluation.get("extractedAnswer", "").strip().upper()
            expected = correct_answer.strip().upper()

            # Only apply MCQ validation if:
            # 1. Expected answer is a single letter (MCQ)
            # 2. Extracted answer is EXACTLY one letter (not part of a phrase)
            # 3. LLM marked it as an MCQ response (not a written explanation)
            is_expected_mcq = len(expected) == 1 and expected.isalpha()
            is_extracted_single_letter = len(extracted) == 1 and extracted.isalpha()

            # Check if LLM detected this as a written explanation vs MCQ answer
            feedback_lower = evaluation.get("feedback", "").lower()
            is_written_explanation = any(phrase in feedback_lower for phrase in [
                "you wrote", "you explained", "you described", "you stated",
                "your explanation", "your description", "don't know", "dont know",
                "unclear", "not sure", "confused", "help", "stuck"
            ])

            if is_expected_mcq and is_extracted_single_letter and not is_written_explanation:
                # This is a genuine MCQ answer - validate it
                is_match = (extracted == expected)
                if is_match and not evaluation["correct"]:
                    logger.info(f"Correcting evaluation: MCQ answer matches (student: {extracted}, expected: {expected})")
                    evaluation["correct"] = True
                    evaluation["score"] = 1.0
                    evaluation["feedback"] = f"Excellent! You correctly chose option {expected}. " + evaluation.get("feedback", "")
                elif not is_match and evaluation["correct"]:
                    logger.info(f"Correcting evaluation: MCQ answer doesn't match (student: {extracted}, expected: {expected})")
                    evaluation["correct"] = False
                    evaluation["score"] = 0.0
                    evaluation["feedback"] = f"Not quite. You chose {extracted}, but the correct answer is {expected}. " + evaluation.get("feedback", "")
            else:
                # Trust LLM's comprehensive evaluation for written work
                logger.info(f"Trusting LLM evaluation for written/explanation content (extracted: '{extracted[:50]}...', is_mcq: {is_expected_mcq})")

        return EvaluateResponse(success=True, evaluation=evaluation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grade submission error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to grade submission")

# Backward compatibility: legacy route alias
@router.post("/evaluate", response_model=EvaluateResponse)
@limiter.limit("120/minute")
async def evaluate_submission_compat(
    request: Request,
    payload: EvaluateRequest,
    db: DatabaseManager = Depends(get_database)
):
    return await grade_submission(request, payload, db)
