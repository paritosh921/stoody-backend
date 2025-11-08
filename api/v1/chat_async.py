"""
Async Chat API for SkillBot
High-performance chat endpoints with AI integration, caching, and rate limiting
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
import hashlib
import json
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from services.async_openai_service import AsyncOpenAIService
from core.database import DatabaseManager
from core.cache import CacheManager
from config_async import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: Optional[Any] = None  # Can be string, date, or anything

    class Config:
        extra = "allow"  # Allow extra fields from frontend

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    sessionId: str = Field(default="default", max_length=100)
    userId: str = Field(default="anonymous", max_length=100)
    mode: str = Field(default="general")
    conversationHistory: List[ChatMessage] = Field(default_factory=list, max_length=20)
    canvasData: Optional[str] = None
    canvasPages: Optional[List[str]] = None
    subject: str = Field(default="general", max_length=50)

    # Optional frontend fields (for compatibility)
    fastMode: Optional[bool] = False
    applyFormatting: Optional[bool] = True
    useAssistants: Optional[bool] = False
    context: Optional[Any] = None
    fileIds: Optional[List[str]] = None

    class Config:
        extra = "allow"  # Allow any extra fields from frontend

    @validator('canvasData', pre=True)
    def validate_canvas_data(cls, v):
        # Accept both data URLs and raw base64; normalize to data URL
        try:
            if v and isinstance(v, str) and not v.startswith('data:image'):
                return f"data:image/png;base64,{v}"
        except Exception:
            pass
        return v

    @validator('canvasPages', pre=True)
    def validate_canvas_pages(cls, v):
        # Accept arrays of strings or objects; normalize to data URLs
        if v is None:
            return v
        if not isinstance(v, list):
            return v
        out: List[str] = []
        for i, item in enumerate(v):
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
                if not isinstance(s, str):
                    continue
                if not s.startswith('data:image'):
                    s = f"data:image/png;base64,{s}"
                out.append(s)
        return out or None

class ChatResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cached: bool = False

class ChatHealthResponse(BaseModel):
    success: bool
    message: str
    model: str
    service: str

# Dependency injection
async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db

async def get_cache(request: Request) -> CacheManager:
    return request.app.state.cache

# Background task for analytics
async def log_chat_analytics(session_id: str, user_id: str, mode: str,
                           response_time: float, cached: bool, db: DatabaseManager):
    """Log chat analytics in background"""
    try:
        analytics_data = {
            "session_id": session_id,
            "user_id": user_id,
            "mode": mode,
            "response_time_ms": response_time * 1000,
            "cached": cached,
            "timestamp": time.time()
        }

        await db.mongo_insert_one("chat_analytics", analytics_data)
    except Exception as e:
        logger.error(f"Failed to log analytics: {str(e)}")

# Background task for chat session tracking
async def track_chat_session(user_id: str, session_id: str, message: str,
                            response: str, mode: str, db: DatabaseManager):
    """Track chat sessions for student monitoring"""
    try:
        # Check if user is a student (user_id should be ObjectId string)
        try:
            student_oid = ObjectId(user_id)
            student = await db.mongo_find_one("students", {"_id": student_oid})

            if not student:
                # Not a student, skip tracking
                return
        except Exception:
            # Invalid ObjectId or user doesn't exist
            return

        # Update or create chat session
        session = await db.mongo_find_one("chat_sessions", {
            "student_id": student_oid,
            "session_id": session_id
        })

        if session:
            # Update existing session
            await db.mongo_update_one(
                "chat_sessions",
                {"_id": session["_id"]},
                {
                    "$set": {
                        "last_message_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    },
                    "$inc": {"message_count": 1}
                }
            )
        else:
            # Get admin_id from JWT for data isolation
            admin_id = current_user.get("admin_id")
            if not admin_id:
                logger.warning(f"Student {user_id} has no admin_id in JWT token")
                admin_id = None

            # Create new session
            session_doc = {
                "student_id": student_oid,
                "session_id": session_id,
                "mode": mode,
                "started_at": datetime.utcnow(),
                "last_message_at": datetime.utcnow(),
                "message_count": 1,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Add admin_id for data isolation if available
            if admin_id:
                session_doc["admin_id"] = admin_id

            await db.mongo_insert_one("chat_sessions", session_doc)

        # Log chat activity
        await db.mongo_insert_one("student_activity_log", {
            "student_id": student_oid,
            "action": "chat_message",
            "timestamp": datetime.utcnow(),
            "metadata": {
                "session_id": session_id,
                "mode": mode,
                "message_preview": message[:100] if len(message) > 100 else message
            }
        })

    except Exception as e:
        logger.error(f"Failed to track chat session: {str(e)}")

@router.post("/", response_model=ChatResponse)
@limiter.limit("60/minute")
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Handle chat requests with AI responses
    Includes caching, rate limiting, and async processing
    """
    start_time = time.time()

    try:
        logger.info(f"Chat request - User: {chat_request.userId}, Session: {chat_request.sessionId}, Mode: {chat_request.mode}")

        # Create cache key for the request
        cache_key_data = {
            "message": chat_request.message,
            "mode": chat_request.mode,
            "subject": chat_request.subject,
            "has_canvas": bool(chat_request.canvasData or chat_request.canvasPages),
            "pages_len": len(chat_request.canvasPages) if chat_request.canvasPages else (1 if chat_request.canvasData else 0),
            "history_length": len(chat_request.conversationHistory)
        }
        cache_key = cache.hash_query(cache_key_data)

        # Check cache first
        cached_response = await cache.get_cached_chat_response(
            chat_request.sessionId, cache_key
        )

        if cached_response:
            response_time = time.time() - start_time

            # Log analytics in background
            background_tasks.add_task(
                log_chat_analytics,
                chat_request.sessionId,
                chat_request.userId,
                chat_request.mode,
                response_time,
                True,
                db
            )

            # Track chat session
            background_tasks.add_task(
                track_chat_session,
                chat_request.userId,
                chat_request.sessionId,
                chat_request.message,
                cached_response,
                chat_request.mode,
                db
            )

            return ChatResponse(
                success=True,
                data={
                    "response": cached_response,
                    "sessionId": chat_request.sessionId,
                    "mode": chat_request.mode,
                    "cached": True,
                    "response_time_ms": response_time * 1000
                },
                cached=True
            )

        # Initialize OpenAI service
        openai_service = AsyncOpenAIService()

        # Prepare messages for AI
        messages = await _prepare_messages(chat_request, openai_service)

        # Get AI response
        ai_response = await openai_service.chat_completion_async(
            messages=messages,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )

        if not ai_response['success']:
            logger.error(f"OpenAI API error: {ai_response['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate AI response"
            )

        response_text = ai_response['response']
        response_time = time.time() - start_time

        # Cache the response asynchronously
        asyncio.create_task(cache.cache_chat_response(
            chat_request.sessionId, cache_key, response_text, 1800  # 30 minutes
        ))

        # Log analytics in background
        background_tasks.add_task(
            log_chat_analytics,
            chat_request.sessionId,
            chat_request.userId,
            chat_request.mode,
            response_time,
            False,
            db
        )

        # Track chat session
        background_tasks.add_task(
            track_chat_session,
            chat_request.userId,
            chat_request.sessionId,
            chat_request.message,
            response_text,
            chat_request.mode,
            db
        )

        return ChatResponse(
            success=True,
            data={
                "response": response_text,
                "usage": ai_response.get('usage'),
                "model": ai_response.get('model'),
                "sessionId": chat_request.sessionId,
                "mode": chat_request.mode,
                "cached": False,
                "response_time_ms": response_time * 1000
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred"
        )

async def _prepare_messages(chat_request: ChatRequest, openai_service: AsyncOpenAIService) -> List[Dict[str, Any]]:
    """Prepare messages for OpenAI API"""
    messages = []

    # Get system prompt
    system_prompts = await openai_service.get_system_prompts_async()
    system_prompt = system_prompts.get(chat_request.mode, system_prompts['general'])

    # Add conversation history (limit to last 10 messages)
    if chat_request.conversationHistory:
        recent_history = chat_request.conversationHistory[-10:]
        for msg in recent_history:
            if msg.role in ['user', 'assistant'] and msg.content:
                messages.append({
                    'role': msg.role,
                    'content': msg.content
                })

    # Add current message (support multi-page canvas)
    images: List[str] = []
    if chat_request.canvasPages:
        images = list(chat_request.canvasPages)
    elif chat_request.canvasData:
        images = [chat_request.canvasData]

    if images:
        enhanced_message = await _create_enhanced_message_with_canvas(
            chat_request.message, images[0]
        )

        messages.append({
            'role': 'user',
            'content': [
                {'type': 'text', 'text': enhanced_message},
                *[{'type': 'image_url', 'image_url': {'url': u}} for u in images]
            ]
        })
    else:
        messages.append({
            'role': 'user',
            'content': chat_request.message
        })

    return messages

async def _create_enhanced_message_with_canvas(message: str, canvas_data: str) -> str:
    """Create enhanced message for canvas analysis"""
    return f"""{message}

🔍 **COMPREHENSIVE ACADEMIC ANALYSIS SYSTEM**

Please analyze the provided image using a systematic multi-pass approach:

## **PASS 1: CONTENT IDENTIFICATION**
- **Subject Area**: Identify the academic field (Math, Physics, Chemistry, Biology)
- **Content Type**: Determine if it's equations, diagrams, problem-solving, or mixed content
- **Complexity Level**: Assess the difficulty level and scope of the material

## **PASS 2: DETAILED TRANSCRIPTION**
- **Text Elements**: Extract all text, equations, labels, and annotations
- **Mathematical Notation**: Identify formulas, variables, operators, and mathematical structures
- **Diagrams**: Describe any charts, graphs, molecular structures, or technical drawings
- **Relationships**: Note connections between elements (arrows, lines, groupings)

## **PASS 3: ACADEMIC INTERPRETATION**
- **Process Analysis**: Explain the complete academic process or solution
- **Step-by-Step Breakdown**: Detail each stage of the work shown
- **Educational Context**: Provide learning insights and explanations
- **Error Detection**: Identify any mistakes or areas for improvement

**FORMATTING**: Use LaTeX for math (\\[ \\] for display, \\( \\) for inline), proper units, and clear structure.
**GOAL**: Provide complete, accurate educational analysis that helps students understand the material."""

@router.get("/health", response_model=ChatHealthResponse)
@limiter.limit("120/minute")
async def chat_health(request: Request):
    """Health check for chat service"""
    try:
        # Test OpenAI service initialization
        openai_service = AsyncOpenAIService()
        model_info = await openai_service.get_model_info_async()

        return ChatHealthResponse(
            success=True,
            message="Chat service is healthy",
            model=model_info.get("model", settings.OPENAI_MODEL),
            service="openai"
        )

    except Exception as e:
        logger.error(f"Chat health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service unhealthy"
        )

@router.get("/models")
@limiter.limit("30/minute")
async def get_available_models(request: Request):
    """Get information about available AI models"""
    try:
        openai_service = AsyncOpenAIService()
        system_prompts = await openai_service.get_system_prompts_async()

        return {
            "success": True,
            "data": {
                "current_model": settings.OPENAI_MODEL,
                "provider": "openai",
                "system_prompts": list(system_prompts.keys()),
                "features": [
                    "Text analysis",
                    "Image analysis",
                    "Mathematical problem solving",
                    "Scientific content interpretation"
                ]
            }
        }

    except Exception as e:
        logger.error(f"Models endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )

@router.get("/analytics")
@limiter.limit("10/minute")
async def get_chat_analytics(
    request: Request,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    hours: int = 24,
    db: DatabaseManager = Depends(get_database)
):
    """Get chat analytics data"""
    try:
        # Build filter
        filter_dict = {
            "timestamp": {"$gte": time.time() - (hours * 3600)}
        }

        if session_id:
            filter_dict["session_id"] = session_id
        if user_id:
            filter_dict["user_id"] = user_id

        # Get analytics data
        analytics = await db.mongo_find(
            "chat_analytics",
            filter_dict,
            sort=[("timestamp", -1)],
            limit=1000
        )

        # Calculate statistics
        total_requests = len(analytics)
        cached_requests = sum(1 for a in analytics if a.get("cached", False))
        avg_response_time = sum(a.get("response_time_ms", 0) for a in analytics) / max(total_requests, 1)

        mode_stats = {}
        for a in analytics:
            mode = a.get("mode", "unknown")
            if mode not in mode_stats:
                mode_stats[mode] = {"count": 0, "avg_response_time": 0}
            mode_stats[mode]["count"] += 1

        return {
            "success": True,
            "data": {
                "total_requests": total_requests,
                "cached_requests": cached_requests,
                "cache_hit_rate": cached_requests / max(total_requests, 1),
                "average_response_time_ms": avg_response_time,
                "mode_statistics": mode_stats,
                "time_range_hours": hours
            }
        }

    except Exception as e:
        logger.error(f"Analytics endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics data"
        )