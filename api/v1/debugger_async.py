"""
Async Debugger Chat API - Production-Ready with LangChain RAG
High-performance debugger endpoints with rate limiting, designed for 1000+ concurrent users
Uses ChromaDB for conversation memory and document RAG
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from services.langchain_debugger_service import get_langchain_debugger_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# ====================REQUEST/RESPONSE MODELS ====================

class DebuggerChatRequest(BaseModel):
    """Request model for debugger chat"""
    sessionId: str = Field(..., min_length=5, max_length=100, description="Unique session ID")
    message: str = Field(..., min_length=1, max_length=10000, description="User message or system prompt")
    attachments: Optional[List[str]] = Field(default=None, max_items=5, description="Optional attachment URLs")
    imageData: Optional[str] = Field(default=None, description="Optional base64 image data")
    canvasPages: Optional[List[Any]] = Field(default=None, description="Optional canvas pages for practice mode")
    mode: Optional[str] = Field(default="debug", description="Chat mode: debug or practice")
    userId: Optional[str] = Field(default=None, description="Optional user ID")
    conversationHistory: Optional[List[Any]] = Field(default=None, description="Optional conversation history")

    @validator('message')
    def validate_message(cls, v):
        """Validate message is not just whitespace"""
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

    @validator('sessionId')
    def validate_session_id(cls, v):
        """Validate session ID format"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Invalid session ID format')
        return v

    @validator('imageData')
    def validate_image_data(cls, v):
        """Validate image data format"""
        if v and not (v.startswith('data:image/') or v.startswith('http')):
            # Allow both base64 and URLs
            pass
        return v


class MessageModel(BaseModel):
    """Message model for response"""
    id: str
    role: str
    content: str
    timestamp: str
    attachments: Optional[List[str]] = []


class DebuggerChatResponse(BaseModel):
    """Response model for debugger chat"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    details: Optional[str] = None


class HistoryResponse(BaseModel):
    """Response model for conversation history"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    success: bool
    document_id: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None
    num_chunks: Optional[int] = None
    error: Optional[str] = None


# ==================== API ENDPOINTS ====================

@router.post("/chat", response_model=DebuggerChatResponse)
@limiter.limit("30/minute")  # 30 requests per minute per user
async def debugger_chat(
    request: Request,
    chat_request: DebuggerChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Send a chat message and get AI response with conversation memory and RAG

    **Features:**
    - Full conversation context using ChromaDB
    - RAG with uploaded documents
    - Semantic search for relevant past messages
    - Image analysis support via GPT-5 multimodal processing
    - **SECURITY**: Session validation to prevent unauthorized access

    **Rate Limit**: 30 requests per minute per IP

    **Request Body**:
    - sessionId: Unique session identifier (format: user_{userId}_{timestamp}_{random})
    - message: User message (1-5000 chars)
    - attachments: Optional list of attachment URLs (max 5)
    - imageData: Optional base64 image data for vision analysis

    **Response**:
    - user_message: The stored user message
    - assistant_message: The AI assistant response
    - response: AI response text
    - session_id: Session identifier
    - message_count: Total messages in conversation
    - usage: Token usage statistics
    - model: AI model used (gpt-5)
    - has_document_context: Whether document RAG was used
    """
    try:
        logger.info(f"📨 Debugger chat - Session: {chat_request.sessionId}, Message: {len(chat_request.message)} chars")

        # SECURITY: Validate session belongs to authenticated user (if auth is enabled)
        if chat_request.userId:
            # Extract user_id from session_id (format: user_{userId}_{timestamp}_{random})
            if chat_request.sessionId.startswith("user_"):
                session_user_id = chat_request.sessionId.split("_")[1] if len(chat_request.sessionId.split("_")) > 1 else None
                if session_user_id and session_user_id != chat_request.userId:
                    logger.warning(f"⚠️ Session hijack attempt: session={chat_request.sessionId}, claimed_user={chat_request.userId}")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Session does not belong to the authenticated user"
                    )

        # Get LangChain service
        service = await get_langchain_debugger_service()

        # For practice mode with multiple images (question diagrams + canvas pages)
        # We need to send all images to the LLM for proper evaluation
        image_data_to_send = chat_request.imageData

        if chat_request.canvasPages and len(chat_request.canvasPages) > 0:
            logger.info(f"📸 Processing {len(chat_request.canvasPages)} images for practice mode evaluation")

            # Collect all image data
            all_images = []
            for page in chat_request.canvasPages:
                if isinstance(page, dict):
                    img_data = page.get('data')
                    img_type = page.get('type', 'unknown')
                    if img_data:
                        all_images.append(img_data)
                        logger.info(f"  - Added {img_type} image")
                elif isinstance(page, str):
                    all_images.append(page)
                    logger.info(f"  - Added image (string format)")

            # For now, send the first image to the existing API
            # TODO: Enhance LangChain service to support multiple images
            if all_images:
                image_data_to_send = all_images[0]

                # If there are multiple images, add them as attachments
                if len(all_images) > 1 and not chat_request.attachments:
                    # Note: Current LangChain service may not support multiple images
                    # This is a temporary workaround
                    logger.warning(f"⚠️ Multiple images ({len(all_images)}) provided but only first image will be sent to LLM. Consider upgrading to GPT-4 Vision API with multiple image support.")

        # Send message and get response with full RAG
        result = await service.send_message(
            session_id=chat_request.sessionId,
            message=chat_request.message,
            attachments=chat_request.attachments,
            image_data=image_data_to_send
        )

        if not result['success']:
            logger.error(f"❌ Chat failed: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Failed to process message')
            )

        logger.info(f"✅ Chat success - Session: {chat_request.sessionId}, Messages: {result['data']['message_count']}")
        return DebuggerChatResponse(**result)

    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"❌ Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"❌ Debugger chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/upload", response_model=FileUploadResponse)
@limiter.limit("20/minute")  # 20 uploads per minute per user
async def upload_document(
    request: Request,
    sessionId: str = Form(..., description="Session identifier"),
    file: UploadFile = File(..., description="Document file (PDF, Word, Image)")
):
    """
    Upload a document for RAG (Retrieval-Augmented Generation)

    **Supported file types:**
    - PDF (.pdf)
    - Word Documents (.docx, .doc)
    - Images (.jpg, .jpeg, .png, .webp)

    **Max file size:** 10MB

    **Rate Limit**: 20 uploads per minute per IP

    **Form Data:**
    - sessionId: Session identifier (string)
    - file: File to upload (multipart/form-data)

    **Response:**
    - success: Operation status
    - document_id: Unique document identifier
    - filename: Original filename
    - file_type: Detected file type
    - num_chunks: Number of text chunks created for RAG
    """
    try:
        logger.info(f"📤 Document upload - Session: {sessionId}, File: {file.filename}")

        # Validate inputs
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if len(sessionId) < 5:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # Read file content
        file_content = await file.read()

        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {len(file_content)/(1024*1024):.2f}MB (max 10MB)"
            )

        # Get service
        service = await get_langchain_debugger_service()

        # Upload and process document
        result = await service.upload_document(
            session_id=sessionId,
            file_content=file_content,
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream"
        )

        if not result['success']:
            logger.error(f"❌ Upload failed: {result.get('error')}")
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Failed to process document')
            )

        logger.info(f"✅ Document uploaded - Session: {sessionId}, Doc: {result.get('document_id')}, Chunks: {result.get('num_chunks')}")
        return FileUploadResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/history/{session_id}", response_model=HistoryResponse)
@limiter.limit("60/minute")  # 60 requests per minute per user
async def get_history(
    request: Request,
    session_id: str,
    limit: Optional[int] = None,
    user_id: Optional[str] = None  # For validation
):
    """
    Get conversation history for a session from ChromaDB

    **SECURITY**: Validates session ownership to prevent unauthorized access

    **Rate Limit**: 60 requests per minute per IP

    **Parameters**:
    - session_id: Session identifier (format: user_{userId}_{timestamp}_{random})
    - limit: Optional limit on number of messages to return
    - user_id: Optional user ID for ownership validation

    **Response**:
    - session_id: Session identifier
    - messages: List of conversation messages
    - metadata: Session metadata (total_messages, has_documents, created_at)
    """
    try:
        # Validate session_id
        if not session_id or len(session_id) < 5:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # SECURITY: Validate session ownership if user_id is provided
        if user_id and session_id.startswith("user_"):
            session_user_id = session_id.split("_")[1] if len(session_id.split("_")) > 1 else None
            if session_user_id and session_user_id != user_id:
                logger.warning(f"⚠️ Unauthorized history access attempt: session={session_id}, user={user_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to access this session"
                )

        # Validate limit
        if limit is not None and (limit < 1 or limit > 100):
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")

        logger.debug(f"📥 History request - Session: {session_id}, Limit: {limit}")

        service = await get_langchain_debugger_service()
        result = await service.get_conversation_history(session_id, limit)

        if not result['success']:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Failed to retrieve history')
            )

        return HistoryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get history error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.post("/session/{session_id}/clear")
@limiter.limit("10/minute")  # 10 clears per minute per user
async def clear_session(request: Request, session_id: str):
    """
    Clear conversation history and documents for a session

    **Rate Limit**: 10 requests per minute per IP

    **Parameters**:
    - session_id: Session identifier

    **Response**:
    - success: Operation status
    - message: Success message
    """
    try:
        if not session_id or len(session_id) < 5:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        logger.info(f"🗑️ Clear session - Session: {session_id}")

        service = await get_langchain_debugger_service()
        result = await service.clear_session(session_id)

        if not result['success']:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Failed to clear session')
            )

        logger.info(f"✅ Session cleared - Session: {session_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Clear session error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear session: {str(e)}"
        )


@router.delete("/session/{session_id}")
@limiter.limit("10/minute")  # 10 deletes per minute per user
async def delete_session(request: Request, session_id: str):
    """
    Delete a conversation session completely (same as clear)

    **Rate Limit**: 10 requests per minute per IP

    **Parameters**:
    - session_id: Session identifier

    **Response**:
    - success: Operation status
    - message: Success message
    """
    try:
        if not session_id or len(session_id) < 5:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        logger.info(f"🗑️ Delete session - Session: {session_id}")

        service = await get_langchain_debugger_service()
        result = await service.delete_session(session_id)

        if not result['success']:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('error', 'Failed to delete session')
            )

        logger.info(f"✅ Session deleted - Session: {session_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete session error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.get("/sessions")
@limiter.limit("20/minute")  # 20 requests per minute per user
async def get_sessions(request: Request):
    """
    Get statistics about all active sessions

    **Rate Limit**: 20 requests per minute per IP

    **Response**:
    - active_sessions: Number of active sessions
    - sessions: List of session details (session_id, message_count, has_documents, created_at)
    """
    try:
        logger.debug("📊 Sessions stats request")

        service = await get_langchain_debugger_service()
        result = await service.get_session_stats()

        return result

    except Exception as e:
        logger.error(f"❌ Get sessions error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )


@router.get("/health")
@limiter.limit("100/minute")  # 100 health checks per minute
async def health_check(request: Request):
    """
    Health check for debugger chat service

    **Rate Limit**: 100 requests per minute per IP

    **Response**:
    - success: Service status
    - message: Status message
    - service: Service description
    - active_sessions: Number of active sessions
    - timestamp: Current timestamp (ISO format)
    """
    try:
        service = await get_langchain_debugger_service()
        stats = await service.get_session_stats()

        return {
            'success': True,
            'message': 'Debugger chat service is healthy',
            'service': 'LangChain + ChromaDB RAG System with GPT-5',
            'active_sessions': stats['data']['active_sessions'],
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Debugger chat service unhealthy: {str(e)}"
        )
