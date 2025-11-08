"""
Async Debugger Chat Service - Production-Ready
Handles debugger mode chat with conversation memory, designed for 1000+ concurrent users
Uses Redis for shared session storage across multiple workers
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import pickle

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Thread-safe conversation memory for chat sessions"""

    def __init__(self, session_id: str, max_history: int = 20):
        """
        Initialize conversation memory

        Args:
            session_id: Unique identifier for the chat session
            max_history: Maximum number of messages to retain
        """
        self.session_id = session_id
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'message_count': 0
        }
        self._lock = asyncio.Lock()
        logger.info(f"Initialized conversation memory for session {session_id} with max_history={max_history}")

    async def add_message(
        self,
        role: str,
        content: str,
        attachments: Optional[List[str]] = None,
        image_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Thread-safely add a message to conversation history with sliding window

        Args:
            role: 'user' or 'assistant'
            content: Message content
            attachments: Optional list of attachment URLs
            image_data: Optional base64 image data for vision messages

        Returns:
            The created message object
        """
        async with self._lock:
            message = {
                'id': f"{self.session_id}_{len(self.messages)}_{int(datetime.now().timestamp() * 1000)}",
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'attachments': attachments or [],
                'image_data': image_data  # Store image data for conversation memory
            }

            self.messages.append(message)
            self.metadata['message_count'] = len(self.messages)
            self.metadata['last_updated'] = datetime.now().isoformat()

            # Sliding window: Keep only recent messages if exceeded max
            if len(self.messages) > self.max_history:
                removed = len(self.messages) - self.max_history
                self.messages = self.messages[-self.max_history:]
                logger.debug(f"Sliding window applied to session {self.session_id}: Removed {removed} old messages, keeping {len(self.messages)}")

            return message

    async def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history

        Args:
            limit: Optional limit on number of messages to return

        Returns:
            List of messages
        """
        async with self._lock:
            if limit:
                return self.messages[-limit:]
            return self.messages.copy()

    async def get_openai_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get messages formatted for OpenAI API with image data support

        Args:
            limit: Number of recent messages to include

        Returns:
            List of messages in OpenAI format with vision support
        """
        async with self._lock:
            recent_messages = self.messages[-limit:] if limit else self.messages

            openai_messages = []
            for msg in recent_messages:
                # Check if message has image data
                if msg.get('image_data'):
                    # Vision message format
                    openai_messages.append({
                        'role': msg['role'],
                        'content': [
                            {'type': 'text', 'text': msg['content']},
                            {'type': 'image_url', 'image_url': {'url': msg['image_data']}}
                        ]
                    })
                else:
                    # Regular text message
                    openai_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

            return openai_messages

    async def clear(self):
        """Clear conversation history"""
        async with self._lock:
            self.messages = []
            self.metadata['message_count'] = 0
            self.metadata['cleared_at'] = datetime.now().isoformat()

    async def to_dict(self) -> Dict[str, Any]:
        """Export conversation to dictionary"""
        async with self._lock:
            return {
                'session_id': self.session_id,
                'messages': self.messages.copy(),
                'metadata': self.metadata.copy()
            }


class AsyncDebuggerChatService:
    """
    Production-ready async service for debugger mode chat
    Supports 1000+ concurrent users with Redis-backed session storage
    Works across multiple uvicorn workers
    """

    # Redis key prefixes
    SESSION_PREFIX = "debugger:session:"
    SESSION_TTL = 86400  # 24 hours

    # System prompt optimized for student learning
    SYSTEM_PROMPT = """You are an expert AI tutor and problem-solving assistant for students. Your role is to:

1. **Problem Solving**: Help students solve academic problems step-by-step without giving away complete answers
2. **Concept Explanation**: Explain complex concepts in simple, understandable terms
3. **Guided Learning**: Ask probing questions to help students think critically
4. **Encouragement**: Provide positive reinforcement and encouragement
5. **Adaptive Teaching**: Adjust your explanations based on the student's level of understanding

**Communication Style:**
- Be friendly, patient, and encouraging
- Use clear, simple language appropriate for students
- Break down complex problems into smaller, manageable steps
- Use examples and analogies when helpful
- Ask questions to check understanding
- Celebrate student progress and insights

**When analyzing images:**
- Carefully examine handwritten work, diagrams, or problems
- Identify the subject area (math, science, etc.)
- Transcribe any text or equations accurately
- Provide detailed explanations of visual content

**Response Format:**
- Use markdown formatting for clarity
- Use LaTeX for ALL mathematical expressions and equations
- For inline math, use SINGLE dollar signs: $E = mc^2$
- For display/block math, use DOUBLE dollar signs on separate lines:

$$
F = ma
$$

- Structure responses with clear headings when appropriate
- Keep responses focused and concise but complete

**CRITICAL - LaTeX Syntax Rules:**
- ALWAYS use $ for inline math: The equation $F = ma$ shows...
- ALWAYS use $$ for display math: $$\\int_0^\\infty e^{-x^2} dx$$
- Use proper LaTeX commands: \\frac{a}{b}, \\sqrt{x}, \\vec{F}, \\theta, \\int, \\sum
- DO NOT use plain text for math - convert ALL equations to LaTeX

Remember: Your goal is to help students learn and understand, not just to provide answers."""

    def __init__(self, openai_service, cache_manager=None):
        """
        Initialize async debugger chat service

        Args:
            openai_service: Instance of AsyncOpenAIService
            cache_manager: Optional CacheManager for Redis storage
        """
        self.openai_service = openai_service
        self.cache_manager = cache_manager
        self.sessions: Dict[str, ConversationMemory] = {}  # In-memory fallback
        self._sessions_lock = asyncio.Lock()

        if cache_manager and cache_manager.redis_client:
            logger.info("Async Debugger Chat Service initialized with Redis storage for multi-worker support")
        else:
            logger.warning("Async Debugger Chat Service initialized with in-memory storage (single worker only)")
            logger.warning("For multi-worker deployments, Redis is required for session persistence")

    async def _load_session_from_redis(self, session_id: str) -> Optional[ConversationMemory]:
        """Load session from Redis - critical for multi-worker deployments"""
        if not self.cache_manager or not self.cache_manager.redis_client:
            return None

        try:
            redis_key = f"{self.SESSION_PREFIX}{session_id}"
            data = await self.cache_manager.redis_client.get(redis_key)

            if data:
                session_data = pickle.loads(data)
                session = ConversationMemory(session_id)
                session.messages = session_data.get('messages', [])
                session.metadata = session_data.get('metadata', session.metadata)
                logger.info(f"Loaded session {session_id} from Redis with {len(session.messages)} messages")
                return session

            logger.debug(f"No session found in Redis for {session_id}")
            return None
        except Exception as e:
            logger.error(f"Error loading session {session_id} from Redis: {str(e)}")
            return None

    async def _save_session_to_redis(self, session: ConversationMemory):
        """Save session to Redis - critical for multi-worker deployments"""
        if not self.cache_manager or not self.cache_manager.redis_client:
            return

        try:
            redis_key = f"{self.SESSION_PREFIX}{session.session_id}"
            session_data = {
                'messages': session.messages,
                'metadata': session.metadata
            }

            serialized = pickle.dumps(session_data)
            await self.cache_manager.redis_client.setex(
                redis_key,
                self.SESSION_TTL,
                serialized
            )
            logger.debug(f"Saved session {session.session_id} to Redis with {len(session.messages)} messages")
        except Exception as e:
            logger.error(f"Error saving session {session.session_id} to Redis: {str(e)}")

    async def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """
        Thread-safely get existing session or create new one
        ALWAYS checks Redis first for multi-worker consistency

        Args:
            session_id: Unique session identifier

        Returns:
            ConversationMemory instance
        """
        async with self._sessions_lock:
            # CRITICAL: For multi-worker deployments, ALWAYS check Redis first
            # In-memory cache is only a performance optimization for same-worker requests
            if self.cache_manager and self.cache_manager.redis_client:
                session = await self._load_session_from_redis(session_id)
                if session:
                    # Update in-memory cache for this worker
                    self.sessions[session_id] = session
                    return session

                # No session in Redis, create new one
                session = ConversationMemory(session_id)
                self.sessions[session_id] = session
                logger.info(f"Created new conversation session: {session_id}")
                # Immediately save to Redis so other workers can access it
                await self._save_session_to_redis(session)
                return session
            else:
                # Fallback to in-memory for single-worker deployments
                if session_id in self.sessions:
                    return self.sessions[session_id]

                session = ConversationMemory(session_id)
                self.sessions[session_id] = session
                logger.info(f"Created new conversation session (in-memory): {session_id}")
                return session

    async def send_message(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[str]] = None,
        image_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message and get AI response with conversation context

        Args:
            session_id: Session identifier
            message: User message
            attachments: Optional list of attachment URLs
            image_data: Optional base64 image data

        Returns:
            Dict containing response and message metadata
        """
        try:
            # Get or create session
            session = await self.get_or_create_session(session_id)

            # Add user message to history with image data if present
            user_msg = await session.add_message('user', message, attachments, image_data)
            logger.info(f"Session {session_id}: Added user message. Total messages in session: {len(session.messages)}")

            # Prepare OpenAI messages with conversation history (sliding window)
            # The get_openai_messages function now automatically includes image data from history
            openai_messages = await session.get_openai_messages(limit=None)  # Use all messages in window
            logger.info(f"Session {session_id}: Sending {len(openai_messages)} messages to AI with full conversation context")

            # Get AI response asynchronously
            response = await self.openai_service.chat_completion_async(
                messages=openai_messages,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1500
            )

            if not response.get('success'):
                logger.error(f"OpenAI API error: {response.get('error')}")
                return {
                    'success': False,
                    'error': 'Failed to generate response',
                    'details': response.get('error')
                }

            # Add assistant response to history
            assistant_content = response['response']
            assistant_msg = await session.add_message('assistant', assistant_content)

            logger.info(f"Session {session_id}: Generated AI response. Total messages in session: {len(session.messages)}")

            # CRITICAL: Save to Redis immediately for multi-worker support
            await self._save_session_to_redis(session)

            return {
                'success': True,
                'data': {
                    'user_message': user_msg,
                    'assistant_message': assistant_msg,
                    'response': assistant_content,
                    'session_id': session_id,
                    'message_count': session.metadata['message_count'],
                    'usage': response.get('usage', {}),
                    'model': response.get('model', 'unknown')
                }
            }

        except Exception as e:
            logger.error(f"Error in send_message: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': 'Internal server error',
                'details': str(e)
            }

    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier
            limit: Optional limit on messages

        Returns:
            Dict containing conversation history
        """
        try:
            session = await self.get_or_create_session(session_id)
            messages = await session.get_messages(limit)

            return {
                'success': True,
                'data': {
                    'session_id': session_id,
                    'messages': messages,
                    'metadata': session.metadata
                }
            }

        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to retrieve history',
                'details': str(e)
            }

    async def clear_session(self, session_id: str) -> Dict[str, Any]:
        """
        Clear conversation history for a session

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        try:
            async with self._sessions_lock:
                if session_id in self.sessions:
                    await self.sessions[session_id].clear()
                    # Update Redis
                    await self._save_session_to_redis(self.sessions[session_id])
                    logger.info(f"Cleared session: {session_id}")
                elif self.cache_manager and self.cache_manager.redis_client:
                    # Clear from Redis
                    redis_key = f"{self.SESSION_PREFIX}{session_id}"
                    await self.cache_manager.redis_client.delete(redis_key)
                    logger.info(f"Cleared session from Redis: {session_id}")

            return {
                'success': True,
                'message': 'Session cleared successfully'
            }

        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to clear session',
                'details': str(e)
            }

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Delete a conversation session

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        try:
            async with self._sessions_lock:
                if session_id in self.sessions:
                    del self.sessions[session_id]
                    logger.info(f"Deleted session from memory: {session_id}")

                # Delete from Redis
                if self.cache_manager and self.cache_manager.redis_client:
                    redis_key = f"{self.SESSION_PREFIX}{session_id}"
                    await self.cache_manager.redis_client.delete(redis_key)
                    logger.info(f"Deleted session from Redis: {session_id}")

            return {
                'success': True,
                'message': 'Session deleted successfully'
            }

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to delete session',
                'details': str(e)
            }

    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active sessions

        Returns:
            Dict containing session statistics
        """
        async with self._sessions_lock:
            return {
                'success': True,
                'data': {
                    'active_sessions': len(self.sessions),
                    'sessions': [
                        {
                            'session_id': session_id,
                            'message_count': session.metadata['message_count'],
                            'created_at': session.metadata['created_at'],
                            'last_updated': session.metadata.get('last_updated')
                        }
                        for session_id, session in self.sessions.items()
                    ]
                }
            }

    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Cleanup sessions older than max_age_hours
        This can be called periodically to prevent memory bloat

        Args:
            max_age_hours: Maximum age of sessions to keep
        """
        try:
            async with self._sessions_lock:
                now = datetime.now()
                sessions_to_delete = []

                for session_id, session in self.sessions.items():
                    created_at = datetime.fromisoformat(session.metadata['created_at'])
                    age_hours = (now - created_at).total_hours()

                    if age_hours > max_age_hours:
                        sessions_to_delete.append(session_id)

                for session_id in sessions_to_delete:
                    del self.sessions[session_id]
                    logger.info(f"Cleaned up old session: {session_id}")

                logger.info(f"Cleaned up {len(sessions_to_delete)} old sessions")

        except Exception as e:
            logger.error(f"Error in cleanup_old_sessions: {str(e)}")


# Singleton instance per worker
_debugger_chat_service = None
_service_lock = asyncio.Lock()


async def get_debugger_chat_service(cache_manager=None):
    """
    Get singleton instance of async debugger chat service

    CRITICAL: For multi-worker deployments, cache_manager MUST be provided
    to enable Redis-based session sharing across workers.
    """
    global _debugger_chat_service

    async with _service_lock:
        if _debugger_chat_service is None:
            from services.async_openai_service import AsyncOpenAIService
            openai_service = AsyncOpenAIService()
            _debugger_chat_service = AsyncDebuggerChatService(openai_service, cache_manager)
            logger.info(f"Initialized debugger chat service singleton for this worker (cache_enabled={cache_manager is not None and cache_manager.redis_client is not None})")
        elif cache_manager is not None and _debugger_chat_service.cache_manager is None:
            # Update singleton with cache_manager if it wasn't available during initialization
            _debugger_chat_service.cache_manager = cache_manager
            logger.info("Updated debugger chat service with cache manager")

        return _debugger_chat_service
