"""
Debugger Chat Service with Conversation Memory Management
Handles chat sessions, memory persistence, and OpenAI integration
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history for chat sessions"""

    def __init__(self, session_id: str, max_history: int = 30):
        """
        Initialize conversation memory with sliding window

        Args:
            session_id: Unique identifier for the chat session
            max_history: Maximum number of message PAIRS (user + assistant) to retain in sliding window
        """
        self.session_id = session_id
        self.max_history = max_history * 2  # Store user+assistant pairs, so double the count (60 messages total)
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'message_count': 0,
            'total_message_count': 0  # Track all messages ever sent
        }
        logger.info(f"📝 Created conversation memory for session {session_id} with sliding window of {max_history} message pairs")

    def add_message(self, role: str, content: str, attachments: Optional[List[str]] = None, image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a message to conversation history with sliding window

        Args:
            role: 'user' or 'assistant'
            content: Message content
            attachments: Optional list of attachment URLs
            image_data: Optional base64 image data for vision messages

        Returns:
            The created message object
        """
        message = {
            'id': f"{self.session_id}_{len(self.messages)}_{int(datetime.now().timestamp())}",
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'attachments': attachments or [],
            'image_data': image_data  # Store image data for conversation memory
        }

        self.messages.append(message)
        self.metadata['message_count'] = len(self.messages)
        self.metadata['total_message_count'] = self.metadata.get('total_message_count', 0) + 1
        self.metadata['last_updated'] = datetime.now().isoformat()

        # Sliding window: Keep only recent messages if exceeded max
        if len(self.messages) > self.max_history:
            removed_count = len(self.messages) - self.max_history
            self.messages = self.messages[-self.max_history:]
            logger.debug(f"Sliding window: Removed {removed_count} old messages. Keeping last {self.max_history} messages for session {self.session_id}")

        return message

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history (without image_data for frontend display)

        Args:
            limit: Optional limit on number of messages to return

        Returns:
            List of messages without image_data (for display purposes)
        """
        messages = self.messages[-limit:] if limit else self.messages
        
        # Remove image_data from messages for frontend (keep it in internal storage)
        return [
            {k: v for k, v in msg.items() if k != 'image_data'}
            for msg in messages
        ]

    def get_openai_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages formatted for OpenAI API with image data support

        Args:
            limit: Number of recent messages to include (None = all messages in window)

        Returns:
            List of messages in OpenAI format with vision support
        """
        recent_messages = self.messages[-limit:] if limit else self.messages

        openai_messages = []
        for msg in recent_messages:
            # Check if message has image data
            if msg.get('image_data'):
                # Vision message format - preserve image context
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

    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.metadata['message_count'] = 0
        self.metadata['cleared_at'] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Export conversation to dictionary"""
        return {
            'session_id': self.session_id,
            'messages': self.messages,
            'metadata': self.metadata
        }


class DebuggerChatService:
    """
    Service for handling debugger mode chat with conversation memory
    """

    # System prompt for debugger mode
    SYSTEM_PROMPT = """You are an expert AI tutor and problem-solving assistant for students. Your role is to:

1. **Problem Solving**: Help students solve academic problems step-by-step without giving away complete answers
2. **Concept Explanation**: Explain complex concepts in simple, understandable terms
3. **Guided Learning**: Ask probing questions to help students think critically
4. **Encouragement**: Provide positive reinforcement and encouragement
5. **Adaptive Teaching**: Adjust your explanations based on the student's level of understanding

**IMPORTANT - Conversation Memory:**
- You have access to the FULL conversation history including all previously uploaded images and documents
- When a student asks about "the image" or "this question", refer back to images they've shared earlier in the conversation
- Reference previous discussions and maintain context throughout the conversation
- If a student asks to explain something from an earlier image, you can see it in the conversation history
- Build on previous explanations and remember what the student has already learned

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
- Remember images from earlier in the conversation when referenced

**Response Format:**
- Use markdown formatting for clarity
- Use LaTeX notation for mathematical expressions: \\( \\) for inline, \\[ \\] for display
- Structure responses with clear headings when appropriate
- Keep responses focused and concise but complete

Remember: Your goal is to help students learn and understand, not just to provide answers. You have full conversation context including all images and documents shared."""

    def __init__(self, openai_service):
        """
        Initialize debugger chat service

        Args:
            openai_service: Instance of OpenAI service
        """
        self.openai_service = openai_service
        self.sessions: Dict[str, ConversationMemory] = {}
        logger.info("Debugger Chat Service initialized")

    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """
        Get existing session or create new one

        Args:
            session_id: Unique session identifier

        Returns:
            ConversationMemory instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(session_id)
            logger.info(f"Created new conversation session: {session_id}")

        return self.sessions[session_id]

    def send_message(
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
            session = self.get_or_create_session(session_id)

            # Add user message to history with image data if present
            user_msg = session.add_message('user', message, attachments, image_data)
            logger.info(f"Session {session_id}: Added user message. Total messages in session: {len(session.messages)}")
            
            # Log if this message has an image
            if image_data:
                logger.info(f"Session {session_id}: Message includes image data (size: {len(image_data)} chars)")

            # Prepare OpenAI messages with conversation history (use all available messages in sliding window)
            # The get_openai_messages function now automatically includes image data from history
            openai_messages = session.get_openai_messages(limit=None)  # Use all messages in the sliding window
            
            # Count messages with images in the context
            image_count = sum(1 for msg in openai_messages if isinstance(msg.get('content'), list))
            logger.info(f"Session {session_id}: Sending {len(openai_messages)} messages to AI ({image_count} with images) - full conversation context preserved")

            # Get AI response
            response = self.openai_service.chat_completion(
                messages=openai_messages,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1500
            )

            if not response['success']:
                logger.error(f"OpenAI API error: {response.get('error')}")
                return {
                    'success': False,
                    'error': 'Failed to generate response',
                    'details': response.get('error')
                }

            # Add assistant response to history
            assistant_content = response['response']
            assistant_msg = session.add_message('assistant', assistant_content)

            logger.info(f"Chat response generated for session {session_id}")

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

    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier
            limit: Optional limit on messages

        Returns:
            Dict containing conversation history
        """
        try:
            session = self.get_or_create_session(session_id)
            messages = session.get_messages(limit)

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

    def clear_session(self, session_id: str) -> Dict[str, Any]:
        """
        Clear conversation history for a session

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        try:
            if session_id in self.sessions:
                self.sessions[session_id].clear()
                logger.info(f"Cleared session: {session_id}")

            return {
                'success': True,
                'message': 'Session cleared successfully'
            }

        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to clear session',
                'details': str(e)
            }

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Delete a conversation session

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session: {session_id}")

            return {
                'success': True,
                'message': 'Session deleted successfully'
            }

        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to delete session',
                'details': str(e)
            }

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active sessions

        Returns:
            Dict containing session statistics
        """
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


# Singleton instance
_debugger_chat_service = None


def get_debugger_chat_service():
    """Get singleton instance of debugger chat service"""
    global _debugger_chat_service

    if _debugger_chat_service is None:
        from services.openai_service import get_openai_service
        openai_service = get_openai_service()
        _debugger_chat_service = DebuggerChatService(openai_service)

    return _debugger_chat_service
