"""
Debugger Chat Routes
Handles debugger mode chat with conversation memory and file uploads
"""

from flask import Blueprint, request, jsonify
import logging
from services.debugger_chat_service import get_debugger_chat_service

# Create debugger blueprint
debugger_bp = Blueprint('debugger', __name__, url_prefix='/api/debugger')

# Configure logging
logger = logging.getLogger(__name__)


@debugger_bp.route('/chat', methods=['POST'])
def chat():
    """
    Handle debugger chat message with conversation memory

    Expected JSON payload:
    {
        "sessionId": "unique_session_id",
        "message": "User message",
        "attachments": ["url1", "url2"],  // optional
        "imageData": "base64_image_data"  // optional
    }

    Returns:
    {
        "success": true,
        "data": {
            "user_message": {...},
            "assistant_message": {...},
            "response": "AI response text",
            "session_id": "...",
            "message_count": 5,
            "usage": {...},
            "model": "gpt-4"
        }
    }
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        # Extract required fields
        session_id = data.get('sessionId')
        message = data.get('message', '').strip()

        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID is required'
            }), 400

        if not message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400

        # Extract optional fields
        attachments = data.get('attachments', [])
        image_data = data.get('imageData')

        logger.info(f"Debugger chat - Session: {session_id}, Message length: {len(message)}")

        # Get debugger chat service
        chat_service = get_debugger_chat_service()

        # Send message and get response
        result = chat_service.send_message(
            session_id=session_id,
            message=message,
            attachments=attachments,
            image_data=image_data
        )

        if not result['success']:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Debugger chat endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@debugger_bp.route('/history/<session_id>', methods=['GET'])
def get_history(session_id: str):
    """
    Get conversation history for a session

    Query params:
        limit: Optional limit on number of messages

    Returns:
    {
        "success": true,
        "data": {
            "session_id": "...",
            "messages": [...],
            "metadata": {...}
        }
    }
    """
    try:
        # Get optional limit parameter
        limit = request.args.get('limit', type=int)

        # Get chat service
        chat_service = get_debugger_chat_service()

        # Get conversation history
        result = chat_service.get_conversation_history(session_id, limit)

        if not result['success']:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Get history error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve history',
            'message': str(e)
        }), 500


@debugger_bp.route('/session/<session_id>/clear', methods=['POST'])
def clear_session(session_id: str):
    """
    Clear conversation history for a session

    Returns:
    {
        "success": true,
        "message": "Session cleared successfully"
    }
    """
    try:
        chat_service = get_debugger_chat_service()
        result = chat_service.clear_session(session_id)

        if not result['success']:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Clear session error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to clear session',
            'message': str(e)
        }), 500


@debugger_bp.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    """
    Delete a conversation session

    Returns:
    {
        "success": true,
        "message": "Session deleted successfully"
    }
    """
    try:
        chat_service = get_debugger_chat_service()
        result = chat_service.delete_session(session_id)

        if not result['success']:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Delete session error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to delete session',
            'message': str(e)
        }), 500


@debugger_bp.route('/sessions', methods=['GET'])
def get_sessions():
    """
    Get statistics about all active sessions

    Returns:
    {
        "success": true,
        "data": {
            "active_sessions": 5,
            "sessions": [...]
        }
    }
    """
    try:
        chat_service = get_debugger_chat_service()
        result = chat_service.get_session_stats()

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve sessions',
            'message': str(e)
        }), 500


@debugger_bp.route('/health', methods=['GET'])
def health():
    """
    Health check for debugger chat service

    Returns:
    {
        "success": true,
        "message": "Debugger chat service is healthy",
        "active_sessions": 5
    }
    """
    try:
        chat_service = get_debugger_chat_service()
        stats = chat_service.get_session_stats()

        return jsonify({
            'success': True,
            'message': 'Debugger chat service is healthy',
            'active_sessions': stats['data']['active_sessions']
        }), 200

    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Debugger chat service unhealthy',
            'message': str(e)
        }), 500
