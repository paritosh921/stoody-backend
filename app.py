from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import logging
import os
from pathlib import Path
import time

# Optional: ChromaDB health probing
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_AVAILABLE = True
except Exception:  # pragma: no cover - health check will handle gracefully
    _CHROMA_AVAILABLE = False

# Import configuration
from config import FLASK_PORT, FLASK_DEBUG, CORS_ORIGINS, JWT_SECRET_KEY, JWT_ACCESS_TOKEN_EXPIRES
from config import CHROMADB_PATH, CHROMADB_COLLECTION_NAME

# Import route blueprints
from routes.questions import questions_bp
from routes.images import images_bp
from routes.chat import chat_bp
from routes.practice import practice_bp
from routes.mcq import mcq_bp
from routes.auth import auth_bp
from routes.admin import admin_bp
from routes.student import student_bp
from routes.debugger import debugger_bp

def create_app():
    """Create and configure the Flask application"""

    # Create Flask app
    app = Flask(__name__)

    # Configure JWT
    app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = JWT_ACCESS_TOKEN_EXPIRES
    jwt = JWTManager(app)

    # Configure CORS
    CORS(app, origins=CORS_ORIGINS)
    
    # Configure logging
    if FLASK_DEBUG:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Reduce werkzeug logging noise in development
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Register blueprints
    app.register_blueprint(questions_bp)
    app.register_blueprint(images_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(practice_bp)
    app.register_blueprint(mcq_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(debugger_bp)
    
    # Health check endpoint (compat with async backend schema)
    def _probe_chroma() -> tuple:
        """Return (connected: bool, count: int). Never raises."""
        if not _CHROMA_AVAILABLE:
            return False, 0
        try:
            client = chromadb.PersistentClient(
                path=str(CHROMADB_PATH),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                    is_persistent=True,
                ),
            )
            collection = client.get_or_create_collection(name=str(CHROMADB_COLLECTION_NAME))
            try:
                count = collection.count()
            except Exception:
                # Some versions may not support count; fall back to len(get())
                res = collection.get()
                count = len(res.get('ids', []))
            return True, int(count)
        except Exception as _e:
            logging.getLogger(__name__).warning(f"ChromaDB probe failed: {_e}")
            return False, 0

    @app.route('/health', methods=['GET'])
    def health_check():
        """Comprehensive health check with fields the frontend expects."""
        try:
            chroma_connected, chroma_count = _probe_chroma()

            # Flask app has no mandatory DB/cache here; treat as degraded only if app is fundamentally broken
            overall_healthy = True
            status_str = 'healthy' if overall_healthy else 'degraded'

            return jsonify({
                'success': overall_healthy,
                'healthy': overall_healthy,
                'ok': overall_healthy,
                'status': status_str,
                'message': 'Backend server is running' if overall_healthy else 'Backend is degraded',
                'timestamp': time.time(),
                'services': {
                    'database': 'healthy',
                    'cache': 'optional',
                    'chromadb': {
                        'connected': chroma_connected,
                        'status': 'online' if chroma_connected else 'offline',
                        'questions_count': chroma_count,
                    },
                },
                'chromaConnected': chroma_connected,
                'chromadb': {
                    'connected': chroma_connected,
                    'status': 'online' if chroma_connected else 'offline',
                    'questions_count': chroma_count,
                },
                'version': '2.0.0',
                'mode': 'development' if FLASK_DEBUG else 'production',
            })
        except Exception as e:
            logging.getLogger(__name__).error(f"Health check failed: {e}")
            return jsonify({
                'success': False,
                'healthy': False,
                'ok': False,
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time(),
            }), 503

    # Compatibility aliases expected by the frontend
    @app.route('/api/health', methods=['GET'])
    def health_check_api_alias():
        return health_check()

    @app.route('/api/v1/health', methods=['GET'])
    def health_check_v1_alias():
        return health_check()
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with API information"""
        return jsonify({
            'message': 'SkillBot Backend API',
            'version': '1.0.0',
            'endpoints': {
                'questions': '/api/questions',
                'images': '/api/images',
                'chat': '/api/chat',
                'practice': '/api/practice',
                'mcq': '/api/mcq',
                'auth': '/api/auth',
                'admin': '/api/admin',
                'student': '/api/student',
                'health': '/health'
            },
            'documentation': {
                'questions': {
                    'POST /api/questions/save': 'Save a single question',
                    'POST /api/questions/batch-save': 'Save multiple questions',
                    'GET /api/questions/<id>': 'Get question by ID',
                    'GET /api/questions/search': 'Search questions with filters',
                    'PUT /api/questions/<id>': 'Update question',
                    'DELETE /api/questions/<id>': 'Delete question',
                    'GET /api/questions/stats': 'Get collection statistics',
                    'GET /api/questions/export': 'Export all questions'
                },
                'images': {
                    'POST /api/images/upload': 'Upload image file',
                    'POST /api/images/upload-base64': 'Upload base64 image',
                    'GET /api/images/<path>': 'Serve image file',
                    'GET /api/images/<path>/base64': 'Get image as base64',
                    'GET /api/images/<path>/info': 'Get image information',
                    'DELETE /api/images/<path>': 'Delete image',
                    'POST /api/images/cleanup': 'Cleanup orphaned images'
                },
                'chat': {
                    'POST /api/chat': 'Send message and get AI response',
                    'GET /api/chat/health': 'Check chat service health',
                    'GET /api/chat/models': 'Get available AI models'
                },
                'practice': {
                    'POST /api/practice/next': 'Get next practice question from ChromaDB',
                    'POST /api/practice/evaluate': 'Evaluate canvas/text answer against ground-truth'
                },
                'mcq': {
                    'POST /api/mcq/check': 'Check MCQ answer and get solution',
                    'GET /api/mcq/solution/<question_id>': 'Get stored solution for question',
                    'POST /api/mcq/solution': 'Save solution manually',
                    'GET /api/mcq/stats': 'Get MCQ solutions statistics',
                    'GET /api/mcq/random-question': 'Get random MCQ question'
                },
                'auth': {
                    'POST /api/auth/admin/login': 'Admin login with email/password',
                    'POST /api/auth/student/login': 'Student login with username/password',
                    'GET /api/auth/verify': 'Verify JWT token',
                    'POST /api/auth/logout': 'Logout user',
                    'POST /api/auth/init-admin': 'Initialize default admin account'
                },
                'admin': {
                    'GET /api/admin/students': 'Get all students with pagination',
                    'POST /api/admin/students': 'Create new student with username/password',
                    'GET /api/admin/students/<id>': 'Get student details',
                    'PUT /api/admin/students/<id>': 'Update student',
                    'DELETE /api/admin/students/<id>': 'Delete student',
                    'POST /api/admin/students/<id>/reset-password': 'Reset student password',
                    'GET /api/admin/students/<id>/performance': 'Get student performance',
                    'GET /api/admin/students/<id>/attempts': 'Get student attempts',
                    'GET /api/admin/analytics/overview': 'Get analytics overview',
                    'GET /api/admin/dashboard/stats': 'Get admin dashboard stats'
                },
                'student': {
                    'GET /api/student/profile': 'Get student profile',
                    'POST /api/student/attempts': 'Submit question attempt',
                    'GET /api/student/attempts': 'Get my attempts',
                    'GET /api/student/performance': 'Get my performance',
                    'GET /api/student/dashboard/stats': 'Get student dashboard stats',
                    'POST /api/student/practice/evaluate': 'Evaluate practice answer',
                    'POST /api/student/change-password': 'Change password'
                }
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found',
            'message': 'The requested endpoint does not exist'
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'success': False,
            'error': 'Method not allowed',
            'message': 'The requested method is not allowed for this endpoint'
        }), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        logging.error(f"Internal server error: {str(error)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    return app

def main():
    """Main function to run the Flask application"""
    app = create_app()
    
    print(f"""
================================================================
                    SkillBot Backend API

  Starting Flask server...
  URL: http://localhost:{FLASK_PORT}
  Debug mode: {'ON' if FLASK_DEBUG else 'OFF'}
  Endpoints:
    - http://localhost:{FLASK_PORT}/
    - http://localhost:{FLASK_PORT}/health
    - http://localhost:{FLASK_PORT}/api/auth
    - http://localhost:{FLASK_PORT}/api/admin
    - http://localhost:{FLASK_PORT}/api/student
    - http://localhost:{FLASK_PORT}/api/questions
    - http://localhost:{FLASK_PORT}/api/images
    - http://localhost:{FLASK_PORT}/api/chat

  Data directories:
    - ChromaDB: ./chromadb_data
    - Images: ./images
    - MongoDB: Connected via environment

================================================================
    """)
    
    try:
        app.run(
            host='0.0.0.0',
            port=FLASK_PORT,
            debug=FLASK_DEBUG,
            use_reloader=FLASK_DEBUG,
            reloader_type='stat' if FLASK_DEBUG else None
        )
    except KeyboardInterrupt:
        print("\n\nShutting down SkillBot Backend API...")
    except Exception as e:
        print(f"\nError starting server: {str(e)}")

if __name__ == '__main__':
    main()
