"""
Admin Management Routes for SkillBot Backend
Handles admin panel operations for student management and analytics
"""

import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from marshmallow import ValidationError

from models import Student, StudentSchema, StudentUpdateSchema, StudentPasswordResetSchema, QuestionAttempt, StudentMetrics
from .auth import admin_required

logger = logging.getLogger(__name__)

# Create blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

@admin_bp.route('/students', methods=['GET'])
@jwt_required()
@admin_required
def get_all_students():
    """Get all students with pagination and search"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        search = request.args.get('search', None)

        # Validate pagination parameters
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 50

        # Get students
        result = Student.get_all_students(page=page, limit=limit, search=search)

        return jsonify({
            'success': True,
            'message': 'Students retrieved successfully',
            'data': result
        }), 200

    except Exception as e:
        logger.error(f"Get students error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve students'
        }), 500

@admin_bp.route('/students', methods=['POST'])
@jwt_required()
@admin_required
def create_student():
    """Create a new student"""
    try:
        user_id = get_jwt_identity()

        # Validate request data
        schema = StudentSchema()
        student_data = schema.load(request.get_json())

        # Check if student ID already exists
        if Student.check_student_id_exists(student_data.student_id):
            return jsonify({
                'success': False,
                'error': 'Duplicate student ID',
                'message': 'Student ID already exists'
            }), 400

        # Check if username already exists
        if Student.check_username_exists(student_data.username):
            return jsonify({
                'success': False,
                'error': 'Duplicate username',
                'message': 'Username already exists'
            }), 400

        # Check if email already exists (if provided)
        if student_data.email and Student.find_by_email(student_data.email):
            return jsonify({
                'success': False,
                'error': 'Duplicate email',
                'message': 'Email already exists'
            }), 400

        # Set created_by to current admin
        student_data.created_by = user_id

        # Save student
        student_id = student_data.save()

        claims = get_jwt()
        logger.info(f"Student created by admin {claims.get('email')}: {student_data.student_id}")

        return jsonify({
            'success': True,
            'message': 'Student created successfully',
            'data': {
                'student_id': str(student_id),
                'student': student_data.to_dict()
            }
        }), 201

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': 'Invalid input data',
            'details': e.messages
        }), 400

    except Exception as e:
        logger.error(f"Create student error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to create student'
        }), 500

@admin_bp.route('/students/<student_id>', methods=['GET'])
@jwt_required()
@admin_required
def get_student(student_id):
    """Get student details by ID"""
    try:
        # Try to find by student_id first, then by MongoDB ObjectId
        student = Student.find_by_student_id(student_id)
        if not student:
            student = Student.find_by_id(student_id)

        if not student:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student not found'
            }), 404

        return jsonify({
            'success': True,
            'message': 'Student retrieved successfully',
            'data': {'student': student.to_dict()}
        }), 200

    except Exception as e:
        logger.error(f"Get student error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve student'
        }), 500

@admin_bp.route('/students/<student_id>', methods=['PUT'])
@jwt_required()
@admin_required
def update_student(student_id):
    """Update student details"""
    try:
        # Find student
        student = Student.find_by_student_id(student_id)
        if not student:
            student = Student.find_by_id(student_id)

        if not student:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student not found'
            }), 404

        # Validate request data
        schema = StudentUpdateSchema()
        update_data = schema.load(request.get_json())

        # Check if username is being changed and already exists
        if 'username' in update_data and update_data['username']:
            existing_student = Student.find_by_username(update_data['username'])
            if existing_student and str(existing_student._id) != str(student._id):
                return jsonify({
                    'success': False,
                    'error': 'Duplicate username',
                    'message': 'Username already exists'
                }), 400

        # Check if email is being changed and already exists
        if 'email' in update_data and update_data['email']:
            existing_student = Student.find_by_email(update_data['email'])
            if existing_student and str(existing_student._id) != str(student._id):
                return jsonify({
                    'success': False,
                    'error': 'Duplicate email',
                    'message': 'Email already exists'
                }), 400

        # Update student fields
        for field, value in update_data.items():
            if hasattr(student, field):
                setattr(student, field, value)

        # Save updated student
        student.save()

        claims = get_jwt()
        logger.info(f"Student updated by admin {claims.get('email')}: {student.student_id}")

        return jsonify({
            'success': True,
            'message': 'Student updated successfully',
            'data': {'student': student.to_dict()}
        }), 200

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': 'Invalid input data',
            'details': e.messages
        }), 400

    except Exception as e:
        logger.error(f"Update student error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to update student'
        }), 500

@admin_bp.route('/students/<student_id>', methods=['DELETE'])
@jwt_required()
@admin_required
def delete_student(student_id):
    """Delete (deactivate) a student"""
    try:
        # Find student
        student = Student.find_by_student_id(student_id)
        if not student:
            student = Student.find_by_id(student_id)

        if not student:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student not found'
            }), 404

        # Soft delete (deactivate)
        student.delete()

        claims = get_jwt()
        logger.info(f"Student deleted by admin {claims.get('email')}: {student.student_id}")

        return jsonify({
            'success': True,
            'message': 'Student deleted successfully'
        }), 200

    except Exception as e:
        logger.error(f"Delete student error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to delete student'
        }), 500

@admin_bp.route('/students/<student_id>/reset-password', methods=['POST'])
@jwt_required()
@admin_required
def reset_student_password(student_id):
    """Reset student password (admin only)"""
    try:
        # Find student
        student = Student.find_by_student_id(student_id)
        if not student:
            student = Student.find_by_id(student_id)

        if not student:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student not found'
            }), 404

        # Validate request data
        schema = StudentPasswordResetSchema()
        password_data = schema.load(request.get_json())

        # Update password
        student.password_hash = Student.hash_password(password_data['new_password'])
        student.save()

        claims = get_jwt()
        logger.info(f"Password reset by admin {claims.get('email')} for student {student.student_id}")

        return jsonify({
            'success': True,
            'message': 'Password reset successfully'
        }), 200

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': 'Invalid input data',
            'details': e.messages
        }), 400

    except Exception as e:
        logger.error(f"Reset password error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to reset password'
        }), 500

@admin_bp.route('/students/<student_id>/performance', methods=['GET'])
@jwt_required()
@admin_required
def get_student_performance(student_id):
    """Get detailed performance analytics for a student"""
    try:
        # Find student
        student = Student.find_by_student_id(student_id)
        if not student:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student not found'
            }), 404

        # Get performance data
        performance = QuestionAttempt.get_student_performance(student.student_id)

        return jsonify({
            'success': True,
            'message': 'Student performance retrieved successfully',
            'data': {
                'student': {
                    'student_id': student.student_id,
                    'name': student.name,
                    'email': student.email,
                    'school': student.school,
                    'grade': student.grade
                },
                'performance': performance
            }
        }), 200

    except Exception as e:
        logger.error(f"Get student performance error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve student performance'
        }), 500

@admin_bp.route('/students/<student_id>/attempts', methods=['GET'])
@jwt_required()
@admin_required
def get_student_attempts(student_id):
    """Get student's question attempts with pagination"""
    try:
        # Find student
        student = Student.find_by_student_id(student_id)
        if not student:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student not found'
            }), 404

        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))

        # Validate pagination parameters
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 50

        # Get attempts
        result = QuestionAttempt.get_student_attempts(student.student_id, page=page, limit=limit)

        return jsonify({
            'success': True,
            'message': 'Student attempts retrieved successfully',
            'data': {
                'student': {
                    'student_id': student.student_id,
                    'name': student.name
                },
                **result
            }
        }), 200

    except Exception as e:
        logger.error(f"Get student attempts error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve student attempts'
        }), 500

@admin_bp.route('/analytics/overview', methods=['GET'])
@jwt_required()
@admin_required
def get_analytics_overview():
    """Get comprehensive analytics overview for all students"""
    try:
        # Get admin_id from JWT for data isolation
        user_id = get_jwt_identity()
        claims = get_jwt()

        if claims.get('user_type') != 'admin':
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Admin access required'
            }), 403

        # Use admin_id from JWT claims, fallback to user_id
        admin_id = claims.get('admin_id', user_id)

        # Get all students performance filtered by admin_id
        all_performance = QuestionAttempt.get_all_students_performance(admin_id=admin_id)

        return jsonify({
            'success': True,
            'message': 'Analytics overview retrieved successfully',
            'data': {
                'students_performance': all_performance,
                'summary': {
                    'total_students': len(all_performance),
                    'total_attempts': sum(p['total_attempts'] for p in all_performance),
                    'avg_accuracy': sum(p['accuracy'] for p in all_performance) / len(all_performance) if all_performance else 0
                }
            }
        }), 200

    except Exception as e:
        logger.error(f"Get analytics overview error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve analytics overview'
        }), 500

@admin_bp.route('/questions/<question_id>/analytics', methods=['GET'])
@jwt_required()
@admin_required
def get_question_analytics(question_id):
    """Get analytics for a specific question"""
    try:
        analytics = QuestionAttempt.get_question_analytics(question_id)

        return jsonify({
            'success': True,
            'message': 'Question analytics retrieved successfully',
            'data': analytics
        }), 200

    except Exception as e:
        logger.error(f"Get question analytics error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve question analytics'
        }), 500

@admin_bp.route('/dashboard/stats', methods=['GET'])
@jwt_required()
@admin_required
def get_dashboard_stats():
    """Get dashboard statistics for admin panel"""
    try:
        # Get admin_id from JWT for data isolation
        user_id = get_jwt_identity()
        claims = get_jwt()

        if claims.get('user_type') != 'admin':
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Admin access required'
            }), 403

        from models import get_collection

        # Use admin_id from JWT claims, fallback to user_id
        admin_id = claims.get('admin_id', user_id)

        # Get basic counts filtered by admin_id
        students_collection = get_collection('students')
        attempts_collection = get_collection('question_attempts')
        documents_collection = get_collection('documents')

        total_students = students_collection.count_documents({'is_active': True, 'admin_id': admin_id})
        total_attempts = attempts_collection.count_documents({'admin_id': admin_id})

        # Get recent activity (last 7 days) filtered by admin_id
        from datetime import datetime, timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)

        recent_students = students_collection.count_documents({
            'created_at': {'$gte': week_ago},
            'admin_id': admin_id
        })

        recent_attempts = attempts_collection.count_documents({
            'created_at': {'$gte': week_ago},
            'admin_id': admin_id
        })

        # Get top performing students (by accuracy) filtered by admin_id
        top_students = QuestionAttempt.get_all_students_performance(admin_id=admin_id)[:10]

        return jsonify({
            'success': True,
            'message': 'Dashboard stats retrieved successfully',
            'data': {
                'total_students': total_students,
                'total_attempts': total_attempts,
                'recent_students': recent_students,
                'recent_attempts': recent_attempts,
                'top_students': top_students
            }
        }), 200

    except Exception as e:
        logger.error(f"Get dashboard stats error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve dashboard stats'
        }), 500

# New endpoints for student monitoring
@admin_bp.route('/monitoring/student-progress', methods=['GET'])
@jwt_required()
@admin_required
def get_student_progress_monitoring():
    """Get student progress data for admin monitoring panel"""
    try:
        # Get all student progress summaries
        progress_data = StudentMetrics.get_all_student_progress()

        return jsonify({
            'success': True,
            'message': 'Student progress data retrieved successfully',
            'data': {
                'student_progress': progress_data
            }
        }), 200

    except Exception as e:
        logger.error(f"Get student progress monitoring error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve student progress data'
        }), 500

@admin_bp.route('/monitoring/recent-activities', methods=['GET'])
@jwt_required()
@admin_required
def get_recent_activities_monitoring():
    """Get recent student activities for admin monitoring panel"""
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        if limit < 1 or limit > 200:
            limit = 50

        # Get recent activities with student names
        activities = StudentMetrics.get_recent_activities_with_names(limit=limit)

        return jsonify({
            'success': True,
            'message': 'Recent activities retrieved successfully',
            'data': {
                'student_activities': activities
            }
        }), 200

    except Exception as e:
        logger.error(f"Get recent activities monitoring error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve recent activities'
        }), 500

@admin_bp.route('/monitoring/dashboard-stats', methods=['GET'])
@jwt_required()
@admin_required
def get_monitoring_dashboard_stats():
    """Get comprehensive dashboard statistics for admin monitoring"""
    try:
        # Get dashboard stats
        dashboard_stats = StudentMetrics.get_admin_dashboard_stats()

        return jsonify({
            'success': True,
            'message': 'Monitoring dashboard stats retrieved successfully',
            'data': dashboard_stats
        }), 200

    except Exception as e:
        logger.error(f"Get monitoring dashboard stats error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve monitoring dashboard stats'
        }), 500

@admin_bp.route('/monitoring/student/<student_id>/details', methods=['GET'])
@jwt_required()
@admin_required
def get_student_monitoring_details(student_id):
    """Get detailed monitoring information for a specific student"""
    try:
        # Get student progress summary
        progress = StudentMetrics.get_student_progress_summary(student_id)

        if not progress:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student not found'
            }), 404

        return jsonify({
            'success': True,
            'message': 'Student monitoring details retrieved successfully',
            'data': progress
        }), 200

    except Exception as e:
        logger.error(f"Get student monitoring details error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve student monitoring details'
        }), 500

# Database status endpoint (diagnostics)
@admin_bp.route('/database/status', methods=['GET'])
@jwt_required()
@admin_required
def get_database_status():
    """Return MongoDB connection status for diagnostics."""
    try:
        from models import get_collection
        mongo_connected = False
        mongo_status = 'offline'
        students_count = 0
        attempts_count = 0

        try:
            students_collection = get_collection('students')
            attempts_collection = get_collection('question_attempts')
            students_count = students_collection.estimated_document_count()
            attempts_count = attempts_collection.estimated_document_count()
            mongo_connected = True
            mongo_status = 'online'
        except Exception as e:
            logger.error(f"MongoDB health check failed: {str(e)}")
            mongo_connected = False
            mongo_status = 'offline'

        return jsonify({
            'success': True,
            'status': {
                'mongodb': {
                    'connected': mongo_connected,
                    'status': mongo_status,
                    'students_count': students_count,
                    'attempts_count': attempts_count
                }
            }
        }), 200

    except Exception as e:
        logger.error(f"Get database status error: {str(e)}")
        return jsonify({
            'success': False,
            'status': {
                'mongodb': {'connected': False, 'status': 'offline'}
            }
        }), 500

# Test data creation endpoint (development only)
@admin_bp.route('/create-test-data', methods=['POST'])
@jwt_required()
@admin_required
def create_test_data():
    """Create test data for development and demonstration"""
    try:
        import subprocess
        import os

        # Get the backend directory path
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(backend_dir, 'create_test_data.py')

        # Run the test data creation script
        result = subprocess.run(['python', script_path],
                              capture_output=True, text=True, cwd=backend_dir)

        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'Test data created successfully',
                'details': result.stdout
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Test data creation failed',
                'details': result.stderr
            }), 500

    except Exception as e:
        logger.error(f"Create test data error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to create test data'
        }), 500
