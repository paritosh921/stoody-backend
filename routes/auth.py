"""
Authentication Routes for SkillBot Backend
Handles admin and student login/logout
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt
from marshmallow import ValidationError

from models import Admin, Student, AdminLoginSchema, StudentLoginSchema, StudentMetrics

logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/admin/login', methods=['POST'])
def admin_login():
    """Admin login endpoint"""
    try:
        # Validate request data
        schema = AdminLoginSchema()
        login_data = schema.load(request.get_json())

        # Find admin by email
        admin = Admin.find_by_email(login_data['email'])
        if not admin:
            return jsonify({
                'success': False,
                'error': 'Invalid credentials',
                'message': 'Email or password is incorrect'
            }), 401

        # Verify password
        if not admin.verify_password(login_data['password']):
            return jsonify({
                'success': False,
                'error': 'Invalid credentials',
                'message': 'Email or password is incorrect'
            }), 401

        # Check if admin is active
        if not admin.is_active:
            return jsonify({
                'success': False,
                'error': 'Account disabled',
                'message': 'Your account has been disabled'
            }), 403

        # Update last login
        admin.update_last_login()

        # Create JWT token
        access_token = create_access_token(
            identity=str(admin._id),
            additional_claims={
                'email': admin.email,
                'role': admin.role,
                'user_type': 'admin',
                'admin_id': str(admin._id),  # Add admin_id for data isolation
                'subdomain': admin.subdomain
            }
        )

        logger.info(f"Admin login successful: {admin.email}")

        return jsonify({
            'success': True,
            'message': 'Login successful',
            'data': {
                'access_token': access_token,
                'user_type': 'admin',
                'user': {
                    'id': str(admin._id),
                    'email': admin.email,
                    'name': admin.name,
                    'role': admin.role
                }
            }
        }), 200

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': 'Invalid input data',
            'details': e.messages
        }), 400

    except Exception as e:
        logger.error(f"Admin login error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

@auth_bp.route('/student/login', methods=['POST'])
def student_login():
    """Student login endpoint"""
    try:
        # Validate request data
        schema = StudentLoginSchema()
        login_data = schema.load(request.get_json())

        # Find student by username
        student = Student.find_by_username(login_data['username'])
        if not student:
            return jsonify({
                'success': False,
                'error': 'Invalid credentials',
                'message': 'Username or password is incorrect'
            }), 401

        # Verify password
        if not student.verify_password(login_data['password']):
            return jsonify({
                'success': False,
                'error': 'Invalid credentials',
                'message': 'Username or password is incorrect'
            }), 401

        # Check if student is active
        if not student.is_active:
            return jsonify({
                'success': False,
                'error': 'Account disabled',
                'message': 'Your account has been disabled'
            }), 403

        # Update last login
        student.update_last_login()

        # Track login session and activity
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        user_agent = request.headers.get('User-Agent')
        session = StudentMetrics.log_student_login(student.student_id, ip_address, user_agent)

        # Create JWT token
        access_token = create_access_token(
            identity=str(student._id),
            additional_claims={
                'student_id': student.student_id,
                'username': student.username,
                'email': student.email,
                'user_type': 'student'
            }
        )

        logger.info(f"Student login successful: {student.username} (ID: {student.student_id})")

        return jsonify({
            'success': True,
            'message': 'Login successful',
            'data': {
                'access_token': access_token,
                'user_type': 'student',
                'user': {
                    'id': str(student._id),
                    'student_id': student.student_id,
                    'username': student.username,
                    'name': student.name,
                    'email': student.email,
                    'grade': student.grade,
                    'school': student.school
                }
            }
        }), 200

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': 'Invalid input data',
            'details': e.messages
        }), 400

    except Exception as e:
        logger.error(f"Student login error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

@auth_bp.route('/verify', methods=['GET'])
@jwt_required()
def verify_token():
    """Verify JWT token and return user info"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()

        if claims.get('user_type') == 'admin':
            admin = Admin.find_by_id(user_id)
            if not admin or not admin.is_active:
                return jsonify({
                    'success': False,
                    'error': 'Invalid token',
                    'message': 'User not found or inactive'
                }), 401

            return jsonify({
                'success': True,
                'data': {
                    'user_type': 'admin',
                    'user': {
                        'id': str(admin._id),
                        'email': admin.email,
                        'name': admin.name,
                        'role': admin.role
                    }
                }
            }), 200

        elif claims.get('user_type') == 'student':
            student = Student.find_by_id(user_id)
            if not student or not student.is_active:
                return jsonify({
                    'success': False,
                    'error': 'Invalid token',
                    'message': 'User not found or inactive'
                }), 401

            return jsonify({
                'success': True,
                'data': {
                    'user_type': 'student',
                    'user': {
                        'id': str(student._id),
                        'student_id': student.student_id,
                        'username': student.username,
                        'name': student.name,
                        'email': student.email,
                        'grade': student.grade,
                        'school': student.school
                    }
                }
            }), 200

        else:
            return jsonify({
                'success': False,
                'error': 'Invalid token',
                'message': 'Unknown user type'
            }), 401

    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout endpoint (token invalidation is handled on frontend)"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()

        # Track logout for students
        if claims.get('user_type') == 'student':
            student_id = claims.get('student_id')
            if student_id:
                StudentMetrics.log_student_logout(student_id)
                logger.info(f"Student logged out: {student_id}")
            else:
                logger.info(f"Student logged out: {user_id}")
        else:
            logger.info(f"User logged out: {claims.get('email', user_id)}")

        return jsonify({
            'success': True,
            'message': 'Logout successful'
        }), 200

    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

@auth_bp.route('/init-admin', methods=['POST'])
def initialize_admin():
    """Initialize default admin account (development only)"""
    try:
        # Create default admin
        admin = Admin.create_default_admin()

        return jsonify({
            'success': True,
            'message': 'Default admin account initialized',
            'data': {
                'email': admin.email,
                'message': 'Use email: admin@skillbot.app and password: admin123 to login'
            }
        }), 200

    except Exception as e:
        logger.error(f"Admin initialization error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to initialize admin account'
        }), 500

# Utility function to require admin role
def admin_required(f):
    """Decorator to require admin authentication"""
    def decorated_function(*args, **kwargs):
        try:
            user_id = get_jwt_identity()
            claims = get_jwt()

            if claims.get('user_type') != 'admin':
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Admin access required'
                }), 403

            # Verify admin still exists and is active
            admin = Admin.find_by_id(user_id)
            if not admin or not admin.is_active:
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Admin account not found or inactive'
                }), 401

            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Admin authorization error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Authorization error',
                'message': 'Failed to verify admin access'
            }), 500

    decorated_function.__name__ = f.__name__
    return decorated_function

# Utility function to require student role
def student_required(f):
    """Decorator to require student authentication"""
    def decorated_function(*args, **kwargs):
        try:
            user_id = get_jwt_identity()
            claims = get_jwt()

            if claims.get('user_type') != 'student':
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Student access required'
                }), 403

            # Verify student still exists and is active
            student = Student.find_by_id(user_id)
            if not student or not student.is_active:
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Student account not found or inactive'
                }), 401

            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Student authorization error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Authorization error',
                'message': 'Failed to verify student access'
            }), 500

    decorated_function.__name__ = f.__name__
    return decorated_function