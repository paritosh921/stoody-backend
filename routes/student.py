"""
Student Routes for SkillBot Backend
Handles student-specific operations and question attempt tracking
"""

import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from marshmallow import ValidationError

from models import QuestionAttempt, QuestionAttemptSchema, Student, StudentPasswordChangeSchema, StudentMetrics
from .auth import student_required

logger = logging.getLogger(__name__)

# Create blueprint
student_bp = Blueprint('student', __name__, url_prefix='/api/student')

@student_bp.route('/profile', methods=['GET'])
@jwt_required()
@student_required
def get_profile():
    """Get student profile information"""
    try:
        user_id = get_jwt_identity()
        student = Student.find_by_id(user_id)

        if not student:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student profile not found'
            }), 404

        return jsonify({
            'success': True,
            'message': 'Profile retrieved successfully',
            'data': {'student': student.to_dict()}
        }), 200

    except Exception as e:
        logger.error(f"Get student profile error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve profile'
        }), 500

@student_bp.route('/attempts', methods=['POST'])
@jwt_required()
@student_required
def submit_attempt():
    """Submit a question attempt"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()

        # Validate request data
        schema = QuestionAttemptSchema()
        attempt_data = schema.load(request.get_json())

        # Set student_id from current user
        attempt_data.student_id = claims.get('student_id')

        # Save attempt
        attempt_id = attempt_data.save()

        # Log the question attempt activity
        StudentMetrics.log_question_attempt(
            student_id=claims.get('student_id'),
            question_id=attempt_data.question_id,
            score=attempt_data.score,
            is_correct=attempt_data.is_correct,
            metadata={
                'question_type': attempt_data.question_type,
                'difficulty': attempt_data.difficulty,
                'subject': attempt_data.subject,
                'mode': attempt_data.mode,
                'time_taken': attempt_data.time_taken,
                'hints_used': attempt_data.hints_used
            }
        )

        logger.info(f"Question attempt submitted by student {claims.get('student_id')}: {attempt_data.question_id}")

        return jsonify({
            'success': True,
            'message': 'Attempt submitted successfully',
            'data': {
                'attempt_id': str(attempt_id),
                'attempt': attempt_data.to_dict()
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
        logger.error(f"Submit attempt error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to submit attempt'
        }), 500

@student_bp.route('/attempts', methods=['GET'])
@jwt_required()
@student_required
def get_my_attempts():
    """Get student's own question attempts with pagination"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()

        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))

        # Validate pagination parameters
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 50

        # Get attempts
        result = QuestionAttempt.get_student_attempts(claims.get('student_id'), page=page, limit=limit)

        return jsonify({
            'success': True,
            'message': 'Attempts retrieved successfully',
            'data': result
        }), 200

    except Exception as e:
        logger.error(f"Get student attempts error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve attempts'
        }), 500

@student_bp.route('/performance', methods=['GET'])
@jwt_required()
@student_required
def get_my_performance():
    """Get student's own performance analytics"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()

        # Get performance data
        performance = QuestionAttempt.get_student_performance(claims.get('student_id'))

        return jsonify({
            'success': True,
            'message': 'Performance retrieved successfully',
            'data': {'performance': performance}
        }), 200

    except Exception as e:
        logger.error(f"Get student performance error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve performance'
        }), 500

@student_bp.route('/dashboard/stats', methods=['GET'])
@jwt_required()
@student_required
def get_dashboard_stats():
    """Get dashboard statistics for student"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()

        # Get performance data
        performance = QuestionAttempt.get_student_performance(claims.get('student_id'))

        # Get recent attempts (last 7 days)
        from models import get_collection
        from datetime import datetime, timedelta

        attempts_collection = get_collection('question_attempts')
        week_ago = datetime.utcnow() - timedelta(days=7)

        recent_attempts = attempts_collection.count_documents({
            'student_id': claims.get('student_id'),
            'created_at': {'$gte': week_ago}
        })

        # Get streak (consecutive days with attempts)
        # This is a simplified version - you might want to implement a more sophisticated streak calculation
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        streak = 0
        check_date = today

        while True:
            day_start = check_date
            day_end = check_date + timedelta(days=1)

            day_attempts = attempts_collection.count_documents({
                'student_id': claims.get('student_id'),
                'created_at': {'$gte': day_start, '$lt': day_end}
            })

            if day_attempts > 0:
                streak += 1
                check_date -= timedelta(days=1)
            else:
                break

            # Limit to prevent infinite loops
            if streak > 365:
                break

        return jsonify({
            'success': True,
            'message': 'Dashboard stats retrieved successfully',
            'data': {
                'total_attempts': performance['total_attempts'],
                'accuracy': performance['accuracy'],
                'avg_score': performance['avg_score'],
                'recent_attempts': recent_attempts,
                'current_streak': streak,
                'difficulty_breakdown': performance['difficulty_breakdown'],
                'subject_performance': performance['subject_performance']
            }
        }), 200

    except Exception as e:
        logger.error(f"Get student dashboard stats error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to retrieve dashboard stats'
        }), 500

@student_bp.route('/practice/evaluate', methods=['POST'])
@jwt_required()
@student_required
def evaluate_practice_answer():
    """Evaluate a practice question answer and provide feedback"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()
        data = request.get_json()

        # Validate required fields
        required_fields = ['question_id', 'question_text', 'student_answer', 'question_type']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing field',
                    'message': f'Field {field} is required'
                }), 400

        # For now, we'll do basic evaluation
        # In a real system, you might integrate with AI services for more sophisticated evaluation
        is_correct = False
        score = 0
        feedback = "Answer submitted successfully"

        # Simple evaluation for MCQ
        if data['question_type'] == 'mcq' and 'correct_answer' in data:
            is_correct = data['student_answer'].strip().lower() == data['correct_answer'].strip().lower()
            score = 100 if is_correct else 0
            feedback = "Correct answer!" if is_correct else "Incorrect answer. Try again!"

        # Create question attempt record
        attempt = QuestionAttempt(
            student_id=claims.get('student_id'),
            question_id=data['question_id'],
            question_type=data['question_type'],
            question_text=data['question_text'],
            student_answer=data['student_answer'],
            correct_answer=data.get('correct_answer'),
            is_correct=is_correct,
            score=score,
            time_taken=data.get('time_taken'),
            hints_used=data.get('hints_used', 0),
            mode='practice',
            difficulty=data.get('difficulty'),
            subject=data.get('subject'),
            topic=data.get('topic')
        )

        attempt.save()

        return jsonify({
            'success': True,
            'message': 'Answer evaluated successfully',
            'data': {
                'is_correct': is_correct,
                'score': score,
                'feedback': feedback,
                'correct_answer': data.get('correct_answer') if not is_correct else None
            }
        }), 200

    except Exception as e:
        logger.error(f"Evaluate practice answer error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to evaluate answer'
        }), 500

@student_bp.route('/change-password', methods=['POST'])
@jwt_required()
@student_required
def change_password():
    """Change student password"""
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()

        # Get current student
        student = Student.find_by_id(user_id)
        if not student:
            return jsonify({
                'success': False,
                'error': 'Student not found',
                'message': 'Student profile not found'
            }), 404

        # Validate request data
        schema = StudentPasswordChangeSchema()
        password_data = schema.load(request.get_json())

        # Verify current password
        if not student.verify_password(password_data['current_password']):
            return jsonify({
                'success': False,
                'error': 'Invalid password',
                'message': 'Current password is incorrect'
            }), 400

        # Update password
        student.password_hash = Student.hash_password(password_data['new_password'])
        student.save()

        logger.info(f"Password changed for student {claims.get('username')}")

        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': 'Invalid input data',
            'details': e.messages
        }), 400

    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Failed to change password'
        }), 500