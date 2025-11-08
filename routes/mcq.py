from flask import Blueprint, request, jsonify, session
from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity
import logging
from typing import Dict, Any

from services.mcq_service import mcq_service
from models import StudentMetrics

mcq_bp = Blueprint('mcq', __name__, url_prefix='/api/mcq')

# Keep track of recently shown questions per session to avoid repetition
recent_questions_cache = {}

@mcq_bp.route('/check', methods=['POST'])
@jwt_required(optional=True)
def check_answer():
    """Check MCQ answer and return result with solution"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        question_id = data.get('question_id')
        selected_answer = data.get('selected_answer')
        time_taken = data.get('time_taken', 0)  # Time in seconds

        if not question_id or not selected_answer:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: question_id and selected_answer'
            }), 400

        success, result, error = mcq_service.check_mcq_answer(question_id, selected_answer)

        if success:
            # Track student activity if authenticated
            try:
                claims = get_jwt()
                if claims and claims.get('user_type') == 'student':
                    student_id = claims.get('student_id')
                    if student_id:
                        is_correct = result.get('is_correct', False)
                        score = 100 if is_correct else 0

                        # Log the question attempt activity
                        StudentMetrics.log_question_attempt(
                            student_id=student_id,
                            question_id=question_id,
                            score=score,
                            is_correct=is_correct,
                            metadata={
                                'question_type': 'mcq',
                                'mode': 'mcq_practice',
                                'time_taken': time_taken,
                                'selected_answer': selected_answer,
                                'correct_answer': result.get('correct_answer'),
                                'difficulty': result.get('difficulty', 'medium')
                            }
                        )
            except Exception as tracking_error:
                # Don't fail the main request if tracking fails
                logging.warning(f"Failed to track MCQ activity: {str(tracking_error)}")

            return jsonify({
                'success': True,
                'result': result
            })
        else:
            return jsonify({
                'success': False,
                'error': error or 'Failed to check MCQ answer'
            }), 500
            
    except Exception as e:
        logging.error(f"Error in check_answer: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@mcq_bp.route('/solution/<question_id>', methods=['GET'])
def get_solution(question_id: str):
    """Get stored solution for a question"""
    try:
        solution = mcq_service.get_solution(question_id)
        
        if solution:
            return jsonify({
                'success': True,
                'solution': solution
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Solution not found'
            }), 404
            
    except Exception as e:
        logging.error(f"Error in get_solution: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@mcq_bp.route('/solution', methods=['POST'])
def save_solution():
    """Save a solution manually"""
    try:
        solution_data = request.get_json()
        
        if not solution_data:
            return jsonify({
                'success': False,
                'error': 'No solution data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['id', 'question_id', 'correct_answer', 'explanation', 'generated_by']
        for field in required_fields:
            if field not in solution_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        success, error = mcq_service.save_solution(solution_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Solution saved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': error or 'Failed to save solution'
            }), 500
            
    except Exception as e:
        logging.error(f"Error in save_solution: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@mcq_bp.route('/stats', methods=['GET'])
def get_statistics():
    """Get MCQ solutions statistics"""
    try:
        stats = mcq_service.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        logging.error(f"Error in get_statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@mcq_bp.route('/random-question', methods=['GET'])
def get_random_question():
    """Get a random MCQ question with options"""
    try:
        # Get query parameters for filtering
        subject = request.args.get('subject')
        difficulty = request.args.get('difficulty')

        # Import here to avoid circular imports
        from services.question_service import question_service

        # Search for questions from Test Series only - get larger sample for better randomization
        questions = question_service.search_questions(
            subject=subject,
            difficulty=difficulty,
            document_type="Test Series",  # MCQ Mode uses Test Series only
            limit=200,  # Get more questions to choose from for better randomization
            include_images=True  # Make sure images are included with base64 data
        )

        # Filter for questions that have options (MCQ questions) OR enhanced options OR questions with multiple images
        mcq_questions = [q for q in questions if
            # Traditional MCQ with options
            (q.get('options') and len(q.get('options', [])) > 1) or
            # Enhanced MCQ with enhanced options (including image options)
            (q.get('enhancedOptions') and len(q.get('enhancedOptions', [])) > 1) or
            # Questions with multiple images (even without options) - can be used as visual questions
            (q.get('images') and len(q.get('images', [])) > 1)
        ]

        if not mcq_questions:
            return jsonify({
                'success': False,
                'error': 'No MCQ questions found with the specified criteria. Searched for questions with options, enhanced options (including image options), or multiple images.'
            }), 404

        # Implement session-based tracking to avoid recent repetitions
        import random
        import time
        from datetime import datetime, timedelta

        # Get client identifier (use IP + User-Agent as simple session)
        client_id = f"{request.remote_addr}_{hash(request.headers.get('User-Agent', ''))}"

        # Clean up old entries (older than 30 minutes)
        current_time = datetime.now()
        if client_id in recent_questions_cache:
            recent_questions_cache[client_id] = [
                (qid, timestamp) for qid, timestamp in recent_questions_cache[client_id]
                if current_time - timestamp < timedelta(minutes=30)
            ]

        # Get recently shown question IDs for this client
        recent_question_ids = set()
        if client_id in recent_questions_cache:
            recent_question_ids = {qid for qid, _ in recent_questions_cache[client_id]}

        # Filter out recently shown questions
        available_questions = [q for q in mcq_questions if q.get('id') not in recent_question_ids]

        # If all questions have been shown recently, reset and use all questions
        if not available_questions:
            available_questions = mcq_questions
            recent_questions_cache[client_id] = []

        # Seed random with current time for better randomization
        random.seed(int(time.time() * 1000000) % 1000000)

        # Shuffle the questions array multiple times for better distribution
        for _ in range(3):
            random.shuffle(available_questions)

        # Prioritize questions with images for better visual experience, but with more balanced selection
        questions_with_images = [q for q in available_questions if q.get('images') and len(q.get('images', [])) > 0]
        questions_without_images = [q for q in available_questions if not q.get('images') or len(q.get('images', [])) == 0]

        # Select a random question with better distribution
        if questions_with_images and questions_without_images:
            # 60% chance for questions with images, 40% for without images
            if random.random() < 0.6:
                selected_question = random.choice(questions_with_images)
            else:
                selected_question = random.choice(questions_without_images)
        elif questions_with_images:
            selected_question = random.choice(questions_with_images)
        else:
            selected_question = random.choice(questions_without_images)

        # Track this question as recently shown
        if client_id not in recent_questions_cache:
            recent_questions_cache[client_id] = []
        recent_questions_cache[client_id].append((selected_question.get('id'), current_time))

        # Remove correct answer from response (don't spoil it!)
        response_question = selected_question.copy()
        if 'correctAnswer' in response_question:
            del response_question['correctAnswer']

        logging.info(f"Selected random question: {response_question.get('id')} from {len(mcq_questions)} available MCQ questions")

        return jsonify({
            'success': True,
            'question': response_question
        })

    except Exception as e:
        logging.error(f"Error in get_random_question: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
