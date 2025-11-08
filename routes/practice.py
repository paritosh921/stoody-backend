from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity
import logging
import random
import re
from typing import Optional, Dict, Any

from services.question_service import question_service
from services.openai_service import get_openai_service
from services.mistral_ocr_service import get_mistral_ocr_service
from models import StudentMetrics


practice_bp = Blueprint('practice', __name__, url_prefix='/api/practice')

logger = logging.getLogger(__name__)


def _shape_question_payload(q: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce question to fields needed by UI."""
    return {
        'id': q.get('id'),
        'text': q.get('text', ''),
        'subject': q.get('subject', ''),
        'difficulty': q.get('difficulty', 'medium'),
        'options': q.get('options') or [],
        'enhancedOptions': q.get('enhancedOptions') or [],
        'correctAnswer': q.get('correctAnswer') or '',
        'images': q.get('images') or [],
        'pdfSource': q.get('pdfSource', ''),
        'metadata': q.get('metadata') or {},
    }


@practice_bp.route('/next', methods=['POST'])
def next_question():
    """Return a random next question from ChromaDB with optional filters.

    Payload (all optional): { subject, difficulty, excludeIds: string[] }
    """
    try:
        data = request.get_json(silent=True) or {}
        subject = data.get('subject')
        difficulty = data.get('difficulty')
        exclude_ids = set(data.get('excludeIds') or [])

        # Pull a pool of candidates from Practice Sets only
        candidates = question_service.search_questions(
            subject=subject,
            difficulty=difficulty,
            document_type="Practice Sets",  # Hustle Mode uses Practice Sets only
            limit=50,
            include_images=True,
        )

        # Filter out excluded
        pool = [q for q in candidates if q.get('id') not in exclude_ids]

        if not pool:
            return jsonify({
                'success': False,
                'error': 'No questions available that match filters'
            }), 404

        # Prefer ones with images when available to leverage canvas
        with_images = [q for q in pool if (q.get('images') or [])]
        choice = random.choice(with_images or pool)

        return jsonify({
            'success': True,
            'question': _shape_question_payload(choice)
        })

    except Exception as e:
        logger.exception('Failed to fetch next practice question')
        return jsonify({'success': False, 'error': str(e)}), 500


@practice_bp.route('/evaluate', methods=['POST'])
@jwt_required(optional=True)
def evaluate_submission():
    """Evaluate student's submission (canvas image and/or text) for a question.

    Payload: {
      questionId: string,
      answerText?: string,
      canvasData?: string (data URL base64),
    }
    Returns: { success, evaluation: { correct, score, extractedAnswer, feedback, reasoning } }
    """
    try:
        payload = request.get_json() or {}
        qid = payload.get('questionId')
        if not qid:
            return jsonify({'success': False, 'error': 'questionId is required'}), 400

        answer_text = payload.get('answerText') or ''
        canvas_data = payload.get('canvasData')  # data URL

        # Normalize/crop canvas data URL to remove large transparent borders
        def _crop_canvas_data_url(data_url: Optional[str]) -> Optional[str]:
            if not data_url or not isinstance(data_url, str):
                return data_url
            if not data_url.startswith('data:image'):
                return data_url
            try:
                import base64, io
                from PIL import Image
                header, b64 = data_url.split(',', 1)
                raw = base64.b64decode(b64)
                with Image.open(io.BytesIO(raw)) as im:
                    if im.mode != 'RGBA':
                        im = im.convert('RGBA')
                    alpha = im.split()[-1]
                    bbox = alpha.getbbox()
                    if bbox:
                        # Add small padding
                        left, top, right, bottom = bbox
                        pad = 10
                        left = max(0, left - pad)
                        top = max(0, top - pad)
                        right = min(im.width, right + pad)
                        bottom = min(im.height, bottom + pad)
                        cropped = im.crop((left, top, right, bottom))
                    else:
                        cropped = im
                    buf = io.BytesIO()
                    cropped.save(buf, format='PNG')
                    enc = base64.b64encode(buf.getvalue()).decode('ascii')
                    return 'data:image/png;base64,' + enc
            except Exception as _e:
                logger.warning(f"Canvas crop failed, using original: {_e}")
                return data_url

        canvas_data = _crop_canvas_data_url(canvas_data)

        q = question_service.get_question(qid, include_images=True)
        if not q:
            return jsonify({'success': False, 'error': 'Question not found'}), 404

        correct_answer = (q.get('correctAnswer') or '').strip().upper()


        # Initialize OpenAI service and OCR service
        openai_service = get_openai_service()
        ocr_service = get_mistral_ocr_service()

        # Step 0: Extract text content from canvas using OCR
        extracted_text = ""
        if canvas_data:
            try:
                logger.info("Extracting text from canvas using OCR...")
                ocr_result = ocr_service.extract_text_from_canvas(canvas_data)

                if ocr_result.get('success'):
                    extracted_text = ocr_result.get('extracted_text', '')
                    logger.info(f"OCR extraction successful: {len(extracted_text)} characters extracted")
                    logger.info(f"Extracted content preview: {extracted_text[:200]}...")
                else:
                    logger.warning(f"OCR extraction failed: {ocr_result.get('error')}")
                    extracted_text = "Could not extract text from canvas image"
            except Exception as e:
                logger.error(f"OCR extraction error: {str(e)}")
                extracted_text = "OCR processing failed"

        system_prompt = """You are an expert academic evaluator with advanced OCR and vision capabilities. Your job is to comprehensively analyze student solutions.

CRITICAL: Return ONLY a single line of valid JSON with NO extra text, NO newlines, NO formatting.

ANALYSIS CAPABILITIES:
- Extract and understand handwritten equations, formulas, and mathematical expressions
- Read handwritten text, definitions, and explanations
- Analyze diagrams, graphs, and visual problem-solving steps
- Recognize scientific notation, chemical formulas, and technical symbols
- Understand multi-step solutions and problem-solving approaches

EVALUATION PROCESS:
1. First, solve the given question yourself to determine the correct answer
2. Extract and interpret ALL content from the student's canvas submission
3. If OCR-extracted text is provided, use it as the primary source of student work
4. Cross-reference the extracted text with the canvas image for accuracy
5. Evaluate the student's approach, calculations, and final answer
6. Provide detailed feedback on correctness, methodology, and areas for improvement

HANDWRITING RECOGNITION GUIDELINES:
- Be generous in interpreting unclear handwriting
- Look for mathematical symbols: +, -, ×, ÷, =, ≠, ≈, ∫, Σ, √, π, ∞, etc.
- Recognize scientific notation: 2.5 × 10³, 6.02 × 10²³, etc.
- Identify equation structures: variables, constants, operations
- Understand chemical formulas: H₂O, CO₂, CH₄, etc.
- Read definitions and explanatory text
- Interpret diagrams and their labels

REQUIRED JSON FORMAT (single line, no spaces around colons):
{"correct":true,"score":0.85,"extractedAnswer":"F = ma (Newton's second law)","feedback":"Excellent work! You correctly identified Newton's second law.","reasoning":"Student provided complete solution with proper equation and explanation. Minor calculation error in numerical value.","extractedContent":"F = m*a\\nWhere F is force, m is mass, a is acceleration\\nF = 5kg * 2m/s² = 10N","analysisType":"comprehensive_solution"}

For multiple choice questions, still identify the chosen option (A, B, C, D) if applicable.
For open-ended questions, evaluate the solution quality and correctness.
Always provide constructive feedback and detailed reasoning.

Return ONLY the JSON line. No other text.
"""

        # Compose user message with comprehensive analysis
        q_lines = [
            f"Question: {q.get('text','')}",
            f"Subject: {q.get('subject','Unknown')}",
            f"Difficulty: {q.get('difficulty','medium')}",
        ]
        options = q.get('options') or []
        if options:
            q_lines.append("Options: " + "; ".join(options))

        q_images = q.get('images') or []

        # Add student submission info with OCR results
        q_lines.append("")
        q_lines.append("=== STUDENT SUBMISSION ANALYSIS ===")
        if answer_text and canvas_data:
            q_lines.append(f"TYPED TEXT: {answer_text}")
            q_lines.append(f"OCR-EXTRACTED CONTENT: {extracted_text}")
            q_lines.append("CANVAS IMAGE: See below for visual verification and additional content")
        elif answer_text:
            q_lines.append(f"TYPED TEXT: {answer_text}")
            q_lines.append("OCR-EXTRACTED CONTENT: No canvas provided")
            q_lines.append("CANVAS IMAGE: None")
        elif canvas_data:
            q_lines.append("TYPED TEXT: None")
            q_lines.append(f"OCR-EXTRACTED CONTENT: {extracted_text}")
            q_lines.append("CANVAS IMAGE: See below - contains student's handwritten solution")
        else:
            q_lines.append("TYPED TEXT: None")
            q_lines.append("OCR-EXTRACTED CONTENT: None")
            q_lines.append("CANVAS IMAGE: None")

        q_lines.append("")
        q_lines.append("EVALUATION TASK:")
        q_lines.append("1) Solve the question yourself to determine the correct answer")
        q_lines.append("2) Analyze ALL student content (typed text + OCR-extracted text + canvas image)")
        q_lines.append("3) Extract equations, formulas, calculations, and explanations from student work")
        q_lines.append("4) Evaluate correctness of approach, calculations, and final answer")
        q_lines.append("5) Provide detailed feedback on methodology and areas for improvement")

        user_content = [{'type': 'text', 'text': "\n".join(q_lines)}]

        # Add question images (physics diagrams) FIRST so AI can solve the problem
        q_images = q.get('images') or []
        for i, img in enumerate(q_images):
            if img.get('base64Data'):
                user_content.append({
                    'type': 'image_url',
                    'image_url': {'url': img['base64Data']}
                })
            elif img.get('path'):
                # Convert relative path to full URL if needed
                img_url = f"http://localhost:5001/api/{img['path']}" if not img['path'].startswith('http') else img['path']
                user_content.append({
                    'type': 'image_url',
                    'image_url': {'url': img_url}
                })

        # Add student's canvas drawing LAST for comparison
        if canvas_data:
            logger.info(f"Adding canvas data to evaluation request (length: {len(canvas_data)} chars)")
            logger.info(f"Canvas data starts with: {canvas_data[:100]}...")
            user_content.append({'type': 'image_url', 'image_url': {'url': canvas_data}})
        else:
            logger.warning("No canvas data provided for evaluation")

        result = openai_service.chat_completion(
            messages=[{'role': 'user', 'content': user_content}],
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=500,
        )

        if not result.get('success'):
            logger.error('OpenAI evaluation error: %s', result.get('error'))
            return jsonify({
                'success': False,
                'error': 'Failed to evaluate submission',
                'details': result.get('error')
            }), 502

        raw = result.get('response') or ''

        # Parse AI response as JSON with improved error handling
        import json
        import re
        
        def clean_and_parse_json(response_text: str) -> Optional[Dict[str, Any]]:
            """Super simple JSON parsing for single-line responses"""
            logger.info(f"AI response: {response_text}")
            
            # Clean the response
            cleaned = response_text.strip()
            
            # Remove any markdown or extra text
            if '```' in cleaned:
                cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
            
            # Find JSON boundaries
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            
            if start == -1 or end == -1 or end <= start:
                logger.error(f"No JSON found in response: {cleaned}")
                return None
            
            # Extract clean JSON
            json_str = cleaned[start:end + 1]
            
            # Simple parsing attempts
            attempts = [
                json_str,                                    # Direct
                json_str.strip(),                           # Stripped
                ' '.join(json_str.split()),                 # Normalized whitespace
                json_str.replace('\n', ' ').replace('\r', ' ').strip()  # No newlines
            ]
            
            for i, candidate in enumerate(attempts):
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict) and 'correct' in result:
                        logger.info(f"✅ JSON parsed successfully (attempt {i+1})")
                        return result
                except json.JSONDecodeError as e:
                    logger.debug(f"Attempt {i+1} failed: {e}")
            
            logger.error(f"❌ All parsing failed. JSON candidate: {json_str}")
            return None

        parsed = clean_and_parse_json(raw)
        
        if parsed:
            # Validate and normalize the evaluation
            evaluation = {
                'correct': bool(parsed.get('correct', False)),
                'score': float(parsed.get('score', 1 if parsed.get('correct', False) else 0)),
                'extractedAnswer': str(parsed.get('extractedAnswer', 'Not found')),
                'feedback': str(parsed.get('feedback', 'No feedback provided')),
                'reasoning': str(parsed.get('reasoning', 'No reasoning provided'))
            }

            # Enhanced validation for comprehensive solutions
            reasoning_text = evaluation.get('reasoning') or ''
            reasoning_lower = reasoning_text.lower()

            # For comprehensive analysis, we trust the AI's evaluation more
            # but still validate against known correct answers when available
            extracted_content = evaluation.get('extractedContent', '')
            extracted_answer = evaluation.get('extractedAnswer', '')

            # If we have a known correct answer, validate against it
            if correct_answer:
                # For multiple choice questions, check if the answer matches
                if correct_answer.upper() in ['A', 'B', 'C', 'D']:
                    expected_letter = correct_answer.upper()
                    evaluation['correctAnswer'] = expected_letter

                    # Check if student's answer matches the expected choice
                    if extracted_answer and expected_letter in extracted_answer.upper():
                        if not evaluation.get('correct', False):
                            logger.info('Correcting AI evaluation based on answer matching')
                            evaluation['correct'] = True
                            evaluation['score'] = 1.0
                            evaluation['feedback'] = f"Excellent! You correctly chose option {expected_letter}."
                            evaluation['reasoning'] = f"The correct answer is {expected_letter}. Student selected the right option."
                    elif extracted_answer and expected_letter not in extracted_answer.upper():
                        if evaluation.get('correct', False):
                            logger.info('Correcting AI evaluation - student chose wrong option')
                            evaluation['correct'] = False
                            evaluation['score'] = 0.0
                            evaluation['feedback'] = f"Not quite right. You chose {extracted_answer}, but the correct answer is {expected_letter}."
                            evaluation['reasoning'] = f"The correct answer is {expected_letter}. Student selected {extracted_answer}."
                else:
                    # For open-ended questions, the AI's evaluation is trusted
                    evaluation['correctAnswer'] = correct_answer
            else:
                evaluation['correctAnswer'] = 'Not specified'

            # Ensure we have extracted content from OCR
            if not extracted_content and extracted_text:
                evaluation['extractedContent'] = extracted_text

            logger.info(f"Comprehensive evaluation completed: correct={evaluation.get('correct')}, score={evaluation.get('score')}, answer='{extracted_answer}'")

            logger.info('Final evaluation: correct=%s, extracted=%s, correct_answer=%s', evaluation['correct'], evaluation.get('extractedAnswer'), evaluation.get('correctAnswer'))
        else:
            # Enhanced fallback evaluation with OCR content
            logger.error(f'JSON parsing failed completely. Raw response: {raw[:500]}...')

            # Use OCR-extracted content as the primary source
            if extracted_text and extracted_text.strip():
                evaluation = {
                    'correct': False,  # Default to False, let manual review determine
                    'score': 0.5,  # Partial credit for attempting
                    'extractedAnswer': extracted_text[:100] + ('...' if len(extracted_text) > 100 else ''),
                    'extractedContent': extracted_text,
                    'feedback': 'Evaluation completed with OCR extraction. Please review manually.',
                    'reasoning': f'JSON parsing failed. OCR extracted: {extracted_text[:200]}...',
                    'analysisType': 'ocr_fallback'
                }
            else:
                evaluation = {
                    'correct': False,
                    'score': 0,
                    'extractedAnswer': 'No content extracted',
                    'extractedContent': extracted_text or 'OCR failed',
                    'feedback': 'Could not extract content from submission.',
                    'reasoning': 'Both OCR extraction and AI evaluation failed.',
                    'analysisType': 'failed_extraction'
                }

        # Track student activity if authenticated
        try:
            claims = get_jwt()
            if claims and claims.get('user_type') == 'student':
                student_id = claims.get('student_id')
                if student_id:
                    score = evaluation.get('score', 0) * 100  # Convert to 0-100 scale
                    is_correct = evaluation.get('correct', False)

                    # Calculate time taken if provided
                    time_taken = payload.get('timeTaken', 0)

                    # Log the problem solving activity
                    StudentMetrics.log_problem_solving_activity(
                        student_id=student_id,
                        score=score,
                        duration_seconds=time_taken,
                        metadata={
                            'question_id': qid,
                            'question_type': 'practice',
                            'mode': 'hustle_mode',
                            'has_canvas': bool(canvas_data),
                            'has_text': bool(answer_text),
                            'extracted_answer': evaluation.get('extractedAnswer'),
                            'difficulty': q.get('difficulty', 'medium'),
                            'subject': q.get('subject', 'unknown')
                        }
                    )

                    # Also log as question attempt
                    StudentMetrics.log_question_attempt(
                        student_id=student_id,
                        question_id=qid,
                        score=score,
                        is_correct=is_correct,
                        metadata={
                            'question_type': 'practice',
                            'mode': 'hustle_mode',
                            'difficulty': q.get('difficulty', 'medium'),
                            'subject': q.get('subject', 'unknown'),
                            'time_taken': time_taken,
                            'has_canvas': bool(canvas_data),
                            'has_text': bool(answer_text)
                        }
                    )
        except Exception as tracking_error:
            # Don't fail the main request if tracking fails
            logger.warning(f"Failed to track practice activity: {str(tracking_error)}")

        return jsonify({'success': True, 'evaluation': evaluation, 'question': _shape_question_payload(q)})

    except Exception as e:
        logger.exception('Failed to evaluate submission')
        return jsonify({'success': False, 'error': str(e)}), 500
