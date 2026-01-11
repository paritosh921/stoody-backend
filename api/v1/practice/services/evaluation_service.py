"""
Practice Module - Evaluation Service
Core logic for evaluating student submissions using AI
"""

import json
import re
import ast
import logging
from typing import Dict, Any, List, Optional

from core.database import DatabaseManager
from ..utils import (
    load_question_doc,
    options_text_from_question,
    figure_images_base64,
    option_images_base64,
)
from ..models import EvaluateRequest

logger = logging.getLogger(__name__)


async def evaluate_student_submission(
    payload: EvaluateRequest,
    current_user: Dict[str, Any],
    db: DatabaseManager
) -> Dict[str, Any]:
    """Evaluate student's submission (canvas image and/or text) for a question with AI tutor feedback.

    ENHANCED VERSION: Uses multi-stage OCR pipeline for reliable handwriting recognition:
    1. Image enhancement (upscaling, contrast, stroke thickening)
    2. Dedicated OCR extraction with confidence scoring
    3. Fallback strategies for low-confidence results
    4. Improved prompting for handwriting analysis

    Args:
        payload: Evaluation request with question ID, answer text, canvas data
        current_user: Current authenticated user
        db: Database manager

    Returns:
        Dict with evaluation data (correct, score, extractedAnswer, feedback, reasoning, etc.)
    """
    qid = payload.questionId
    answer_text = (payload.answerText or "").strip()
    canvas_data = payload.canvasData

    # Detect if user is B2C
    user_type = current_user.get("user_type", "")
    is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

    logger.info(f"Evaluating submission for Q:{qid}, user_type:{user_type}, is_b2c:{is_b2c}")

    # Normalize canvas data header
    if canvas_data and not canvas_data.startswith("data:image"):
        canvas_data = f"data:image/png;base64,{canvas_data}"

    # Fetch question
    question_doc = await load_question_doc(db, qid, is_b2c=is_b2c)
    if not question_doc:
        return {"error": "Question not found", "status_code": 404}

    # Pull correct answer
    ca_primary = question_doc.get("correctAnswer")
    ca_alt = question_doc.get("correct_answer")
    correct_answer = str((ca_primary if ca_primary is not None else (ca_alt if ca_alt is not None else ""))).strip()

    # Extract question text and options
    question_text = str(question_doc.get("text", ""))
    options_text = options_text_from_question(question_doc)

    # Determine if this is MCQ
    is_mcq = bool(options_text)

    # Initialize AI service
    from services.async_openai_service import AsyncOpenAIService
    ai = AsyncOpenAIService()

    # Prepare images
    question_images = await figure_images_base64(question_doc, db, is_b2c)
    option_images_data = await option_images_base64(question_doc, db, is_b2c)
    option_images = [oi["image"] for oi in option_images_data]
    all_question_images = question_images + option_images

    logger.info(f"Total question images: {len(all_question_images)} (figures: {len(question_images)}, option images: {len(option_images)})")

    # Student Canvas Images - Enhanced Processing
    student_images_raw = []
    if payload.canvasPages and len(payload.canvasPages) > 0:
        student_images_raw = payload.canvasPages
    elif canvas_data:
        student_images_raw = [canvas_data]

    # Stage 1: Canvas OCR Extraction
    ocr_extracted_text = ""
    ocr_confidence = 0.0
    student_images = student_images_raw

    if student_images_raw:
        try:
            from services.canvas_ocr_service import get_canvas_ocr_service
            from utils.image_processor import enhance_canvas_images_batch

            logger.info(f"Enhancing {len(student_images_raw)} canvas images...")
            enhanced_student_images = enhance_canvas_images_batch(student_images_raw, target_width=1500)

            ocr_service = get_canvas_ocr_service()
            ocr_result = await ocr_service.extract_text_from_canvas(
                canvas_pages=enhanced_student_images,
                question_context=question_text,
                options_context=options_text if is_mcq else None,
                is_mcq=is_mcq
            )

            ocr_extracted_text = ocr_result.extracted_text
            ocr_confidence = ocr_result.confidence

            logger.info(f"OCR Extraction: '{ocr_extracted_text}' (confidence: {ocr_confidence:.2f}, method: {ocr_result.method})")
            student_images = enhanced_student_images

        except ImportError as ie:
            logger.warning(f"Canvas OCR service not available: {ie}. Using raw images.")
        except Exception as ocr_err:
            logger.error(f"OCR extraction failed: {ocr_err}. Continuing with raw images.")

    # Stage 2: Combined Evaluation with Enhanced Prompt
    combined_answer = answer_text
    if ocr_extracted_text and ocr_confidence > 0.3:
        if answer_text:
            combined_answer = f"{answer_text} (Canvas OCR: {ocr_extracted_text})"
        else:
            combined_answer = ocr_extracted_text

    # Combine all images - student images first
    all_images = student_images + all_question_images
    num_q_images = len(all_question_images)
    num_fig_images = len(question_images)
    num_opt_images = len(option_images)
    num_s_images = len(student_images)

    logger.info(f"Question {qid}: text_len={len(question_text)}, correct='{correct_answer}', is_mcq={is_mcq}")
    logger.info(f"Image breakdown: student={num_s_images}, figures={num_fig_images}, options={num_opt_images}")

    has_correct_answer = bool(correct_answer and correct_answer.strip())

    # Build evaluation prompt
    prompt = _build_evaluation_prompt(
        question_text=question_text,
        options_text=options_text,
        correct_answer=correct_answer,
        has_correct_answer=has_correct_answer,
        is_mcq=is_mcq,
        answer_text=answer_text,
        ocr_extracted_text=ocr_extracted_text,
        ocr_confidence=ocr_confidence,
        num_s_images=num_s_images,
        num_q_images=num_q_images,
        num_fig_images=num_fig_images,
        num_opt_images=num_opt_images
    )

    # Build system prompt
    system_prompt = _build_system_prompt(has_correct_answer)

    logger.info(f"Sending evaluation to LLM for Q:{qid}. Images: {len(all_images)}. OCR: '{ocr_extracted_text}'")

    # Call LLM
    if all_images:
        response = await ai.analyze_images_and_text_async(
            all_images,
            prompt,
            max_tokens=1200,
            system_prompt=system_prompt
        )
    else:
        response = await ai.chat_completion_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )

    raw_response = (response.get("response") or "").strip()
    logger.info(f"LLM Raw Response (first 500 chars): {raw_response[:500]}")

    # Parse response
    evaluation_data = _parse_evaluation_response(
        raw_response=raw_response,
        correct_answer=correct_answer,
        has_correct_answer=has_correct_answer,
        ocr_extracted_text=ocr_extracted_text,
        ocr_confidence=ocr_confidence
    )

    logger.info(f"Evaluation complete for Q:{qid}. Correct: {evaluation_data['correct']}, Extracted: '{evaluation_data['extractedAnswer']}'")

    return evaluation_data


def _build_evaluation_prompt(
    question_text: str,
    options_text: str,
    correct_answer: str,
    has_correct_answer: bool,
    is_mcq: bool,
    answer_text: str,
    ocr_extracted_text: str,
    ocr_confidence: float,
    num_s_images: int,
    num_q_images: int,
    num_fig_images: int,
    num_opt_images: int
) -> str:
    """Build the evaluation prompt for the LLM."""
    prompt = (
        "You are an expert tutor evaluating a student's handwritten answer. "
        "Your task is to determine if their answer is CORRECT or INCORRECT.\n\n"
    )

    # Add image guide
    if num_s_images > 0 and num_q_images > 0:
        prompt += f"IMAGE GUIDE: You will see {num_s_images + num_q_images} images total.\n"
        prompt += f"  - Images 1 to {num_s_images}: STUDENT'S HANDWRITTEN WORK (analyze these carefully)\n"
        img_offset = num_s_images + 1
        if num_fig_images > 0:
            prompt += f"  - Images {img_offset} to {img_offset + num_fig_images - 1}: QUESTION DIAGRAMS/FIGURES\n"
            img_offset += num_fig_images
        if num_opt_images > 0:
            prompt += f"  - Images {img_offset} to {img_offset + num_opt_images - 1}: MCQ OPTION IMAGES\n"
        prompt += "\nIMPORTANT: You MUST examine the QUESTION DIAGRAMS to understand the problem correctly!\n\n"
    elif num_s_images > 0:
        prompt += f"IMAGE GUIDE: You will see {num_s_images} image(s) of the STUDENT'S HANDWRITTEN WORK.\n\n"
    elif num_q_images > 0:
        prompt += f"IMAGE GUIDE: You will see {num_q_images} image(s) of QUESTION DIAGRAMS.\n\n"

    # Question section
    prompt += "=" * 43 + "\n"
    prompt += "QUESTION:\n"
    prompt += "=" * 43 + "\n"
    prompt += f"{question_text}\n\n"

    # Options if MCQ
    if options_text:
        prompt += "OPTIONS:\n"
        prompt += f"{options_text}\n\n"

    # Correct answer section
    prompt += "=" * 43 + "\n"
    prompt += "CORRECT ANSWER:\n"
    prompt += "=" * 43 + "\n"

    if has_correct_answer:
        prompt += f"{correct_answer}\n\n"
    else:
        prompt += "NO CORRECT ANSWER PROVIDED BY ADMIN\n\n"
        prompt += "YOU MUST SOLVE THIS QUESTION YOURSELF:\n"
        if is_mcq:
            prompt += "1. Read the question carefully and analyze each option.\n"
            prompt += "2. Determine which option (A, B, C, or D) is correct.\n"
            prompt += "3. Use this as your reference to evaluate the student's answer.\n\n"
        else:
            prompt += "1. Read the question carefully.\n"
            prompt += "2. Solve it step-by-step to find the correct answer.\n"
            prompt += "3. Use your solution as the reference to evaluate the student's answer.\n\n"

    # Student submission section
    prompt += "=" * 43 + "\n"
    prompt += "STUDENT'S SUBMISSION:\n"
    prompt += "=" * 43 + "\n"

    if answer_text:
        prompt += f"Typed Answer: {answer_text}\n"
    if ocr_extracted_text:
        conf_label = "HIGH" if ocr_confidence > 0.7 else ("MEDIUM" if ocr_confidence > 0.4 else "LOW")
        prompt += f"OCR Detected Text ({conf_label} confidence): {ocr_extracted_text}\n"
    if num_s_images > 0:
        prompt += f"Handwritten Canvas: {num_s_images} page(s) submitted - EXAMINE CAREFULLY.\n"
    if not answer_text and not ocr_extracted_text and num_s_images == 0:
        prompt += "(No answer submitted)\n"
    prompt += "\n"

    # Evaluation rules
    if is_mcq:
        prompt += "EVALUATION RULES (Multiple Choice Question):\n"
        prompt += "1. Look for a LETTER (A, B, C, D, etc.) in the student's handwriting - this is their FINAL answer.\n"
        prompt += "2. IMPORTANT: Students often show their WORK before writing their final letter.\n"
        prompt += "3. If you see calculations/work but NO final letter, evaluate if their work leads to the correct answer.\n"
        if has_correct_answer:
            prompt += f"4. The student is CORRECT if their final letter matches '{correct_answer}'\n"
        prompt += "\n"
    else:
        prompt += "EVALUATION RULES (Subjective Question):\n"
        prompt += "1. Transcribe all text, numbers, and equations from the student's handwriting.\n"
        prompt += "2. Look for their FINAL answer (often boxed, circled, or underlined).\n"
        if has_correct_answer:
            prompt += f"3. Compare their answer to the correct answer: '{correct_answer[:100]}'\n"
        prompt += "4. Partial credit (score 0.5): if they're on the right track but made a small error.\n\n"

    # JSON output format
    prompt += "=" * 43 + "\n"
    prompt += "RETURN THIS JSON EXACTLY:\n"
    prompt += "=" * 43 + "\n"
    prompt += "{\n"
    prompt += '  "extracted_answer": "The student\'s FINAL answer",\n'
    prompt += '  "work_shown": "Summary of any calculations or work the student showed",\n'
    prompt += '  "is_correct": true or false,\n'
    prompt += '  "score": 0.0 to 1.0,\n'
    if not has_correct_answer:
        prompt += '  "solved_answer": "The correct answer YOU calculated",\n'
    prompt += '  "what_went_wrong": "Explanation of mistake if incorrect",\n'
    prompt += '  "correct_solution": "Step-by-step solution if incorrect",\n'
    prompt += '  "feedback": "Encouraging feedback for the student",\n'
    prompt += '  "reasoning": "Your evaluation logic"\n'
    prompt += "}\n\n"
    prompt += "IMPORTANT: Output ONLY the JSON, no markdown formatting.\n"

    return prompt


def _build_system_prompt(has_correct_answer: bool) -> str:
    """Build the system prompt for evaluation."""
    if has_correct_answer:
        return (
            "You are an expert answer evaluator specializing in reading handwritten student work. "
            "CRITICAL: Your ONLY job is to determine if the student's answer is CORRECT or INCORRECT. "
            "You MUST compare the student's answer to the CORRECT ANSWER provided. "
            "For MCQ, a single letter (A/B/C/D) is the answer - focus on identifying that letter. "
            "Always output ONLY valid JSON without markdown code blocks. "
            "Be generous in interpreting messy handwriting but strict in evaluating correctness."
        )
    else:
        return (
            "You are an expert tutor who can both SOLVE questions AND evaluate student answers. "
            "CRITICAL: Since NO CORRECT ANSWER was provided, you MUST first SOLVE the question yourself. "
            "1. First, solve the question to determine the correct answer. "
            "2. Then, read and interpret the student's handwritten work. "
            "3. Compare the student's answer to YOUR solution. "
            "4. Include your 'solved_answer' in the JSON response. "
            "Always output ONLY valid JSON without markdown code blocks."
        )


def _parse_evaluation_response(
    raw_response: str,
    correct_answer: str,
    has_correct_answer: bool,
    ocr_extracted_text: str,
    ocr_confidence: float
) -> Dict[str, Any]:
    """Parse the LLM evaluation response into structured data."""
    evaluation_data = {
        "correct": False,
        "score": 0.0,
        "extractedAnswer": "",
        "feedback": "",
        "reasoning": "",
        "answerSource": "ai_eval",
        "ocrConfidence": ocr_confidence,
        "ocrExtractedText": ocr_extracted_text,
        "correctAnswer": correct_answer
    }

    try:
        parsed = None

        # Pre-process: Remove markdown code blocks
        clean_response = raw_response
        if "```json" in raw_response:
            clean_response = re.sub(r"```json\s*", "", raw_response)
            clean_response = re.sub(r"```\s*", "", clean_response).strip()
        elif "```" in raw_response:
            clean_response = re.sub(r"```\s*", "", raw_response).strip()

        # Attempt 1: Direct JSON parse
        try:
            parsed = json.loads(clean_response)
        except Exception:
            pass

        # Attempt 2: Regex extraction
        if not parsed:
            m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean_response, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    try:
                        parsed = ast.literal_eval(m.group(0))
                    except Exception:
                        pass

        # Attempt 3: Simple brace extraction
        if not parsed:
            m = re.search(r"\{[^{}]*\}", clean_response, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    pass

        if parsed and isinstance(parsed, dict):
            # Parse is_correct
            is_correct_val = parsed.get("is_correct", parsed.get("correct", False))
            if isinstance(is_correct_val, bool):
                evaluation_data["correct"] = is_correct_val
            elif isinstance(is_correct_val, str):
                evaluation_data["correct"] = is_correct_val.lower() in ("true", "yes", "correct", "1")
            else:
                evaluation_data["correct"] = bool(is_correct_val)

            # Parse score
            llm_score = parsed.get("score")
            if llm_score is not None:
                try:
                    evaluation_data["score"] = float(llm_score)
                except (ValueError, TypeError):
                    evaluation_data["score"] = 1.0 if evaluation_data["correct"] else 0.0
            else:
                evaluation_data["score"] = 1.0 if evaluation_data["correct"] else 0.0

            evaluation_data["extractedAnswer"] = str(parsed.get("extracted_answer", "")).strip()
            evaluation_data["feedback"] = str(parsed.get("feedback", "")).strip()
            evaluation_data["reasoning"] = str(parsed.get("reasoning", "")).strip()

            # Extract additional fields
            if parsed.get("work_shown"):
                evaluation_data["workShown"] = str(parsed["work_shown"]).strip()
            if parsed.get("what_went_wrong"):
                evaluation_data["whatWentWrong"] = str(parsed["what_went_wrong"]).strip()
            if parsed.get("correct_solution"):
                evaluation_data["correctSolution"] = str(parsed["correct_solution"]).strip()

            # Handle solved_answer
            solved_answer = parsed.get("solved_answer", "")
            if solved_answer and not has_correct_answer:
                evaluation_data["correctAnswer"] = str(solved_answer).strip()
                evaluation_data["answerSource"] = "llm_solved"

            # Fallback to OCR if LLM didn't extract
            if not evaluation_data["extractedAnswer"] and ocr_extracted_text:
                evaluation_data["extractedAnswer"] = ocr_extracted_text
                evaluation_data["answerSource"] = "ocr_extraction"

        else:
            logger.warning(f"Could not parse JSON from LLM response. Raw: {raw_response[:200]}")
            evaluation_data["feedback"] = raw_response
            evaluation_data["reasoning"] = "Could not parse JSON from LLM response."
            if ocr_extracted_text:
                evaluation_data["extractedAnswer"] = ocr_extracted_text
                evaluation_data["answerSource"] = "ocr_fallback"

    except Exception as parse_err:
        logger.error(f"Failed to parse LLM evaluation JSON: {parse_err}")
        evaluation_data["feedback"] = raw_response
        evaluation_data["reasoning"] = f"JSON parse error: {parse_err}"
        if ocr_extracted_text:
            evaluation_data["extractedAnswer"] = ocr_extracted_text
            evaluation_data["answerSource"] = "ocr_fallback"

    if not evaluation_data["feedback"]:
        evaluation_data["feedback"] = raw_response

    return evaluation_data


async def grade_student_submission(
    payload: EvaluateRequest,
    current_user: Dict[str, Any],
    db: DatabaseManager
) -> Dict[str, Any]:
    """Comprehensive evaluation of student submissions using LLM analysis.

    Supports multiple choice, written solutions, mathematical derivations,
    diagrams, definitions, and conceptual answers.

    Args:
        payload: Evaluation request
        current_user: Current authenticated user
        db: Database manager

    Returns:
        Dict with evaluation data
    """
    from services.async_openai_service import AsyncOpenAIService
    ai = AsyncOpenAIService()

    qid = payload.questionId
    answer_text = (payload.answerText or "").strip()
    canvas_data = payload.canvasData
    if canvas_data and not canvas_data.startswith("data:image"):
        canvas_data = f"data:image/png;base64,{canvas_data}"

    # Get admin_id for data isolation
    from api.v1.questions_async import get_admin_id_from_user
    admin_id = get_admin_id_from_user(current_user)

    # Get question from admin's collection
    from services.question_service import QuestionService
    question_service = QuestionService(admin_id)
    question_obj = question_service.get_question(qid)

    if not question_obj:
        return {"error": "Question not found", "status_code": 404}

    q = question_obj.to_dict()

    ca_primary = q.get("correctAnswer")
    ca_alt = q.get("correct_answer")
    correct_answer = str((ca_primary if ca_primary is not None else (ca_alt if ca_alt is not None else ""))).strip()

    # Build system prompt for comprehensive evaluation
    system_prompt = _build_comprehensive_system_prompt()

    # Build context
    question_text = str(q.get("text", ""))
    subject = q.get("subject", "Unknown")
    difficulty = q.get("difficulty", "medium")

    context_parts = [
        f"Question: {question_text}",
        f"Subject: {subject}",
        f"Difficulty: {difficulty}",
    ]

    options_list = options_text_from_question(q)
    if options_list:
        context_parts.append(f"\nOptions:\n{options_list}")

    context_parts.append("\n=== STUDENT SUBMISSION ===")
    if answer_text and canvas_data:
        context_parts.append(f"TYPED TEXT: {answer_text}")
        context_parts.append("CANVAS: See image below")
    elif answer_text:
        context_parts.append(f"TYPED TEXT: {answer_text}")
        context_parts.append("CANVAS: None provided")
    elif canvas_data:
        context_parts.append("TYPED TEXT: None")
        context_parts.append("CANVAS: See image below")
    else:
        context_parts.append("TYPED TEXT: None")
        context_parts.append("CANVAS: None")

    context_parts.append("\n=== EVALUATION TASK ===")
    context_parts.append("1. Solve the question yourself")
    context_parts.append("2. Analyze ALL student content")
    context_parts.append("3. Extract equations, formulas, calculations")
    context_parts.append("4. Evaluate correctness")
    context_parts.append("5. Provide detailed feedback")

    prompt_text = "\n".join(context_parts)

    # Collect images
    images_for_eval: List[str] = []

    # Detect if user is B2C
    user_type = current_user.get("user_type", "")
    is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

    figures = await figure_images_base64(q, db, is_b2c)
    for fig in figures[:2]:
        if fig:
            images_for_eval.append(fig)
    student_pages = payload.canvasPages or ([] if not canvas_data else [canvas_data])
    images_for_eval.extend(student_pages)

    logger.info(f"Comprehensive evaluation: Q:{qid}, Subject:{subject}, Images:{len(images_for_eval)}")

    # Call LLM
    if images_for_eval:
        res = await ai.analyze_images_and_text_async(
            images_for_eval,
            prompt_text,
            max_tokens=800,
            system_prompt=system_prompt
        )
    else:
        res = await ai.evaluate_answer_async(
            question=question_text,
            student_answer=answer_text,
            correct_answer=correct_answer
        )

    raw_response = (res.get("response") or "").strip()
    logger.info(f"LLM evaluation response: {raw_response[:500]}...")

    # Parse response
    evaluation = _parse_grade_response(raw_response, answer_text)

    # Validate against correct answer for MCQ
    if correct_answer:
        evaluation = _validate_mcq_answer(evaluation, correct_answer)

    return evaluation


def _build_comprehensive_system_prompt() -> str:
    """Build system prompt for comprehensive grading."""
    return """You are an expert academic evaluator. Your job is to comprehensively analyze student solutions.

CRITICAL: Return ONLY a single line of valid JSON with NO extra text, NO newlines, NO formatting.

ANALYSIS CAPABILITIES:
- Understand handwritten equations, formulas, and mathematical expressions
- Read handwritten text, definitions, and explanations
- Analyze diagrams, graphs, and visual problem-solving steps
- Recognize scientific notation, chemical formulas, and technical symbols

EVALUATION PROCESS:
1. First, solve the given question yourself
2. Extract and interpret ALL content from the student's submission
3. Evaluate the student's approach, calculations, and final answer
4. Provide detailed feedback

REQUIRED JSON FORMAT (single line):
{"correct":false,"score":0.0,"extractedAnswer":"answer","feedback":"detailed feedback","reasoning":"evaluation logic"}

Return ONLY the JSON line. No other text."""


def _parse_grade_response(raw_response: str, answer_text: str) -> Dict[str, Any]:
    """Parse grading response from LLM."""
    evaluation = None
    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            evaluation = {
                "correct": bool(parsed.get("correct", False)),
                "score": float(parsed.get("score", 0.0)),
                "extractedAnswer": str(parsed.get("extractedAnswer", "Not found")),
                "feedback": str(parsed.get("feedback", "No feedback provided")),
                "reasoning": str(parsed.get("reasoning", "No reasoning provided"))
            }
        except Exception as parse_error:
            logger.warning(f"JSON parse failed: {parse_error}")

    if not evaluation:
        evaluation = {
            "correct": False,
            "score": 0.5,
            "extractedAnswer": answer_text or "See canvas",
            "feedback": raw_response[:500] if raw_response else "Unable to evaluate.",
            "reasoning": "Response format needs review."
        }

    return evaluation


def _validate_mcq_answer(evaluation: Dict[str, Any], correct_answer: str) -> Dict[str, Any]:
    """Validate MCQ answer against correct answer."""
    extracted = evaluation.get("extractedAnswer", "").strip().upper()
    expected = correct_answer.strip().upper()

    is_expected_mcq = len(expected) == 1 and expected.isalpha()
    is_extracted_single_letter = len(extracted) == 1 and extracted.isalpha()

    feedback_lower = evaluation.get("feedback", "").lower()
    is_written_explanation = any(phrase in feedback_lower for phrase in [
        "you wrote", "you explained", "don't know", "unclear", "not sure"
    ])

    if is_expected_mcq and is_extracted_single_letter and not is_written_explanation:
        is_match = (extracted == expected)
        if is_match and not evaluation["correct"]:
            evaluation["correct"] = True
            evaluation["score"] = 1.0
            evaluation["feedback"] = f"Excellent! You correctly chose option {expected}. " + evaluation.get("feedback", "")
        elif not is_match and evaluation["correct"]:
            evaluation["correct"] = False
            evaluation["score"] = 0.0
            evaluation["feedback"] = f"Not quite. You chose {extracted}, but the correct answer is {expected}. " + evaluation.get("feedback", "")

    return evaluation
