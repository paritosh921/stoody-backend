import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert educational AI that evaluates student answers. "
    "Analyze the student's work and return a JSON object with exactly these fields:\n"
    '- "score": a number from 0.0 to 1.0 representing correctness\n'
    '- "is_correct": true if the answer is fully correct, false otherwise\n'
    '- "student_answer": a concise summary of what the student wrote\n'
    '- "work_shown": a description of the work/steps the student showed\n'
    '- "what_went_wrong": if incorrect, explain what went wrong; null if correct\n'
    '- "correct_solution": the correct solution/explanation\n'
    "Return ONLY valid JSON, no markdown, no explanation."
)

SUBMISSIONS_COLLECTION = "online_class_submissions"


def _build_analysis_prompt(lock: Dict[str, Any], answer_text: Optional[str]) -> str:
    parts = []
    q_text = lock.get("question_text")
    if q_text:
        parts.append(f"Question: {q_text}")
    q_bbox = lock.get("question_bbox")
    if q_bbox:
        parts.append(f"Question region: {q_bbox}")
    if answer_text:
        parts.append(f"Student answer text: {answer_text}")
    parts.append(
        "Evaluate this student answer. If canvas images are attached, "
        "examine them for handwritten work. Return your analysis as JSON."
    )
    return "\n\n".join(parts)


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


def _clamp_score(value) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, num))


def _success_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.utcnow()
    return {
        "analysis_status": "completed",
        "score": _clamp_score(parsed.get("score", 0)),
        "is_correct": _normalize_bool(parsed.get("is_correct")),
        "student_answer": _optional_text(parsed.get("student_answer")),
        "work_shown": _optional_text(parsed.get("work_shown")),
        "what_went_wrong": parsed.get("what_went_wrong"),
        "correct_solution": _optional_text(parsed.get("correct_solution")),
        "analysis_completed_at": now,
        "analysis_error": None,
        "analysis_failed_at": None,
        "updated_at": now,
    }


def _failure_fields(error: Exception) -> Dict[str, Any]:
    now = datetime.utcnow()
    return {
        "analysis_status": "failed",
        "analysis_error": str(error)[:200],
        "analysis_failed_at": now,
        "score": None,
        "is_correct": None,
        "student_answer": None,
        "work_shown": None,
        "what_went_wrong": None,
        "correct_solution": None,
        "analysis_completed_at": None,
        "updated_at": now,
    }


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


async def run_submission_analysis(
    db,
    current_user: Dict[str, Any],
    lock: Dict[str, Any],
    submission: Dict[str, Any],
) -> Dict[str, Any]:
    from api.v1.practice_async import (
        _gate_text_call,
        _gate_vision_call,
        robust_json_parse,
    )

    canvas_pages = submission.get("canvas_pages") or []
    answer_text = submission.get("answer_text")
    prompt = _build_analysis_prompt(lock, answer_text)
    submission_id = submission["submission_id"]

    try:
        if canvas_pages:
            result = await _gate_vision_call(
                db=db,
                current_user=current_user,
                images=canvas_pages,
                prompt=prompt,
                system_prompt=ANALYSIS_SYSTEM_PROMPT,
                max_tokens=800,
                temperature=0.2,
            )
        else:
            result = await _gate_text_call(
                db=db,
                current_user=current_user,
                prompt=prompt,
                system_prompt=ANALYSIS_SYSTEM_PROMPT,
                max_tokens=800,
                temperature=0.2,
            )

        raw_response = result.get("response", "") if result else ""
        parsed = robust_json_parse(raw_response)

        if parsed is None:
            raise ValueError("LLM response could not be parsed as JSON")

        fields = _success_fields(parsed)

    except Exception as exc:
        logger.warning(
            "Analysis failed for submission %s: %s", submission_id, exc
        )
        fields = _failure_fields(exc)

    await db.mongo_update_one(
        SUBMISSIONS_COLLECTION,
        {"submission_id": submission_id},
        {"$set": fields},
    )
    submission.update(fields)
    return submission
