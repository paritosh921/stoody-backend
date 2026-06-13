from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

QUESTION_TYPE_LABELS = {
    "objective",
    "mcq",
    "multiple_choice",
    "numerical",
    "integer",
    "subjective",
    "short_answer",
    "case_study",
    "assertion_reason",
    "diagram_based",
    "other",
}


def _safe_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _question_number(question: Dict[str, Any], fallback: int) -> int:
    for key in ("question_number", "extraction_order"):
        number = _safe_int(question.get(key))
        if number is not None:
            return number
    metadata = question.get("metadata") if isinstance(question.get("metadata"), dict) else {}
    number = _safe_int(metadata.get("question_number"))
    return number if number is not None else fallback


def _question_text(question: Dict[str, Any]) -> str:
    return str(question.get("text") or question.get("question_text") or "").strip()


def _normalise_question_type(value: Any, question: Optional[Dict[str, Any]] = None) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "objective": "objective",
        "mcq": "objective",
        "multiple_choice": "objective",
        "multiplechoice": "objective",
        "single_choice": "objective",
        "integer": "numerical",
        "numeric": "numerical",
        "numerical": "numerical",
        "subjective": "subjective",
        "descriptive": "subjective",
        "short_answer": "short_answer",
        "shortanswer": "short_answer",
        "case": "case_study",
        "case_study": "case_study",
        "assertion_reason": "assertion_reason",
        "assertion": "assertion_reason",
        "diagram": "diagram_based",
        "diagram_based": "diagram_based",
    }
    if raw in aliases:
        return aliases[raw]

    if question:
        options = question.get("options") or question.get("enhanced_options") or []
        if isinstance(options, list) and len(options) >= 2:
            return "objective"
        text = _question_text(question).lower()
        if any(token in text for token in ("calculate", "find the value", "evaluate", "numerical")):
            return "numerical"
    return "other"


def _expected_marks_for_question(
    question_number: int,
    marking_scheme: List[Dict[str, Any]],
    fallback: Optional[float] = None,
) -> Optional[float]:
    for item in marking_scheme or []:
        from_q = _safe_int(item.get("from") or item.get("from_"))
        to_q = _safe_int(item.get("to"))
        marks = _safe_float(item.get("marks"))
        if from_q is not None and to_q is not None and marks is not None and from_q <= question_number <= to_q:
            return marks
    return fallback


def _fallback_subtopic(question: Dict[str, Any]) -> str:
    text = _question_text(question)
    if not text:
        return "General"
    lowered = text.lower()
    phrase_patterns = [
        (r"\bdimensional\s+analysis\b", "Dimensional Analysis"),
        (r"\bsignificant\s+figures?\b", "Significant Figures"),
        (r"\b(error|errors)\b", "Error Analysis"),
        (r"\bprojectile\b", "Projectile Motion"),
        (r"\bfree\s+body\b", "Free Body Diagram"),
        (r"\bnewton'?s?\s+(law|laws)\b", "Newton's Laws"),
        (r"\bkinematics?\b", "Kinematics"),
        (r"\bwork\s+energy\b|\benergy\b", "Work and Energy"),
        (r"\bcurrent\s+electricity\b", "Current Electricity"),
        (r"\bmagnetism\b", "Magnetism"),
        (r"\bprobability\b", "Probability"),
        (r"\btrigonometry\b", "Trigonometry"),
        (r"\bdifferentiation\b|\bderivative\b", "Differentiation"),
        (r"\bintegration\b|\bintegral\b", "Integration"),
    ]
    for pattern, label in phrase_patterns:
        if re.search(pattern, lowered):
            return label
    return "General"


def _parse_llm_json(text: str) -> Dict[str, Any]:
    clean = str(text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        parsed = json.loads(clean)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(clean[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def _classify_with_llm(
    questions: List[Dict[str, Any]],
    *,
    subject: Optional[str],
    standard: Optional[str],
) -> Dict[int, Dict[str, Any]]:
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not groq_key and not openai_key:
        return {}

    try:
        from openai import AsyncOpenAI
    except Exception as exc:
        logger.warning("OpenAI client unavailable for tally question map classification: %s", exc)
        return {}

    if groq_key:
        client = AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    else:
        client = AsyncOpenAI(api_key=openai_key)
        model = os.getenv("OCR_FALLBACK_MODEL", "gpt-5-mini")

    compact_questions = []
    for index, question in enumerate(questions[:80], start=1):
        compact_questions.append(
            {
                "question_number": _question_number(question, index),
                "text": _question_text(question)[:1400],
                "existing_question_type": question.get("question_type"),
            }
        )

    prompt = (
        "Classify each exam question into the most specific sub-topic visible from the question text. "
        "Do not return broad chapter names when a narrower concept is clear. "
        "Use concise 2-5 word sub-topic labels.\n\n"
        f"Subject: {subject or 'General'}\n"
        f"Class/standard: {standard or 'Unknown'}\n\n"
        "Return ONLY JSON in this shape:\n"
        '{"items":[{"question_number":1,"sub_topic":"Dimensional Analysis",'
        '"question_type":"numerical","confidence":0.0}]}\n\n'
        "Allowed question_type values: objective, numerical, subjective, short_answer, "
        "case_study, assertion_reason, diagram_based, other.\n\n"
        f"Questions:\n{json.dumps(compact_questions, ensure_ascii=False)}"
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
        )
        content = response.choices[0].message.content if response.choices else ""
        parsed = _parse_llm_json(content or "")
    except Exception as exc:
        logger.warning("Tally question map AI classification failed: %s", exc)
        return {}

    by_number: Dict[int, Dict[str, Any]] = {}
    for item in parsed.get("items") or []:
        if not isinstance(item, dict):
            continue
        number = _safe_int(item.get("question_number"))
        if number is None:
            continue
        sub_topic = str(item.get("sub_topic") or "").strip()[:80]
        if not sub_topic:
            continue
        confidence = item.get("confidence")
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.0
        by_number[number] = {
            "sub_topic": sub_topic,
            "question_type": _normalise_question_type(item.get("question_type")),
            "confidence": confidence_value,
            "source": "ai",
        }
    return by_number


async def build_tally_question_map(
    *,
    tally_document_id: str,
    source_document_id: str,
    questions: List[Dict[str, Any]],
    subject: Optional[str] = None,
    standard: Optional[str] = None,
    course_plan: Optional[str] = None,
    marking_scheme: Optional[List[Dict[str, Any]]] = None,
    fallback_max_marks: Optional[float] = None,
    generated_by: Optional[str] = None,
) -> Dict[str, Any]:
    ordered_questions = sorted(
        questions or [],
        key=lambda q: (_question_number(q, 10**6), str(q.get("id") or "")),
    )
    classifications = await _classify_with_llm(
        ordered_questions,
        subject=subject,
        standard=standard,
    )

    items: List[Dict[str, Any]] = []
    warnings: List[str] = []
    seen_numbers = set()
    for index, question in enumerate(ordered_questions, start=1):
        number = _question_number(question, index)
        if number in seen_numbers:
            warnings.append(f"Duplicate question number Q{number} found in source OCR.")
        seen_numbers.add(number)

        classification = classifications.get(number) or {}
        sub_topic = classification.get("sub_topic") or _fallback_subtopic(question)
        question_type = classification.get("question_type") or _normalise_question_type(
            question.get("question_type"),
            question,
        )
        text = _question_text(question)
        max_marks = _expected_marks_for_question(
            number,
            marking_scheme or [],
            fallback_max_marks,
        )
        if max_marks is None:
            max_marks = _safe_float(question.get("points"))

        items.append(
            {
                "question_number": number,
                "question_id": str(question.get("id") or question.get("_id") or ""),
                "question_text": text,
                "question_text_preview": text[:240],
                "sub_topic": str(sub_topic or "General")[:80],
                "question_type": _normalise_question_type(question_type, question),
                "max_marks": max_marks,
                "confidence": float(classification.get("confidence") or 0.0),
                "source": classification.get("source") or "fallback",
            }
        )

    now = datetime.utcnow()
    status = "ready" if items else "empty"
    if not items:
        warnings.append("No extracted questions were available for the selected source paper.")

    return {
        "_id": tally_document_id,
        "tally_document_id": tally_document_id,
        "source_document_id": source_document_id,
        "status": status,
        "items": items,
        "warnings": warnings,
        "subject": subject,
        "standard": standard,
        "course_plan": course_plan,
        "generated_by": generated_by,
        "generated_at": now,
        "updated_at": now,
    }
