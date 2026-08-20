"""Immutable PCR marking-policy and criterion helpers.

The AI is allowed to *evaluate evidence*, not invent the teacher's marking
scheme.  Authoring stores a small structured rubric per question and the
document stores the strictness/temperature policy.  Finalisation validates and
freezes both into the paper snapshot; conducted sessions only read that
snapshot.
"""

from __future__ import annotations

import json
import math
import re
import hashlib
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional


POLICY_VERSION = "criterion-rubric-v2"
METHOD_POLICY_VERSION = "method-policy-v1"
ASSESSMENT_UNIT_VERSION = "assessment-unit-v1"
RESPONSE_SELECTION_VERSION = "response-selection-v1"
STRUCTURED_RUBRIC_MODE = "criterion_rubric_v1"
LEGACY_AI_MODE = "legacy_ai_v1"
MAX_AI_TEMPERATURE = 0.20

ANY_VALID_METHOD = "any_valid_method"
SPECIFIED_METHOD_REQUIRED = "specified_method_required"
NO_METHOD_REQUIRED = "no_method_required"
METHOD_POLICY_MODES = {
    ANY_VALID_METHOD,
    SPECIFIED_METHOD_REQUIRED,
    NO_METHOD_REQUIRED,
}

POINT_BASED_SCORING = "point_based"
HOLISTIC_BANDED_SCORING = "holistic_banded"
SCORING_MODELS = {POINT_BASED_SCORING, HOLISTIC_BANDED_SCORING}


# These are marking standards, not model-temperature aliases.  Temperature is
# deliberately kept low so a repeated assessment is consistent and auditable.
STRICTNESS_PROFILES: Dict[str, str] = {
    "lenient": (
        "Accept scientifically or academically valid alternative methods and "
        "minor OCR/notation defects when the intended work is clear. Award "
        "available method credit where the criterion is substantially met. "
        "For open-ended responses, judge meaning and task fulfilment rather than exact wording."
    ),
    "balanced": (
        "Apply ordinary teacher partial credit. Accept valid alternative "
        "methods, but require the criterion's stated outcome or evidence "
        "before awarding its marks. If a legacy criterion is worth more than "
        "one mark, award proportional credit for each independently correct "
        "visible step supported by the reference solution; do not turn it into "
        "an all-or-nothing result unless the method policy explicitly says that "
        "only the result is required. Apply the subject, class, board, response genre, "
        "and teacher guidance shown in the frozen paper context. For essays, speeches, "
        "paragraphs, and other open responses, accept original valid content and equivalent "
        "wording while applying each content, organization, language, style, or format criterion."
    ),
    "strict": (
        "Apply the criterion exactly. Do not infer missing method, units, "
        "working, or conclusion. If handwriting/OCR makes evidence ambiguous, "
        "mark it for teacher review instead of guessing a deduction. Do not require exact "
        "wording unless the frozen teacher criterion explicitly requires it."
    ),
}

_CRITERION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MARK_TOLERANCE = 0.01

_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "पाँच": 5,
    "छह": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
}
_COUNT_TOKEN_PATTERN = "|".join(
    sorted((re.escape(value) for value in _COUNT_WORDS), key=len, reverse=True)
)
_ATTEMPT_ANY_PATTERNS = (
    re.compile(
        rf"\b(?:answer|attempt|solve|write)\s+(?:only\s+)?any\s+"
        rf"(?P<count>\d+|{_COUNT_TOKEN_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:answer|attempt|solve|write)\s+(?P<count>\d+|{_COUNT_TOKEN_PATTERN})\s+"
        r"(?:of|out\s+of)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:किन्हीं|किन्ही|किसी|कोई)\s*(?P<count>\d+|{_COUNT_TOKEN_PATTERN})\s*"
        r"(?:प्रश्न|प्रश्नों|भाग|खंड|विकल्प)",
        re.IGNORECASE,
    ),
)

_INSTRUCTION_CUES = (
    re.compile(r"\bthis\s+(?:question\s+)?paper\s+is\s+divided\b", re.IGNORECASE),
    re.compile(r"\ball\s+(?:the\s+)?questions?\s+are\s+compulsory\b", re.IGNORECASE),
    re.compile(r"\bmarks?\s+(?:are|is)\s+indicated\b", re.IGNORECASE),
    re.compile(r"\b(?:attempt|answer)\s+all\s+(?:the\s+)?questions?\b", re.IGNORECASE),
    re.compile(r"\bwrite\s+(?:the\s+)?answers?\s+(?:in|on)\s+(?:the\s+)?answer\s+", re.IGNORECASE),
    re.compile(r"\bread\s+(?:the\s+)?instructions?\s+carefully\b", re.IGNORECASE),
    re.compile(r"(?:सभी|समस्त)\s+प्रश्न\s+अनिवार्य", re.IGNORECASE),
    re.compile(r"प्रश्न.?पत्र\s+(?:को|में).*(?:खंड|भाग)", re.IGNORECASE),
)
_ASSESSABLE_TASK_CUE = re.compile(
    r"(?:\?|\b(?:find|calculate|explain|describe|discuss|compare|define|prove|derive|"
    r"write\s+(?:a|an|about)|draft|identify|state|evaluate|analyse|analyze)\b|"
    r"(?:ज्ञात|गणना|समझाइए|वर्णन|चर्चा|तुलना|परिभाषित|सिद्ध|व्याख्या|लिखिए|बताइए))",
    re.IGNORECASE,
)


def normalize_scoring_model(value: Any) -> str:
    """Canonicalise how marks are divided inside one assessable response."""

    raw = str(value or POINT_BASED_SCORING).strip().lower().replace("-", "_")
    aliases = {
        "atomic": POINT_BASED_SCORING,
        "analytical": POINT_BASED_SCORING,
        "criterion": POINT_BASED_SCORING,
        "criteria": POINT_BASED_SCORING,
        "holistic": HOLISTIC_BANDED_SCORING,
        "banded": HOLISTIC_BANDED_SCORING,
        "range": HOLISTIC_BANDED_SCORING,
    }
    model = aliases.get(raw, raw)
    if model not in SCORING_MODELS:
        raise ValueError("Scoring model must be point_based or holistic_banded")
    return model


def default_structured_marking_policy() -> Dict[str, Any]:
    """Return the policy used for newly authored PCR papers."""

    return {
        "version": POLICY_VERSION,
        "mode": STRUCTURED_RUBRIC_MODE,
        "strictness": "balanced",
        "temperature": 0.0,
    }


def legacy_marking_policy() -> Dict[str, Any]:
    """Return a compatibility policy for already-finalized legacy papers."""

    return {
        "version": "legacy-ai-v1",
        "mode": LEGACY_AI_MODE,
        "strictness": "balanced",
        "temperature": 0.10,
    }


def default_method_policy() -> Dict[str, Any]:
    """Return the method contract for a normal subjective question.

    The teacher solution is an example of a valid route, not the only route.
    A stricter method requirement must be explicitly authored and frozen with
    the paper; the grader must never infer it merely from the worked solution.
    """

    return {
        "version": METHOD_POLICY_VERSION,
        "mode": ANY_VALID_METHOD,
        "required_method": None,
        "allow_error_carried_forward": True,
    }


def normalize_method_policy(value: Any) -> Dict[str, Any]:
    """Canonicalise one question's method and follow-through contract."""

    if value is None or value == "":
        return default_method_policy()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("PCR method policy must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("PCR method policy must be an object")

    raw_mode = str(value.get("mode") or ANY_VALID_METHOD).strip().lower()
    aliases = {
        "any": ANY_VALID_METHOD,
        "alternative_methods": ANY_VALID_METHOD,
        "any-equivalent-method": ANY_VALID_METHOD,
        "specified": SPECIFIED_METHOD_REQUIRED,
        "required": SPECIFIED_METHOD_REQUIRED,
        "required_method": SPECIFIED_METHOD_REQUIRED,
        "answer_only": NO_METHOD_REQUIRED,
        "result_only": NO_METHOD_REQUIRED,
        "no_working_required": NO_METHOD_REQUIRED,
    }
    mode = aliases.get(raw_mode, raw_mode)
    if mode not in METHOD_POLICY_MODES:
        raise ValueError(
            "Method policy mode must be any_valid_method, "
            "specified_method_required, or no_method_required"
        )

    required_method = str(value.get("required_method") or "").strip()
    if mode == SPECIFIED_METHOD_REQUIRED and not required_method:
        raise ValueError("Name the method that the question explicitly requires")
    if mode != SPECIFIED_METHOD_REQUIRED:
        required_method = ""
    if len(required_method) > 800:
        raise ValueError("Required method description is too long")

    raw_follow_through = value.get("allow_error_carried_forward", True)
    if not isinstance(raw_follow_through, bool):
        raise ValueError("allow_error_carried_forward must be true or false")

    return {
        "version": METHOD_POLICY_VERSION,
        "mode": mode,
        "required_method": required_method or None,
        "allow_error_carried_forward": raw_follow_through,
    }


def method_policy_instruction(policy: Any) -> str:
    """Return a server-owned examiner instruction for one frozen question."""

    normalized = normalize_method_policy(policy)
    mode = normalized["mode"]
    if mode == SPECIFIED_METHOD_REQUIRED:
        method = normalized["required_method"]
        instruction = (
            f"The question explicitly requires this method: {method}. "
            "Award method-dependent criteria only when that method is visibly used, "
            "while still awarding independent accuracy or conclusion criteria when met."
        )
    elif mode == NO_METHOD_REQUIRED:
        instruction = (
            "No working method is required. Judge the visible result and any other "
            "locked criteria; do not withhold marks merely because working is absent."
        )
    else:
        instruction = (
            "Accept any mathematically, scientifically, or academically valid method. "
            "The teacher solution is one example only; method difference is never a "
            "reason to deduct marks when the locked criterion is satisfied."
        )

    if normalized["allow_error_carried_forward"]:
        instruction += (
            " Apply error-carried-forward marking: after a clearly identified earlier "
            "error, award later method or reasoning criteria when the subsequent work is "
            "internally correct for the student's own value, unless a criterion explicitly "
            "requires the correct earlier value."
        )
    else:
        instruction += " Do not apply error-carried-forward credit for this question."
    return instruction


def normalize_marking_policy(
    value: Optional[Mapping[str, Any]],
    *,
    default_structured: bool = False,
) -> Dict[str, Any]:
    """Validate and canonicalise one PCR document/session marking policy.

    ``default_structured=False`` is intentional: documents created before this
    feature remain readable and are not suddenly made invalid.  New documents
    and explicit policy saves use ``default_structured=True``.
    """

    if value is None:
        return (
            default_structured_marking_policy()
            if default_structured
            else legacy_marking_policy()
        )
    if not isinstance(value, Mapping):
        raise ValueError("PCR marking policy must be an object")

    fallback = default_structured_marking_policy() if default_structured else legacy_marking_policy()
    raw_mode = str(value.get("mode") or fallback["mode"]).strip().lower()
    aliases = {
        "structured": STRUCTURED_RUBRIC_MODE,
        "criteria": STRUCTURED_RUBRIC_MODE,
        "criterion_rubric": STRUCTURED_RUBRIC_MODE,
        "criterion-rubric-v1": STRUCTURED_RUBRIC_MODE,
        "legacy": LEGACY_AI_MODE,
    }
    mode = aliases.get(raw_mode, raw_mode)
    if mode not in {STRUCTURED_RUBRIC_MODE, LEGACY_AI_MODE}:
        raise ValueError("Unsupported PCR marking policy mode")

    strictness = str(value.get("strictness") or fallback["strictness"]).strip().lower()
    if strictness not in STRICTNESS_PROFILES:
        raise ValueError("Strictness must be lenient, balanced, or strict")

    raw_temperature = value.get("temperature", fallback["temperature"])
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("AI temperature must be a number between 0 and 0.20") from exc
    if not math.isfinite(temperature) or not 0.0 <= temperature <= MAX_AI_TEMPERATURE:
        raise ValueError("AI temperature must be between 0 and 0.20")

    return {
        "version": POLICY_VERSION if mode == STRUCTURED_RUBRIC_MODE else "legacy-ai-v1",
        "mode": mode,
        "strictness": strictness,
        "temperature": round(temperature, 2),
    }


def is_structured_rubric_policy(policy: Optional[Mapping[str, Any]]) -> bool:
    """Return whether a policy requires locked criterion-by-criterion marks."""

    return normalize_marking_policy(policy).get("mode") == STRUCTURED_RUBRIC_MODE


def strictness_instruction(strictness: str) -> str:
    """Return the server-owned instruction for a marking standard."""

    return STRICTNESS_PROFILES.get(str(strictness).strip().lower(), STRICTNESS_PROFILES["balanced"])


def _as_json_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Marking criteria must be valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("Marking criteria must be a list")
    return value


def normalize_marking_criteria(
    value: Any,
    *,
    assign_missing_ids: bool = True,
) -> List[Dict[str, Any]]:
    """Return a safe, serialisable list of teacher-authored criteria.

    Draft rows are allowed to be incomplete while a teacher is editing.  The
    stronger :func:`validate_marking_criteria` check is used only at the paper
    finalisation boundary.
    """

    raw_items = _as_json_list(value)
    criteria: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for position, item in enumerate(raw_items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Criterion {position} must be an object")

        raw_id = str(item.get("criterion_id") or item.get("id") or "").strip()
        if not raw_id and assign_missing_ids:
            raw_id = f"criterion_{position}"
        if raw_id and not _CRITERION_ID_RE.fullmatch(raw_id):
            raise ValueError(
                f"Criterion {position} has an invalid id; use letters, numbers, _ or -"
            )
        if raw_id in used_ids:
            raise ValueError(f"Criterion ids must be unique ({raw_id})")
        if raw_id:
            used_ids.add(raw_id)

        raw_marks = item.get("max_marks", item.get("marks", 0))
        try:
            max_marks = float(raw_marks or 0)
        except (TypeError, ValueError):
            max_marks = 0.0
        if not math.isfinite(max_marks):
            raise ValueError(f"Criterion {position} has an invalid mark value")

        description = str(
            item.get("description") or item.get("criterion") or item.get("step") or ""
        ).strip()
        acceptable_evidence = str(
            item.get("acceptable_evidence")
            or item.get("evidence")
            or item.get("expected_evidence")
            or ""
        ).strip()
        if len(description) > 1600 or len(acceptable_evidence) > 1600:
            raise ValueError(f"Criterion {position} is too long")

        criteria.append(
            {
                "criterion_id": raw_id,
                "description": description,
                "max_marks": round(max_marks, 2),
                "acceptable_evidence": acceptable_evidence,
            }
        )
    return criteria


def validate_marking_criteria(
    criteria: Iterable[Mapping[str, Any]],
    question_max_marks: Any,
    *,
    require_atomic: bool = False,
) -> List[str]:
    """Return finalisation errors for a locked teacher marking rubric."""

    normalized = normalize_marking_criteria(list(criteria), assign_missing_ids=False)
    if not normalized:
        return ["add at least one criterion with marks"]

    try:
        max_marks = float(question_max_marks)
    except (TypeError, ValueError):
        max_marks = 0.0
    if not math.isfinite(max_marks) or max_marks <= 0:
        return ["assign question marks greater than zero before setting criteria"]

    errors: List[str] = []
    total = 0.0
    seen: set[str] = set()
    for position, criterion in enumerate(normalized, start=1):
        criterion_id = str(criterion.get("criterion_id") or "").strip()
        if not criterion_id:
            errors.append(f"criterion {position}: missing criterion id")
        elif criterion_id in seen:
            errors.append(f"criterion {position}: duplicate criterion id {criterion_id}")
        seen.add(criterion_id)

        if not str(criterion.get("description") or "").strip():
            errors.append(f"criterion {position}: describe what earns the marks")
        try:
            marks = float(criterion.get("max_marks") or 0)
        except (TypeError, ValueError):
            marks = 0.0
        if not math.isfinite(marks) or marks <= 0:
            errors.append(f"criterion {position}: assign marks greater than zero")
        elif require_atomic and marks > 1.0 + _MARK_TOLERANCE:
            errors.append(
                f"criterion {position}: split this {marks:g}-mark row into "
                "independently assessable criteria worth at most 1 mark each"
            )
        total += marks

    if abs(total - max_marks) > _MARK_TOLERANCE:
        errors.append(
            f"criterion marks total {total:g}, but this question is worth {max_marks:g}"
        )
    return errors


def normalize_assessment_units(
    value: Any,
    *,
    assign_missing_ids: bool = True,
) -> List[Dict[str, Any]]:
    """Return canonical leaf-level scoring units for one displayed question.

    A printed question may contain several separately answered and marked
    subparts.  Keeping those leaves explicit prevents one broad rubric row from
    hiding which subpart earned or lost marks while preserving the parent
    question's display identity and total.
    """

    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Assessment units must be valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("Assessment units must be a list")

    units: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Assessment unit {position} must be an object")
        unit_id = str(item.get("unit_id") or item.get("id") or "").strip()
        if not unit_id and assign_missing_ids:
            unit_id = f"unit_{position}"
        if unit_id and not _CRITERION_ID_RE.fullmatch(unit_id):
            raise ValueError(
                f"Assessment unit {position} has an invalid id; use letters, numbers, _ or -"
            )
        if unit_id in seen_ids:
            raise ValueError(f"Assessment unit ids must be unique ({unit_id})")
        if unit_id:
            seen_ids.add(unit_id)

        label = str(item.get("label") or item.get("question_label") or "").strip()
        prompt = str(
            item.get("prompt")
            or item.get("question_text")
            or item.get("text")
            or ""
        ).strip()
        reference_solution = str(
            item.get("reference_solution")
            or item.get("answer")
            or item.get("solution")
            or ""
        ).strip()
        if len(label) > 160:
            raise ValueError(f"Assessment unit {position} label is too long")
        if len(prompt) > 12000:
            raise ValueError(f"Assessment unit {position} prompt is too long")
        if len(reference_solution) > 24000:
            raise ValueError(f"Assessment unit {position} reference solution is too long")

        raw_marks = item.get("max_marks", item.get("marks", item.get("points", 0)))
        try:
            max_marks = float(raw_marks or 0)
        except (TypeError, ValueError):
            max_marks = 0.0
        if not math.isfinite(max_marks):
            raise ValueError(f"Assessment unit {position} has an invalid mark value")

        figure_refs = item.get("figure_refs") or item.get("image_refs") or []
        if not isinstance(figure_refs, list):
            raise ValueError(f"Assessment unit {position} figure_refs must be a list")
        normalized_refs = []
        for ref in figure_refs[:20]:
            value_text = str(ref or "").strip()
            if value_text and value_text not in normalized_refs:
                normalized_refs.append(value_text[:200])

        units.append(
            {
                "version": ASSESSMENT_UNIT_VERSION,
                "unit_id": unit_id,
                "label": label or f"Part {position}",
                "prompt": prompt,
                "max_marks": round(max_marks, 2),
                "scoring_model": normalize_scoring_model(item.get("scoring_model")),
                "reference_solution": reference_solution,
                "marking_criteria": normalize_marking_criteria(
                    item.get("marking_criteria", item.get("criteria", [])),
                    assign_missing_ids=assign_missing_ids,
                ),
                "method_policy": normalize_method_policy(item.get("method_policy")),
                "figure_refs": normalized_refs,
            }
        )
    return units


def _positive_count(value: Any) -> Optional[int]:
    raw = str(value or "").strip().casefold()
    if not raw:
        return None
    if raw.isdigit():
        count = int(raw)
    else:
        count = _COUNT_WORDS.get(raw)
    return count if count and count > 0 else None


def normalize_response_selection(
    value: Any,
    *,
    available_unit_ids: Optional[Iterable[Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize the question-level rule that selects optional response units.

    This rule is deliberately stored beside, rather than inside, assessment
    units.  Every alternative can retain its honest mark value while the
    parent question still owns the maximum selectable budget.
    """

    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Response selection must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Response selection must be an object")
    mode = str(value.get("mode") or "").strip().lower().replace("-", "_")
    if mode not in {"attempt_any"}:
        raise ValueError("Response selection mode must be attempt_any")
    required_count = _positive_count(value.get("required_count"))
    if required_count is None:
        raise ValueError("Response selection requires a positive required_count")

    allowed_ids = [
        str(item or "").strip()
        for item in (available_unit_ids or [])
        if str(item or "").strip()
    ]
    raw_ids = value.get("available_unit_ids")
    if raw_ids is None:
        unit_ids = list(allowed_ids)
    elif not isinstance(raw_ids, list):
        raise ValueError("Response selection available_unit_ids must be a list")
    else:
        unit_ids = [str(item or "").strip() for item in raw_ids if str(item or "").strip()]
    if not unit_ids:
        raise ValueError("Response selection requires available assessment units")
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Response selection assessment unit ids must be unique")
    if allowed_ids and (set(unit_ids) - set(allowed_ids)):
        raise ValueError("Response selection references an unknown assessment unit")
    if required_count >= len(unit_ids):
        raise ValueError("attempt_any must select fewer than the available assessment units")

    return {
        "version": RESPONSE_SELECTION_VERSION,
        "mode": "attempt_any",
        "required_count": required_count,
        "available_unit_ids": unit_ids,
    }


def derive_response_selection(
    question_text: Any,
    units: Any,
    question_max_marks: Any,
    *,
    explicit: Any = None,
) -> Optional[Dict[str, Any]]:
    """Derive a strict ``attempt any N`` rule without mutating stored data."""

    normalized_units = normalize_assessment_units(units, assign_missing_ids=True)
    unit_ids = [str(unit.get("unit_id") or "") for unit in normalized_units]
    if explicit is not None and explicit != "":
        selection = normalize_response_selection(
            explicit,
            available_unit_ids=unit_ids,
        )
    else:
        text = " ".join(str(question_text or "").split())
        required_count: Optional[int] = None
        for pattern in _ATTEMPT_ANY_PATTERNS:
            match = pattern.search(text)
            if match:
                required_count = _positive_count(match.group("count"))
                break
        if required_count is None or required_count >= len(unit_ids):
            return None
        selection = normalize_response_selection(
            {
                "mode": "attempt_any",
                "required_count": required_count,
                "available_unit_ids": unit_ids,
            },
            available_unit_ids=unit_ids,
        )

    try:
        question_marks = float(question_max_marks)
    except (TypeError, ValueError) as exc:
        raise ValueError("Question marks must be a positive number") from exc
    if not math.isfinite(question_marks) or question_marks <= 0:
        raise ValueError("Question marks must be a positive number")
    required_count = int(selection["required_count"])
    per_unit_marks = round(question_marks / required_count, 2)
    if per_unit_marks <= 0 or not math.isclose(
        per_unit_marks * required_count,
        question_marks,
        abs_tol=_MARK_TOLERANCE,
    ):
        raise ValueError(
            "Question marks cannot be divided exactly across the required optional responses"
        )
    return {**selection, "per_unit_marks": per_unit_marks}


def instruction_only_question_reason(question_text: Any) -> str:
    """Identify high-confidence paper directions accidentally catalogued as a question."""

    text = " ".join(str(question_text or "").split())
    if not text:
        return "Question text is empty"
    cue_count = sum(1 for pattern in _INSTRUCTION_CUES if pattern.search(text))
    if cue_count < 2 or _ASSESSABLE_TASK_CUE.search(text):
        return ""
    return "Appears to contain paper instructions rather than an assessable question"


def _mark_quantum(total_marks: float, item_count: int) -> Optional[float]:
    """Choose the coarsest conventional increment that can fund every item."""

    for quantum in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
        ticks = round(total_marks / quantum)
        if (
            ticks >= item_count
            and math.isclose(ticks * quantum, total_marks, abs_tol=_MARK_TOLERANCE)
        ):
            return quantum
    return None


def _allocate_mark_budget(
    total_marks: float,
    weights: Iterable[Any],
) -> Optional[List[float]]:
    """Allocate an exact budget using relative semantic weights.

    Model-produced numbers are treated only as relative importance. The
    server owns the mark ledger, uses conventional school-mark increments,
    and guarantees an exact total. ``None`` means the budget cannot give every
    requested row a positive mark and the caller should collapse the rows.
    """

    normalized_weights: List[float] = []
    for raw_weight in weights:
        try:
            weight = float(raw_weight or 0)
        except (TypeError, ValueError):
            weight = 0.0
        normalized_weights.append(weight if math.isfinite(weight) and weight > 0 else 1.0)
    if not normalized_weights:
        return []

    quantum = _mark_quantum(total_marks, len(normalized_weights))
    if quantum is None:
        return None
    total_ticks = int(round(total_marks / quantum))
    base_ticks = [1] * len(normalized_weights)
    remaining = total_ticks - len(normalized_weights)
    if remaining <= 0:
        return [round(quantum, 2) for _ in normalized_weights]

    weight_total = sum(normalized_weights)
    raw_extras = [remaining * weight / weight_total for weight in normalized_weights]
    extra_ticks = [int(math.floor(value)) for value in raw_extras]
    undistributed = remaining - sum(extra_ticks)
    priority = sorted(
        range(len(raw_extras)),
        key=lambda index: (raw_extras[index] - extra_ticks[index], normalized_weights[index], -index),
        reverse=True,
    )
    for index in priority[:undistributed]:
        extra_ticks[index] += 1
    return [
        round((base_ticks[index] + extra_ticks[index]) * quantum, 2)
        for index in range(len(normalized_weights))
    ]


def _inclusive_criterion(
    criteria: Iterable[Mapping[str, Any]],
    *,
    max_marks: float,
) -> Dict[str, Any]:
    normalized = normalize_marking_criteria(list(criteria), assign_missing_ids=True)
    descriptions = [
        str(item.get("description") or "").strip()
        for item in normalized
        if str(item.get("description") or "").strip()
    ]
    evidence = [
        str(item.get("acceptable_evidence") or "").strip()
        for item in normalized
        if str(item.get("acceptable_evidence") or "").strip()
    ]
    description = "Provides a correct and sufficiently complete response to the required question parts."
    if descriptions:
        description += " Required achievements: " + "; ".join(descriptions)
    acceptable_evidence = "Accept the reference solution and clearly equivalent valid responses."
    if evidence:
        acceptable_evidence += " Evidence may include: " + "; ".join(evidence)
    return {
        "criterion_id": "complete_response",
        "description": description[:1600],
        "max_marks": round(max_marks, 2),
        "acceptable_evidence": acceptable_evidence[:1600],
    }


def _compile_unit_criteria(unit: Dict[str, Any]) -> Dict[str, Any]:
    compiled = dict(unit)
    criteria = list(compiled.get("marking_criteria") or [])
    unit_marks = float(compiled.get("max_marks") or 0)
    if not criteria:
        return compiled

    allocations = _allocate_mark_budget(
        unit_marks,
        [item.get("max_marks") for item in criteria],
    )
    if allocations is None or (unit_marks <= 1.0 and len(criteria) > 1):
        compiled["marking_criteria"] = [
            _inclusive_criterion(criteria, max_marks=unit_marks)
        ]
        compiled["scoring_model"] = (
            POINT_BASED_SCORING if unit_marks <= 1.0 else HOLISTIC_BANDED_SCORING
        )
        return compiled

    compiled_criteria: List[Dict[str, Any]] = []
    for criterion, marks in zip(criteria, allocations):
        row = dict(criterion)
        row["max_marks"] = marks
        compiled_criteria.append(row)
    compiled["marking_criteria"] = compiled_criteria
    if (
        compiled.get("scoring_model") == POINT_BASED_SCORING
        and any(float(item.get("max_marks") or 0) > 1.0 for item in compiled_criteria)
    ):
        compiled["scoring_model"] = HOLISTIC_BANDED_SCORING
    return compiled


def compile_assessment_units_to_budget(
    value: Any,
    question_max_marks: Any,
    *,
    question_text: str = "",
    explicit_unit_marks: Optional[List[float]] = None,
    response_selection: Any = None,
) -> List[Dict[str, Any]]:
    """Compile semantic assessment units into a server-owned mark ledger.

    Printed subpart marks, when supplied, remain exact. Otherwise the model's
    suggested numbers are relative weights only. A one-mark question without
    printed subpart allocation becomes one inclusive question-level unit, so
    several requested parts can never accidentally inflate the total marks.
    """

    units = normalize_assessment_units(value, assign_missing_ids=True)
    if not units:
        return []
    try:
        question_marks = float(question_max_marks)
    except (TypeError, ValueError) as exc:
        raise ValueError("Question marks must be a positive number") from exc
    if not math.isfinite(question_marks) or question_marks <= 0:
        raise ValueError("Question marks must be a positive number")

    selection = derive_response_selection(
        question_text,
        units,
        question_marks,
        explicit=response_selection,
    )
    printed_allocations = list(explicit_unit_marks or [])
    if selection:
        per_unit_marks = float(selection["per_unit_marks"])
        if printed_allocations:
            if len(printed_allocations) != len(units):
                raise ValueError(
                    "Printed optional-part marks do not match the available responses"
                )
            if any(
                not math.isclose(float(value), per_unit_marks, abs_tol=_MARK_TOLERANCE)
                for value in printed_allocations
            ):
                raise ValueError(
                    "Printed optional-part marks do not match the selectable question budget"
                )
        allocations = [per_unit_marks for _unit in units]
    elif printed_allocations:
        if len(printed_allocations) != len(units):
            return units
        if not math.isclose(sum(printed_allocations), question_marks, abs_tol=_MARK_TOLERANCE):
            raise ValueError("Printed subpart marks do not match the question total")
        allocations = [round(float(value), 2) for value in printed_allocations]
    else:
        allocations = _allocate_mark_budget(
            question_marks,
            [unit.get("max_marks") for unit in units],
        )

    if allocations is None or (question_marks <= 1.0 and len(units) > 1 and not printed_allocations):
        labels = [str(unit.get("label") or "").strip() for unit in units]
        reference_sections = []
        all_criteria: List[Dict[str, Any]] = []
        figure_refs: List[str] = []
        for position, unit in enumerate(units, start=1):
            label = str(unit.get("label") or f"Part {position}").strip()
            reference = str(unit.get("reference_solution") or "").strip()
            if reference:
                reference_sections.append(f"{label}: {reference}")
            all_criteria.extend(unit.get("marking_criteria") or [])
            for figure_ref in unit.get("figure_refs") or []:
                if figure_ref not in figure_refs:
                    figure_refs.append(figure_ref)
        method_policies = [unit.get("method_policy") for unit in units]
        method_policy = (
            method_policies[0]
            if method_policies and all(item == method_policies[0] for item in method_policies)
            else default_method_policy()
        )
        return normalize_assessment_units(
            [
                {
                    "unit_id": "unit_1",
                    "label": "Whole question",
                    "prompt": str(question_text or "").strip()
                    or "Complete all required parts of the question.",
                    "max_marks": question_marks,
                    "scoring_model": (
                        POINT_BASED_SCORING
                        if question_marks <= 1.0
                        else HOLISTIC_BANDED_SCORING
                    ),
                    "reference_solution": "\n".join(reference_sections),
                    "marking_criteria": [
                        _inclusive_criterion(all_criteria, max_marks=question_marks)
                    ],
                    "method_policy": method_policy,
                    "figure_refs": figure_refs,
                }
            ],
            assign_missing_ids=False,
        )

    compiled_units: List[Dict[str, Any]] = []
    for unit, marks in zip(units, allocations):
        compiled = dict(unit)
        compiled["max_marks"] = marks
        compiled_units.append(_compile_unit_criteria(compiled))
    return normalize_assessment_units(compiled_units, assign_missing_ids=False)


def validate_assessment_units(
    units: Iterable[Mapping[str, Any]],
    question_max_marks: Any,
    *,
    require_reference_solution: bool = False,
    question_text: str = "",
    response_selection: Any = None,
) -> List[str]:
    """Validate the complete marks ledger below one displayed question."""

    normalized = normalize_assessment_units(list(units), assign_missing_ids=False)
    if not normalized:
        return ["add at least one assessment unit"]
    try:
        question_marks = float(question_max_marks)
    except (TypeError, ValueError):
        question_marks = 0.0
    if not math.isfinite(question_marks) or question_marks <= 0:
        return ["assign question marks greater than zero before setting assessment units"]

    errors: List[str] = []
    try:
        selection = derive_response_selection(
            question_text,
            normalized,
            question_marks,
            explicit=response_selection,
        )
    except ValueError as exc:
        selection = None
        errors.append(str(exc))
    unit_total = 0.0
    seen_ids: set[str] = set()
    for position, unit in enumerate(normalized, start=1):
        unit_id = str(unit.get("unit_id") or "").strip()
        label = str(unit.get("label") or f"Part {position}").strip()
        if not unit_id:
            errors.append(f"assessment unit {position}: missing unit id")
        elif unit_id in seen_ids:
            errors.append(f"assessment unit {position}: duplicate unit id {unit_id}")
        seen_ids.add(unit_id)
        if not str(unit.get("prompt") or "").strip():
            errors.append(f"assessment unit {label}: missing prompt")
        marks = float(unit.get("max_marks") or 0)
        if marks <= 0:
            errors.append(f"assessment unit {label}: assign marks greater than zero")
        unit_total += marks
        if require_reference_solution and not str(unit.get("reference_solution") or "").strip():
            errors.append(f"assessment unit {label}: add a reference solution")
        criterion_errors = validate_marking_criteria(
            unit.get("marking_criteria") or [],
            marks,
            require_atomic=(unit.get("scoring_model") == POINT_BASED_SCORING),
        )
        errors.extend(f"assessment unit {label}: {error}" for error in criterion_errors)

    if selection:
        expected_marks = float(selection["per_unit_marks"])
        selectable_ids = set(selection["available_unit_ids"])
        known_ids = {str(unit.get("unit_id") or "") for unit in normalized}
        if selectable_ids != known_ids:
            errors.append("response selection must cover every optional assessment unit")
        for unit in normalized:
            marks = float(unit.get("max_marks") or 0)
            if not math.isclose(marks, expected_marks, abs_tol=_MARK_TOLERANCE):
                errors.append(
                    f"assessment unit {unit.get('label')}: expected {expected_marks:g} marks "
                    "for each selectable response"
                )
        selected_total = expected_marks * int(selection["required_count"])
        if not math.isclose(selected_total, question_marks, abs_tol=_MARK_TOLERANCE):
            errors.append(
                f"selected assessment-unit marks total {selected_total:g}, but this question "
                f"is worth {question_marks:g}"
            )
    elif abs(unit_total - question_marks) > _MARK_TOLERANCE:
        errors.append(
            f"assessment unit marks total {unit_total:g}, but this question is worth {question_marks:g}"
        )
    return errors


def _flattened_criterion_id(unit_id: str, criterion_id: str, used: set[str]) -> str:
    raw = f"{unit_id}__{criterion_id}"
    if len(raw) > 64:
        suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        raw = f"{raw[:53]}_{suffix}"
    candidate = raw
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{raw[:64-len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def flatten_assessment_unit_criteria(value: Any) -> List[Dict[str, Any]]:
    """Flatten child criteria for existing question-level grading consumers."""

    units = normalize_assessment_units(value, assign_missing_ids=False)
    flattened: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    single_unit = len(units) == 1
    for unit in units:
        label = str(unit.get("label") or "").strip()
        for criterion in unit.get("marking_criteria") or []:
            description = str(criterion.get("description") or "").strip()
            evidence = str(criterion.get("acceptable_evidence") or "").strip()
            flattened.append(
                {
                    "criterion_id": (
                        str(criterion.get("criterion_id") or "criterion")
                        if single_unit
                        else _flattened_criterion_id(
                            str(unit.get("unit_id") or "unit"),
                            str(criterion.get("criterion_id") or "criterion"),
                            used_ids,
                        )
                    ),
                    "description": (
                        description
                        if single_unit
                        else (f"[{label}] {description}" if label else description)
                    ),
                    "max_marks": float(criterion.get("max_marks") or 0),
                    "acceptable_evidence": (
                        evidence
                        if single_unit
                        else (f"[{label}] {evidence}" if label and evidence else evidence)
                    ),
                    "assessment_unit_id": unit.get("unit_id"),
                    "assessment_unit_label": label,
                    "scoring_model": unit.get("scoring_model"),
                }
            )
    return flattened


def compose_assessment_unit_reference_solution(value: Any) -> str:
    units = normalize_assessment_units(value, assign_missing_ids=False)
    if len(units) == 1:
        return str(units[0].get("reference_solution") or "").strip()
    sections = []
    for unit in units:
        solution = str(unit.get("reference_solution") or "").strip()
        if solution:
            sections.append(f"{unit.get('label')}: {solution}")
    return "\n\n".join(sections)


def snapshot_criteria(value: Any) -> List[Dict[str, Any]]:
    """Make an immutable copy of criteria after finalisation validation."""

    return deepcopy(normalize_marking_criteria(value, assign_missing_ids=False))


def snapshot_method_policy(value: Any) -> Dict[str, Any]:
    """Make an immutable copy of a validated question method policy."""

    return deepcopy(normalize_method_policy(value))


def snapshot_assessment_units(value: Any) -> List[Dict[str, Any]]:
    """Freeze validated assessment units into an immutable paper snapshot."""

    return deepcopy(normalize_assessment_units(value, assign_missing_ids=False))
