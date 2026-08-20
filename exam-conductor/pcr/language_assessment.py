"""Language-writing feedback contract for PCR visual grading.

Exam marks and developmental writing feedback have different ownership.  The
locked marking criteria own the score; this module only describes the compact,
seven-dimension feedback profile returned alongside that score.  The profile is
derived at read time so existing exams need no data migration.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence


LANGUAGE_FEEDBACK_VERSION = "language-feedback-v1"

DIMENSIONS = (
    ("understanding", "Understanding"),
    ("content", "Content"),
    ("structure_organization", "Structure & Organization"),
    ("language_grammar", "Language & Grammar"),
    ("clarity_expression", "Clarity & Expression"),
    ("tone_style", "Tone & Style"),
    ("conciseness_precision", "Conciseness & Precision"),
)

LEVELS = (
    "excellent",
    "secure",
    "developing",
    "needs_improvement",
    "not_assessed",
    "not_applicable",
)

RESPONSE_FAMILIES = {
    "creative_writing",
    "functional_writing",
    "comprehension",
    "grammar_vocabulary",
    "translation",
    "literature_response",
    "short_language_response",
}

_LANGUAGE_SUBJECTS = {
    "arabic",
    "assamese",
    "bengali",
    "english",
    "french",
    "german",
    "gujarati",
    "hindi",
    "japanese",
    "kannada",
    "kashmiri",
    "konkani",
    "malayalam",
    "manipuri",
    "marathi",
    "nepali",
    "odia",
    "oriya",
    "persian",
    "punjabi",
    "sanskrit",
    "sindhi",
    "spanish",
    "tamil",
    "telugu",
    "urdu",
}

_NATIVE_LANGUAGE_NAMES = (
    "हिंदी",
    "हिन्दी",
    "अंग्रेजी",
    "अंग्रेज़ी",
    "संस्कृत",
    "मराठी",
    "नेपाली",
    "اردو",
    "ਪੰਜਾਬੀ",
    "বাংলা",
    "অসমীয়া",
    "ગુજરાતી",
    "ଓଡ଼ିଆ",
    "தமிழ்",
    "తెలుగు",
    "ಕನ್ನಡ",
    "മലയാളം",
)

# Fail closed when a medium/language label appears alongside a different
# academic subject (for example, "English Medium Physics").  The script used
# for teaching or answering never changes the subject being assessed.
_NON_LANGUAGE_SUBJECT_PATTERN = re.compile(
    r"\b(?:physics|chemistry|mathematics|maths|math|biology|science|computer|"
    r"informatics|history|geography|civics|economics|commerce|accountancy|"
    r"business|environmental|engineering|medicine|psychology|sociology|"
    r"political)\b|भौतिक|रसायन|गणित|जीव.?विज्ञान|विज्ञान|इतिहास|भूगोल|अर्थशास्त्र",
    re.IGNORECASE,
)

_FAMILY_PATTERNS = (
    (
        "translation",
        re.compile(r"\b(?:translate|translation)\b|अनुवाद|तर्जुमा", re.IGNORECASE),
    ),
    (
        "functional_writing",
        re.compile(
            r"\b(?:letter|application|notice|report|email|e-mail|article|advertisement|message)\b"
            r"|पत्र|आवेदन|सूचना|रिपोर्ट|प्रतिवेदन|विज्ञापन|संदेश|ईमेल",
            re.IGNORECASE,
        ),
    ),
    (
        "creative_writing",
        re.compile(
            r"\b(?:essay|paragraph|speech|story|diary|debate|composition|creative writing)\b"
            r"|निबंध|अनुच्छेद|भाषण|कहानी|डायरी|वाद.?विवाद|रचनात्मक",
            re.IGNORECASE,
        ),
    ),
    (
        "comprehension",
        re.compile(
            r"\b(?:comprehension|unseen passage|read the passage|passage)\b"
            r"|अपठित|गद्यांश|पद्यांश|अनुच्छेद को पढ़|काव्यांश",
            re.IGNORECASE,
        ),
    ),
    (
        "grammar_vocabulary",
        re.compile(
            r"\b(?:grammar|vocabulary|synonym|antonym|spelling|punctuation|tense|voice|narration|"
            r"fill in the blanks?|one word|idiom|phrase|parts? of speech)\b"
            r"|व्याकरण|पर्यायवाची|विलोम|वर्तनी|विराम.?चिह्न|काल|वाच्य|संधि|समास|"
            r"मुहावरा|लोकोक्ति|लिंग|वचन|कारक|उपसर्ग|प्रत्यय|रिक्त स्थान",
            re.IGNORECASE,
        ),
    ),
    (
        "literature_response",
        re.compile(
            r"\b(?:poem|poetry|prose|literature|character sketch|theme|literary|explain the lines)\b"
            r"|कविता|काव्य|साहित्य|चरित्र.?चित्रण|भावार्थ|संदर्भ|प्रसंग|पाठ के आधार",
            re.IGNORECASE,
        ),
    ),
)

_STEM_TASK_PATTERN = re.compile(
    r"\b(?:calculate|compute|solve\s+(?:the|for)|derive|prove|equation|formula|velocity|"
    r"acceleration|force|energy|mole|reaction|compound|cell|organism|theorem|graph|"
    r"coordinate|probability|algorithm|program)\b|"
    r"गणना|समीकरण|सूत्र|वेग|त्वरण|बल|ऊर्जा|अभिक्रिया|प्रमेय|प्रायिकता",
    re.IGNORECASE,
)

_APPLICABLE_BY_FAMILY = {
    "creative_writing": {item[0] for item in DIMENSIONS},
    "functional_writing": {item[0] for item in DIMENSIONS},
    "comprehension": {
        "understanding",
        "content",
        "language_grammar",
        "clarity_expression",
        "conciseness_precision",
    },
    "grammar_vocabulary": {
        "understanding",
        "language_grammar",
        "clarity_expression",
        "conciseness_precision",
    },
    "translation": {
        "understanding",
        "content",
        "language_grammar",
        "clarity_expression",
        "tone_style",
        "conciseness_precision",
    },
    "literature_response": {
        "understanding",
        "content",
        "structure_organization",
        "language_grammar",
        "clarity_expression",
        "tone_style",
        "conciseness_precision",
    },
    "short_language_response": {
        "understanding",
        "content",
        "language_grammar",
        "clarity_expression",
        "conciseness_precision",
    },
}


def _normal_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().replace("_", " ").split())


def _is_language_subject(subject: Any) -> bool:
    normalized = _normal_text(subject)
    if not normalized:
        return False
    if _NON_LANGUAGE_SUBJECT_PATTERN.search(normalized):
        return False
    tokens = set(re.findall(r"[a-z]+", normalized))
    return (
        bool(tokens.intersection(_LANGUAGE_SUBJECTS))
        or normalized in _LANGUAGE_SUBJECTS
        or any(name in normalized for name in _NATIVE_LANGUAGE_NAMES)
    )


def _explicit_profile(question: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = question.get("language_feedback_profile")
    if not isinstance(raw, Mapping):
        return None
    if raw.get("enabled") is False:
        return {"enabled": False, "version": LANGUAGE_FEEDBACK_VERSION}
    family = _normal_text(raw.get("response_family")).replace(" ", "_")
    if family not in RESPONSE_FAMILIES:
        family = "short_language_response"
    raw_dimensions = raw.get("dimensions")
    applicability: Dict[str, bool] = {}
    if isinstance(raw_dimensions, Sequence) and not isinstance(raw_dimensions, (str, bytes)):
        for item in raw_dimensions:
            if not isinstance(item, Mapping):
                continue
            dimension_id = str(item.get("dimension_id") or "").strip()
            if dimension_id in {entry[0] for entry in DIMENSIONS}:
                applicability[dimension_id] = bool(item.get("applicable", True))
    return _profile(family, applicability=applicability or None)


def _profile(
    family: str,
    *,
    applicability: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    applicable = _APPLICABLE_BY_FAMILY[family]
    dimensions = []
    for dimension_id, label in DIMENSIONS:
        is_applicable = (
            bool(applicability[dimension_id])
            if applicability and dimension_id in applicability
            else dimension_id in applicable
        )
        dimensions.append(
            {
                "dimension_id": dimension_id,
                "label": label,
                "applicable": is_applicable,
            }
        )
    return {
        "enabled": True,
        "version": LANGUAGE_FEEDBACK_VERSION,
        "response_family": family,
        "dimensions": dimensions,
    }


def _task_text(question: Mapping[str, Any]) -> str:
    return "\n".join(
        str(question.get(key) or "")
        for key in ("question_text", "rubric", "reference_solution")
    )


def _response_family(task_text: str) -> str:
    for candidate, pattern in _FAMILY_PATTERNS:
        if pattern.search(task_text):
            return candidate
    return "short_language_response"


def infer_language_paper(questions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Infer a language paper from immutable task semantics, not answer script.

    Metadata imported from older PDFs is sometimes wrong.  This conservative
    paper-level check requires repeated unmistakable language tasks and avoids
    switching ordinary STEM papers merely because their prose is English.
    The result is ephemeral and therefore needs no database migration.
    """

    subjective: list[Mapping[str, Any]] = []
    strong_question_ids: list[str] = []
    stem_count = 0
    family_counts: Dict[str, int] = {}
    for position, question in enumerate(questions, start=1):
        grading_mode = _normal_text(
            question.get("grading_mode") or question.get("question_type")
        ).replace(" ", "_")
        if grading_mode in {"objective", "mcq", "integer"}:
            continue
        subjective.append(question)
        text = _task_text(question)
        if _STEM_TASK_PATTERN.search(text):
            stem_count += 1
        family = _response_family(text)
        if family == "short_language_response":
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        strong_question_ids.append(
            str(question.get("question_id") or question.get("question_number") or position)
        )

    total = len(subjective)
    strong_count = len(strong_question_ids)
    strong_ratio = (strong_count / total) if total else 0.0
    # Two independent language tasks plus a meaningful share of the paper are
    # required. Clear STEM operations dominating the same catalogue veto the
    # inference; an explicit correct language subject remains handled above.
    enabled = bool(
        total
        and strong_count >= 2
        and strong_ratio >= 0.35
        and stem_count < max(2, strong_count)
    )
    return {
        "enabled": enabled,
        "version": LANGUAGE_FEEDBACK_VERSION,
        "basis": "immutable_question_semantics" if enabled else "insufficient_evidence",
        "strong_question_ids": strong_question_ids,
        "strong_question_count": strong_count,
        "subjective_question_count": total,
        "stem_question_count": stem_count,
        "family_counts": family_counts,
    }


def language_feedback_profile(question: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a migration-free language-feedback profile for one question."""

    grading_mode = _normal_text(
        question.get("grading_mode") or question.get("question_type")
    ).replace(" ", "_")
    if grading_mode in {"objective", "mcq", "integer"}:
        return {"enabled": False, "version": LANGUAGE_FEEDBACK_VERSION}
    explicit = _explicit_profile(question)
    if explicit is not None:
        return explicit
    if not (
        _is_language_subject(question.get("subject"))
        or question.get("language_subject_inferred") is True
    ):
        return {"enabled": False, "version": LANGUAGE_FEEDBACK_VERSION}

    task_text = _task_text(question)
    family = _response_family(task_text)
    return _profile(family)


def format_language_feedback(value: Any) -> str:
    """Curate validated dimensions into the existing feedback text surface."""

    if not isinstance(value, Mapping):
        return ""
    lines: list[str] = []
    summary = str(value.get("summary") or "").strip()
    if summary:
        lines.append(summary[:700])
    dimensions = value.get("dimensions")
    if isinstance(dimensions, Sequence) and not isinstance(dimensions, (str, bytes)):
        for item in dimensions:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("applicability") or "") != "applicable":
                continue
            label = str(item.get("label") or "").strip()
            level = str(item.get("level") or "not_assessed").strip().replace("_", " ")
            evidence = str(item.get("evidence") or "").strip()
            feedback = str(item.get("feedback") or "").strip()
            detail = " ".join(part for part in (evidence, feedback) if part)
            lines.append(f"- **{label} ({level.title()}):** {detail or 'Not assessed.'}")
    actions = [
        str(item).strip()
        for item in (value.get("priority_actions") or [])
        if str(item).strip()
    ][:3]
    if actions:
        lines.append("**Priority improvements:** " + "; ".join(actions))
    example = str(value.get("example_revision") or "").strip()
    if example:
        lines.append("**Example improvement:** " + example[:800])
    return "\n".join(lines).strip()


def language_feedback_schema(profile: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the strict compact schema embedded in one question response."""

    if not profile.get("enabled"):
        return None
    dimension_properties: Dict[str, Any] = {}
    for dimension in profile.get("dimensions") or []:
        if not isinstance(dimension, Mapping):
            continue
        dimension_id = str(dimension.get("dimension_id") or "")
        applicable = bool(dimension.get("applicable"))
        dimension_properties[dimension_id] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "applicability": {
                    "type": "string",
                    "enum": ["applicable" if applicable else "not_applicable"],
                },
                "level": {
                    "type": "string",
                    "enum": list(LEVELS[:-1]) if applicable else ["not_applicable"],
                },
                "evidence": {"type": "string", "maxLength": 400},
                "feedback": {"type": "string", "maxLength": 500},
            },
            "required": ["applicability", "level", "evidence", "feedback"],
        }
    dimension_ids = [dimension_id for dimension_id, _label in DIMENSIONS]
    if set(dimension_properties) != set(dimension_ids):
        return None
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"type": "string", "enum": [LANGUAGE_FEEDBACK_VERSION]},
            "response_family": {
                "type": "string",
                "enum": [str(profile.get("response_family"))],
            },
            "feedback_language": {"type": "string", "maxLength": 80},
            "summary": {"type": "string", "maxLength": 700},
            "priority_actions": {
                "type": "array",
                "items": {"type": "string", "maxLength": 300},
                "maxItems": 3,
            },
            "example_revision": {"type": "string", "maxLength": 800},
            "dimensions": {
                "type": "object",
                "additionalProperties": False,
                "properties": dimension_properties,
                "required": dimension_ids,
            },
        },
        "required": [
            "version",
            "response_family",
            "feedback_language",
            "summary",
            "priority_actions",
            "example_revision",
            "dimensions",
        ],
    }


def normalize_language_feedback(
    value: Any,
    *,
    profile: Mapping[str, Any],
    attempted: bool,
) -> Optional[Dict[str, Any]]:
    """Return safe diagnostic feedback without ever invalidating exam marks."""

    if not attempted or not profile.get("enabled") or not isinstance(value, Mapping):
        return None
    expected = {
        str(item.get("dimension_id")): bool(item.get("applicable"))
        for item in profile.get("dimensions") or []
        if isinstance(item, Mapping)
    }
    raw_by_id: Dict[str, Mapping[str, Any]] = {}
    raw_dimensions = value.get("dimensions")
    if isinstance(raw_dimensions, Mapping):
        raw_by_id = {
            str(dimension_id): item
            for dimension_id, item in raw_dimensions.items()
            if isinstance(item, Mapping)
        }
    elif isinstance(raw_dimensions, list):
        # Accept the pre-v1 draft shape defensively; persisted output is always
        # normalized to the stable array consumed by the APIs and UI.
        for item in raw_dimensions:
            if not isinstance(item, Mapping):
                continue
            dimension_id = str(item.get("dimension_id") or "").strip()
            if dimension_id in raw_by_id:
                return None
            raw_by_id[dimension_id] = item
    else:
        return None
    if set(raw_by_id) != set(expected) or len(expected) != len(DIMENSIONS):
        return None

    dimensions = []
    for dimension_id, label in DIMENSIONS:
        raw = raw_by_id[dimension_id]
        applicable = expected[dimension_id]
        level = str(raw.get("level") or "").strip().lower()
        if applicable and level not in LEVELS[:-1]:
            return None
        if not applicable:
            level = "not_applicable"
        dimensions.append(
            {
                "dimension_id": dimension_id,
                "label": label,
                "applicability": "applicable" if applicable else "not_applicable",
                "level": level,
                "evidence": str(raw.get("evidence") or "").strip()[:400]
                if applicable
                else "",
                "feedback": str(raw.get("feedback") or "").strip()[:500]
                if applicable
                else "",
            }
        )
    return {
        "version": LANGUAGE_FEEDBACK_VERSION,
        "response_family": str(profile.get("response_family") or "short_language_response"),
        "feedback_language": str(value.get("feedback_language") or "").strip()[:80],
        "summary": str(value.get("summary") or "").strip()[:700],
        "priority_actions": [
            str(item).strip()[:300]
            for item in (value.get("priority_actions") or [])
            if str(item).strip()
        ][:3],
        "example_revision": str(value.get("example_revision") or "").strip()[:800],
        "dimensions": dimensions,
    }
