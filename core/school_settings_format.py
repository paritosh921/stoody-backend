"""Formatting helpers for school academic settings."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


_CLASS_PREFIX_RE = re.compile(r"^\s*class\s+", re.IGNORECASE)
_NUMERIC_CLASS_RE = re.compile(r"^(0|[1-9]\d*)$")
_ROMAN_CLASS_RE = re.compile(
    r"^(?=[IVXLCDM]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)


def _format_class_label(label: str) -> str:
    return label.upper() if _ROMAN_CLASS_RE.fullmatch(label) else label


def normalize_class_label(value: Any) -> str:
    """Return the stored class label without presentation-only prefixes."""
    label = str(value or "").strip()
    while label:
        normalized = _CLASS_PREFIX_RE.sub("", label, count=1).strip()
        if normalized == label:
            return _format_class_label(label)
        label = normalized
    return ""


def is_class_number_format(value: Any) -> bool:
    label = normalize_class_label(value)
    return bool(label and (_NUMERIC_CLASS_RE.fullmatch(label) or _ROMAN_CLASS_RE.fullmatch(label)))


def clean_class_values(values: Optional[Iterable[Any]]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for value in values or []:
        item = normalize_class_label(value)
        if item and is_class_number_format(item) and item not in seen:
            cleaned.append(item)
            seen.add(item)
    return cleaned


def validate_class_values(values: Optional[Iterable[Any]]) -> List[str]:
    invalid: List[str] = []
    seen_invalid = set()
    for value in values or []:
        item = normalize_class_label(value)
        if item and not is_class_number_format(item) and item not in seen_invalid:
            invalid.append(item)
            seen_invalid.add(item)
    if invalid:
        examples = ", ".join(invalid)
        raise ValueError(
            "Class must be entered as a number or roman numeral, for example 4 or IV. "
            f"Invalid values: {examples}."
        )
    return clean_class_values(values)


def clean_class_sections(
    class_sections: Optional[Dict[str, Iterable[Any]]],
    *,
    classes: Optional[Iterable[Any]] = None,
    sections: Optional[Iterable[Any]] = None,
) -> Dict[str, List[str]]:
    allowed_classes = set(clean_class_values(classes)) if classes is not None else None
    allowed_sections = {str(section).strip() for section in sections or [] if str(section).strip()}
    cleaned: Dict[str, List[str]] = {}

    for raw_class, raw_sections in (class_sections or {}).items():
        class_label = normalize_class_label(raw_class)
        if not class_label or (allowed_classes is not None and class_label not in allowed_classes):
            continue

        selected: List[str] = []
        seen_sections = set()
        for section in raw_sections or []:
            section_label = str(section).strip()
            if not section_label or section_label in seen_sections:
                continue
            if allowed_sections and section_label not in allowed_sections:
                continue
            selected.append(section_label)
            seen_sections.add(section_label)
        cleaned[class_label] = selected

    return cleaned
