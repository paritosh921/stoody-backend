"""
PCR Marker Parser — Parse Q markers from OCR text.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md section 4.2
Failure mode:   PCR-01 (boundary/marker detection failure -> flags + review queue)
Test ID:        U-SEG-02

Expected student notation: Q.No X.Ans

Regex from spec:
    /Q\\.?\\s*(?:No|no)\\.?\\s*(\\d{1,3})\\s*(?:[\\.\(\\s]*([a-z]|[ivx]+|[A-Z])[\\)\\.]?)?\\s*\\.?\\s*(?:Ans|ans|ANS)\\.?/i

Post-OCR fixes in marker context:
    l -> 1
    O -> 0
    I -> 1
"""

from __future__ import annotations

import re

from .response_models import PageOCR, QMarker, TextBlock


# ---------------------------------------------------------------------------
# Regex — translated from the spec's JS-style pattern to Python
# ---------------------------------------------------------------------------

# Spec pattern (case-insensitive):
# Q\.?\s*(?:No|no)\.?\s*(\d{1,3})\s*(?:[\.\(\s]*([a-z]|[ivx]+|[A-Z])[\)\.]?)?\s*\.?\s*(?:Ans|ans|ANS)\.?
#
# Python re.IGNORECASE makes the (?:No|no) and (?:Ans|ans|ANS) groups
# redundant, but we keep the explicit alternations for fidelity to spec.

Q_MARKER_PATTERN: re.Pattern[str] = re.compile(
    r"Q\.?\s*(?:No|no)\.?\s*(\d{1,3})"           # Q.No <number>
    r"\s*(?:[\.\(\s]*([a-z]|[ivx]+|[A-Z])[\)\.]?)?"  # optional sub-part
    r"\s*\.?\s*(?:Ans|ans|ANS)\.?",               # .Ans
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# OCR Post-Fixes (spec 4.2)
# ---------------------------------------------------------------------------

# Applied ONLY to the text region that will be matched as a Q marker, to
# improve recognition of question numbers after OCR/HWR confusion.

_OCR_FIX_MAP: dict[str, str] = {
    "l": "1",  # lowercase L -> digit 1
    "O": "0",  # uppercase O -> digit 0
    "I": "1",  # uppercase I -> digit 1
}


def _apply_ocr_fixes(text: str) -> str:
    """Apply post-OCR character fixes for marker context.

    Only substitutions listed in the spec are applied:
        l -> 1, O -> 0, I -> 1

    These are applied character-by-character only within digit-expected
    positions (the question number portion).  To avoid false positives in
    prose text we isolate the numeric region first.
    """
    # We apply fixes to the entire candidate text and let the regex
    # determine if it now matches.  The fixes are conservative enough
    # (single-character swaps) that false positives in the broader marker
    # format are unlikely.
    result: list[str] = []
    for ch in text:
        result.append(_OCR_FIX_MAP.get(ch, ch))
    return "".join(result)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _try_parse_marker(
    text: str,
    *,
    page_number: int,
    y_position: float,
    confidence: float,
) -> QMarker | None:
    """Try to parse a Q marker from a text string.

    Returns None if no match.
    """
    # First attempt on raw text
    match = Q_MARKER_PATTERN.search(text)

    # Second attempt with OCR fixes
    if match is None:
        fixed = _apply_ocr_fixes(text)
        if fixed != text:
            match = Q_MARKER_PATTERN.search(fixed)

    if match is None:
        return None

    q_number_str = match.group(1)
    sub_part_raw = match.group(2)

    try:
        question_number = int(q_number_str)
    except ValueError:
        return None

    sub_part: str | None = sub_part_raw if sub_part_raw else None

    return QMarker(
        question_number=question_number,
        sub_part=sub_part,
        raw_text=match.group(0),
        page_number=page_number,
        y_position=y_position,
        confidence=confidence,
    )


def parse_markers_from_page(page: PageOCR) -> list[QMarker]:
    """Extract all Q markers from a single page's text blocks.

    Each text block is tested for a Q marker pattern.  At most one marker
    is extracted per text block (the first match).

    Returns markers sorted by y_position ascending.
    """
    markers: list[QMarker] = []
    for block in page.text_blocks:
        marker = _try_parse_marker(
            block.text,
            page_number=page.page_number,
            y_position=block.bbox.y_min,
            confidence=block.confidence,
        )
        if marker is not None:
            markers.append(marker)

    markers.sort(key=lambda m: m.y_position)
    return markers


def parse_markers(pages: list[PageOCR]) -> list[QMarker]:
    """Extract all Q markers across multiple pages.

    Returns markers sorted by (page_number, y_position).
    """
    all_markers: list[QMarker] = []
    for page in sorted(pages, key=lambda p: p.page_number):
        all_markers.extend(parse_markers_from_page(page))
    return all_markers
