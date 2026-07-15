"""
PCR Marker Parser — Parse Q markers from OCR text.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md section 4.2
Failure mode:   PCR-01 (boundary/marker detection failure -> flags + review queue)
Test ID:        U-SEG-02

Preferred student notation: Q.No X.Ans.  Common handwritten variants such as
Q1, Q. 1, Question 1, and Question No. 1 are accepted as well.

Real student answer books frequently use plain numbered labels instead of a Q
prefix (``1)``, ``2.``, ``Ans 3)``).  The content-section answer-sheet mapper
already treats those as first-class anchors; PCR must do the same or multi-
question copies collapse into the wrong question / form-header noise.

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

# The original strict spec notation remains accepted, but student copies are
# frequently labelled simply "Q1" or "Question 1".  Marker recognition must
# be tolerant while still requiring an explicit Q/Question prefix so normal
# numbered working is never mistaken for a new answer.

Q_MARKER_PATTERN: re.Pattern[str] = re.compile(
    r"\bQ(?:uestion)?\.?\s*(?:No\.?\s*)?(\d{1,3})"  # Q1 / Q.No 1 / Question 1
    r"\s*(?:\(\s*([a-z]|[ivx]+|[A-Z])\s*\)|\.\s*([a-z]|[ivx]+|[A-Z])(?=\s*\.?\s*(?:Ans(?:wer)?)\b|\s*$))?"  # optional (a) / .a, not .Ans
    r"\s*\.?\s*(?:Ans(?:wer)?)?\.?",                    # optional .Ans / .Answer
    re.IGNORECASE,
)

# Content-section style numbered answer anchors used on handwritten answer
# books: "1)", "2.", "Ans 3)", "Answer: 4.", "Sol 5)".  Require a terminal
# "." or ")" so prose like "The answer is 42" never becomes a marker.
ANSWER_NUMBER_MARKER_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:(?:ans(?:wer)?|sol(?:ution)?)\s*[:.\-]?\s*)?"
    r"(\d{1,3})\s*[\.\)]"  # 1) / 2. / Ans 3)
    r"(?:\s+|$)"  # space before answer body, or label-only line
    r"(?!\s*(?:of|\/)\s*\d)",  # reject "1 of 10" / "1/10" page counters
    re.IGNORECASE | re.MULTILINE,
)

# Any marker pattern used when splitting a multi-answer OCR blob.
ANY_ANSWER_MARKER_PATTERN: re.Pattern[str] = re.compile(
    r"(?:"
    r"\bQ(?:uestion)?\.?\s*(?:No\.?\s*)?\d{1,3}"
    r"|"
    r"^\s*(?:(?:ans(?:wer)?|sol(?:ution)?)\s*[:.\-]?\s*)?\d{1,3}\s*[\.\)](?:\s+|$)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Printed answer-book form chrome that must never be graded as student work.
_FORM_HEADER_TOKEN_RE = re.compile(
    r"^(?:"
    r"name|date|page|class|roll(?:\s*no\.?)?|section|subject|school|"
    r"student(?:\s*name)?|admission(?:\s*no\.?)?|exam|paper|marks|"
    r"answer\s*book|answer\s*sheet|prayaan|stoody|book|"
    r"signature|invigilator|candidate"
    r")[\s:.\-]*$",
    re.IGNORECASE,
)
_FORM_HEADER_LINE_RE = re.compile(
    r"^(?:"
    r"(?:name|date|page|class|roll(?:\s*no\.?)?|section|subject|school|"
    r"student(?:\s*name)?|answer\s*book|answer\s*sheet|prayaan|stoody)"
    r"(?:\s+(?:name|date|page|class|roll(?:\s*no\.?)?|section|subject|"
    r"school|student|answer\s*book|answer\s*sheet|prayaan|stoody|book))*"
    r")\s*$",
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


def is_form_header_text(text: str) -> bool:
    """Return True when OCR text is printed answer-book chrome, not working.

    Production failures show the first page header ("Prayaan Answer Book Date
    Page") being treated as Q1 evidence.  Those tokens must never reach the
    marker or the auto-grader as student content.
    """
    cleaned = " ".join(str(text or "").split()).strip(" :-_\t")
    if not cleaned:
        return True
    if _FORM_HEADER_LINE_RE.fullmatch(cleaned):
        return True
    tokens = [token for token in re.split(r"[\s/|,;]+", cleaned) if token]
    if not tokens:
        return True
    # Entire line is only form-field labels / book title words.
    return all(_FORM_HEADER_TOKEN_RE.fullmatch(token) for token in tokens)


def strip_form_header_noise(text: str) -> str:
    """Drop leading/trailing form-header lines from an OCR blob."""
    if not text:
        return ""
    lines = [line.strip() for line in str(text).splitlines()]
    kept = [line for line in lines if line and not is_form_header_text(line)]
    if kept:
        return "\n".join(kept).strip()
    # Single-line blobs often arrive as space-joined form labels.
    if is_form_header_text(text):
        return ""
    # Fall back to token filtering for space-joined header chrome.
    tokens = [token for token in str(text).split() if not is_form_header_text(token)]
    return " ".join(tokens).strip()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _match_q_marker(text: str) -> re.Match[str] | None:
    match = Q_MARKER_PATTERN.search(text)
    if match is not None:
        return match
    fixed = _apply_ocr_fixes(text)
    if fixed != text:
        return Q_MARKER_PATTERN.search(fixed)
    return None


def _match_answer_number_marker(text: str) -> re.Match[str] | None:
    """Match content-style numbered answer labels at the start of a block."""
    if is_form_header_text(text):
        return None
    match = ANSWER_NUMBER_MARKER_PATTERN.search(text)
    if match is not None:
        return match
    fixed = _apply_ocr_fixes(text)
    if fixed != text:
        return ANSWER_NUMBER_MARKER_PATTERN.search(fixed)
    return None


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
    if is_form_header_text(text):
        return None

    # Prefer explicit Q/Question markers (spec path).
    match = _match_q_marker(text)
    sub_part: str | None = None
    if match is not None:
        q_number_str = match.group(1)
        # Group 2 is the usual parenthesised sub-part, while group 3 supports
        # compact labels such as ``Q1.a``.  Keep one normalized output field.
        sub_part_raw = match.group(2) or match.group(3)
        sub_part = sub_part_raw if sub_part_raw else None
        raw_text = match.group(0)
    else:
        # Fall back to plain numbered answer labels used on answer books.
        match = _match_answer_number_marker(text)
        if match is None:
            return None
        q_number_str = match.group(1)
        raw_text = match.group(0)

    try:
        question_number = int(q_number_str)
    except ValueError:
        return None

    if question_number < 1:
        return None

    return QMarker(
        question_number=question_number,
        sub_part=sub_part,
        raw_text=raw_text,
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
    seen_numbers: set[int] = set()
    for block in page.text_blocks:
        if is_form_header_text(block.text):
            continue
        marker = _try_parse_marker(
            block.text,
            page_number=page.page_number,
            y_position=block.bbox.y_min,
            confidence=block.confidence,
        )
        if marker is None:
            continue
        # Keep the first (top-most) occurrence of each number on the page so a
        # repeated working line does not create duplicate answer ownership.
        if marker.question_number in seen_numbers:
            continue
        seen_numbers.add(marker.question_number)
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


def find_answer_marker_spans(text: str) -> list[re.Match[str]]:
    """Return non-overlapping marker matches suitable for splitting a blob."""
    if not text:
        return []
    # Prefer explicit Q markers inside a block; if none, fall back to numbered
    # answer labels.  Mixing both often double-splits one answer.
    q_matches = list(Q_MARKER_PATTERN.finditer(text))
    if len(q_matches) > 1:
        return q_matches
    if len(q_matches) == 1:
        return q_matches
    return list(ANSWER_NUMBER_MARKER_PATTERN.finditer(text))
