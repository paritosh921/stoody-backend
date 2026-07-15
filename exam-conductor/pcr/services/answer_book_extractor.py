"""Deterministic extraction of numbered student answers from OCR pages.

Student answer books almost always look like::

    Prayaan Answer Book
    1) 30.067
    2) 3630
    3) small diagram ...
    4) Partition ...
    5) Rohit goes to park ...

The content-section answer-sheet pipeline already maps this shape well by
splitting on ``N)`` / ``N.`` labels.  PCR was losing those answers because it
either:
  * required a ``Q`` prefix for markers, or
  * replaced good numbered splits with a fragile multi-page vision remap, or
  * treated sparse handwriting geometry as blocking diagram content.

This module is the **primary** association path for camera/PDF answer copies:
it never awards marks; it only produces evidence-backed ``DetectedResponse``
rows keyed by paper question numbers.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..domain.marker_parser import is_form_header_text, strip_form_header_noise
from ..domain.response_models import (
    ContentType,
    DetectedResponse,
    Flag,
    FlagSeverity,
    FlagType,
    PageOCR,
    SourcePageRef,
    TextBlock,
)


# Line-start labels: "1)", "2.", "Ans 3)", "Q4."
_LINE_ANSWER_START = re.compile(
    r"^\s*(?:(?:ans(?:wer)?|sol(?:ution)?|q(?:uestion)?)\s*[:.\-]?\s*)?"
    r"(\d{1,3})\s*[\.\)]\s*(.*)$",
    re.IGNORECASE,
)

# Same pattern but also mid-string after whitespace (single OCR blob).
_INLINE_ANSWER_START = re.compile(
    r"(?:^|(?<=\s))"
    r"(?:(?:ans(?:wer)?|sol(?:ution)?|q(?:uestion)?)\s*[:.\-]?\s*)?"
    r"(\d{1,3})\s*[\.\)]\s+",
    re.IGNORECASE,
)

# Reject page counters like "1 of 2" / "Page 1/2"
_PAGE_COUNTER = re.compile(
    r"^\s*(?:page\s*)?\d{1,3}\s*(?:of|/)\s*\d{1,3}\s*$",
    re.IGNORECASE,
)


@dataclass
class NumberedAnswerBlock:
    question_number: int
    text: str
    page_number: int
    y_start: float
    y_end: float
    confidence: float


def extract_numbered_answers_from_pages(
    pages: Sequence[PageOCR],
    *,
    valid_question_numbers: Iterable[int],
) -> List[NumberedAnswerBlock]:
    """Split OCR page text into numbered answer blocks for paper Q numbers."""
    valid = {int(n) for n in valid_question_numbers if int(n) >= 1}
    if not pages or not valid:
        return []

    ordered_lines = _collect_ocr_lines(pages)
    if not ordered_lines:
        return []

    # Prefer line-oriented split (most reliable for answer books).
    blocks = _split_lines_into_answers(ordered_lines, valid)
    if len(blocks) < 1:
        # Fallback: whole-document string split for single-blob OCR.
        blocks = _split_blob_into_answers(ordered_lines, valid)

    # Keep first occurrence of each question number (top-to-bottom).
    by_number: Dict[int, NumberedAnswerBlock] = {}
    for block in blocks:
        if block.question_number not in by_number:
            by_number[block.question_number] = block
        else:
            # Continuation without a new label: append body.
            prev = by_number[block.question_number]
            merged_text = f"{prev.text}\n{block.text}".strip()
            by_number[block.question_number] = NumberedAnswerBlock(
                question_number=prev.question_number,
                text=merged_text,
                page_number=prev.page_number,
                y_start=prev.y_start,
                y_end=max(prev.y_end, block.y_end),
                confidence=min(prev.confidence, block.confidence),
            )

    return [
        by_number[n]
        for n in sorted(by_number.keys())
        if strip_form_header_noise(by_number[n].text)
    ]


def build_responses_from_numbered_answers(
    answers: Sequence[NumberedAnswerBlock],
) -> Tuple[List[DetectedResponse], Dict[str, Dict[str, Any]]]:
    """Convert numbered blocks into scoreable DetectedResponse rows."""
    responses: List[DetectedResponse] = []
    assignment: Dict[str, Dict[str, Any]] = {}
    for answer in answers:
        text = strip_form_header_noise(answer.text)
        if not text or is_form_header_text(text):
            continue
        response_id = f"RESP-BOOK-{uuid.uuid4().hex[:12]}"
        region = SourcePageRef(
            page_number=answer.page_number,
            y_start=max(0.0, float(answer.y_start)),
            y_end=max(float(answer.y_start) + 1.0, float(answer.y_end)),
        )
        response = DetectedResponse(
            response_id=response_id,
            question_number=int(answer.question_number),
            sub_part=None,
            detected_text=text,
            source_pages=[region],
            content_type=ContentType.TEXT_ONLY,
            text_coverage_ratio=1.0,
            segmentation_confidence=float(answer.confidence),
            ocr_confidence=float(answer.confidence),
            flags=[],
            word_count=len(text.split()),
            is_continuation=False,
        )
        responses.append(response)
        assignment[response_id] = {
            "method": "answer_book_numbered_extract",
            "question_number": int(answer.question_number),
            "confidence": float(answer.confidence),
            "mapping_basis": "explicit_label",
            "manual_review_required": False,
        }
    return responses, assignment


def try_extract_answer_book_responses(
    pages: Sequence[PageOCR],
    numbered_questions: Sequence[Tuple[int, Dict[str, Any]]],
) -> Optional[Tuple[List[DetectedResponse], Dict[str, Dict[str, Any]]]]:
    """Return responses when the OCR looks like a numbered student answer book.

    Requires at least one valid paper-question number so a single clear
    ``1) ...`` answer is still scored.  Returns ``None`` when the copy does
    not contain usable numbered labels (fall through to vision mapping).
    """
    valid = [int(n) for n, _q in numbered_questions]
    answers = extract_numbered_answers_from_pages(
        pages, valid_question_numbers=valid
    )
    if not answers:
        return None
    responses, assignment = build_responses_from_numbered_answers(answers)
    if not responses:
        return None
    return responses, assignment


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _collect_ocr_lines(
    pages: Sequence[PageOCR],
) -> List[Tuple[int, float, float, str]]:
    """Return (page, y_min, y_max, text) rows top-to-bottom, header-stripped."""
    rows: List[Tuple[int, float, float, str]] = []
    for page in sorted(pages, key=lambda p: p.page_number):
        blocks = sorted(
            page.text_blocks,
            key=lambda b: (b.bbox.y_min, b.bbox.x_min),
        )
        for block in blocks:
            raw = str(block.text or "").strip()
            if not raw:
                continue
            # Expand multi-line OCR blocks into individual logical lines so
            # "1) a\\n2) b" inside one bbox still splits.
            for line in raw.splitlines():
                cleaned = strip_form_header_noise(line.strip())
                if not cleaned or is_form_header_text(cleaned):
                    continue
                if _PAGE_COUNTER.match(cleaned):
                    continue
                rows.append(
                    (
                        int(page.page_number),
                        float(block.bbox.y_min),
                        float(block.bbox.y_max),
                        cleaned,
                    )
                )
    return rows


def _split_lines_into_answers(
    lines: Sequence[Tuple[int, float, float, str]],
    valid: set[int],
) -> List[NumberedAnswerBlock]:
    blocks: List[NumberedAnswerBlock] = []
    current: Optional[Dict[str, Any]] = None

    for page_number, y_min, y_max, text in lines:
        # Inline multi-answer line first: "1) 30.067 2) 3630 3) diagram"
        # Must win over a single line-start match, otherwise Q2..Qn collapse
        # into the body of Q1.
        inline = list(_INLINE_ANSWER_START.finditer(text))
        if len(inline) >= 2:
            if current is not None:
                blocks.append(_to_block(current))
                current = None
            for idx, m in enumerate(inline):
                number = int(m.group(1))
                if number not in valid:
                    continue
                start = m.end()
                end = inline[idx + 1].start() if idx + 1 < len(inline) else len(text)
                body = text[start:end].strip()
                if not body:
                    body = m.group(0).strip()
                blocks.append(
                    NumberedAnswerBlock(
                        question_number=number,
                        text=body,
                        page_number=page_number,
                        y_start=y_min,
                        y_end=y_max,
                        confidence=0.9,
                    )
                )
            continue

        match = _LINE_ANSWER_START.match(text)
        if match:
            number = int(match.group(1))
            body = (match.group(2) or "").strip()
            if number not in valid:
                # Not a paper question label — treat as continuation text if we
                # already have an open answer (working lines like "2+2=4").
                if current is not None:
                    current["text"] = f"{current['text']}\n{text}".strip()
                    current["y_end"] = max(current["y_end"], y_max)
                continue
            if current is not None:
                blocks.append(_to_block(current))
            current = {
                "question_number": number,
                "text": body or text,
                "page_number": page_number,
                "y_start": y_min,
                "y_end": y_max,
                "confidence": 0.93,
            }
            continue

        if current is not None:
            current["text"] = f"{current['text']}\n{text}".strip()
            current["y_end"] = max(current["y_end"], y_max)

    if current is not None:
        blocks.append(_to_block(current))
    return blocks


def _split_blob_into_answers(
    lines: Sequence[Tuple[int, float, float, str]],
    valid: set[int],
) -> List[NumberedAnswerBlock]:
    if not lines:
        return []
    blob = "\n".join(text for _p, _a, _b, text in lines)
    matches = list(_INLINE_ANSWER_START.finditer(blob))
    if not matches:
        return []

    # Approximate page/y from first line for region metadata.
    page_number = lines[0][0]
    y_start = lines[0][1]
    y_end = lines[-1][2]
    height = max(y_end - y_start, 1.0)
    blocks: List[NumberedAnswerBlock] = []
    for idx, m in enumerate(matches):
        number = int(m.group(1))
        if number not in valid:
            continue
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(blob)
        body = blob[start:end].strip()
        if not body:
            continue
        frac = idx / max(len(matches), 1)
        frac_next = (idx + 1) / max(len(matches), 1)
        blocks.append(
            NumberedAnswerBlock(
                question_number=number,
                text=body,
                page_number=page_number,
                y_start=y_start + frac * height,
                y_end=y_start + frac_next * height,
                confidence=0.88,
            )
        )
    return blocks


def _to_block(current: Dict[str, Any]) -> NumberedAnswerBlock:
    return NumberedAnswerBlock(
        question_number=int(current["question_number"]),
        text=str(current.get("text") or "").strip(),
        page_number=int(current["page_number"]),
        y_start=float(current["y_start"]),
        y_end=float(current["y_end"]),
        confidence=float(current.get("confidence") or 0.9),
    )
