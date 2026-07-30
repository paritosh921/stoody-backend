"""Conservative grouping for full-document answer-sheet OCR text."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


ANSWER_START_RE = re.compile(
    r"^\s*(?:(?:ans(?:wer)?|sol(?:ution)?|exp(?:lanation)?)\s*[:\-]?\s*)?(\d{1,3})[\.\)]\s*(.*)$",
    re.IGNORECASE,
)
ANCHOR_CUE_RE = re.compile(r"\b(ans(?:wer)?|sol(?:ution)?|exp(?:lanation)?|worked|exp)\b", re.IGNORECASE)
NUMBERED_ITEM_RE = re.compile(r"^\s*(\d{1,3})[\.\)]\s*(.*)$")


class AnswerSheetBlockNormalizer:
    """Group page OCR text into numbered answer blocks when cues are clear."""

    def normalize(
        self,
        *,
        pages: List[Dict[str, Any]],
        layout_report: Optional[Dict[str, Any]] = None,
        anchor_text: Optional[str] = None,
        question_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        blocks: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        anchor_pattern = self._anchor_pattern(anchor_text)

        for page in pages or []:
            page_no = int(page.get("index", page.get("page", 0)) or 0)
            for line in str(page.get("markdown", "") or "").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                match = ANSWER_START_RE.match(stripped)
                starts_with_anchor = bool(anchor_pattern and anchor_pattern.search(stripped))
                if match and (starts_with_anchor or ANCHOR_CUE_RE.search(stripped)):
                    if current:
                        blocks.append(current)
                    current = {
                        "number": match.group(1),
                        "text": match.group(2).strip(),
                        "page": page_no,
                        "confidence": 0.82 if starts_with_anchor else 0.72,
                        "reasons": [],
                    }
                    continue
                if current:
                    current["text"] = f"{current['text']}\n{stripped}".strip()
                elif starts_with_anchor:
                    current = {
                        "number": None,
                        "text": stripped,
                        "page": page_no,
                        "confidence": 0.45,
                        "reasons": ["anchor_without_answer_number"],
                    }
        if current:
            blocks.append(current)

        # Some schools upload an annotated copy of the question paper as the
        # answer key: each numbered question is followed by its worked answer,
        # but there are no "Answer 1" / "Solution 1" labels.  Do not infer that
        # shape from arbitrary numbered working steps.  It is only accepted
        # when the saved question list gives us a complete, consecutive set of
        # expected question numbers.  The resulting mappings remain review-only
        # because the OCR cannot reliably separate every prompt from its answer.
        annotated_blocks = self._annotated_question_paper_blocks(
            pages=pages,
            question_docs=question_docs or [],
        )
        annotated_question_paper_detected = len(annotated_blocks) > len(blocks)
        if annotated_question_paper_detected:
            blocks = annotated_blocks

        duplicate_numbers = sorted(
            {
                str(block.get("number"))
                for block in blocks
                if block.get("number") and [b.get("number") for b in blocks].count(block.get("number")) > 1
            }
        )
        manual_review_required = bool(duplicate_numbers)
        return {
            "answers": blocks,
            "answer_count": len(blocks),
            "duplicate_numbers": duplicate_numbers,
            "manual_review_required": manual_review_required,
            "normalizer": (
                "annotated_question_paper_block_normalizer"
                if annotated_question_paper_detected
                else "answer_sheet_block_normalizer"
            ),
            "annotated_question_paper_detected": annotated_question_paper_detected,
        }

    def _annotated_question_paper_blocks(
        self,
        *,
        pages: List[Dict[str, Any]],
        question_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        expected_numbers = self._expected_question_numbers(question_docs)
        if len(expected_numbers) < 2:
            return []

        blocks: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        expected_index = 0
        for page in pages or []:
            page_no = int(page.get("index", page.get("page", 0)) or 0)
            for line in str(page.get("markdown", "") or "").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                match = NUMBERED_ITEM_RE.match(stripped)
                expected_number = (
                    expected_numbers[expected_index]
                    if expected_index < len(expected_numbers)
                    else None
                )
                if match and match.group(1) == expected_number:
                    if current:
                        blocks.append(current)
                    current = {
                        "number": expected_number,
                        "text": match.group(2).strip(),
                        "page": page_no,
                        "confidence": 0.62,
                        "manual_review_required": True,
                        "reasons": ["annotated_question_paper_requires_review"],
                    }
                    expected_index += 1
                    continue
                if current:
                    current["text"] = f"{current['text']}\n{stripped}".strip()

        if current:
            blocks.append(current)

        # A partial sequence is more likely to be numbered working than a
        # question-and-solution document, so leave it to the normal parser.
        if [str(block.get("number") or "") for block in blocks] != expected_numbers:
            return []
        return blocks

    def _expected_question_numbers(self, question_docs: List[Dict[str, Any]]) -> List[str]:
        expected: List[str] = []
        for index, question in enumerate(question_docs or [], start=1):
            raw_number = question.get("question_number") or question.get("extraction_order") or index
            try:
                number = str(int(raw_number))
            except (TypeError, ValueError):
                return []
            expected.append(number)
        return expected

    def _layout_has_answer_anchor(self, layout_report: Optional[Dict[str, Any]], number: str) -> bool:
        if not layout_report:
            return False
        for page in layout_report.get("pages", []) or []:
            for anchor in page.get("answer_anchors", []) or []:
                if str(anchor.get("number")) == str(number):
                    return True
        return False

    def _anchor_pattern(self, anchor_text: Optional[str]) -> Optional[re.Pattern[str]]:
        anchor = str(anchor_text or "").strip()
        if not anchor:
            return None
        return re.compile(r"^\s*" + re.escape(anchor) + r"\b", re.IGNORECASE)
