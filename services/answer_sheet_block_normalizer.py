"""Conservative grouping for full-document answer-sheet OCR text."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


ANSWER_START_RE = re.compile(
    r"^\s*(?:(?:ans(?:wer)?|sol(?:ution)?|exp(?:lanation)?)\s*[:\-]?\s*)?(\d{1,3})[\.\)]\s*(.*)$",
    re.IGNORECASE,
)
ANCHOR_CUE_RE = re.compile(r"\b(ans(?:wer)?|sol(?:ution)?|exp(?:lanation)?|worked|exp)\b", re.IGNORECASE)


class AnswerSheetBlockNormalizer:
    """Group page OCR text into numbered answer blocks when cues are clear."""

    def normalize(
        self,
        *,
        pages: List[Dict[str, Any]],
        layout_report: Optional[Dict[str, Any]] = None,
        anchor_text: Optional[str] = None,
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
            "normalizer": "answer_sheet_block_normalizer",
        }

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
