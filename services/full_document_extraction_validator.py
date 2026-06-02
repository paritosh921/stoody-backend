"""Quality validation for automatic full-document OCR outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FullDocumentExtractionValidator:
    """Compute review metadata without changing OCR job lifecycle status."""

    def validate_questions(
        self,
        *,
        questions: List[Any],
        layout_report: Optional[Dict[str, Any]] = None,
        expected_option_count: Optional[int] = None,
        skip_option_extraction: bool = False,
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        question_warnings: List[Dict[str, Any]] = []
        question_numbers: List[str] = []
        anchor_count = self._anchor_count(layout_report, "question_anchors")

        for index, question in enumerate(questions or []):
            qid = self._get(question, "id") or f"question-{index + 1}"
            metadata = self._metadata(question)
            qnum = str(self._get(question, "number") or metadata.get("number") or metadata.get("question_number") or "").strip()
            if qnum:
                question_numbers.append(qnum)

            q_reasons: List[str] = []
            text = str(self._get(question, "text") or "").strip()
            options = list(self._get(question, "options") or [])
            if not text:
                q_reasons.append("missing_question_text")
            if not qnum:
                q_reasons.append("missing_question_number")
            if not skip_option_extraction and expected_option_count is not None:
                if len(options) != expected_option_count:
                    q_reasons.append("option_count_mismatch")
                if any(not str(option or "").strip() for option in options):
                    q_reasons.append("empty_option")
                lowered = [str(option or "").strip().lower() for option in options if str(option or "").strip()]
                if len(set(lowered)) != len(lowered):
                    q_reasons.append("duplicate_option_text")
            if self._get(question, "metadata", {}).get("has_figure") and self._layout_has_risk(layout_report, "formula_or_image_dependency"):
                q_reasons.append("figure_or_formula_review")

            if q_reasons:
                question_warnings.append({"question_id": qid, "number": qnum or None, "reasons": q_reasons})
                reasons.extend(q_reasons)

        duplicate_numbers = sorted({number for number in question_numbers if question_numbers.count(number) > 1})
        if duplicate_numbers:
            reasons.append("duplicate_question_numbers")
        if anchor_count and len(questions or []) < anchor_count:
            reasons.append("question_count_lower_than_layout_anchors")

        return self._summary(
            total=len(questions or []),
            warnings=question_warnings,
            reasons=reasons,
            anchor_count=anchor_count,
            scope="question",
        )

    def validate_answer_sheet(
        self,
        *,
        extracted_text: str,
        page_summaries: List[Dict[str, Any]],
        layout_report: Optional[Dict[str, Any]] = None,
        mapped_count: int = 0,
        question_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        answer_anchor_count = self._anchor_count(layout_report, "answer_anchors")
        text_length = len(str(extracted_text or "").strip())
        if text_length == 0:
            reasons.append("empty_answer_sheet_text")
        if page_summaries and text_length < max(80, len(page_summaries) * 40):
            reasons.append("very_low_answer_text_density")
        if answer_anchor_count and mapped_count and mapped_count < answer_anchor_count:
            reasons.append("mapped_answer_count_lower_than_answer_anchors")
        if question_count is not None and answer_anchor_count and answer_anchor_count != question_count:
            reasons.append("answer_anchor_count_differs_from_question_count")
        if self._layout_has_risk(layout_report, "formula_or_image_dependency"):
            reasons.append("formula_or_image_dependency_review")

        warnings = [{"scope": "answer_sheet", "reasons": sorted(set(reasons))}] if reasons else []
        return self._summary(
            total=len(page_summaries or []),
            warnings=warnings,
            reasons=reasons,
            anchor_count=answer_anchor_count,
            scope="answer",
        )

    def _summary(
        self,
        *,
        total: int,
        warnings: List[Dict[str, Any]],
        reasons: List[str],
        anchor_count: int,
        scope: str,
    ) -> Dict[str, Any]:
        unique_reasons = sorted(set(reasons))
        warning_count = len(warnings)
        if total <= 0:
            score = 0.0
        else:
            score = max(0.0, min(1.0, 1.0 - (warning_count / max(1, total)) - (0.05 * len(unique_reasons))))
        status = "trusted_draft"
        if unique_reasons:
            status = "needs_review"
        if score < 0.75 or "question_count_lower_than_layout_anchors" in unique_reasons:
            status = "manual_segmentation_recommended"
        if total == 0 or "empty_answer_sheet_text" in unique_reasons:
            status = "failed_validation"
        return {
            "scope": scope,
            "status": status,
            "score": round(score, 2),
            "total_items": total,
            "warning_count": warning_count,
            "anchor_count": anchor_count,
            "reasons": unique_reasons,
            "warnings": warnings[:100],
            "manual_segmentation_recommended": status == "manual_segmentation_recommended",
        }

    def _anchor_count(self, layout_report: Optional[Dict[str, Any]], key: str) -> int:
        if not layout_report:
            return 0
        return sum(len(page.get(key, []) or []) for page in layout_report.get("pages", []) or [])

    def _layout_has_risk(self, layout_report: Optional[Dict[str, Any]], risk: str) -> bool:
        if not layout_report:
            return False
        return risk in (layout_report.get("document_layout_risks") or [])

    def _get(self, obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _metadata(self, obj: Any) -> Dict[str, Any]:
        metadata = self._get(obj, "metadata", {})
        return metadata if isinstance(metadata, dict) else {}
