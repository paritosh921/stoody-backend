"""Validation layer for structured manual-region OCR output."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ExtractionValidator:
    """Validate question extraction before results are treated as trusted."""

    def validate_question(
        self,
        *,
        question_text: str,
        options: List[str],
        layout_report: Optional[Dict[str, Any]] = None,
        expected_option_count: Optional[int] = None,
        has_figure: bool = False,
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        normalized_options = [str(option or "").strip() for option in (options or [])]
        non_empty_options = [option for option in normalized_options if option]

        if not str(question_text or "").strip():
            reasons.append("missing_question_text")

        if expected_option_count is not None and len(normalized_options) != expected_option_count:
            reasons.append("option_count_mismatch")

        if any(not option for option in normalized_options):
            reasons.append("empty_option")

        lowered = [option.lower() for option in non_empty_options]
        if len(set(lowered)) != len(lowered):
            reasons.append("duplicate_option_text")

        if expected_option_count and len(non_empty_options) < expected_option_count:
            question_lower = str(question_text or "").lower()
            swallowed = sum(1 for marker in ("a.", "b.", "c.", "d.", "(a)", "(b)", "(c)", "(d)") if marker in question_lower)
            if swallowed >= 2:
                reasons.append("option_text_swallowed_into_stem")

        layout_risks = (layout_report or {}).get("layout_risks", [])
        if "formula_or_image_dependency" in layout_risks and not has_figure:
            reasons.append("unresolved_formula_or_image_dependency")

        manual_review_required = bool(reasons)
        if "staggered_options" in layout_risks and reasons:
            manual_review_required = True

        return {
            "valid": not reasons,
            "manual_review_required": manual_review_required,
            "reasons": reasons,
            "expected_option_count": expected_option_count,
            "actual_option_count": len(normalized_options),
        }
