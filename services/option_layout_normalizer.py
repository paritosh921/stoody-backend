"""Deterministic correction for simple staggered MCQ option layouts."""

from __future__ import annotations

from typing import Any, Dict, List

from services.layout_preflight_service import group_text_items_into_lines


class OptionLayoutNormalizer:
    """Attach label-only option markers to nearby unassigned option text."""

    def correct(
        self,
        *,
        text_items: List[Dict[str, Any]],
        layout_report: Dict[str, Any],
        min_confidence: float = 0.75,
    ) -> Dict[str, Any]:
        if "formula_or_image_dependency" in (layout_report or {}).get("layout_risks", []):
            return {
                "options_by_label": {},
                "corrections": [],
                "confidence": 0.0,
                "manual_review_required": True,
                "reason": "formula_or_image_dependency",
            }

        lines = group_text_items_into_lines(text_items)
        options_by_label: Dict[str, str] = {}
        corrections: List[Dict[str, Any]] = []
        consumed_unlabelled: set[int] = set()
        ambiguous = False

        for idx, line in enumerate(lines):
            label = line.get("option_label")
            if not label:
                continue
            option_text = str(line.get("option_text", "") or "").strip()
            if option_text:
                options_by_label[label] = option_text
                continue

            candidate_indices = self._previous_unassigned_text_lines(lines, idx, consumed_unlabelled)
            if not candidate_indices:
                ambiguous = True
                continue

            candidate_lines = [lines[candidate_idx] for candidate_idx in candidate_indices]
            confidence = min(self._confidence(candidate, line) for candidate in candidate_lines)
            if confidence < min_confidence:
                ambiguous = True
                continue
            for candidate_idx in candidate_indices:
                consumed_unlabelled.add(candidate_idx)
            source_text = "\n".join(candidate["text"] for candidate in candidate_lines)
            options_by_label[label] = source_text
            corrections.append(
                {
                    "correction": "label_only_line_attached_to_previous_text",
                    "label": label,
                    "source_line": source_text,
                    "confidence": confidence,
                }
            )

        expected = {"a", "b", "c", "d"}
        overall_confidence = min(
            [c["confidence"] for c in corrections] or [1.0 if expected.issubset(options_by_label) else 0.0]
        )
        manual_review_required = ambiguous or bool(expected - set(options_by_label))
        if corrections and overall_confidence < min_confidence:
            manual_review_required = True

        return {
            "options_by_label": {
                label: options_by_label[label]
                for label in sorted(options_by_label)
            },
            "corrections": corrections,
            "confidence": round(float(overall_confidence), 2),
            "manual_review_required": manual_review_required,
        }

    def _previous_unassigned_text_lines(
        self,
        lines: List[Dict[str, Any]],
        label_idx: int,
        consumed_unlabelled: set[int],
    ) -> List[int]:
        candidate_indices: List[int] = []
        for idx in range(label_idx - 1, -1, -1):
            if idx in consumed_unlabelled:
                continue
            line = lines[idx]
            if line.get("option_label"):
                break
            if str(line.get("text", "") or "").strip():
                candidate_indices.append(idx)
                continue
            if candidate_indices:
                break
        return list(reversed(candidate_indices))

    def _confidence(self, text_line: Dict[str, Any], label_line: Dict[str, Any]) -> float:
        x_gap = abs(float(text_line.get("x", 0)) - float(label_line.get("x", 0)))
        y_gap = abs(float(label_line.get("y", 0)) - float(text_line.get("y", 0)))
        confidence = 0.92
        if x_gap > 35:
            confidence -= 0.12
        if y_gap > 32:
            confidence -= 0.20
        return round(max(0.0, min(1.0, confidence)), 2)
