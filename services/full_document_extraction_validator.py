"""Quality validation for automatic full-document OCR outputs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


CRITICAL_REASONS = {"missing_question_text", "question_count_lower_than_layout_anchors"}
HIGH_REASONS = {"missing_options_detected", "too_few_options_for_objective_question"}
MEDIUM_REASONS = {"incomplete_question_text", "figure_or_formula_review", "missing_question_number"}
LOW_REASONS = {"duplicate_question_numbers", "empty_option", "duplicate_option_text"}
QUESTION_ANCHOR_RE = re.compile(r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})[\.\)]\s+", re.IGNORECASE)
OPTION_LABEL_RE = re.compile(r"^\s*[\(\[]?([a-zA-Z])[\.\)]\s*(.*)$")
SUBPART_VERB_RE = re.compile(
    r"^\s*(prove|show|derive|explain|calculate|find|determine|write|draw|state|discuss|verify)\b",
    re.IGNORECASE,
)


class FullDocumentExtractionValidator:
    """Compute review metadata without changing OCR job lifecycle status."""

    def validate_questions(
        self,
        *,
        questions: List[Any],
        layout_report: Optional[Dict[str, Any]] = None,
        ocr_result: Optional[Dict[str, Any]] = None,
        option_evidence_by_question: Optional[Dict[str, Dict[str, Any]]] = None,
        skip_option_extraction: bool = False,
        objective_questions: bool = True,
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        question_warnings: List[Dict[str, Any]] = []
        question_numbers: List[str] = []
        anchor_count = self._anchor_count(layout_report, "question_anchors")
        option_evidence_by_question = option_evidence_by_question or self.option_evidence_by_question(
            layout_report,
            ocr_result=ocr_result,
        )

        for index, question in enumerate(questions or []):
            qid = self._get(question, "id") or f"question-{index + 1}"
            metadata = self._metadata(question)
            qnum = str(self._get(question, "number") or metadata.get("number") or metadata.get("question_number") or "").strip()
            if qnum:
                question_numbers.append(qnum)

            q_reasons: List[str] = []
            text = str(self._get(question, "text") or "").strip()
            options = list(self._get(question, "options") or [])
            observed_option_count = len([option for option in options if str(option or "").strip()])
            evidence = option_evidence_by_question.get(qnum) if qnum else None
            expected_option_count = self._expected_option_count(evidence)
            missing_option_labels: List[str] = []
            if not text:
                q_reasons.append("missing_question_text")
            elif self._incomplete_question_text(text):
                q_reasons.append("incomplete_question_text")
            if not qnum:
                q_reasons.append("missing_question_number")
            if not skip_option_extraction and objective_questions:
                if expected_option_count is not None and observed_option_count < expected_option_count:
                    q_reasons.append("missing_options_detected")
                    found_labels = [
                        chr(ord("A") + idx)
                        for idx in range(observed_option_count)
                    ]
                    expected_labels = [
                        chr(ord("A") + idx)
                        for idx in range(expected_option_count)
                    ]
                    missing_option_labels = [label for label in expected_labels if label not in found_labels]
                elif expected_option_count is None and observed_option_count <= 1:
                    q_reasons.append("too_few_options_for_objective_question")
                if any(not str(option or "").strip() for option in options):
                    q_reasons.append("empty_option")
                lowered = [str(option or "").strip().lower() for option in options if str(option or "").strip()]
                if len(set(lowered)) != len(lowered):
                    q_reasons.append("duplicate_option_text")
            if self._get(question, "metadata", {}).get("has_figure") and self._layout_has_risk(layout_report, "formula_or_image_dependency"):
                q_reasons.append("figure_or_formula_review")

            if q_reasons:
                question_warnings.append(
                    {
                        "question_id": qid,
                        "number": qnum or None,
                        "reasons": q_reasons,
                        "reason_severities": {reason: self._severity(reason) for reason in q_reasons},
                        "manual_review_required": True,
                        "manual_segmentation_recommended": any(reason in CRITICAL_REASONS or reason in HIGH_REASONS for reason in q_reasons),
                        "observed_option_count": observed_option_count,
                        "expected_option_count": expected_option_count,
                        "missing_option_labels": missing_option_labels,
                        "option_evidence": evidence or {},
                    }
                )
                reasons.extend(q_reasons)

        duplicate_numbers = sorted({number for number in question_numbers if question_numbers.count(number) > 1})
        if duplicate_numbers:
            reasons.append("duplicate_question_numbers")
        if anchor_count and len(questions or []) < anchor_count:
            reasons.append("question_count_lower_than_layout_anchors")
            question_warnings.append(
                {
                    "question_id": None,
                    "number": None,
                    "reasons": ["question_count_lower_than_layout_anchors"],
                    "reason_severities": {"question_count_lower_than_layout_anchors": "critical"},
                    "manual_review_required": True,
                    "manual_segmentation_recommended": True,
                    "observed_question_count": len(questions or []),
                    "expected_question_count": anchor_count,
                }
            )

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
        if answer_anchor_count and mapped_count < answer_anchor_count:
            reasons.append("mapped_answer_count_lower_than_answer_anchors")
        if question_count and mapped_count == 0:
            reasons.append("no_answers_mapped")
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
        severity_points = sum(self._severity_weight(reason) for reason in reasons)
        high_issue_count = sum(1 for reason in reasons if reason in HIGH_REASONS)
        has_critical = any(reason in CRITICAL_REASONS for reason in reasons)
        if total <= 0:
            score = 0.0
        else:
            score = max(0.0, min(1.0, 1.0 - (severity_points / max(1.0, total * 3.0))))
        status = "trusted_draft"
        if unique_reasons:
            status = "needs_review"
        if has_critical or score < 0.75 or high_issue_count >= max(2, int(max(total, 1) * 0.2)):
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

    def option_evidence_by_question(
        self,
        layout_report: Optional[Dict[str, Any]],
        *,
        ocr_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        evidence = self._layout_option_evidence_by_question(layout_report)
        ocr_evidence = self._ocr_option_evidence_by_question(ocr_result)
        return self._merge_option_evidence(evidence, ocr_evidence)

    def _layout_option_evidence_by_question(self, layout_report: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        evidence: Dict[str, Dict[str, Any]] = {}
        if not layout_report:
            return evidence
        for page in layout_report.get("pages", []) or []:
            for item in page.get("question_option_evidence", []) or []:
                number = str(item.get("question_number") or "").strip()
                if number:
                    evidence[number] = item
        return evidence

    def _ocr_option_evidence_by_question(self, ocr_result: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not ocr_result:
            return {}
        lines: List[Dict[str, Any]] = []
        for page_index, page in enumerate(ocr_result.get("pages", []) or []):
            markdown = str(page.get("markdown") or "")
            for line_index, line in enumerate(markdown.splitlines()):
                stripped = line.strip()
                if stripped:
                    lines.append({"text": stripped, "page": page_index + 1, "index": len(lines), "line": line_index})
        question_lines: List[Dict[str, Any]] = []
        option_lines: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines):
            text = str(line.get("text") or "")
            q_match = QUESTION_ANCHOR_RE.match(text)
            if q_match:
                question_lines.append({"index": idx, "number": q_match.group(1)})
            option_match = OPTION_LABEL_RE.match(text)
            if option_match:
                option_lines.append(
                    {
                        "index": idx,
                        "label": option_match.group(1).lower(),
                        "text": option_match.group(2).strip(),
                    }
                )

        evidence: Dict[str, Dict[str, Any]] = {}
        for idx, question in enumerate(question_lines):
            start = int(question["index"])
            end = int(question_lines[idx + 1]["index"]) if idx + 1 < len(question_lines) else 10**9
            scoped_options = [
                option
                for option in option_lines
                if start < int(option["index"]) < end
            ]
            labels: List[str] = []
            for option in scoped_options:
                label = str(option.get("label") or "").lower()
                if label and label not in labels:
                    labels.append(label)
            if not labels:
                continue
            expected_count: Optional[int] = None
            confidence = 0.0
            missing_labels: List[str] = []
            if len(labels) >= 2 and labels[0] == "a":
                ordinals = [ord(label) - ord("a") for label in labels if len(label) == 1]
                contiguous = ordinals == list(range(0, len(ordinals)))
                option_like_ratio = (
                    sum(1 for option in scoped_options if self._option_text_looks_like_choice(option.get("text")))
                    / max(1, len(scoped_options))
                )
                if contiguous and len(labels) >= 3 and option_like_ratio >= 0.75:
                    expected_count = len(labels)
                    confidence = round(min(0.92, 0.64 + (0.04 * len(labels)) + (0.12 * option_like_ratio)), 2)
                elif contiguous:
                    confidence = 0.5
                elif ordinals:
                    missing_labels = [
                        chr(ord("A") + ordinal)
                        for ordinal in range(0, max(ordinals) + 1)
                        if ordinal not in ordinals
                    ]
                    confidence = 0.4
            evidence[str(question.get("number"))] = {
                "question_number": str(question.get("number")),
                "option_labels_found": [label.upper() for label in labels],
                "expected_option_count": expected_count,
                "evidence_confidence": confidence,
                "missing_option_labels": missing_labels,
                "source": "ocr_markdown",
            }
        return evidence

    def _merge_option_evidence(
        self,
        layout_evidence: Dict[str, Dict[str, Any]],
        ocr_evidence: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for number in sorted(set(layout_evidence) | set(ocr_evidence), key=lambda value: (len(value), value)):
            layout_item = layout_evidence.get(number) or {}
            ocr_item = ocr_evidence.get(number) or {}
            chosen = self._stronger_option_evidence(layout_item, ocr_item)
            if not chosen:
                continue
            labels: List[str] = []
            for item in (layout_item, ocr_item):
                for label in item.get("option_labels_found", []) or []:
                    label = str(label or "").upper()
                    if label and label not in labels:
                        labels.append(label)
            sources = [
                source
                for source in [
                    layout_item.get("source") or ("layout" if layout_item else None),
                    ocr_item.get("source") or ("ocr_markdown" if ocr_item else None),
                ]
                if source
            ]
            combined = dict(chosen)
            combined["option_labels_found"] = labels or chosen.get("option_labels_found", [])
            combined["sources"] = sources
            if "source" not in combined and sources:
                combined["source"] = sources[0]
            merged[number] = combined
        return merged

    def _stronger_option_evidence(self, first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
        if not first:
            return second
        if not second:
            return first
        first_expected = self._expected_option_count(first)
        second_expected = self._expected_option_count(second)
        if first_expected is not None and second_expected is None:
            return first
        if second_expected is not None and first_expected is None:
            return second
        first_confidence = float(first.get("evidence_confidence") or 0)
        second_confidence = float(second.get("evidence_confidence") or 0)
        return first if first_confidence >= second_confidence else second

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

    def _expected_option_count(self, evidence: Optional[Dict[str, Any]]) -> Optional[int]:
        if not evidence:
            return None
        confidence = float(evidence.get("evidence_confidence") or 0)
        expected = evidence.get("expected_option_count")
        if expected is None or confidence < 0.7:
            return None
        try:
            return int(expected)
        except (TypeError, ValueError):
            return None

    def _incomplete_question_text(self, text: str) -> bool:
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return True
        words = [word for word in normalized.split(" ") if word]
        if len(words) < 3 or len(normalized) < 12:
            return True
        trailing = normalized.rstrip().lower()
        if trailing.endswith((",", ";", ":", "-", "(", "[")):
            return True
        last_word = trailing.split()[-1].strip(".,;:()[]")
        return last_word in {"of", "the", "and", "or", "if", "then", "where", "when", "which"}

    def _option_text_looks_like_choice(self, text: Any) -> bool:
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return False
        if len(normalized) > 160:
            return False
        if normalized.endswith(":"):
            return False
        return not SUBPART_VERB_RE.match(normalized)

    def _severity(self, reason: str) -> str:
        if reason in CRITICAL_REASONS:
            return "critical"
        if reason in HIGH_REASONS:
            return "high"
        if reason in MEDIUM_REASONS:
            return "medium"
        return "low"

    def _severity_weight(self, reason: str) -> float:
        severity = self._severity(reason)
        return {
            "critical": 3.0,
            "high": 1.5,
            "medium": 0.75,
            "low": 0.25,
        }[severity]
