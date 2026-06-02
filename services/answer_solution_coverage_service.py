"""Document-level answer/solution coverage readiness."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


class AnswerSolutionCoverageService:
    """Compute answer readiness independently from question OCR quality."""

    LOW_COVERAGE_THRESHOLD = 0.8

    def compute(
        self,
        *,
        document: Dict[str, Any],
        questions: List[Dict[str, Any]],
        mappings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current_question_ids = {
            str(question.get("id") or question.get("question_id") or "")
            for question in (questions or [])
            if question.get("id") or question.get("question_id")
        }
        question_count = len(current_question_ids) if current_question_ids else len(questions or [])
        all_answer_mappings = [
            mapping
            for mapping in (mappings or [])
            if str(mapping.get("answer_text") or "").strip()
        ]
        answer_source = self._answer_source(document=document, mappings=all_answer_mappings)
        mapped = [
            mapping
            for mapping in all_answer_mappings
            if self._mapping_matches_answer_source(mapping, answer_source)
        ]
        stale_mapping_count = len(
            [
                mapping
                for mapping in mapped
                if current_question_ids
                and self._mapping_question_id(mapping)
                and self._mapping_question_id(mapping) not in current_question_ids
            ]
        )
        mapped_question_ids = {
            self._mapping_question_id(mapping)
            for mapping in mapped
            if self._mapping_question_id(mapping)
            and (not current_question_ids or self._mapping_question_id(mapping) in current_question_ids)
        }
        mapped_answer_count = len(mapped_question_ids)
        manual_review_count = len(
            [
                mapping
                for mapping in mapped
                if mapping.get("manual_review_required")
                and self._mapping_question_id(mapping) in mapped_question_ids
            ]
        )
        expected = self._answers_expected(document=document, answer_source=answer_source)
        score = min(1.0, round(mapped_answer_count / question_count, 2)) if question_count else 0.0
        reasons: List[str] = []
        manual_segmentation_recommended = False

        if not expected:
            status = "not_expected"
        elif question_count == 0:
            status = "pending"
            reasons.append("question_ocr_pending")
        elif self._processing_pending(document=document, answer_source=answer_source) and mapped_answer_count == 0:
            status = "pending"
            reasons.append(self._pending_reason(answer_source))
        elif mapped_answer_count == 0:
            status = "not_ready"
            reasons.append("no_answers_mapped")
        elif manual_review_count > 0:
            status = "needs_review"
            reasons.append("mapped_answers_need_manual_review")
        elif score < self.LOW_COVERAGE_THRESHOLD:
            status = "not_ready"
            reasons.append("low_answer_mapping_coverage")
        elif mapped_answer_count < question_count:
            status = "needs_review"
            reasons.append("some_questions_without_mapped_answers")
        else:
            status = "ready"
        if stale_mapping_count:
            reasons.append("stale_answer_mappings_ignored")

        if (
            document.get("answer_sheet_path")
            and answer_source in {"upload", "manual"}
            and status == "not_ready"
        ):
            manual_segmentation_recommended = True
            if "manual_answer_segmentation_recommended" not in reasons:
                reasons.append("manual_answer_segmentation_recommended")

        summary = {
            "question_count": question_count,
            "mapped_answer_count": mapped_answer_count,
            "manual_review_count": manual_review_count,
            "stale_mapping_count": stale_mapping_count,
            "answer_source": answer_source,
            "reasons": sorted(set(reasons)),
            "manual_segmentation_recommended": manual_segmentation_recommended,
        }
        return {
            "answer_solution_coverage_status": status,
            "answer_solution_coverage_score": score,
            "answer_solution_coverage_summary": summary,
            "answer_solution_coverage_updated_at": datetime.utcnow(),
        }

    def _answers_expected(self, *, document: Dict[str, Any], answer_source: str) -> bool:
        mode = str(document.get("answer_solution_mode") or "").strip().lower()
        if mode in {"upload", "auto"}:
            return True
        return bool(document.get("answer_sheet_path")) or answer_source != "none"

    def _answer_source(self, *, document: Dict[str, Any], mappings: List[Dict[str, Any]]) -> str:
        mode = str(document.get("answer_solution_mode") or "").strip().lower()
        sources = {
            str(mapping.get("source") or "").strip().lower()
            for mapping in (mappings or [])
            if mapping.get("source")
        }
        strategies = {
            str(mapping.get("mapping_strategy") or "").strip().lower()
            for mapping in (mappings or [])
            if mapping.get("mapping_strategy")
        }
        if mode == "auto":
            return "generated"
        if mode == "upload" or document.get("answer_sheet_path"):
            if "manual_answer_segmentation" in sources or document.get("answer_sheet_processed_regions_count"):
                return "manual"
            return "upload"
        if "ai_generated" in sources or document.get("generated_solutions_count"):
            return "generated"
        if "manual_answer_segmentation" in sources or document.get("answer_sheet_processed_regions_count"):
            return "manual"
        if strategies:
            return "manual"
        return "none"

    def _mapping_matches_answer_source(self, mapping: Dict[str, Any], answer_source: str) -> bool:
        source = str(mapping.get("source") or "").strip().lower()
        strategy = str(mapping.get("mapping_strategy") or "").strip().lower()
        if answer_source == "generated":
            return source == "ai_generated" or strategy == "ai_generated_solution"
        if source == "ai_generated" or strategy == "ai_generated_solution":
            return False
        if answer_source == "manual":
            return source in {"manual_answer_segmentation", ""} or strategy in {"question_number", "region_order"}
        if answer_source == "upload":
            return source in {"answer_sheet", "upload", "uploaded_answer_sheet", "manual_answer_segmentation", ""} or strategy in {"question_number", "region_order"}
        return False

    def _mapping_question_id(self, mapping: Dict[str, Any]) -> str:
        return str(mapping.get("question_id") or mapping.get("question_region_id") or "").strip()

    def _processing_pending(self, *, document: Dict[str, Any], answer_source: str) -> bool:
        if answer_source == "generated":
            return str(document.get("generated_solutions_status") or "not_generated") in {
                "not_generated",
                "processing",
            }
        if answer_source == "upload":
            return str(document.get("answer_sheet_ocr_status") or "not_processed") in {
                "not_processed",
                "processing",
            }
        return False

    def _pending_reason(self, answer_source: str) -> str:
        if answer_source == "generated":
            return "generated_solutions_pending"
        if answer_source == "upload":
            return "answer_sheet_ocr_pending"
        return "answer_mapping_pending"
