"""Map full-document answer-sheet OCR blocks to saved question IDs."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.answer_sheet_vision_mapper import AnswerSheetVisionMapper


class AnswerSheetMappingService:
    """Persist question-addressable mappings from uploaded full answer-sheet OCR."""

    SOURCE = "answer_sheet_full_ocr"
    ACCEPT_THRESHOLD = 0.82
    VISION_ACCEPT_THRESHOLD = 0.86
    REVIEW_THRESHOLD = 0.75

    def __init__(self, vision_mapper: Optional[Any] = None):
        self.vision_mapper = vision_mapper if vision_mapper is not None else AnswerSheetVisionMapper()
        self.vision_enabled = str(os.getenv("ANSWER_MAPPING_VISION_ENABLED", "true")).lower() not in {
            "0",
            "false",
            "no",
        }

    async def map_full_document_blocks(
        self,
        *,
        db: Any,
        is_b2c: bool,
        document_id: str,
        question_docs: List[Dict[str, Any]],
        answer_blocks: List[Dict[str, Any]],
        page_summaries: Optional[List[Dict[str, Any]]] = None,
        layout_report: Optional[Dict[str, Any]] = None,
        pdf_bytes: Optional[bytes] = None,
        gateway_context: Optional[Dict[str, Any]] = None,
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        questions = self._sorted_questions(question_docs)
        blocks = self._normalise_blocks(answer_blocks)
        existing_mappings = await self._find_existing_mappings(db, is_b2c, document_id)
        protected_question_ids = {
            self._mapping_question_id(mapping)
            for mapping in existing_mappings
            if self._is_manual_mapping(mapping) and self._mapping_question_id(mapping)
        }

        deterministic = self._build_deterministic_mappings(
            document_id=document_id,
            questions=questions,
            blocks=blocks,
            protected_question_ids=protected_question_ids,
        )
        vision_reasons = self._vision_reasons(
            questions=questions,
            blocks=blocks,
            mappings=deterministic,
            layout_report=layout_report,
            page_summaries=page_summaries or [],
        )

        vision_result: Dict[str, Any] = {"used": False, "mappings": [], "reasons": vision_reasons}
        if self.vision_enabled and pdf_bytes and hasattr(self.vision_mapper, "extract_by_question"):
            try:
                vision_result = await self.vision_mapper.extract_by_question(
                    pdf_bytes=pdf_bytes,
                    question_docs=questions,
                    page_summaries=page_summaries or [],
                    layout_report=layout_report,
                    gateway_context=gateway_context,
                )
                if vision_result.get("mappings"):
                    deterministic = self._merge_vision_mappings(
                        document_id=document_id,
                        questions=questions,
                        blocks=blocks,
                        deterministic=deterministic,
                        vision_mappings=vision_result.get("mappings") or [],
                        protected_question_ids=protected_question_ids,
                        vision_result=vision_result,
                    )
            except Exception as exc:
                vision_result = {
                    "used": False,
                    "mode": "question_anchored",
                    "error": str(exc),
                    "mappings": [],
                    "reasons": vision_reasons,
                }

        if (
            self.vision_enabled
            and pdf_bytes
            and vision_reasons
            and not (vision_result.get("used") and vision_result.get("mappings"))
        ):
            try:
                vision_result = await self.vision_mapper.map(
                    pdf_bytes=pdf_bytes,
                    question_docs=questions,
                    answer_blocks=blocks,
                    candidate_mappings=deterministic,
                    layout_report=layout_report,
                    reasons=vision_reasons,
                    gateway_context=gateway_context,
                )
                deterministic = self._merge_vision_mappings(
                    document_id=document_id,
                    questions=questions,
                    blocks=blocks,
                    deterministic=deterministic,
                    vision_mappings=vision_result.get("mappings") or [],
                    protected_question_ids=protected_question_ids,
                    vision_result=vision_result,
                )
            except Exception as exc:
                vision_result = {
                    "used": False,
                    "error": str(exc),
                    "mappings": [],
                    "reasons": vision_reasons,
                }

        acceptance_policy = self._document_acceptance_policy(
            questions=questions,
            blocks=blocks,
            mappings=deterministic,
            vision_reasons=vision_reasons,
            vision_result=vision_result,
        )
        deterministic = self._apply_acceptance_policy(
            mappings=deterministic,
            acceptance_policy=acceptance_policy,
        )

        mappings_to_persist = [
            mapping
            for mapping in deterministic
            if mapping.get("question_id")
            and mapping.get("answer_text")
            and str(mapping.get("question_id")) not in protected_question_ids
        ]

        if replace_existing:
            await self._clear_existing_full_ocr_mappings(db, is_b2c, document_id)
        persisted_mappings: List[Dict[str, Any]] = []
        for mapping in mappings_to_persist:
            if await self._upsert_mapping(db, is_b2c, mapping):
                persisted_mappings.append(mapping)

        trusted_mapped_count = len(
            [
                mapping
                for mapping in persisted_mappings
                if mapping.get("answer_text")
                and not mapping.get("manual_review_required")
                and mapping.get("review_status") == "accepted"
            ]
        )
        manual_review_count = len(
            [
                mapping
                for mapping in persisted_mappings
                if mapping.get("manual_review_required") or mapping.get("review_status") == "needs_review"
            ]
        )
        unmatched_answer_count = len([mapping for mapping in deterministic if not mapping.get("question_id")])
        question_anchored_complete = (
            str(vision_result.get("mode") or "") == "question_anchored"
            and len(
                {
                    str(mapping.get("question_id") or "")
                    for mapping in persisted_mappings
                    if mapping.get("question_id") and str(mapping.get("answer_text") or "").strip()
                }
            )
            == len(questions)
        )
        summary = {
            "source": self.SOURCE,
            "answer_blocks_count": len(blocks),
            "question_count": len(questions),
            "candidate_mapping_count": len(mappings_to_persist),
            "persisted_mapping_count": len(persisted_mappings),
            "mapped_count": trusted_mapped_count,
            "manual_review_count": manual_review_count,
            "unmatched_answer_count": unmatched_answer_count,
            "protected_manual_mapping_count": len(protected_question_ids),
            "vision_used": bool(vision_result.get("used")),
            "vision_model": vision_result.get("model"),
            "vision_provider": vision_result.get("provider"),
            "vision_mode": vision_result.get("mode"),
            "vision_reasons": vision_reasons,
            "vision_error": vision_result.get("error"),
            "auto_acceptance_blocked": acceptance_policy.get("auto_acceptance_blocked"),
            "auto_acceptance_blockers": acceptance_policy.get("auto_acceptance_blockers"),
            "manual_segmentation_recommended": bool(
                manual_review_count
                or unmatched_answer_count
                or (len(blocks) != len(questions) and not question_anchored_complete)
                or vision_result.get("error")
                or acceptance_policy.get("auto_acceptance_blocked")
            ),
        }
        return {
            "mappings": persisted_mappings,
            "mapped_count": trusted_mapped_count,
            "manual_review_count": manual_review_count,
            "summary": summary,
        }

    def _build_deterministic_mappings(
        self,
        *,
        document_id: str,
        questions: List[Dict[str, Any]],
        blocks: List[Dict[str, Any]],
        protected_question_ids: set[str],
    ) -> List[Dict[str, Any]]:
        questions_by_number = self._questions_by_number(questions)
        used_question_ids: set[str] = set()
        number_counts = self._answer_number_counts(blocks)
        count_aligned = len(questions) == len(blocks)
        mappings: List[Dict[str, Any]] = []

        for index, block in enumerate(blocks):
            answer_number = str(block.get("number") or "").strip()
            question = questions_by_number.get(answer_number) if answer_number else None
            strategy = "answer_number"
            reasons: List[str] = []
            if question and str(question.get("id") or "") in used_question_ids:
                question = None
                reasons.append("duplicate_question_number_match")
            if question is None:
                strategy = "document_order"
                question = questions[index] if index < len(questions) else None
                if not count_aligned:
                    reasons.append("answer_question_count_mismatch")
            question_id = str((question or {}).get("id") or (question or {}).get("question_id") or "").strip()
            if question_id in protected_question_ids:
                reasons.append("manual_mapping_has_higher_authority")
            if question_id:
                used_question_ids.add(question_id)

            text = str(block.get("text") or block.get("answer_text") or "").strip()
            confidence = self._deterministic_confidence(
                strategy=strategy,
                count_aligned=count_aligned,
                block=block,
                duplicate_number=bool(answer_number and number_counts.get(answer_number, 0) > 1),
                question_id=question_id,
                answer_text=text,
            )
            if not answer_number:
                reasons.append("missing_answer_number")
            elif number_counts.get(answer_number, 0) > 1:
                reasons.append("duplicate_answer_number")
            if not question_id:
                reasons.append("no_question_match")
            if len(text) < 20:
                reasons.append("answer_text_too_short")
            if confidence < self.REVIEW_THRESHOLD:
                reasons.append("low_mapping_confidence")

            manual_review_required = bool(reasons) or confidence < self.ACCEPT_THRESHOLD
            review_status = "accepted" if not manual_review_required and confidence >= self.ACCEPT_THRESHOLD else "needs_review"
            answer_block_id = str(block.get("block_id") or f"answer_block_{index + 1}")
            mappings.append(
                {
                    "mapping_id": f"{document_id}:{question_id or 'unmapped'}:{answer_block_id}",
                    "document_id": document_id,
                    "question_region_id": question_id,
                    "question_id": question_id,
                    "answer_region_id": answer_block_id,
                    "answer_block_id": answer_block_id,
                    "answer_number": answer_number or None,
                    "answer_text": text,
                    "mapping_strategy": strategy,
                    "confidence": confidence,
                    "manual_review_required": manual_review_required,
                    "review_status": review_status,
                    "mapping_reasons": sorted(set(reasons)),
                    "source": self.SOURCE,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
            )
        return mappings

    def _merge_vision_mappings(
        self,
        *,
        document_id: str,
        questions: List[Dict[str, Any]],
        blocks: List[Dict[str, Any]],
        deterministic: List[Dict[str, Any]],
        vision_mappings: List[Dict[str, Any]],
        protected_question_ids: set[str],
        vision_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        by_question = {
            str(mapping.get("question_id") or ""): mapping
            for mapping in deterministic
            if mapping.get("question_id")
        }
        block_by_id = {str(block.get("block_id") or ""): block for block in blocks}
        question_ids = {
            str(question.get("id") or question.get("question_id") or "")
            for question in questions
            if question.get("id") or question.get("question_id")
        }

        for candidate in vision_mappings or []:
            question_id = str(candidate.get("question_id") or "").strip()
            if not question_id or question_id not in question_ids or question_id in protected_question_ids:
                continue
            answer_block_id = str(candidate.get("answer_block_id") or "").strip()
            block = block_by_id.get(answer_block_id, {})
            answer_text = str(candidate.get("answer_text") or block.get("text") or "").strip()
            if not answer_text:
                continue
            confidence = self._float(candidate.get("confidence"), 0.0)
            strategy = str(candidate.get("mapping_strategy") or "gpt_vision_mapper")
            reasons = ["question_anchored_vision_used" if strategy == "gpt_question_anchored" else "vision_mapper_used"]
            candidate_review_required = bool(candidate.get("manual_review_required"))
            if strategy == "gpt_question_anchored" and self._question_anchored_can_auto_accept(candidate):
                candidate_review_required = False
            if candidate_review_required:
                reasons.append("vision_mapper_requested_review")
            if confidence < self.VISION_ACCEPT_THRESHOLD:
                reasons.append("vision_confidence_below_accept_threshold")
            review_required = candidate_review_required or confidence < self.VISION_ACCEPT_THRESHOLD
            review_status = "accepted" if not review_required else "needs_review"
            existing = by_question.get(question_id)
            if (
                strategy != "gpt_question_anchored"
                and existing
                and existing.get("confidence", 0) > confidence
                and not existing.get("manual_review_required")
            ):
                continue
            answer_item_id = self._answer_item_id(
                candidate=candidate,
                answer_block_id=answer_block_id,
                question_id=question_id,
            )
            mapping = {
                "mapping_id": f"{document_id}:{question_id}:{answer_item_id}",
                "document_id": document_id,
                "question_region_id": question_id,
                "question_id": question_id,
                "answer_region_id": answer_item_id,
                "answer_block_id": answer_block_id or None,
                "answer_item_id": answer_item_id,
                "answer_number": candidate.get("answer_number") or block.get("number"),
                "answer_text": answer_text,
                "mapping_strategy": strategy,
                "confidence": max(0.0, min(1.0, confidence)),
                "manual_review_required": review_required,
                "review_status": review_status,
                "mapping_reasons": sorted(set(reasons)),
                "mapping_evidence": candidate.get("evidence") or "",
                "mapping_notes": candidate.get("notes") or "",
                "correct_answer_candidate": self._normalise_correct_answer(candidate.get("correct_answer")),
                "correct_answer_confidence": self._float(candidate.get("correct_answer_confidence"), 0.0),
                "final_answer_text": str(candidate.get("final_answer_text") or "").strip(),
                "solution_image_notes": str(candidate.get("solution_image_notes") or "").strip(),
                "solution_images": candidate.get("solution_images") if isinstance(candidate.get("solution_images"), list) else [],
                "answer_page_numbers": candidate.get("page_numbers") if isinstance(candidate.get("page_numbers"), list) else [],
                "source": self.SOURCE,
                "mapper_provider": vision_result.get("provider"),
                "mapper_model": vision_result.get("model"),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            by_question[question_id] = mapping

        merged_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for mapping in deterministic:
            question_id = str(mapping.get("question_id") or "")
            key = (question_id, str(mapping.get("answer_region_id") or ""))
            if question_id and by_question.get(question_id) is not mapping:
                continue
            merged_by_key[key] = mapping
        for mapping in by_question.values():
            key = (str(mapping.get("question_id") or ""), str(mapping.get("answer_region_id") or ""))
            merged_by_key[key] = mapping
        return list(merged_by_key.values())

    def _document_acceptance_policy(
        self,
        *,
        questions: List[Dict[str, Any]],
        blocks: List[Dict[str, Any]],
        mappings: List[Dict[str, Any]],
        vision_reasons: List[str],
        vision_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        question_ids = {
            str(question.get("id") or question.get("question_id") or "")
            for question in questions
            if question.get("id") or question.get("question_id")
        }
        mapped_question_ids = {
            str(mapping.get("question_id") or "")
            for mapping in mappings
            if mapping.get("question_id") and str(mapping.get("answer_text") or "").strip()
        }
        blockers: List[str] = []
        question_anchored = str(vision_result.get("mode") or "") == "question_anchored"
        complete_question_anchored = (
            question_anchored
            and bool(question_ids)
            and mapped_question_ids == question_ids
        )
        if len(blocks) != len(questions) and not complete_question_anchored:
            blockers.append("answer_question_count_mismatch")
        if question_ids and mapped_question_ids != question_ids:
            blockers.append("partial_mapping_coverage")
        if vision_reasons and not complete_question_anchored:
            blockers.append("vision_risk_review_required")
        if vision_result.get("error"):
            blockers.append("vision_mapper_error")
        if any(mapping.get("manual_review_required") for mapping in mappings) and not complete_question_anchored:
            blockers.append("mapping_candidate_requires_review")
        return {
            "auto_acceptance_blocked": bool(blockers),
            "auto_acceptance_blockers": sorted(set(blockers)),
        }

    def _apply_acceptance_policy(
        self,
        *,
        mappings: List[Dict[str, Any]],
        acceptance_policy: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not acceptance_policy.get("auto_acceptance_blocked"):
            return mappings
        blockers = acceptance_policy.get("auto_acceptance_blockers") or []
        updated: List[Dict[str, Any]] = []
        for mapping in mappings:
            if not mapping.get("question_id"):
                updated.append(mapping)
                continue
            reasons = set(mapping.get("mapping_reasons") or [])
            reasons.add("document_level_review_required")
            reasons.update(str(blocker) for blocker in blockers)
            updated.append(
                {
                    **mapping,
                    "manual_review_required": True,
                    "review_status": "needs_review",
                    "mapping_reasons": sorted(reasons),
                    "updated_at": datetime.utcnow(),
                }
            )
        return updated

    def _vision_reasons(
        self,
        *,
        questions: List[Dict[str, Any]],
        blocks: List[Dict[str, Any]],
        mappings: List[Dict[str, Any]],
        layout_report: Optional[Dict[str, Any]],
        page_summaries: List[Dict[str, Any]],
    ) -> List[str]:
        reasons: List[str] = []
        if not questions or not blocks:
            return reasons
        if len(questions) != len(blocks):
            reasons.append("answer_question_count_mismatch")
        if any(not block.get("number") for block in blocks):
            reasons.append("missing_answer_numbers")
        if any(count > 1 for count in self._answer_number_counts(blocks).values()):
            reasons.append("duplicate_answer_numbers")
        if any(self._float(mapping.get("confidence"), 0) < self.REVIEW_THRESHOLD for mapping in mappings):
            reasons.append("low_deterministic_mapping_confidence")
        if any(len(str(mapping.get("answer_text") or "").strip()) < 20 for mapping in mappings):
            reasons.append("short_answer_text")
        if any(not str(page.get("markdown") or "").strip() for page in page_summaries or []):
            reasons.append("empty_ocr_page_text")
        layout_risks = self._layout_risks(layout_report)
        if layout_risks.intersection(
            {
                "multi_column",
                "table_layout",
                "formula_or_image_dependency",
                "reading_order_risk",
                "answer_anchor_conflict",
            }
        ):
            reasons.append("complex_answer_sheet_layout")
        return sorted(set(reasons))

    def _normalise_blocks(self, answer_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for index, block in enumerate(answer_blocks or [], start=1):
            if not isinstance(block, dict):
                continue
            normalised = dict(block)
            normalised["block_id"] = str(block.get("block_id") or block.get("id") or f"answer_block_{index}")
            normalised["text"] = str(block.get("text") or block.get("answer_text") or "").strip()
            if block.get("number") is not None:
                normalised["number"] = str(block.get("number")).strip()
            blocks.append(normalised)
        return blocks

    def _sorted_questions(self, question_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(question_docs or [], key=self._question_sort_key)

    def _question_sort_key(self, question: Dict[str, Any]) -> tuple:
        region = question.get("region_metadata") or {}
        explicit_order = self._int_or_none(question.get("extraction_order") or question.get("question_number"))
        if explicit_order is not None:
            return (0, explicit_order, 0.0, str(question.get("id") or question.get("question_id") or ""))
        return (
            1,
            int(question.get("page_number") or region.get("page") or 0),
            float(region.get("y") or 0),
            str(question.get("id") or question.get("question_id") or ""),
        )

    def _questions_by_number(self, questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for index, question in enumerate(questions, start=1):
            candidates = {str(index)}
            candidates.update(self._numbers_from_text(str(question.get("id") or question.get("question_id") or "")))
            candidates.update(self._numbers_from_text(str(question.get("text") or question.get("question_text") or "")))
            for number in candidates:
                if number and number not in indexed:
                    indexed[number] = question
        return indexed

    def _numbers_from_text(self, text: str) -> List[str]:
        numbers: List[str] = []
        for pattern in (
            r"^\s*(?:q(?:uestion)?\.?\s*)?(\d{1,3})[\.\)]\s+",
            r"\bq(?:uestion)?\.?\s*(\d{1,3})\b",
        ):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                numbers.append(str(int(match.group(1))))
        return numbers

    def _answer_number_counts(self, blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for block in blocks or []:
            number = str(block.get("number") or "").strip()
            if number:
                counts[number] = counts.get(number, 0) + 1
        return counts

    def _deterministic_confidence(
        self,
        *,
        strategy: str,
        count_aligned: bool,
        block: Dict[str, Any],
        duplicate_number: bool,
        question_id: str,
        answer_text: str,
    ) -> float:
        if not question_id or not answer_text:
            return 0.25
        base = 0.9 if strategy == "answer_number" else (0.83 if count_aligned else 0.55)
        if duplicate_number:
            base = min(base, 0.58)
        block_confidence = self._float(block.get("confidence"), 0.65)
        if len(answer_text) < 20:
            base = min(base, 0.6)
        return round(max(0.0, min(1.0, min(base, 0.45 + (block_confidence * 0.65)))), 2)

    def _layout_risks(self, layout_report: Optional[Dict[str, Any]]) -> set[str]:
        risks = set(str(risk) for risk in (layout_report or {}).get("layout_risks", []) or [])
        risks.update(str(risk) for risk in (layout_report or {}).get("document_layout_risks", []) or [])
        for page in (layout_report or {}).get("pages", []) or []:
            risks.update(str(risk) for risk in page.get("layout_risks", []) or [])
        return risks

    async def _find_existing_mappings(self, db: Any, is_b2c: bool, document_id: str) -> List[Dict[str, Any]]:
        if is_b2c and hasattr(db, "b2c_find"):
            return await db.b2c_find("answer_question_mappings", {"document_id": document_id})
        if hasattr(db, "mongo_find"):
            return await db.mongo_find("answer_question_mappings", {"document_id": document_id})
        return []

    async def _clear_existing_full_ocr_mappings(self, db: Any, is_b2c: bool, document_id: str) -> None:
        query = {"document_id": document_id, "source": self.SOURCE}
        if is_b2c:
            await db.b2c_delete_many("answer_question_mappings", query)
        else:
            await db.mongo_delete_many("answer_question_mappings", query)

    async def _upsert_mapping(self, db: Any, is_b2c: bool, mapping: Dict[str, Any]) -> bool:
        query = {
            "document_id": mapping["document_id"],
            "question_id": mapping["question_id"],
            "source": self.SOURCE,
        }
        if is_b2c:
            return bool(await db.b2c_update_one("answer_question_mappings", query, {"$set": mapping}, upsert=True))
        return bool(await db.mongo_update_one("answer_question_mappings", query, {"$set": mapping}, upsert=True))

    def _is_manual_mapping(self, mapping: Dict[str, Any]) -> bool:
        source = str(mapping.get("source") or "").strip().lower()
        return source == "manual_answer_segmentation"

    def _mapping_question_id(self, mapping: Dict[str, Any]) -> str:
        return str(mapping.get("question_id") or mapping.get("question_region_id") or "").strip()

    def _answer_item_id(self, *, candidate: Dict[str, Any], answer_block_id: str, question_id: str) -> str:
        raw = str(candidate.get("answer_item_id") or "").strip()
        if not raw:
            raw = f"{answer_block_id or 'vision'}:{question_id}"
        safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw).strip("-")
        if question_id not in safe:
            safe = f"{safe}:{question_id}" if safe else f"vision:{question_id}"
        return safe

    def _normalise_correct_answer(self, value: Any) -> str:
        label = str(value or "").strip().upper()
        if label in {"A", "B", "C", "D", "E", "F"}:
            return label
        if label in {"1", "2", "3", "4", "5", "6"}:
            return chr(64 + int(label))
        return ""

    def _question_anchored_can_auto_accept(self, candidate: Dict[str, Any]) -> bool:
        """Accept a complete per-question vision extraction even if the model was conservative."""
        confidence = self._float(candidate.get("confidence"), 0.0)
        correct_confidence = self._float(candidate.get("correct_answer_confidence"), 0.0)
        answer_text = str(candidate.get("answer_text") or "").strip()
        correct_answer = self._normalise_correct_answer(candidate.get("correct_answer"))
        return (
            bool(answer_text)
            and bool(correct_answer)
            and confidence >= self.VISION_ACCEPT_THRESHOLD
            and correct_confidence >= self.VISION_ACCEPT_THRESHOLD
        )

    def _float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _int_or_none(self, value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
