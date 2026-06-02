"""Map OCR'd answer-sheet regions to saved question IDs."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional


class AnswerQuestionMappingService:
    """Persist question-addressable worked-answer mappings."""

    async def map_region_order(
        self,
        *,
        db: Any,
        is_b2c: bool,
        document_id: str,
        question_regions: List[Dict[str, Any]],
        answer_regions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        question_regions_sorted = sorted(
            question_regions or [],
            key=lambda r: (int(r.get("pageNumber", 0) or 0), float(r.get("y", 0) or 0), float(r.get("x", 0) or 0)),
        )
        answer_regions_sorted = sorted(
            answer_regions or [],
            key=lambda r: (int(r.get("pageNumber", 0) or 0), float(r.get("y", 0) or 0), float(r.get("x", 0) or 0)),
        )
        await self._clear_existing_manual_mappings(db, is_b2c, document_id)
        if not question_regions_sorted or not answer_regions_sorted:
            return []

        count_aligned = len(question_regions_sorted) == len(answer_regions_sorted)
        mappings: List[Dict[str, Any]] = []
        questions_by_number = self._questions_by_number(question_regions_sorted)
        used_question_ids: set[str] = set()
        for idx, answer_region in enumerate(answer_regions_sorted):
            question_region = self._match_by_question_number(
                answer_region,
                questions_by_number,
                used_question_ids,
            )
            strategy = "question_number" if question_region else "region_order"
            if question_region is None:
                question_region = question_regions_sorted[idx] if idx < len(question_regions_sorted) else None
            question_region_id = question_region.get("id") if question_region else None
            if question_region_id:
                used_question_ids.add(str(question_region_id))
            answer_text = str(answer_region.get("extractedText", "") or "").strip()
            confidence = 0.96 if strategy == "question_number" else (0.94 if count_aligned and question_region_id else 0.45)
            manual_review_required = (
                not count_aligned
                or not question_region_id
                or not answer_text
                or bool(answer_region.get("manualReviewRequired"))
            )
            if strategy == "question_number" and question_region_id and answer_text:
                manual_review_required = bool(answer_region.get("manualReviewRequired"))
            review_status = "needs_review" if manual_review_required else "accepted"
            mapping = {
                "mapping_id": f"{document_id}:{question_region_id or 'unmapped'}:{answer_region.get('id')}",
                "document_id": document_id,
                "question_region_id": question_region_id,
                "question_id": question_region_id,
                "answer_region_id": answer_region.get("id"),
                "answer_text": answer_text,
                "mapping_strategy": strategy,
                "confidence": confidence,
                "manual_review_required": manual_review_required,
                "review_status": review_status,
                "source": "manual_answer_segmentation",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            mappings.append(mapping)
            await self._upsert_mapping(db, is_b2c, mapping)
        return mappings

    async def _upsert_mapping(self, db: Any, is_b2c: bool, mapping: Dict[str, Any]) -> None:
        if mapping.get("question_id"):
            query = {
                "document_id": mapping["document_id"],
                "question_id": mapping["question_id"],
                "source": "manual_answer_segmentation",
            }
        else:
            query = {
                "document_id": mapping["document_id"],
                "answer_region_id": mapping["answer_region_id"],
                "source": "manual_answer_segmentation",
            }
        update = {"$set": mapping}
        if is_b2c:
            await db.b2c_update_one("answer_question_mappings", query, update, upsert=True)
        else:
            await db.mongo_update_one("answer_question_mappings", query, update, upsert=True)

    async def _clear_existing_manual_mappings(self, db: Any, is_b2c: bool, document_id: str) -> None:
        query = {
            "document_id": document_id,
            "$or": [
                {"source": "manual_answer_segmentation"},
                {
                    "source": {"$exists": False},
                    "mapping_strategy": {"$in": ["question_number", "region_order"]},
                },
            ],
        }
        if is_b2c:
            await db.b2c_delete_many("answer_question_mappings", query)
        else:
            await db.mongo_delete_many("answer_question_mappings", query)

    def _questions_by_number(self, question_regions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for idx, region in enumerate(question_regions, start=1):
            candidates = {
                str(idx),
                *self._numbers_from_text(str(region.get("label", "") or "")),
                *self._numbers_from_text(str(region.get("extractedText", "") or "")),
                *self._numbers_from_text(str(region.get("id", "") or "")),
            }
            for number in candidates:
                if number and number not in indexed:
                    indexed[number] = region
        return indexed

    def _match_by_question_number(
        self,
        answer_region: Dict[str, Any],
        questions_by_number: Dict[str, Dict[str, Any]],
        used_question_ids: set[str],
    ) -> Optional[Dict[str, Any]]:
        for number in self._numbers_from_text(str(answer_region.get("extractedText", "") or "")):
            region = questions_by_number.get(number)
            if region and str(region.get("id")) not in used_question_ids:
                return region
        return None

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
