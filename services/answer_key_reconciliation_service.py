"""Extract and reconcile correct-answer keys from uploaded answer sheets."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class AnswerKeyReconciliationService:
    """Apply uploaded answer-key labels only when extraction is complete and safe."""

    VALID_LABELS = {"A", "B", "C", "D", "E", "F"}
    AUTO_APPLY_THRESHOLD = 0.86

    async def reconcile(
        self,
        *,
        db: Any,
        is_b2c: bool,
        document_id: str,
        question_docs: List[Dict[str, Any]],
        page_summaries: Optional[List[Dict[str, Any]]] = None,
        mappings: Optional[List[Dict[str, Any]]] = None,
        mapping_summary: Optional[Dict[str, Any]] = None,
        replace_existing: bool = False,
    ) -> Dict[str, Any]:
        questions = self._sorted_questions(question_docs)
        candidates, extraction_notes = self.extract_candidates(
            question_docs=questions,
            page_summaries=page_summaries or [],
            mappings=mappings or [],
        )
        current_question_ids = {
            str(question.get("id") or question.get("question_id") or "")
            for question in questions
            if question.get("id") or question.get("question_id")
        }
        candidates = {
            question_id: candidate
            for question_id, candidate in candidates.items()
            if question_id in current_question_ids
        }

        question_count = len(questions)
        extracted_count = len(candidates)
        duplicate_count = len([note for note in extraction_notes if note.get("reason") == "duplicate_answer_key_candidate"])
        conflict_count = 0
        already_set_count = 0
        auto_applied_count = 0
        skipped_count = 0

        full_coverage = question_count > 0 and extracted_count == question_count
        auto_apply_allowed = full_coverage and duplicate_count == 0
        candidate_payload: List[Dict[str, Any]] = []
        now = datetime.utcnow()
        for question in questions:
            question_id = str(question.get("id") or question.get("question_id") or "")
            candidate = candidates.get(question_id)
            if not question_id or not candidate:
                skipped_count += 1
                continue

            current_answer = self._normalise_label(question.get("correct_answer"))
            candidate_label = self._normalise_label(candidate.get("correct_answer"))
            if not candidate_label:
                skipped_count += 1
                continue

            candidate_record = {
                **candidate,
                "question_id": question_id,
                "applied": False,
                "needs_review": True,
            }
            if current_answer:
                if current_answer == candidate_label:
                    already_set_count += 1
                    candidate_record["needs_review"] = False
                    candidate_record["matches_existing"] = True
                else:
                    conflict_count += 1
                    candidate_record["conflict_with_existing"] = current_answer
                candidate_payload.append(candidate_record)
                continue

            confidence = self._float(candidate.get("confidence"), 0.0)
            can_apply = (
                auto_apply_allowed
                and confidence >= self.AUTO_APPLY_THRESHOLD
            )
            if can_apply:
                update = {
                    "correct_answer": candidate_label,
                    "correct_answer_source": "answer_sheet_ocr",
                    "correct_answer_confidence": confidence,
                    "correct_answer_extracted_at": now,
                    "correct_answer_extraction_method": candidate.get("source"),
                }
                await self._update_question(db, is_b2c, document_id, question_id, update)
                auto_applied_count += 1
                candidate_record["applied"] = True
                candidate_record["needs_review"] = False
            else:
                skipped_count += 1
            candidate_payload.append(candidate_record)

        review_required_count = len([candidate for candidate in candidate_payload if candidate.get("needs_review")])
        missing_count = max(0, question_count - extracted_count)
        reasons: List[str] = []
        if missing_count:
            reasons.append("missing_answer_key_candidates")
        if duplicate_count:
            reasons.append("duplicate_answer_key_candidates")
        if conflict_count:
            reasons.append("existing_correct_answer_conflicts")
        if not auto_apply_allowed and extracted_count:
            reasons.append("answer_key_requires_review")
        if auto_applied_count == 0 and question_count and not extracted_count:
            reasons.append("no_answer_key_candidates")

        status = "ready" if question_count and (auto_applied_count + already_set_count) == question_count else "needs_review"
        if not extracted_count:
            status = "not_found"

        summary = {
            "status": status,
            "question_count": question_count,
            "extracted_count": extracted_count,
            "auto_applied_count": auto_applied_count,
            "already_set_count": already_set_count,
            "review_required_count": review_required_count,
            "missing_count": missing_count,
            "duplicate_count": duplicate_count,
            "conflict_count": conflict_count,
            "auto_apply_allowed": auto_apply_allowed,
            "reasons": sorted(set(reasons)),
            "extraction_notes": extraction_notes[:50],
            "updated_at": now,
        }
        return {
            "summary": summary,
            "candidates": candidate_payload,
        }

    def extract_candidates(
        self,
        *,
        question_docs: List[Dict[str, Any]],
        page_summaries: List[Dict[str, Any]],
        mappings: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        questions = self._sorted_questions(question_docs)
        question_by_number = self._questions_by_number(questions)
        candidates: Dict[str, Dict[str, Any]] = {}
        notes: List[Dict[str, Any]] = []

        for candidate in self._extract_from_mappings(mappings):
            self._add_candidate(candidates, notes, candidate)
        for candidate in self._extract_from_pages(page_summaries, question_by_number):
            self._add_candidate(candidates, notes, candidate)

        return candidates, notes

    def _extract_from_pages(
        self,
        page_summaries: List[Dict[str, Any]],
        question_by_number: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for page in page_summaries or []:
            page_no = int(page.get("index", page.get("page", 0)) or 0)
            for line in str(page.get("markdown") or "").splitlines():
                stripped = self._strip_markdown_line(line)
                if not stripped:
                    continue
                for number, label in self._answer_pairs_from_line(stripped):
                    question = question_by_number.get(str(int(number))) if str(number).isdigit() else None
                    question_id = str((question or {}).get("id") or (question or {}).get("question_id") or "")
                    if not question_id:
                        continue
                    candidates.append(
                        {
                            "question_id": question_id,
                            "question_number": str(int(number)),
                            "correct_answer": label,
                            "confidence": 0.92,
                            "source": "answer_key_table",
                            "manual_review_required": False,
                            "evidence": stripped[:240],
                            "page": page_no,
                        }
                    )
        return candidates

    def _extract_from_mappings(self, mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for mapping in mappings or []:
            question_id = str(mapping.get("question_id") or mapping.get("question_region_id") or "")
            if not question_id:
                continue
            raw_candidate = (
                mapping.get("correct_answer_candidate")
                or mapping.get("correct_answer")
                or mapping.get("extracted_correct_answer")
            )
            label = self._normalise_label(raw_candidate)
            source = "mapped_solution_answer_key"
            confidence = self._float(mapping.get("correct_answer_confidence"), 0.0)
            evidence = str(mapping.get("mapping_evidence") or "")
            if not label:
                label, evidence = self._label_from_answer_text(str(mapping.get("answer_text") or ""))
                confidence = max(confidence, 0.78 if label else 0.0)
                source = "mapped_solution_text_regex"
            if not label:
                continue
            mapping_review = str(mapping.get("review_status") or "").strip().lower()
            mapping_review_required = bool(mapping.get("manual_review_required")) or mapping_review in {
                "draft",
                "needs_review",
                "rejected",
            }
            candidates.append(
                {
                    "question_id": question_id,
                    "question_number": str(mapping.get("answer_number") or ""),
                    "correct_answer": label,
                    "confidence": max(confidence, self._float(mapping.get("confidence"), 0.0) if raw_candidate else confidence),
                    "source": source,
                    "manual_review_required": mapping_review_required,
                    "evidence": evidence[:240],
                    "mapping_id": mapping.get("mapping_id"),
                }
            )
        return candidates

    def _answer_pairs_from_line(self, line: str) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        normalized = line.replace("|", " ")
        patterns = [
            r"\b(?:q(?:uestion)?\.?\s*)?(\d{1,3})\s*[:.\)\-]\s*(?:ans(?:wer)?\s*)?[:.\-]?\s*\(?([A-Fa-f])\)?\b",
            r"\b(\d{1,3})\s+(?:ans(?:wer)?\s*)?[:.\-]?\s*\(?([A-Fa-f])\)?\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                label = self._normalise_label(match.group(2))
                if label:
                    pairs.append((match.group(1), label))
        if pairs:
            return self._dedupe_pairs(pairs)

        compact = re.findall(r"\b(\d{1,3})\s*[-:]\s*([A-Fa-f])\b", normalized)
        return self._dedupe_pairs([(number, self._normalise_label(label)) for number, label in compact])

    def _label_from_answer_text(self, text: str) -> Tuple[str, str]:
        patterns = [
            r"\b(?:correct\s*)?(?:answer|ans|option|choice)\s*(?:is|:|\-)?\s*\(?([A-Fa-f])\)?\b",
            r"\boption\s+([A-Fa-f])\b.{0,30}\b(?:correct|right)\b",
            r"\(([A-Fa-f])\)\s*(?:is\s*)?(?:correct|right)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                label = self._normalise_label(match.group(1))
                if label:
                    return label, match.group(0)[:240]
        return "", ""

    def _add_candidate(
        self,
        candidates: Dict[str, Dict[str, Any]],
        notes: List[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> None:
        question_id = str(candidate.get("question_id") or "")
        label = self._normalise_label(candidate.get("correct_answer"))
        if not question_id or not label:
            return
        candidate = {**candidate, "correct_answer": label}
        existing = candidates.get(question_id)
        if existing:
            existing_label = self._normalise_label(existing.get("correct_answer"))
            if existing_label and existing_label != label:
                preferred = self._preferred_conflicting_candidate(existing, candidate)
                if preferred == "existing":
                    notes.append(
                        {
                            "question_id": question_id,
                            "reason": "ignored_lower_priority_answer_key_candidate",
                            "kept": existing_label,
                            "ignored": label,
                            "kept_source": existing.get("source"),
                            "ignored_source": candidate.get("source"),
                        }
                    )
                    return
                if preferred == "candidate":
                    notes.append(
                        {
                            "question_id": question_id,
                            "reason": "replaced_lower_priority_answer_key_candidate",
                            "replaced": existing_label,
                            "candidate": label,
                            "replaced_source": existing.get("source"),
                            "candidate_source": candidate.get("source"),
                        }
                    )
                    candidates[question_id] = candidate
                    return
                notes.append(
                    {
                        "question_id": question_id,
                        "reason": "duplicate_answer_key_candidate",
                        "existing": existing_label,
                        "candidate": label,
                    }
                )
                existing["manual_review_required"] = True
                existing["confidence"] = min(self._float(existing.get("confidence"), 0.0), 0.6)
                return
            if self._float(candidate.get("confidence"), 0.0) <= self._float(existing.get("confidence"), 0.0):
                return
        candidates[question_id] = candidate

    def _preferred_conflicting_candidate(
        self,
        existing: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> str:
        """Prefer accepted per-question mapped keys over weak page regex collisions."""
        existing_priority = self._candidate_priority(existing)
        candidate_priority = self._candidate_priority(candidate)
        if existing_priority[0] >= candidate_priority[0] + 2:
            return "existing"
        if candidate_priority[0] >= existing_priority[0] + 2:
            return "candidate"
        if existing_priority[0] == candidate_priority[0]:
            if existing_priority[1] >= candidate_priority[1] + 0.18:
                return "existing"
            if candidate_priority[1] >= existing_priority[1] + 0.18:
                return "candidate"
        return ""

    def _candidate_priority(self, candidate: Dict[str, Any]) -> Tuple[int, float]:
        source = str(candidate.get("source") or "").strip().lower()
        confidence = self._float(candidate.get("confidence"), 0.0)
        manual_review_required = bool(candidate.get("manual_review_required"))
        if source == "mapped_solution_answer_key" and not manual_review_required:
            return (5, confidence)
        if source == "mapped_solution_text_regex" and not manual_review_required:
            return (4, confidence)
        if source == "answer_key_table":
            return (3, confidence)
        return (1, confidence)

    def _mapping_summary_requires_review(self, mapping_summary: Dict[str, Any]) -> bool:
        if not mapping_summary:
            return False
        if mapping_summary.get("auto_acceptance_blocked"):
            return True
        if mapping_summary.get("manual_segmentation_recommended"):
            return True
        if int(mapping_summary.get("manual_review_count") or 0) > 0:
            return True
        return False

    async def _update_question(
        self,
        db: Any,
        is_b2c: bool,
        document_id: str,
        question_id: str,
        update: Dict[str, Any],
    ) -> None:
        query = {"document_id": document_id, "id": question_id}
        if is_b2c:
            await db.b2c_update_one("questions", query, {"$set": update})
        else:
            await db.mongo_update_one("questions", query, {"$set": update})

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
            candidates.update(self._numbers_from_text(str(question.get("text") or question.get("question_text") or "")))
            for number in candidates:
                indexed.setdefault(number, question)
        return indexed

    def _numbers_from_text(self, text: str) -> List[str]:
        numbers: List[str] = []
        for pattern in (
            r"^\s*(?:q(?:uestion)?\.?\s*)?(\d{1,3})[\.\)]\s+",
            r"\bq(?:uestion)?\.?\s*(\d{1,3})\b",
        ):
            match = re.search(pattern, text or "", re.IGNORECASE)
            if match:
                numbers.append(str(int(match.group(1))))
        return numbers

    def _normalise_label(self, value: Any) -> str:
        label = str(value or "").strip().upper()
        if label in self.VALID_LABELS:
            return label
        if label in {"1", "2", "3", "4", "5", "6"}:
            return chr(64 + int(label))
        return ""

    def _strip_markdown_line(self, line: str) -> str:
        stripped = str(line or "").strip()
        stripped = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped)
        return stripped.strip()

    def _dedupe_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        seen: set[Tuple[str, str]] = set()
        deduped: List[Tuple[str, str]] = []
        for number, label in pairs:
            if not label:
                continue
            key = (str(int(number)) if str(number).isdigit() else str(number), label)
            if key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

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
