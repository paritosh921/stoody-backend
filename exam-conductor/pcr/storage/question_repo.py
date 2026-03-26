"""
PCR Question Repository
=======================

Async MongoDB persistence for ``evalpen_questions`` — teacher-uploaded
question metadata including complexity, eval template, and diagram weight.

Questions are **mutable** (unlike detected responses and submissions).
Teachers can update question metadata, rubrics, and evaluation templates.
However, updates should be auditable through the application layer when
they affect downstream evaluation behavior.

Ownership Declaration
---------------------
- Writes: ``evalpen_questions``
- Reads from: exam metadata (exam_id)
- Never writes to: ``evalpen_submissions``, ``evalpen_answer_pages``,
  ``evalpen_detected_responses``, practice collections
- Transactional boundaries: question metadata persisted atomically

References
----------
- Storage model: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md  (Section 7.4)
- Complexity:    new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md  (Section 5.2)
- Templates:     new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md  (Section 5.3)
- Failure modes: PCR-03 (diagram-heavy auto-scored -> classification blocks)
- Test IDs:      U-EVAL-01
- API schema:    new-docs/api/eval-solutions.openapi.yaml (QuestionMetadata)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, IndexModel
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection name — PCR_EVAL_ENGINE_SPEC 7.4
# ---------------------------------------------------------------------------

QUESTIONS_COLLECTION = "evalpen_questions"


class QuestionRepository:
    """Async repository for PCR question metadata persistence.

    Question metadata includes complexity routing info, eval template
    selection, diagram expectations, and rubric data.  This is
    teacher-uploaded content that controls how PCR evaluates student
    responses.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        Tenant-scoped Motor database (``skb_<tenant>``).
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._questions = db[QUESTIONS_COLLECTION]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create required indexes (idempotent).

        Indexes match PCR_EVAL_ENGINE_SPEC Section 7.4:
        - ``{ question_id: 1 }`` unique
        - ``{ exam_id: 1 }``
        """
        await self._questions.create_indexes(
            [
                IndexModel(
                    [("question_id", ASCENDING)],
                    unique=True,
                    name="uniq_question_id",
                ),
                IndexModel(
                    [("exam_id", ASCENDING)],
                    name="idx_exam_id",
                ),
            ]
        )
        logger.info("Ensured indexes on %s", QUESTIONS_COLLECTION)

    # ------------------------------------------------------------------
    # Write operations (upsert — mutable teacher content)
    # ------------------------------------------------------------------

    async def upsert_question(
        self, doc: Dict[str, Any]
    ) -> tuple[Dict[str, Any], bool]:
        """Insert or update question metadata.

        Questions are mutable — teachers can update complexity, eval
        template, rubric, diagram expectations, etc.  This uses MongoDB
        ``replace_one`` with ``upsert=True`` so the latest metadata always
        wins.

        Returns
        -------
        (document, was_update)
            The persisted document and a boolean indicating whether this
            was an update (True) vs a new insert (False).
        """
        question_id = doc["question_id"]

        doc.setdefault("created_at", datetime.now(timezone.utc))
        doc["updated_at"] = datetime.now(timezone.utc)

        result = await self._questions.replace_one(
            {"question_id": question_id},
            doc,
            upsert=True,
        )

        was_update = result.matched_count > 0
        if was_update:
            logger.info("Updated question metadata %s", question_id)
        else:
            logger.info("Inserted question metadata %s", question_id)

        return doc, was_update

    async def upsert_questions_bulk(
        self, docs: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        """Bulk upsert question metadata.

        Returns
        -------
        (inserted_count, updated_count)
        """
        if not docs:
            return 0, 0

        inserted = 0
        updated = 0

        for doc in docs:
            _, was_update = await self.upsert_question(doc)
            if was_update:
                updated += 1
            else:
                inserted += 1

        return inserted, updated

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_question(
        self, question_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single question by its canonical ID.

        Used by the evaluation engine to determine complexity routing,
        eval template, max marks, and diagram weight.
        """
        return await self._questions.find_one(
            {"question_id": question_id}
        )

    async def get_questions_by_exam(
        self, exam_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch all questions for an exam.

        Returns questions ordered by ``question_id`` for deterministic
        presentation.
        """
        cursor = (
            self._questions.find({"exam_id": exam_id})
            .sort("question_id", ASCENDING)
        )
        return await cursor.to_list(length=500)

    async def get_question_count(self, exam_id: str) -> int:
        """Return the number of questions registered for an exam.

        Useful for clubbed response detection (H3 — missing question
        heuristic, PCR-02).
        """
        return await self._questions.count_documents(
            {"exam_id": exam_id}
        )

    async def get_questions_expecting_diagram(
        self, exam_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch questions that expect a diagram response.

        Used by the content classifier to raise
        ``expected_diagram_missing`` flags when a diagram is expected
        but not detected (PCR-03).
        """
        cursor = self._questions.find(
            {"exam_id": exam_id, "expects_diagram": True}
        )
        return await cursor.to_list(length=500)

    # ------------------------------------------------------------------
    # Delete (admin-only, for exam reconfiguration)
    # ------------------------------------------------------------------

    async def delete_question(self, question_id: str) -> bool:
        """Delete a question metadata document.

        This is an administrative operation for exam reconfiguration.
        Existing evaluations that reference this question are NOT deleted
        — they retain their ``question_id`` for audit purposes.

        Returns True if the document was found and deleted.
        """
        result = await self._questions.delete_one(
            {"question_id": question_id}
        )
        if result.deleted_count > 0:
            logger.info("Deleted question metadata %s", question_id)
            return True

        logger.warning(
            "No question found to delete: %s", question_id
        )
        return False
