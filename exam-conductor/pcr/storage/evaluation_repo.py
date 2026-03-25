"""
PCR Evaluation Repository
=========================

Async MongoDB persistence for ``evalpen_evaluations`` — LLM evaluation results
with step marks, feedback, token usage, and append-only audit trail.

Audit trail contract (TAMPER_PROOF_SPEC Layer 3):
    Evaluations support an **append-only** ``audit_trail`` array.  Score
    overrides, flag resolutions, and manual corrections append new entries
    to this array — they never replace previous entries.  Each audit entry
    captures ``actor_id``, ``timestamp``, ``action``, ``before``, ``after``,
    and ``reason`` (TAMPER_PROOF_SPEC Section 5).

Ownership Declaration
---------------------
- Writes: ``evalpen_evaluations``
- Reads from: ``evalpen_detected_responses`` (via DetectedResponseRepository),
  ``evalpen_questions`` (via QuestionRepository),
  ``evalpen_solutions`` (via SolutionRepository)
- Never writes to: ``evalpen_submissions``, ``evalpen_answer_pages``,
  ``evalpen_detected_responses`` (detected text), practice collections
- Transactional boundaries: evaluation output + gate usage refs + audit metadata

References
----------
- Storage model: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md  (Section 7.3)
- Integrity:     new-docs/architecture/TAMPER_PROOF_SPEC.md     (Layer 3)
- Ownership:     new-docs/governance/STATE_OWNERSHIP_MAP.md
- Failure modes: PCR-03 (diagram-heavy auto-scored incorrectly)
- Test IDs:      U-EVAL-01, I-PCR-01, I-TAMP-03
- API schema:    new-docs/api/eval-evaluate.openapi.yaml (EvaluationDetail)
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
# Collection name — PCR_EVAL_ENGINE_SPEC 7.3
# ---------------------------------------------------------------------------

EVALUATIONS_COLLECTION = "evalpen_evaluations"


class DuplicateEvaluationError(Exception):
    """Raised when a duplicate evaluation is inserted for a response that
    already has a different evaluation result.

    Each response should have exactly one evaluation.  Re-evaluations go
    through the override flow with an audit trail, not a second insert.
    """


class EvaluationRepository:
    """Async repository for PCR evaluation result persistence.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        Tenant-scoped Motor database (``skb_<tenant>``).
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._evaluations = db[EVALUATIONS_COLLECTION]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create required indexes (idempotent).

        Indexes match PCR_EVAL_ENGINE_SPEC Section 7.3:
        - ``{ evaluation_id: 1 }`` unique
        - ``{ response_id: 1 }`` unique  (one eval per response)
        - ``{ student_id: 1 }``
        - ``{ question_id: 1 }``
        """
        await self._evaluations.create_indexes(
            [
                IndexModel(
                    [("evaluation_id", ASCENDING)],
                    unique=True,
                    name="uniq_evaluation_id",
                ),
                IndexModel(
                    [("response_id", ASCENDING)],
                    unique=True,
                    name="uniq_response_id",
                ),
                IndexModel(
                    [("student_id", ASCENDING)],
                    name="idx_student_id",
                ),
                IndexModel(
                    [("question_id", ASCENDING)],
                    name="idx_question_id",
                ),
            ]
        )
        logger.info("Ensured indexes on %s", EVALUATIONS_COLLECTION)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def insert_evaluation(
        self, doc: Dict[str, Any]
    ) -> tuple[Dict[str, Any], bool]:
        """Persist an evaluation result.

        The document should include at minimum:
        - ``evaluation_id``, ``response_id``, ``question_id``, ``student_id``
        - ``eval_path``, ``model_used``
        - ``total_score``, ``max_score``, ``scoreable_max``
        - ``step_marks[]``, ``overall_feedback``
        - ``reference_solution``, ``token_usage``, ``raw_llm_response``
        - ``audit_trail[]`` (initialized with the creation entry)

        Returns
        -------
        (document, already_existed)
            The persisted (or pre-existing) document and a boolean indicating
            duplicate detection.

        Raises
        ------
        DuplicateEvaluationError
            If an evaluation already exists for the same ``response_id``
            with a different ``evaluation_id`` (data inconsistency).
        """
        evaluation_id = doc["evaluation_id"]
        response_id = doc["response_id"]

        # Ensure audit trail is initialized
        if "audit_trail" not in doc:
            doc["audit_trail"] = [
                {
                    "actor_id": "system",
                    "timestamp": datetime.now(timezone.utc),
                    "action": "evaluation_created",
                    "before": None,
                    "after": {
                        "total_score": doc.get("total_score"),
                        "max_score": doc.get("max_score"),
                    },
                    "reason": "Initial automated evaluation",
                }
            ]

        doc.setdefault("created_at", datetime.now(timezone.utc))

        try:
            await self._evaluations.insert_one(doc)
            logger.info(
                "Persisted evaluation %s for response %s",
                evaluation_id,
                response_id,
            )
            return doc, False
        except DuplicateKeyError:
            # Check which unique constraint was violated
            existing_by_eval = await self._evaluations.find_one(
                {"evaluation_id": evaluation_id}
            )
            if existing_by_eval is not None:
                # Same evaluation_id — idempotent
                logger.info(
                    "Duplicate evaluation %s — idempotent return",
                    evaluation_id,
                )
                return existing_by_eval, True

            # Different evaluation_id but same response_id — conflict
            existing_by_resp = await self._evaluations.find_one(
                {"response_id": response_id}
            )
            if existing_by_resp is not None:
                raise DuplicateEvaluationError(
                    f"Response {response_id} already has evaluation "
                    f"{existing_by_resp['evaluation_id']}. Use the "
                    f"override flow with audit trail instead of inserting "
                    f"a second evaluation."
                )

            # Race condition fallback — retry once
            await self._evaluations.insert_one(doc)
            return doc, False

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_evaluation(
        self, evaluation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single evaluation by its canonical ID.

        Used by ``GET /api/v1/evalpen/evaluations/{evaluation_id}``.
        """
        return await self._evaluations.find_one(
            {"evaluation_id": evaluation_id}
        )

    async def get_evaluation_by_response(
        self, response_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch the evaluation for a specific detected response.

        Each response has at most one evaluation (enforced by unique index).
        """
        return await self._evaluations.find_one(
            {"response_id": response_id}
        )

    async def get_evaluations_by_student(
        self,
        student_id: str,
        *,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch all evaluations for a student."""
        cursor = (
            self._evaluations.find({"student_id": student_id})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def get_evaluations_by_question(
        self,
        question_id: str,
        *,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch all evaluations for a specific question across students."""
        cursor = (
            self._evaluations.find({"question_id": question_id})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------------
    # Append-only audit trail (TAMPER_PROOF_SPEC Layer 3)
    # ------------------------------------------------------------------

    async def append_audit_entry(
        self,
        evaluation_id: str,
        *,
        actor_id: str,
        action: str,
        before: Any = None,
        after: Any = None,
        reason: str,
    ) -> bool:
        """Append an audit entry to the evaluation's ``audit_trail``.

        This is the ONLY way to record overrides, flag resolutions, and
        manual corrections (TAMPER_PROOF_SPEC Layer 3, Section 5).

        Audit entries are append-only — ``$push`` ensures previous entries
        are never removed or modified.

        Parameters
        ----------
        evaluation_id:
            The evaluation to append the audit entry to.
        actor_id:
            Who performed the action (user ID, ``system``, etc.).
        action:
            The action type (e.g. ``score_override``, ``flag_resolved``,
            ``manual_correction``).
        before:
            State before the action (for diffing).
        after:
            State after the action.
        reason:
            Human-readable justification (required by TAMPER_PROOF_SPEC
            Section 6, Rule 4).

        Returns
        -------
        bool
            True if the audit entry was appended successfully.
        """
        audit_entry = {
            "actor_id": actor_id,
            "timestamp": datetime.now(timezone.utc),
            "action": action,
            "before": before,
            "after": after,
            "reason": reason,
        }

        result = await self._evaluations.update_one(
            {"evaluation_id": evaluation_id},
            {"$push": {"audit_trail": audit_entry}},
        )

        if result.modified_count > 0:
            logger.info(
                "Appended audit entry action=%s to evaluation %s by %s",
                action,
                evaluation_id,
                actor_id,
            )
            return True

        logger.warning(
            "Failed to append audit entry to evaluation %s — not found",
            evaluation_id,
        )
        return False

    # ------------------------------------------------------------------
    # Score override with audit (TAMPER_PROOF_SPEC Section 6, Rule 4)
    # ------------------------------------------------------------------

    async def override_score(
        self,
        evaluation_id: str,
        *,
        new_total_score: float,
        actor_id: str,
        reason: str,
    ) -> bool:
        """Override the total score on an evaluation with a mandatory
        audit trail entry.

        Score overrides require an ``actor_id`` and ``reason``
        (TAMPER_PROOF_SPEC Section 6, Rule 4).

        The override is performed atomically: the score is updated and
        the audit entry is appended in a single ``update_one`` operation.

        Returns True if the override was applied.
        """
        # Fetch current score for the before/after audit record
        existing = await self._evaluations.find_one(
            {"evaluation_id": evaluation_id},
            projection={"total_score": 1},
        )
        if existing is None:
            logger.warning(
                "Cannot override score — evaluation %s not found",
                evaluation_id,
            )
            return False

        old_score = existing.get("total_score")

        audit_entry = {
            "actor_id": actor_id,
            "timestamp": datetime.now(timezone.utc),
            "action": "score_override",
            "before": {"total_score": old_score},
            "after": {"total_score": new_total_score},
            "reason": reason,
        }

        result = await self._evaluations.update_one(
            {"evaluation_id": evaluation_id},
            {
                "$set": {"total_score": new_total_score},
                "$push": {"audit_trail": audit_entry},
            },
        )

        if result.modified_count > 0:
            logger.info(
                "Score override on evaluation %s: %.2f -> %.2f by %s (%s)",
                evaluation_id,
                old_score,
                new_total_score,
                actor_id,
                reason,
            )
            return True

        return False
