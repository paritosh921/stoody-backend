"""
PCR Detected Response Repository
=================================

Async MongoDB persistence for ``evalpen_detected_responses`` — segmented
student responses with flags.

Immutability contract (TAMPER_PROOF_SPEC Layer 1):
    Detected response documents are immutable after initial write.  The
    ``detected_text`` and segmentation data MUST NOT be overwritten.  Any
    correction goes through the review / override flow with an audit trail
    (TAMPER_PROOF_SPEC Layer 3, Section 6).

The only mutable field on a detected response is ``eval_status``, which
tracks whether the response has been evaluated, blocked, etc.  This is
analogous to ``segmentation_status`` on submissions — a processing status
field, not part of the canonical detected content.

Ownership Declaration
---------------------
- Writes: ``evalpen_detected_responses``
- Reads from: ``evalpen_submissions`` (via SubmissionRepository)
- Never writes to: ``evalpen_submissions``, ``evalpen_answer_pages``,
  practice collections
- Transactional boundaries: detected response + flags persisted atomically

References
----------
- Storage model: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md  (Section 7.2)
- Integrity:     new-docs/architecture/TAMPER_PROOF_SPEC.md     (Layer 1)
- Ownership:     new-docs/governance/STATE_OWNERSHIP_MAP.md
- Failure modes: PCR-01 (boundary/marker failure -> flags + review queue)
- Test IDs:      I-PCR-01, I-PCR-02, I-TAMP-02, I-TAMP-03
- API schema:    new-docs/api/eval-submissions.openapi.yaml (DetectedResponse)
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
# Collection name — PCR_EVAL_ENGINE_SPEC 7.2
# ---------------------------------------------------------------------------

DETECTED_RESPONSES_COLLECTION = "evalpen_detected_responses"


class ImmutableResponseError(Exception):
    """Raised when an attempt is made to mutate an immutable detected response.

    Detected text and segmentation data are protected by TAMPER_PROOF_SPEC
    Layer 1.  Only ``eval_status`` may be updated through defined transitions.
    """


class DetectedResponseRepository:
    """Async repository for PCR detected response persistence.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        Tenant-scoped Motor database (``skb_<tenant>``).
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._responses = db[DETECTED_RESPONSES_COLLECTION]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create required indexes (idempotent).

        Indexes match PCR_EVAL_ENGINE_SPEC Section 7.2:
        - ``{ response_id: 1 }`` unique
        - ``{ submission_id: 1 }``
        - ``{ eval_status: 1 }``
        - ``{ "flags.severity": 1 }``
        """
        await self._responses.create_indexes(
            [
                IndexModel(
                    [("response_id", ASCENDING)],
                    unique=True,
                    name="uniq_response_id",
                ),
                IndexModel(
                    [("submission_id", ASCENDING)],
                    name="idx_submission_id",
                ),
                IndexModel(
                    [("eval_status", ASCENDING)],
                    name="idx_eval_status",
                ),
                IndexModel(
                    [("flags.severity", ASCENDING)],
                    name="idx_flags_severity",
                ),
            ]
        )
        logger.info(
            "Ensured indexes on %s", DETECTED_RESPONSES_COLLECTION
        )

    # ------------------------------------------------------------------
    # Write operations (write-once for detected content)
    # ------------------------------------------------------------------

    async def insert_response(
        self, doc: Dict[str, Any]
    ) -> tuple[Dict[str, Any], bool]:
        """Persist a detected response with write-once semantics.

        The document is stored with ``_immutable: true`` to protect the
        detected text and segmentation data (TAMPER_PROOF_SPEC Layer 1).

        Returns
        -------
        (document, already_existed)
            The persisted (or pre-existing) document and a boolean indicating
            duplicate detection.

        Raises
        ------
        ImmutableResponseError
            If a response with the same ``response_id`` exists but has
            different ``detected_text`` (integrity violation).
        """
        response_id = doc["response_id"]

        # Enforce immutability marker
        doc.setdefault("_immutable", True)
        doc.setdefault("created_at", datetime.now(timezone.utc))

        try:
            await self._responses.insert_one(doc)
            logger.info("Persisted detected response %s", response_id)
            return doc, False
        except DuplicateKeyError:
            existing = await self._responses.find_one(
                {"response_id": response_id}
            )
            if existing is None:
                # Race condition — retry once
                await self._responses.insert_one(doc)
                return doc, False

            # Verify content consistency on duplicate
            if existing.get("detected_text") != doc.get("detected_text"):
                raise ImmutableResponseError(
                    f"Detected response {response_id} already exists with "
                    f"different detected_text. This violates immutability "
                    f"(TAMPER_PROOF_SPEC Layer 1)."
                )

            logger.info(
                "Duplicate detected response %s — idempotent return",
                response_id,
            )
            return existing, True

    async def insert_responses_bulk(
        self, docs: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        """Bulk-insert detected responses.  Skips duplicates.

        Returns
        -------
        (inserted_count, duplicate_count)
        """
        if not docs:
            return 0, 0

        inserted = 0
        duplicates = 0

        for doc in docs:
            _, already_existed = await self.insert_response(doc)
            if already_existed:
                duplicates += 1
            else:
                inserted += 1

        return inserted, duplicates

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_response(
        self, response_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single detected response by its canonical ID.

        This is the server-side fetch path for TAMPER_PROOF_SPEC Layer 2.
        The evaluation engine must call this to get the canonical detected
        text rather than trusting client-supplied answer text (I-TAMP-02).
        """
        return await self._responses.find_one(
            {"response_id": response_id}
        )

    async def get_responses_by_submission(
        self,
        submission_id: str,
        *,
        include_superseded: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch all detected responses for a submission.

        Used by the submission detail endpoint
        (``GET /api/v1/evalpen/submissions/{submission_id}/responses``).
        """
        query: Dict[str, Any] = {"submission_id": submission_id}
        if not include_superseded:
            query["eval_status"] = {"$ne": "superseded"}

        cursor = self._responses.find(query).sort("question_id", ASCENDING)
        return await cursor.to_list(length=500)

    async def get_responses_by_status(
        self,
        eval_status: str,
        *,
        submission_id: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch detected responses filtered by eval status.

        Useful for finding all ``blocked`` responses (I-PCR-02) or all
        ``pending`` responses ready for evaluation.
        """
        query: Dict[str, Any] = {"eval_status": eval_status}
        if submission_id is not None:
            query["submission_id"] = submission_id

        cursor = (
            self._responses.find(query)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def get_responses_with_blocking_flags(
        self,
        *,
        submission_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch responses that have at least one blocking flag.

        Blocking flags prevent auto-evaluation (I-PCR-02, PCR-01).
        Used by the review queue.
        """
        query: Dict[str, Any] = {"flags.severity": "blocking"}
        query["eval_status"] = {"$ne": "superseded"}
        if submission_id is not None:
            query["submission_id"] = submission_id

        cursor = self._responses.find(query).limit(limit)
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------------
    # Status update (non-immutable field)
    # ------------------------------------------------------------------

    async def update_eval_status(
        self, response_id: str, eval_status: str
    ) -> bool:
        """Update the ``eval_status`` on a detected response.

        ``eval_status`` is a processing-status field, NOT part of the
        immutable detected content.  Valid transitions:

        - ``pending`` -> ``ready`` | ``ready_with_warnings`` | ``blocked``
        - ``ready`` -> ``evaluated`` | ``not_attempted`` | ``manual_review``
        - ``ready_with_warnings`` -> ``evaluated`` | ``manual_review``
        - ``blocked`` -> ``manual_review``

        The detected text, flags, and segmentation data remain untouched.

        Returns True if the document was found and updated.
        """
        result = await self._responses.update_one(
            {"response_id": response_id},
            {"$set": {"eval_status": eval_status}},
        )
        if result.modified_count > 0:
            logger.info(
                "Updated eval_status=%s for response %s",
                eval_status,
                response_id,
            )
            return True

        logger.warning(
            "No response found or status unchanged for %s (eval_status=%s)",
            response_id,
            eval_status,
        )
        return False

    async def supersede_responses_for_submission(
        self,
        submission_id: str,
        *,
        keep_response_ids: List[str],
        reason: str,
    ) -> int:
        """Mark previous detected responses for a submission as superseded.

        Reprocessing a submission may produce new random response IDs after OCR
        or segmentation improvements.  The previous detected text remains
        immutable; this method only marks old PCR-derived rows inactive for
        normal tutor/review/evaluation reads.
        """
        now = datetime.now(timezone.utc)
        query: Dict[str, Any] = {
            "submission_id": submission_id,
            "eval_status": {"$ne": "superseded"},
        }
        if keep_response_ids:
            query["response_id"] = {"$nin": keep_response_ids}

        docs = await self._responses.find(
            query,
            {"response_id": 1, "eval_status": 1},
        ).to_list(length=1000)

        modified = 0
        for doc in docs:
            response_id = doc.get("response_id")
            if not response_id:
                continue
            result = await self._responses.update_one(
                {
                    "response_id": response_id,
                    "eval_status": doc.get("eval_status"),
                },
                {
                    "$set": {
                        "eval_status": "superseded",
                        "superseded_at": now,
                        "superseded_reason": reason,
                    },
                    "$push": {
                        "audit_trail": {
                            "actor_id": "system",
                            "timestamp": now,
                            "action": "detected_response_superseded",
                            "before": {
                                "eval_status": doc.get("eval_status"),
                            },
                            "after": {"eval_status": "superseded"},
                            "reason": reason,
                        }
                    },
                },
            )
            modified += int(result.modified_count)

        if modified:
            logger.info(
                "Superseded %d previous detected responses for submission %s",
                modified,
                submission_id,
            )
        return modified

    # ------------------------------------------------------------------
    # Immutability enforcement
    # ------------------------------------------------------------------

    async def assert_not_immutable(self, response_id: str) -> None:
        """Raise ``ImmutableResponseError`` if the response is immutable.

        Guard for any future administrative operation that needs to verify
        immutability before proceeding.
        """
        doc = await self._responses.find_one(
            {"response_id": response_id},
            projection={"_immutable": 1},
        )
        if doc is not None and doc.get("_immutable", False):
            raise ImmutableResponseError(
                f"Detected response {response_id} is immutable and "
                f"its detected content cannot be modified "
                f"(TAMPER_PROOF_SPEC Layer 1)."
            )
