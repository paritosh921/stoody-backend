"""
PCR Submission Repository
=========================

Async MongoDB persistence for ``evalpen_submissions`` — the PCR conducted-exam
submission records.

This repository is a **read-only consumer** of submissions created by the
shared ingest substrate (``ingest/repository.py``).  PCR storage does not
create or mutate the raw submission documents; it only reads them and updates
the non-immutable ``segmentation_status`` field after PCR segmentation
completes.

Ownership Declaration
---------------------
- Writes: ``segmentation_status`` field only (processing status, not artifact content)
- Reads from: ``evalpen_submissions`` (owned by ingest substrate)
- Never writes to: raw artifact fields, ``content_hash``, ``_immutable``
- Transactional boundaries: status update is atomic per document

Why this exists alongside IngestRepository
------------------------------------------
The ingest repository owns creation and write-once semantics.  This PCR-scoped
repository provides PCR-specific query patterns (e.g., filter by
``segmentation_status``) and a clear ownership boundary: the PCR engine only
touches ``segmentation_status``, never the canonical artifact data.

References
----------
- Storage model: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md  (Section 7.1)
- Integrity:     new-docs/architecture/TAMPER_PROOF_SPEC.md     (Layer 1, Layer 2)
- Ownership:     new-docs/governance/STATE_OWNERSHIP_MAP.md
- Failure modes: PCR-01 (boundary/marker failure -> flags + review queue)
- Test IDs:      I-PCR-01, I-TAMP-02
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection name — matches PCR_EVAL_ENGINE_SPEC 7.1 and ingest substrate
# ---------------------------------------------------------------------------

SUBMISSIONS_COLLECTION = "evalpen_submissions"


class SubmissionRepository:
    """Read-oriented repository for PCR access to conducted-exam submissions.

    The PCR engine uses this to:
    1. Fetch submission records for segmentation input (Layer 2 server-side fetch).
    2. Update ``segmentation_status`` after segmentation completes.
    3. Query submissions pending PCR processing.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        Tenant-scoped Motor database (``skb_<tenant>``).
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._submissions = db[SUBMISSIONS_COLLECTION]

    # ------------------------------------------------------------------
    # Read operations (TAMPER_PROOF_SPEC Layer 2 — server-side fetch)
    # ------------------------------------------------------------------

    async def get_submission(
        self, submission_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single submission by its canonical ID.

        This is the server-side fetch path required by TAMPER_PROOF_SPEC
        Layer 2.  PCR must always fetch from this repository rather than
        trusting client-supplied submission data.
        """
        return await self._submissions.find_one(
            {"submission_id": submission_id}
        )

    async def get_submission_by_exam_student(
        self, exam_id: str, student_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a submission by exam + student pair."""
        return await self._submissions.find_one(
            {"exam_id": exam_id, "student_id": student_id}
        )

    async def list_submissions_by_status(
        self,
        segmentation_status: str,
        *,
        admin_id: Optional[str] = None,
        exam_id: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """List submissions filtered by segmentation status.

        This is the primary query for the PCR processing queue — fetch all
        ``pending`` submissions that need segmentation.

        Parameters
        ----------
        segmentation_status:
            One of ``pending``, ``complete``, ``failed``.
        admin_id:
            Optional filter by tenant admin.
        exam_id:
            Optional filter by exam.
        limit:
            Maximum results to return.
        skip:
            Pagination offset.
        """
        query: Dict[str, Any] = {
            "segmentation_status": segmentation_status,
        }
        if admin_id is not None:
            query["admin_id"] = admin_id
        if exam_id is not None:
            query["exam_id"] = exam_id

        cursor = (
            self._submissions.find(query)
            .sort("submitted_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def list_submissions(
        self,
        *,
        admin_id: Optional[str] = None,
        exam_id: Optional[str] = None,
        exam_ids: Optional[List[str]] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """List submissions with optional admin/exam filters.

        Used by the PCR submission listing endpoint
        (``GET /api/v1/evalpen/submissions``).
        """
        query: Dict[str, Any] = {}
        if admin_id is not None:
            query["admin_id"] = admin_id
        if exam_ids is not None:
            query["exam_id"] = {"$in": exam_ids}
        elif exam_id is not None:
            query["exam_id"] = exam_id

        cursor = (
            self._submissions.find(query)
            .sort("submitted_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------------
    # Status update (non-immutable field)
    # ------------------------------------------------------------------

    async def update_segmentation_status(
        self, submission_id: str, status: str
    ) -> bool:
        """Update ``segmentation_status`` after PCR segmentation.

        ``segmentation_status`` is explicitly NOT covered by the immutability
        flag — it is a processing-status field set by the PCR engine, not
        part of the canonical artifact content.  The ``content_hash`` and raw
        artifact data remain untouched.

        Returns True if the document was found and updated.
        """
        result = await self._submissions.update_one(
            {"submission_id": submission_id},
            {"$set": {"segmentation_status": status}},
        )
        if result.modified_count > 0:
            logger.info(
                "Updated segmentation_status=%s for submission %s",
                status,
                submission_id,
            )
            return True

        logger.warning(
            "No submission found or status unchanged for %s (status=%s)",
            submission_id,
            status,
        )
        return False
