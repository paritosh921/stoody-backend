"""
Ingest Repository
=================

Async MongoDB persistence layer for canonical conducted-exam artifacts.

Collections
-----------
- ``evalpen_submissions``  -- submission-level records
- ``evalpen_answer_pages`` -- per-page raw artifact records

Write-once semantics (TAMPER_PROOF_SPEC Layer 1):
    Documents carrying ``_immutable: true`` MUST NOT be updated or replaced.
    The repository enforces this at the application layer by checking the
    flag before any mutation.  A unique index on ``submission_id`` (and
    ``page_id``) provides a second safety net at the database level.

Idempotent ingest (ING-03):
    ``insert_submission`` and ``insert_answer_page`` return the existing
    document when a duplicate key is detected, rather than raising.

Index strategy:
    ``ensure_indexes`` creates the minimal required indexes on first use.
    It is safe to call repeatedly (idempotent).

References
----------
- Architecture: new-docs/architecture/DUAL_MODE_ARCHITECTURE.md (Section 3)
- Integrity:    new-docs/architecture/TAMPER_PROOF_SPEC.md (Layer 1)
- Failure modes: ING-01 (loss prevention), ING-03 (duplicate handling)
- Test IDs: I-ING-01 (write-once), I-ING-02 (idempotent duplicate)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, IndexModel
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection names — canonical, referenced by DUAL_MODE_ARCHITECTURE.md
# ---------------------------------------------------------------------------

SUBMISSIONS_COLLECTION = "evalpen_submissions"
ANSWER_PAGES_COLLECTION = "evalpen_answer_pages"


class ImmutableDocumentError(Exception):
    """Raised when an attempt is made to mutate an immutable document.

    This is a hard violation of TAMPER_PROOF_SPEC Layer 1 and should
    never be swallowed silently.
    """


class IngestRepository:
    """Async repository for conducted-exam ingest artifacts.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        A tenant-scoped Motor database instance (``skb_<tenant>``).
        Obtained via ``DatabaseManager.get_tenant_db(db_name)``.
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._submissions = db[SUBMISSIONS_COLLECTION]
        self._answer_pages = db[ANSWER_PAGES_COLLECTION]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create required indexes (idempotent).

        Indexes match the spec in DUAL_MODE_ARCHITECTURE.md Section 4.6
        (adapted for the shared evalpen collections).
        """
        # evalpen_submissions indexes
        await self._submissions.create_indexes(
            [
                IndexModel(
                    [("submission_id", ASCENDING)],
                    unique=True,
                    name="uniq_submission_id",
                ),
                IndexModel(
                    [("exam_id", ASCENDING), ("student_id", ASCENDING)],
                    unique=True,
                    name="uniq_exam_student",
                ),
                IndexModel(
                    [("admin_id", ASCENDING), ("exam_id", ASCENDING)],
                    name="idx_admin_exam",
                ),
            ]
        )

        # evalpen_answer_pages indexes
        await self._answer_pages.create_indexes(
            [
                IndexModel(
                    [("page_id", ASCENDING)],
                    unique=True,
                    name="uniq_page_id",
                ),
                IndexModel(
                    [("submission_id", ASCENDING), ("page_number", ASCENDING)],
                    unique=True,
                    name="uniq_submission_page",
                ),
                IndexModel(
                    [("exam_id", ASCENDING), ("student_id", ASCENDING)],
                    name="idx_page_exam_student",
                ),
            ]
        )

        logger.info(
            "Ensured ingest indexes on %s and %s",
            SUBMISSIONS_COLLECTION,
            ANSWER_PAGES_COLLECTION,
        )

    # ------------------------------------------------------------------
    # Submission CRUD
    # ------------------------------------------------------------------

    async def insert_submission(
        self, doc: Dict[str, Any]
    ) -> tuple[Dict[str, Any], bool]:
        """Persist a submission document with write-once semantics.

        Deduplication is enforced by two unique indexes:

        1. ``uniq_submission_id`` — on ``submission_id``
        2. ``uniq_exam_student``  — on ``(exam_id, student_id)``

        A ``DuplicateKeyError`` on *either* index triggers the idempotent
        return path (ING-03): the existing document is returned after
        verifying ``content_hash`` consistency.

        Returns
        -------
        (document, already_existed)
            The persisted (or pre-existing) document and a boolean flag
            indicating whether a duplicate was detected (ING-03).

        Raises
        ------
        ImmutableDocumentError
            If a document with the same key(s) exists but carries a
            different ``content_hash`` (integrity violation — should never
            happen in normal operation).
        """
        submission_id = doc["submission_id"]
        exam_id = doc["exam_id"]
        student_id = doc["student_id"]
        try:
            await self._submissions.insert_one(doc)
            logger.info("Persisted submission %s", submission_id)
            return doc, False
        except DuplicateKeyError:
            # ING-03: idempotent duplicate handling.
            # The duplicate may be on submission_id OR on (exam_id, student_id).
            # Try both lookups so we find the existing document regardless of
            # which unique index triggered the error.
            existing = await self._submissions.find_one(
                {"submission_id": submission_id}
            )
            if existing is None:
                # Duplicate was on the (exam_id, student_id) compound index
                existing = await self._submissions.find_one(
                    {"exam_id": exam_id, "student_id": student_id}
                )
            if existing is None:
                # Extremely unlikely race: doc vanished between insert and
                # find.  Retry the insert once.
                await self._submissions.insert_one(doc)
                return doc, False

            # Verify content consistency on duplicate
            if existing.get("content_hash") != doc.get("content_hash"):
                raise ImmutableDocumentError(
                    f"Submission for exam_id={exam_id} student_id={student_id} "
                    f"already exists with a different content_hash. "
                    f"Existing: {existing.get('content_hash')}, "
                    f"new: {doc.get('content_hash')}. "
                    f"This violates immutability (TAMPER_PROOF_SPEC Layer 1)."
                )

            logger.info(
                "Duplicate submission detected (exam_id=%s, student_id=%s) "
                "— idempotent return (ING-03)",
                exam_id,
                student_id,
            )
            return existing, True

    async def get_submission(
        self, submission_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single submission by its canonical ID."""
        return await self._submissions.find_one(
            {"submission_id": submission_id}
        )

    async def get_submission_by_exam_student(
        self, exam_id: str, student_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a submission by exam + student (common lookup for engines)."""
        return await self._submissions.find_one(
            {"exam_id": exam_id, "student_id": student_id}
        )

    async def list_submissions(
        self,
        *,
        admin_id: Optional[str] = None,
        exam_id: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """List submissions with optional admin/exam filters."""
        query: Dict[str, Any] = {}
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

    # ------------------------------------------------------------------
    # Answer page CRUD
    # ------------------------------------------------------------------

    async def insert_answer_page(
        self, doc: Dict[str, Any]
    ) -> tuple[Dict[str, Any], bool]:
        """Persist a single answer-page document with write-once semantics.

        Duplicate handling mirrors ``insert_submission`` (ING-03).
        """
        page_id = doc["page_id"]
        submission_id = doc.get("submission_id")
        page_number = doc.get("page_number")
        try:
            await self._answer_pages.insert_one(doc)
            logger.info("Persisted answer page %s", page_id)
            return doc, False
        except DuplicateKeyError:
            # DuplicateKeyError may come from either the page_id unique index
            # or the (submission_id, page_number) compound unique index.
            # Try both lookup paths (ING-03).
            existing = await self._answer_pages.find_one({"page_id": page_id})
            if existing is None and submission_id and page_number is not None:
                existing = await self._answer_pages.find_one({
                    "submission_id": submission_id,
                    "page_number": page_number,
                })
            if existing is None:
                # Race condition edge case — retry once
                await self._answer_pages.insert_one(doc)
                return doc, False

            if existing.get("content_hash") != doc.get("content_hash"):
                raise ImmutableDocumentError(
                    f"Answer page {page_id} (submission={submission_id}, "
                    f"page={page_number}) already exists with a "
                    f"different content_hash. "
                    f"This violates immutability (TAMPER_PROOF_SPEC Layer 1)."
                )

            logger.info(
                "Duplicate answer page %s (submission=%s, page=%s) "
                "detected — idempotent return (ING-03)",
                page_id, submission_id, page_number,
            )
            return existing, True

    async def insert_answer_pages_bulk(
        self, docs: List[Dict[str, Any]]
    ) -> tuple[int, int, List[str]]:
        """Bulk-insert answer pages.  Skips duplicates (ING-03).

        Returns
        -------
        (inserted_count, duplicate_count, inserted_page_ids)

        Returning the exact inserted IDs is required for safe compensating
        cleanup.  Treating every attempted ID as newly inserted can delete an
        older immutable page when a later submission write fails.
        """
        if not docs:
            return 0, 0, []

        inserted = 0
        duplicates = 0
        inserted_page_ids: List[str] = []

        for doc in docs:
            _, already_existed = await self.insert_answer_page(doc)
            if already_existed:
                duplicates += 1
            else:
                inserted += 1
                inserted_page_ids.append(str(doc["page_id"]))

        return inserted, duplicates, inserted_page_ids

    async def get_answer_pages(
        self, submission_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch all answer pages for a submission, ordered by page number."""
        cursor = (
            self._answer_pages.find({"submission_id": submission_id})
            .sort("page_number", 1)
        )
        return await cursor.to_list(length=500)

    async def get_answer_page(
        self, page_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single answer page by its canonical ID."""
        return await self._answer_pages.find_one({"page_id": page_id})

    async def delete_answer_pages_by_ids(
        self, page_ids: List[str]
    ) -> int:
        """Delete answer pages by their page IDs (orphan cleanup).

        This is used exclusively by the service layer to roll back
        orphaned pages when a submission insert fails after pages have
        already been committed.  It is NOT a general-purpose delete —
        immutability enforcement does not apply here because the parent
        submission was never persisted, so these pages are dangling.

        Returns the number of documents deleted.
        """
        if not page_ids:
            return 0
        result = await self._answer_pages.delete_many(
            {"page_id": {"$in": page_ids}}
        )
        logger.info(
            "Cleaned up %d orphaned answer pages (of %d requested)",
            result.deleted_count,
            len(page_ids),
        )
        return result.deleted_count

    # ------------------------------------------------------------------
    # Immutability enforcement
    # ------------------------------------------------------------------

    async def assert_not_immutable(
        self, collection_name: str, doc_id_field: str, doc_id_value: str
    ) -> None:
        """Raise ``ImmutableDocumentError`` if the target document is immutable.

        This is a guard intended for any future administrative or migration
        operation that needs to verify immutability before proceeding.

        Parameters
        ----------
        collection_name:
            Either ``evalpen_submissions`` or ``evalpen_answer_pages``.
        doc_id_field:
            The field name used as the document identifier (e.g. ``submission_id``).
        doc_id_value:
            The value of the identifier to look up.
        """
        collection = self._db[collection_name]
        doc = await collection.find_one(
            {doc_id_field: doc_id_value},
            projection={"_immutable": 1},
        )
        if doc is not None and doc.get("_immutable", False):
            raise ImmutableDocumentError(
                f"Document {doc_id_field}={doc_id_value} in "
                f"{collection_name} is immutable and cannot be modified."
            )

    # ------------------------------------------------------------------
    # Status update (non-immutable field)
    # ------------------------------------------------------------------

    async def update_segmentation_status(
        self, submission_id: str, status: str
    ) -> bool:
        """Update the segmentation_status on a submission.

        ``segmentation_status`` is explicitly NOT covered by the immutability
        flag — it is a processing-status field set by downstream engines, not
        part of the canonical artifact content.  The ``content_hash`` and raw
        artifact data remain untouched.

        Returns True if the document was found and updated.
        """
        result = await self._submissions.update_one(
            {"submission_id": submission_id},
            {"$set": {"segmentation_status": status}},
        )
        return result.modified_count > 0
