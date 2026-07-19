"""
Ingest Service
==============

Orchestrates the ingestion of conducted-exam artifacts into canonical
MongoDB persistence.  This is the primary entry point for upstream callers
(hub upload endpoints in SWM-010 / SWM-012).

Responsibilities
----------------
1. Validate and normalize incoming artifact data.
2. Compute per-page and submission-level content hashes (TAMPER_PROOF_SPEC).
3. Persist pages and submission via the repository layer (with orphan
   cleanup on failure — see ``ingest_submission`` for details).
4. Return an ``IngestResult`` envelope for the API handler.

Non-responsibilities
--------------------
- No DCR/PCR evaluation logic.
- No LLM calls.
- No practice persistence.

References
----------
- Architecture:  new-docs/architecture/DUAL_MODE_ARCHITECTURE.md (Section 3)
- Integrity:     new-docs/architecture/TAMPER_PROOF_SPEC.md (Layer 1)
- Ownership:     new-docs/governance/STATE_OWNERSHIP_MAP.md
- Failure modes: ING-01 (loss), ING-02 (mis-attribution), ING-03 (duplicates)
- Test IDs:      U-ING-01, U-ING-02, I-ING-01, I-ING-02
- Event schema:  new-docs/contracts/events/eval.submission.received.schema.json
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from .hashing import compute_content_hash, compute_page_hash
from .models import (
    AnswerPage,
    ArtifactSource,
    ConductedExamSubmission,
    IngestResult,
    PageRef,
    SubmissionStatus,
)
from .repository import IngestRepository

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """Generate a globally unique, URL-safe identifier."""
    return uuid.uuid4().hex


def _deterministic_submission_id(exam_id: str, student_id: str) -> str:
    """Derive a stable submission_id from (exam_id, student_id).

    Using a deterministic ID ensures that re-ingesting the same
    (exam, student) pair always targets the same submission_id,
    which aligns the submission_id unique index with the compound
    (exam_id, student_id) dedup index (ING-03).
    """
    h = hashlib.sha256()
    h.update(f"submission:{exam_id}:{student_id}".encode("utf-8"))
    return h.hexdigest()[:32]  # 32 hex chars, same length as uuid4().hex


def _deterministic_page_id(submission_id: str, page_number: int) -> str:
    """Derive a stable page_id from (submission_id, page_number).

    Ensures re-ingest of the same exam/student/page produces the same
    page_id, so the unique page_id index catches true duplicates (ING-03).
    """
    h = hashlib.sha256()
    h.update(f"page:{submission_id}:{page_number}".encode("utf-8"))
    return h.hexdigest()[:32]


class IngestService:
    """High-level ingest operations for conducted-exam artifacts.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        Tenant-scoped Motor database (``skb_<tenant>``).
        Obtained via ``DatabaseManager.get_tenant_db(db_name)``.

    Usage
    -----
    ::

        db = await db_manager.get_tenant_db(db_name)
        service = IngestService(db)
        await service.initialize()
        result = await service.ingest_submission(...)
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._repo = IngestRepository(db)

    async def initialize(self) -> None:
        """Create required indexes.  Safe to call repeatedly."""
        await self._repo.ensure_indexes()

    # ------------------------------------------------------------------
    # Core ingest
    # ------------------------------------------------------------------

    async def ingest_submission(
        self,
        *,
        exam_id: str,
        student_id: str,
        admin_id: str,
        source: ArtifactSource | str,
        pen_mac: Optional[str] = None,
        hub_id: Optional[str] = None,
        pages: Optional[List[Dict[str, Any]]] = None,
        page_refs: Optional[List[Dict[str, Any]]] = None,
    ) -> IngestResult:
        """Ingest a complete conducted-exam submission.

        This is the main entry point.  It performs the following steps:

        1. Validate provenance fields (ING-02).
        2. Persist each page as an ``evalpen_answer_pages`` document with
           its own ``content_hash`` (U-ING-02).
        3. Compute a submission-level ``content_hash`` from the ordered
           page hashes.
        4. Persist the ``evalpen_submissions`` document.
        5. Return an ``IngestResult`` envelope.

        All persistence is write-once (ING-01) and idempotent (ING-03).

        Parameters
        ----------
        exam_id : str
            Conducted exam identifier.
        student_id : str
            Student identity.
        admin_id : str
            Tenant admin who owns this exam context.
        source : ArtifactSource | str
            Origin of the artifact (``ble_pen`` or ``camera``).
        pen_mac : str, optional
            BLE pen MAC address.  Required when ``source`` is ``ble_pen``.
        pages : list[dict], optional
            List of raw page payloads.  Each dict may contain:
            - ``page_number`` (int, required, 1-based)
            - ``raw_strokes`` (list[dict], for pen path)
            - ``raw_image_ref`` (str, for camera path)
        page_refs : list[dict], optional
            Lightweight page references when raw data was pre-uploaded
            separately (matches ``CreateSubmissionRequest.page_refs``).

        Returns
        -------
        IngestResult
            Contains ``submission_id``, ``content_hash``, ``page_count``,
            ``segmentation_status``, and ``already_existed``.
        """
        # Normalize source enum
        if isinstance(source, str):
            source = ArtifactSource(source)

        # Validate provenance (ING-02)
        if source == ArtifactSource.BLE_PEN and not pen_mac:
            raise ValueError(
                "pen_mac is required when source is ble_pen (ING-02 provenance)"
            )

        now = datetime.now(timezone.utc)
        submission_id = _deterministic_submission_id(exam_id, student_id)

        # ----- Persist pages -----
        page_hashes: List[str] = []
        page_ref_models: List[PageRef] = []
        pages_to_insert: List[Dict[str, Any]] = []

        raw_pages = pages or []
        for page_data in raw_pages:
            page_number = page_data["page_number"]
            raw_strokes = page_data.get("raw_strokes")
            raw_image_ref = page_data.get("raw_image_ref")
            asset_sha256 = page_data.get("asset_sha256") or page_data.get("content_hash")

            page_hash = compute_page_hash(
                page_number=page_number,
                raw_strokes=raw_strokes,
                raw_image_ref=raw_image_ref,
                asset_sha256=asset_sha256,
            )
            page_hashes.append(page_hash)

            page_model = AnswerPage(
                page_id=_deterministic_page_id(submission_id, page_number),
                submission_id=submission_id,
                exam_id=exam_id,
                student_id=student_id,
                admin_id=admin_id,
                page_number=page_number,
                source=source,
                pen_mac=pen_mac,
                raw_strokes=raw_strokes,
                raw_image_ref=raw_image_ref,
                asset_sha256=(str(asset_sha256).strip().lower() if asset_sha256 else None),
                image_width_px=page_data.get("image_width_px"),
                image_height_px=page_data.get("image_height_px"),
                original_filename=page_data.get("original_filename"),
                upload_id=page_data.get("upload_id"),
                content_type=page_data.get("content_type"),
                file_size_bytes=page_data.get("file_size_bytes"),
                content_hash=page_hash,
                created_at=now,
            )
            pages_to_insert.append(page_model.to_mongo_doc())

            page_ref_models.append(
                PageRef(page_num=page_number, raw_asset_ref=raw_image_ref)
            )

        # Handle lightweight page_refs (pre-uploaded assets)
        if page_refs and not raw_pages:
            for pr in page_refs:
                page_num = pr.get("page_num", pr.get("page_number", 0))
                asset_ref = pr.get("raw_asset_ref")
                asset_sha256 = pr.get("asset_sha256") or pr.get("content_hash")
                page_hash = compute_page_hash(
                    page_number=page_num,
                    raw_image_ref=asset_ref,
                    asset_sha256=asset_sha256,
                )
                page_hashes.append(page_hash)
                page_ref_models.append(
                    PageRef(page_num=page_num, raw_asset_ref=asset_ref)
                )

        # ----- Compute submission-level hash -----
        content_hash = compute_content_hash(
            exam_id=exam_id,
            student_id=student_id,
            page_hashes=page_hashes,
        )

        # ----- Build submission model -----
        submission = ConductedExamSubmission(
            submission_id=submission_id,
            exam_id=exam_id,
            student_id=student_id,
            admin_id=admin_id,
            source=source,
            pen_mac=pen_mac,
            hub_id=hub_id,
            page_count=len(page_hashes),
            page_refs=page_ref_models,
            content_hash=content_hash,
            upload_status="acknowledged",
            segmentation_status=SubmissionStatus.PENDING,
            submitted_at=now,
            created_at=now,
        )

        # ----- Persist pages first (ING-01: data before metadata) -----
        # Pages are inserted before the submission document.  If the
        # subsequent submission insert fails, orphaned pages are cleaned
        # up in the except block below.  This is *not* a true MongoDB
        # multi-document transaction (we only hold an AsyncIOMotorDatabase,
        # not the client), but it provides best-effort atomicity with
        # deterministic cleanup.
        inserted_page_ids: List[str] = []

        if pages_to_insert:
            inserted, duplicates, inserted_page_ids = await self._repo.insert_answer_pages_bulk(
                pages_to_insert
            )
            logger.info(
                "Answer pages for submission %s: %d inserted, %d duplicates",
                submission_id,
                inserted,
                duplicates,
            )

        # ----- Persist submission (with orphan cleanup on failure) -----
        try:
            sub_doc, already_existed = await self._repo.insert_submission(
                submission.to_mongo_doc()
            )
        except Exception:
            # Submission insert failed after pages were committed.
            # Clean up orphaned pages to maintain consistency.
            if inserted_page_ids:
                logger.warning(
                    "Submission insert failed for %s — cleaning up %d "
                    "orphaned answer pages",
                    submission_id,
                    len(inserted_page_ids),
                )
                try:
                    await self._repo.delete_answer_pages_by_ids(
                        inserted_page_ids
                    )
                except Exception:
                    logger.error(
                        "Failed to clean up orphaned pages for submission %s. "
                        "Page IDs: %s",
                        submission_id,
                        inserted_page_ids,
                        exc_info=True,
                    )
            raise

        if already_existed:
            # Return the existing submission's details (ING-03 idempotency)
            return IngestResult(
                submission_id=sub_doc["submission_id"],
                content_hash=sub_doc["content_hash"],
                page_count=sub_doc.get("page_count", 0),
                segmentation_status=SubmissionStatus(
                    sub_doc.get("segmentation_status", "pending")
                ),
                already_existed=True,
            )

        logger.info(
            "Ingested submission %s for exam=%s student=%s admin=%s "
            "source=%s pages=%d hash=%s",
            submission_id,
            exam_id,
            student_id,
            admin_id,
            source,
            len(page_hashes),
            content_hash[:16],
        )

        return IngestResult(
            submission_id=submission_id,
            content_hash=content_hash,
            page_count=len(page_hashes),
            segmentation_status=SubmissionStatus.PENDING,
            already_existed=False,
        )

    # ------------------------------------------------------------------
    # Read helpers (for DCR / PCR engines — server-side fetch)
    # ------------------------------------------------------------------

    async def get_submission(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a canonical submission by ID.

        This is the server-side fetch path required by TAMPER_PROOF_SPEC
        Layer 2.  Engines call this instead of trusting client-supplied data.
        """
        return await self._repo.get_submission(submission_id)

    async def get_submission_by_exam_student(
        self, exam_id: str, student_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a submission by exam + student combination.

        Convenience method for engine lookup flows.
        """
        return await self._repo.get_submission_by_exam_student(
            exam_id, student_id
        )

    async def get_answer_pages(
        self, submission_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch all answer pages for a submission (ordered by page number).

        Engines use this to retrieve the canonical raw artifacts for
        recognition / OCR / segmentation.
        """
        return await self._repo.get_answer_pages(submission_id)

    async def get_answer_page(self, page_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single answer page by its canonical page ID."""
        return await self._repo.get_answer_page(page_id)

    async def list_submissions(
        self,
        *,
        admin_id: Optional[str] = None,
        exam_id: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """List submissions with optional filters.

        Used by the submission listing API endpoint
        (``GET /api/v1/evalpen/submissions``).
        """
        return await self._repo.list_submissions(
            admin_id=admin_id,
            exam_id=exam_id,
            limit=limit,
            skip=skip,
        )

    async def update_segmentation_status(
        self, submission_id: str, status: SubmissionStatus | str
    ) -> bool:
        """Update the downstream processing status on a submission.

        Called by DCR/PCR engines after segmentation/processing completes.
        This field is NOT covered by immutability — it is a processing
        status, not part of the canonical artifact content.
        """
        if isinstance(status, SubmissionStatus):
            status = status.value
        return await self._repo.update_segmentation_status(
            submission_id, status
        )

    # ------------------------------------------------------------------
    # Event construction helper
    # ------------------------------------------------------------------

    @staticmethod
    def build_submission_received_event(
        result: IngestResult,
        *,
        exam_id: str,
        student_id: str,
        source: ArtifactSource | str,
    ) -> Dict[str, Any]:
        """Build an ``eval.submission.received`` event payload.

        Matches ``new-docs/contracts/events/eval.submission.received.schema.json``.

        The caller (route handler) is responsible for publishing this event
        to whatever transport is configured (in-process, message queue, etc.).
        """
        if isinstance(source, ArtifactSource):
            source_str = "pen" if source == ArtifactSource.BLE_PEN else "camera"
        else:
            source_str = "pen" if source == "ble_pen" else "camera"

        return {
            "event_id": _generate_id(),
            "event_type": "eval.submission.received",
            "event_version": "2.0.0",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "submission_id": result.submission_id,
            "exam_id": exam_id,
            "student_id": student_id,
            "source": source_str,
        }
