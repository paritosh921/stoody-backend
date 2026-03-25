"""
DCR MongoDB repository.

Handles persistence for ``exampen_dcr_results`` and read access to the shared
ingest collections ``evalpen_submissions`` and ``evalpen_answer_pages``
(owned by the ingest substrate — read-only here).

Storage: MongoDB only (C1).  Per-tenant DB ``skb_<tenant>``.
Collections: DUAL_MODE_ARCHITECTURE.md §4.6
Ownership: STATE_OWNERSHIP_MAP.md — DCR engine writes DCR results.

Test IDs: I-DCR-01 (canonical artifact -> DCR result commit)
Failure modes: DCR-03 (scope guard — DCR never writes to ingest artifacts)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, IndexModel

from .models import (
    AuditAction,
    DCRAuditEntry,
    DCRResult,
    DCRSubmission,
    DCRSubmissionPage,
    MatchType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------
# Ingest substrate collections (read-only from DCR — C5, DCR-03)
SUBMISSIONS_COLLECTION = "evalpen_submissions"
ANSWER_PAGES_COLLECTION = "evalpen_answer_pages"
# DCR-owned results collection
RESULTS_COLLECTION = "exampen_dcr_results"


class DCRRepository:
    """
    Async MongoDB repository for the DCR engine.

    Accepts a Motor ``AsyncIOMotorDatabase`` representing the tenant DB
    (resolved via ``DatabaseManager.get_tenant_db(db_name)``).
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._submissions = db[SUBMISSIONS_COLLECTION]
        self._answer_pages = db[ANSWER_PAGES_COLLECTION]
        self._results = db[RESULTS_COLLECTION]

    # ------------------------------------------------------------------
    # Index management (idempotent)
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """
        Create indexes for DCR-owned collections.

        Safe to call multiple times — Motor/PyMongo skips existing indexes
        with identical specs.

        Note: Indexes on ``evalpen_submissions`` and ``evalpen_answer_pages``
        are managed by the ingest repository (IngestRepository.ensure_indexes).
        DCR only ensures indexes on its own ``exampen_dcr_results`` collection.
        """
        # exampen_dcr_results indexes (DCR-owned)
        await self._results.create_indexes([
            IndexModel(
                [
                    ("exam_id", ASCENDING),
                    ("student_id", ASCENDING),
                    ("question_id", ASCENDING),
                ],
                unique=True,
                name="uniq_exam_student_question",
            ),
        ])

        logger.info(
            "DCR indexes ensured on %s.%s",
            self._db.name,
            RESULTS_COLLECTION,
        )

    # ------------------------------------------------------------------
    # Submissions & answer pages — READ-ONLY (C5, DCR-03)
    # ------------------------------------------------------------------

    async def get_submission(self, submission_id: str) -> Optional[DCRSubmission]:
        """
        Fetch a single submission from ``evalpen_submissions`` by ``submission_id``.

        Returns ``None`` if not found.  DCR never mutates submission docs.
        """
        doc = await self._submissions.find_one({"submission_id": submission_id})
        if doc is None:
            return None
        return _doc_to_submission(doc)

    async def get_submission_by_exam_student(
        self,
        exam_id: str,
        student_id: str,
    ) -> Optional[DCRSubmission]:
        """
        Fetch the submission for a given exam + student pair.

        There should be at most one canonical submission per (exam, student).
        """
        doc = await self._submissions.find_one({
            "exam_id": exam_id,
            "student_id": student_id,
        })
        if doc is None:
            return None
        return _doc_to_submission(doc)

    async def get_submission_pages(
        self,
        submission_id: str,
    ) -> List[DCRSubmissionPage]:
        """
        Fetch all answer pages for a submission from ``evalpen_answer_pages``.

        Returns pages ordered by ``page_number`` (ascending).
        DCR never mutates answer page docs (C5, DCR-03).
        """
        cursor = (
            self._answer_pages.find({"submission_id": submission_id})
            .sort("page_number", 1)
        )
        docs = await cursor.to_list(length=500)
        return [_doc_to_page(d) for d in docs]

    # ------------------------------------------------------------------
    # Results — WRITE (DCR engine is the writable owner)
    # ------------------------------------------------------------------

    async def upsert_result(self, result: DCRResult) -> bool:
        """
        Upsert a single DCR result document.

        Transactional boundary: recognized text + match result + score must be
        committed atomically (STATE_OWNERSHIP_MAP.md §4).

        Uses the unique compound index ``{ exam_id, student_id, question_id }``
        as the upsert key.

        Returns ``True`` if the document was inserted or modified.
        """
        now = datetime.now(timezone.utc)
        doc = result.model_dump(mode="json")
        doc["updated_at"] = now.isoformat()

        # Convert audit_trail entries so they serialize cleanly
        if "audit_trail" in doc:
            for entry in doc["audit_trail"]:
                if isinstance(entry.get("occurred_at"), datetime):
                    entry["occurred_at"] = entry["occurred_at"].isoformat()

        filter_key = {
            "exam_id": result.exam_id,
            "student_id": result.student_id,
            "question_id": result.question_id,
        }

        update_result = await self._results.update_one(
            filter_key,
            {
                "$set": {
                    "recognized_text": result.recognized_text,
                    "confidence": result.confidence,
                    "match_type": result.match_type.value,
                    "score": result.score,
                    "max_score": result.max_score,
                    "updated_at": now.isoformat(),
                },
                "$push": {
                    "audit_trail": {
                        "$each": [
                            entry.model_dump(mode="json")
                            for entry in result.audit_trail
                        ],
                    },
                },
                "$setOnInsert": {
                    "exam_id": result.exam_id,
                    "student_id": result.student_id,
                    "question_id": result.question_id,
                    "created_at": now.isoformat(),
                },
            },
            upsert=True,
        )
        modified = update_result.modified_count > 0
        upserted = update_result.upserted_id is not None
        return modified or upserted

    async def upsert_results_batch(self, results: List[DCRResult]) -> int:
        """
        Upsert a batch of DCR results.

        Returns the number of documents inserted or modified.
        """
        count = 0
        for result in results:
            if await self.upsert_result(result):
                count += 1
        return count

    async def get_result(
        self,
        exam_id: str,
        student_id: str,
        question_id: str,
    ) -> Optional[DCRResult]:
        """Fetch a single stored DCR result."""
        doc = await self._results.find_one({
            "exam_id": exam_id,
            "student_id": student_id,
            "question_id": question_id,
        })
        if doc is None:
            return None
        return _doc_to_result(doc)

    async def get_results_for_exam_student(
        self,
        exam_id: str,
        student_id: str,
    ) -> List[DCRResult]:
        """Fetch all DCR results for an exam + student pair."""
        cursor = self._results.find({
            "exam_id": exam_id,
            "student_id": student_id,
        })
        docs = await cursor.to_list(length=500)
        return [_doc_to_result(d) for d in docs]

    async def get_total_score(
        self,
        exam_id: str,
        student_id: str,
    ) -> Dict[str, float]:
        """
        Compute aggregate total_score and total_max_score for an exam + student.

        Returns ``{"total_score": float, "total_max_score": float}``.
        """
        pipeline = [
            {"$match": {"exam_id": exam_id, "student_id": student_id}},
            {
                "$group": {
                    "_id": None,
                    "total_score": {"$sum": "$score"},
                    "total_max_score": {"$sum": "$max_score"},
                }
            },
        ]
        cursor = self._results.aggregate(pipeline)
        result_list = await cursor.to_list(length=1)
        if not result_list:
            return {"total_score": 0.0, "total_max_score": 0.0}
        row = result_list[0]
        return {
            "total_score": float(row.get("total_score", 0)),
            "total_max_score": float(row.get("total_max_score", 0)),
        }

    async def append_audit_entry(
        self,
        exam_id: str,
        student_id: str,
        question_id: str,
        entry: DCRAuditEntry,
    ) -> bool:
        """
        Append a single audit trail entry to an existing result.

        Used by manual override and rescore flows.
        """
        update_result = await self._results.update_one(
            {
                "exam_id": exam_id,
                "student_id": student_id,
                "question_id": question_id,
            },
            {
                "$push": {"audit_trail": entry.model_dump(mode="json")},
                "$set": {"updated_at": datetime.now(timezone.utc).isoformat()},
            },
        )
        return update_result.modified_count > 0


# ---------------------------------------------------------------------------
# Internal helpers — document ↔ model mapping
# ---------------------------------------------------------------------------

def _doc_to_submission(doc: Dict[str, Any]) -> DCRSubmission:
    """Convert a raw ``evalpen_submissions`` MongoDB document to a ``DCRSubmission``."""
    from .models import PageRef

    page_refs = [
        PageRef(
            page_num=pr["page_num"],
            raw_asset_ref=pr.get("raw_asset_ref"),
        )
        for pr in doc.get("page_refs", [])
    ]

    return DCRSubmission(
        submission_id=doc["submission_id"],
        exam_id=doc["exam_id"],
        student_id=doc["student_id"],
        admin_id=doc["admin_id"],
        source=doc.get("source"),
        pen_mac=doc.get("pen_mac"),
        page_count=doc.get("page_count", 0),
        page_refs=page_refs,
        content_hash=doc.get("content_hash"),
        _immutable=doc.get("_immutable", True),
        segmentation_status=doc.get("segmentation_status"),
        submitted_at=_parse_dt(doc.get("submitted_at")),
    )


def _doc_to_page(doc: Dict[str, Any]) -> DCRSubmissionPage:
    return DCRSubmissionPage(
        page_id=doc.get("page_id", ""),
        submission_id=doc.get("submission_id", ""),
        exam_id=doc.get("exam_id", ""),
        student_id=doc.get("student_id", ""),
        admin_id=doc.get("admin_id", ""),
        page_number=doc["page_number"],
        source=doc.get("source"),
        pen_mac=doc.get("pen_mac"),
        raw_strokes=doc.get("raw_strokes"),
        raw_image_ref=doc.get("raw_image_ref"),
        content_hash=doc.get("content_hash", ""),
        _immutable=doc.get("_immutable", True),
    )


def _doc_to_result(doc: Dict[str, Any]) -> DCRResult:
    """Convert a raw MongoDB document to a ``DCRResult`` model."""
    audit_trail_raw = doc.get("audit_trail", [])
    audit_entries: list[DCRAuditEntry] = []
    for entry_doc in audit_trail_raw:
        audit_entries.append(
            DCRAuditEntry(
                action=AuditAction(entry_doc["action"]),
                actor=entry_doc.get("actor"),
                previous_score=entry_doc.get("previous_score"),
                new_score=entry_doc.get("new_score"),
                previous_match_type=(
                    MatchType(entry_doc["previous_match_type"])
                    if entry_doc.get("previous_match_type")
                    else None
                ),
                new_match_type=(
                    MatchType(entry_doc["new_match_type"])
                    if entry_doc.get("new_match_type")
                    else None
                ),
                gate_call_ref=entry_doc.get("gate_call_ref"),
                note=entry_doc.get("note"),
                occurred_at=_parse_dt(entry_doc.get("occurred_at")),
            )
        )

    return DCRResult(
        exam_id=doc["exam_id"],
        student_id=doc["student_id"],
        question_id=doc["question_id"],
        recognized_text=doc["recognized_text"],
        confidence=doc["confidence"],
        match_type=MatchType(doc["match_type"]),
        score=doc["score"],
        max_score=doc["max_score"],
        audit_trail=audit_entries,
        created_at=_parse_dt(doc.get("created_at")),
        updated_at=_parse_dt(doc.get("updated_at")),
    )


def _parse_dt(value: Any) -> datetime:
    """Best-effort datetime parsing from a Mongo document field."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)
