"""
EvalPen Teacher BFF (Backend-for-Frontend) API — Thin read-only
aggregation layer for teacher exam dashboard views.

Provides two endpoints:
  1. GET /exams — List all exams with per-exam aggregation counts
  2. GET /exams/{exam_id}/queue — Submission queue bucketed by status

Architecture:
    Read-only aggregation over evalpen_submissions,
    evalpen_detected_responses, evalpen_evaluations,
    evalpen_questions, exampen_dcr_results, and exampen_processing_jobs.
    No writes — this module is a pure reader.

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  NONE
    - Reads from: evalpen_submissions, evalpen_detected_responses,
                  evalpen_evaluations, evalpen_questions,
                  exampen_dcr_results, exampen_processing_jobs,
                  documents (finalized offline exams)
    - Never writes to: any collection

Hard constraints:
    - C1: MongoDB only
    - C5: Ownership boundaries — BFF endpoints are pure readers
    - Read-only: no $set, $push, insert, or update operations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from api.v1.exam_orch_async import (
    _current_tutor_id,
    _is_exam_visible_to_tutor,
    _is_tutor_admin_role,
)
from utils.tutor_scoping import get_tutor_scoped_students
from services.exampen_submission_readiness import (
    assess_submissions_readiness,
    readiness_message,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency: require admin or tutor role for teacher BFF endpoints."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for teacher dashboard",
        )
    return current_user


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ExamSummaryItem(BaseModel):
    """Per-exam aggregation row for the teacher exam list."""

    exam_id: str
    title: str
    exam_type: Optional[str] = None
    prepared_document_id: Optional[str] = None
    lifecycle_state: Optional[str] = None
    total_students: int = 0
    evaluated_count: int = 0
    blocked_count: int = 0
    published_count: int = 0
    # Prepared-exam fields (from finalized documents)
    status: str = "active"  # "prepared" | "active"
    exam_mode: Optional[str] = None  # "dcr" | "pcr" | None
    created_at: Optional[str] = None
    finalized_at: Optional[str] = None
    question_count: int = 0


class ExamListResponse(BaseModel):
    """Response for GET /exams."""

    items: List[ExamSummaryItem] = Field(default_factory=list)


class QueueItem(BaseModel):
    """A single submission entry within a queue bucket."""

    submission_id: str
    student_id: str
    response_count: int = 0
    page_count: int = 0
    source: Optional[str] = None
    status_summary: str = ""
    has_dcr_results: bool = False
    # PCR copy-upload jobs are durable and may still be queued before OCR
    # creates individual responses. Exposing that state prevents a teacher
    # from seeing a misleading generic "Pending" message.
    processing_status: Optional[str] = None
    processing_error: Optional[str] = None


class ExamQueueResponse(BaseModel):
    """Response for GET /exams/{exam_id}/queue."""

    pending: List[QueueItem] = Field(default_factory=list)
    blocked: List[QueueItem] = Field(default_factory=list)
    needs_review: List[QueueItem] = Field(default_factory=list)
    ready_to_publish: List[QueueItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: resolve tenant DB
# ---------------------------------------------------------------------------

async def _get_tenant_db(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Any:
    """Resolve the tenant database from the authenticated user's JWT claims."""
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context missing from authentication token",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


# ---------------------------------------------------------------------------
# Helper: tutor visibility scoping
# ---------------------------------------------------------------------------

async def _get_tutor_scoped_student_ids(
    current_user: Dict[str, Any],
    db: DatabaseManager,
) -> Optional[List[str]]:
    """Return the list of student_id strings visible to the current user.

    - Admins / b2c_admins: returns ``None`` (no filtering — full access).
    - Tutors: resolves scoped students via ``get_tutor_scoped_students()``
      and returns their ``student_id`` values.

    Raises 403 if a tutor has no admin_id context or the tutor document
    is missing.
    """
    user_type = current_user.get("user_type")
    if user_type in {"admin", "b2c_admin"}:
        return None  # admins see everything

    # Must be tutor — resolve scoped students
    from bson import ObjectId

    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    admin_id = current_user.get("admin_id")

    if not tutor_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor identity could not be determined",
        )

    try:
        admin_oid = ObjectId(admin_id) if admin_id else None
    except Exception:
        admin_oid = None

    if admin_oid is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor admin context missing — cannot determine visibility scope",
        )

    scoped_students = await get_tutor_scoped_students(
        tutor_id=str(tutor_id),
        admin_oid=admin_oid,
        db=db,
    )

    return [
        s.get("student_id")
        for s in scoped_students
        if s.get("student_id")
    ]


def _visible_exam_query_for_user(current_user: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ExamPen exam visibility query for the current actor."""
    if _is_tutor_admin_role(current_user):
        return {}

    tutor_id = _current_tutor_id(current_user)
    if tutor_id is None:
        return {"exam_id": {"$in": []}}

    return {
        "$or": [
            {"created_by_tutor_id": tutor_id},
            {"teacher_ids": tutor_id},
            {"teacher_ids": []},
            {"teacher_ids": None},
            {"teacher_ids": {"$exists": False}},
        ]
    }


async def _require_exam_visible_or_legacy_student_scope(
    tenant_db: Any,
    exam_id: str,
    current_user: Dict[str, Any],
) -> bool:
    """Return True if a tutor-visible exam doc exists.

    If no orchestration exam exists, callers may fall back to the older
    student-scope behavior for legacy submission-only data. If an exam doc does
    exist, its tutor ownership fields are authoritative.
    """
    if _is_tutor_admin_role(current_user):
        return True

    tutor_id = _current_tutor_id(current_user)
    if tutor_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor identity could not be determined",
        )

    exam_doc = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        projection={
            "exam_id": 1,
            "created_by_tutor_id": 1,
            "teacher_ids": 1,
        },
    )
    if exam_doc is None:
        return False
    if not _is_exam_visible_to_tutor(exam_doc, tutor_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Exam is not visible to this tutor",
        )
    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/exams",
    response_model=ExamListResponse,
    summary="List exams with per-exam aggregation counts",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def list_exams(
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamListResponse:
    """List all exams visible to the teacher with aggregated status counts.

    Groups submissions by ``exam_id`` and computes:
      - ``total_students`` — distinct student_ids with submissions
      - ``evaluated_count`` — submissions where ALL responses have
        eval_status != "pending"
      - ``blocked_count`` — submissions with any unresolved blocking flags
      - ``published_count`` — submissions with publication_status = "published"

    Exam title is resolved from ``evalpen_questions`` (first question's
    exam metadata) with a fallback to the raw exam_id.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        def _normalize_exam_type(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            return str(value).upper()

        # ----- Fetch prepared exams from finalized documents -----
        # These are offline exams that have been finalized but may have
        # no student submissions yet.  We merge them into the list so
        # teachers can see every exam they have prepared.
        prepared_items: List[ExamSummaryItem] = []
        prepared_doc_meta: Dict[str, Dict[str, Any]] = {}

        doc_query: Dict[str, Any] = {
            "$or": [
                {"exam_finalized": True},
                {"exam_mode": "dcr"},
            ],
        }
        # Tutor scoping: match existing document visibility model —
        # tutors see docs mapped to them OR docs open to all tutors
        # (teacher_ids is empty, null, or missing).
        if current_user.get("user_type") == "tutor":
            tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
            if tutor_id:
                doc_query = {
                    "$and": [
                        doc_query,
                        {"$or": [
                            {"teacher_ids": {"$in": [str(tutor_id)]}},
                            {"teacher_ids": []},
                            {"teacher_ids": None},
                            {"teacher_ids": {"$exists": False}},
                        ]},
                    ]
                }

        doc_cursor = tenant_db["documents"].find(
            doc_query,
            projection={
                "document_id": 1,
                "title": 1,
                "exam_mode": 1,
                "exam_finalized_at": 1,
            },
        )
        finalized_docs = await doc_cursor.to_list(length=5000)

        # Get live question counts per document_id (avoids stale
        # extracted_questions_count which drifts after edits).
        finalized_doc_ids = [
            d.get("document_id", "") for d in finalized_docs if d.get("document_id")
        ]
        live_q_counts: Dict[str, int] = {}
        if finalized_doc_ids:
            qc_cursor = tenant_db["questions"].aggregate([
                {"$match": {"document_id": {"$in": finalized_doc_ids}}},
                {"$group": {"_id": "$document_id", "count": {"$sum": 1}}},
            ])
            for qc in await qc_cursor.to_list(length=5000):
                live_q_counts[qc["_id"]] = qc["count"]

        for fdoc in finalized_docs:
            doc_id = fdoc.get("document_id", "")
            if doc_id:
                prepared_doc_meta[doc_id] = {
                    "title": fdoc.get("title", doc_id),
                    "exam_mode": fdoc.get("exam_mode"),
                    "finalized_at": (
                        fdoc["exam_finalized_at"].isoformat()
                        if fdoc.get("exam_finalized_at")
                        else None
                    ),
                    "question_count": live_q_counts.get(doc_id, 0),
                }
                prepared_items.append(
                    ExamSummaryItem(
                        exam_id=doc_id,
                        title=fdoc.get("title", doc_id),
                        exam_type=_normalize_exam_type(
                            fdoc.get("exam_mode")
                        ),
                        prepared_document_id=doc_id,
                        total_students=0,
                        evaluated_count=0,
                        blocked_count=0,
                        published_count=0,
                        status="prepared",
                        exam_mode=fdoc.get("exam_mode"),
                        created_at=(
                            fdoc["exam_finalized_at"].isoformat()
                            if fdoc.get("exam_finalized_at")
                            else None
                        ),
                        finalized_at=(
                            fdoc["exam_finalized_at"].isoformat()
                            if fdoc.get("exam_finalized_at")
                            else None
                        ),
                        question_count=live_q_counts.get(doc_id, 0),
                    )
                )

        # ----- Fetch orchestration exams even before submissions exist -----
        active_exam_query = _visible_exam_query_for_user(current_user)

        active_exam_docs = await tenant_db["exampen_exams"].find(
            active_exam_query,
            projection={
                "exam_id": 1,
                "exam_type": 1,
                "lifecycle_state": 1,
                "prepared_document_id": 1,
                "created_by_tutor_id": 1,
                "teacher_ids": 1,
                "created_at": 1,
            },
        ).to_list(length=5000)
        active_exam_map: Dict[str, Dict[str, Any]] = {
            doc["exam_id"]: doc
            for doc in active_exam_docs
            if doc.get("exam_id")
        }

        def _build_active_exam_item(
            exam_id: str,
            exam_doc: Dict[str, Any],
            *,
            total_students: int = 0,
            evaluated_count: int = 0,
            blocked_count: int = 0,
            published_count: int = 0,
            title_override: Optional[str] = None,
            exam_type_override: Optional[str] = None,
        ) -> ExamSummaryItem:
            prepared_document_id = exam_doc.get("prepared_document_id")
            prepared_meta = prepared_doc_meta.get(prepared_document_id or "", {})
            exam_type = (
                exam_type_override
                or _normalize_exam_type(exam_doc.get("exam_type"))
                or _normalize_exam_type(prepared_meta.get("exam_mode"))
            )
            title = (
                title_override
                or prepared_meta.get("title")
                or exam_id
            )
            return ExamSummaryItem(
                exam_id=exam_id,
                title=title,
                exam_type=exam_type,
                prepared_document_id=prepared_document_id,
                lifecycle_state=exam_doc.get("lifecycle_state"),
                total_students=total_students,
                evaluated_count=evaluated_count,
                blocked_count=blocked_count,
                published_count=published_count,
                status="active",
                exam_mode=prepared_meta.get("exam_mode") or exam_doc.get("exam_type"),
                created_at=(
                    exam_doc["created_at"].isoformat()
                    if hasattr(exam_doc.get("created_at"), "isoformat")
                    else str(exam_doc["created_at"])
                    if exam_doc.get("created_at")
                    else prepared_meta.get("finalized_at")
                ),
                finalized_at=prepared_meta.get("finalized_at"),
                question_count=prepared_meta.get("question_count", 0),
            )

        # ----- Fetch submissions (tutor-scoped) -----
        sub_query: Dict[str, Any] = {}
        if scoped_ids is not None:
            visible_exam_ids = list(active_exam_map.keys())
            if visible_exam_ids:
                sub_query["exam_id"] = {"$in": visible_exam_ids}
            else:
                sub_query["exam_id"] = {"$in": []}

        submissions_cursor = tenant_db["evalpen_submissions"].find(
            sub_query,
            projection={
                "submission_id": 1,
                "exam_id": 1,
                "student_id": 1,
                "publication_status": 1,
                "page_count": 1,
                "source": 1,
            },
        )
        submissions = await submissions_cursor.to_list(length=5000)

        if not submissions:
            items: List[ExamSummaryItem] = [
                _build_active_exam_item(exam_id, exam_doc)
                for exam_id, exam_doc in active_exam_map.items()
            ]
            linked_prepared_ids = {
                doc.get("prepared_document_id")
                for doc in active_exam_docs
                if doc.get("prepared_document_id")
            }
            for prep in prepared_items:
                if (
                    prep.exam_id not in linked_prepared_ids
                    and prep.exam_id not in active_exam_map
                ):
                    items.append(prep)
            items.sort(
                key=lambda item: (
                    item.created_at or item.finalized_at or "",
                    item.exam_id,
                ),
                reverse=True,
            )
            return ExamListResponse(items=items)

        # ----- Group submissions by exam_id -----
        # exam_id -> list of submission dicts
        exams_map: Dict[str, List[Dict[str, Any]]] = {}
        for sub in submissions:
            eid = sub.get("exam_id", "")
            if eid:
                exams_map.setdefault(eid, []).append(sub)

        # ----- Collect all submission_ids for batch response lookup -----
        all_sub_ids = [s.get("submission_id", "") for s in submissions]

        # ----- Fetch all responses for these submissions -----
        resp_cursor = tenant_db["evalpen_detected_responses"].find(
            {
                "submission_id": {"$in": all_sub_ids},
                "eval_status": {"$ne": "superseded"},
                "superseded_at": {"$exists": False},
            },
            projection={
                "response_id": 1,
                "submission_id": 1,
                "eval_status": 1,
                "flags": 1,
            },
        )
        all_responses = await resp_cursor.to_list(length=50000)

        # Group responses by submission_id
        responses_by_sub: Dict[str, List[Dict[str, Any]]] = {}
        for resp in all_responses:
            sid = resp.get("submission_id", "")
            responses_by_sub.setdefault(sid, []).append(resp)

        # ----- Resolve exam titles from evalpen_questions -----
        exam_ids = list(exams_map.keys())
        title_cursor = tenant_db["evalpen_questions"].aggregate([
            {"$match": {"exam_id": {"$in": exam_ids}}},
            {"$group": {
                "_id": "$exam_id",
                "title": {"$first": "$exam_title"},
                "exam_type": {"$first": "$exam_type"},
            }},
        ])
        title_docs = await title_cursor.to_list(length=5000)
        title_map: Dict[str, Dict[str, Any]] = {
            d["_id"]: d for d in title_docs
        }

        # ----- Determine which exams have DCR answer keys (metadata-driven) -----
        # Use exampen_answer_keys to detect DCR exams, NOT exampen_dcr_results.
        # This way a fresh DCR exam (before any results) is still recognized.
        ak_cursor = tenant_db["exampen_answer_keys"].aggregate([
            {"$match": {"exam_id": {"$in": exam_ids}}},
            {"$group": {"_id": "$exam_id"}},
        ])
        ak_docs = await ak_cursor.to_list(length=5000)
        exams_with_dcr: set = {d["_id"] for d in ak_docs}

        # ----- DCR: count distinct (exam_id, student_id) with results -----
        dcr_cursor = tenant_db["exampen_dcr_results"].aggregate([
            {"$match": {"exam_id": {"$in": exam_ids}}},
            {"$group": {
                "_id": {
                    "exam_id": "$exam_id",
                    "student_id": "$student_id",
                },
            }},
        ])
        dcr_pairs = await dcr_cursor.to_list(length=50000)

        # exam_id -> set of student_ids that have DCR results
        dcr_students_by_exam: Dict[str, set] = {}
        for pair in dcr_pairs:
            eid = pair["_id"]["exam_id"]
            sid = pair["_id"]["student_id"]
            dcr_students_by_exam.setdefault(eid, set()).add(sid)

        # ----- Build per-exam aggregation -----
        items: List[ExamSummaryItem] = []

        for exam_id, exam_subs in exams_map.items():
            student_ids_set: set = set()
            evaluated = 0
            blocked = 0
            published = 0
            exam_has_dcr = exam_id in exams_with_dcr
            dcr_students = dcr_students_by_exam.get(exam_id, set())

            for sub in exam_subs:
                sub_id = sub.get("submission_id", "")
                student_id = sub.get("student_id", "")
                if student_id:
                    student_ids_set.add(student_id)

                # Publication check
                if sub.get("publication_status") == "published":
                    published += 1

                # Get responses for this submission
                sub_responses = responses_by_sub.get(sub_id, [])

                # Check if ALL PCR responses are evaluated (none pending)
                pcr_all_evaluated = sub_responses and all(
                    r.get("eval_status", "pending") != "pending"
                    for r in sub_responses
                )

                # A student is "fully evaluated" only if:
                # 1. All PCR responses are evaluated, AND
                # 2. DCR results exist (if exam has DCR questions)
                dcr_complete = (
                    not exam_has_dcr
                    or student_id in dcr_students
                )
                if pcr_all_evaluated and dcr_complete:
                    evaluated += 1

                # Check for any unresolved blocking flags
                has_unresolved_blocking = False
                for resp in sub_responses:
                    flags = resp.get("flags", [])
                    for f in flags:
                        if (
                            f.get("severity") == "blocking"
                            and not f.get("resolution", {}).get(
                                "resolved", False
                            )
                        ):
                            has_unresolved_blocking = True
                            break
                    if has_unresolved_blocking:
                        break
                if has_unresolved_blocking:
                    blocked += 1

            # Resolve title
            title_info = title_map.get(exam_id, {})
            exam_doc = active_exam_map.get(exam_id, {})
            prepared_meta = prepared_doc_meta.get(
                (exam_doc.get("prepared_document_id") or ""),
                {},
            )
            title = (
                title_info.get("title")
                or prepared_meta.get("title")
                or exam_id
            )
            exam_type = (
                _normalize_exam_type(title_info.get("exam_type"))
                or _normalize_exam_type(exam_doc.get("exam_type"))
                or _normalize_exam_type(prepared_meta.get("exam_mode"))
            )

            if exam_doc:
                items.append(
                    _build_active_exam_item(
                        exam_id,
                        exam_doc,
                        total_students=len(student_ids_set),
                        evaluated_count=evaluated,
                        blocked_count=blocked,
                        published_count=published,
                        title_override=title,
                        exam_type_override=exam_type,
                    )
                )
            else:
                items.append(
                    ExamSummaryItem(
                        exam_id=exam_id,
                        title=title,
                        exam_type=exam_type,
                        total_students=len(student_ids_set),
                        evaluated_count=evaluated,
                        blocked_count=blocked,
                        published_count=published,
                        status="active",
                    )
                )

        # Add active exams that exist in orchestration but have no submissions yet
        submission_exam_ids = {item.exam_id for item in items}
        for exam_id, exam_doc in active_exam_map.items():
            if exam_id not in submission_exam_ids:
                items.append(
                    _build_active_exam_item(exam_id, exam_doc)
                )

        linked_prepared_ids = {
            doc.get("prepared_document_id")
            for doc in active_exam_docs
            if doc.get("prepared_document_id")
        }

        # Merge prepared exams that have no linked orchestration exam yet
        for prep in prepared_items:
            if (
                prep.exam_id not in submission_exam_ids
                and prep.exam_id not in active_exam_map
                and prep.exam_id not in linked_prepared_ids
            ):
                items.append(prep)

        # Teacher lists are chronological: newest exam first.
        items.sort(
            key=lambda item: (
                item.created_at or item.finalized_at or "",
                item.exam_id,
            ),
            reverse=True,
        )

        return ExamListResponse(items=items)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to list exams for teacher BFF: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exam list",
        )


@router.get(
    "/exams/{exam_id}/queue",
    response_model=ExamQueueResponse,
    summary="Get submission queue for an exam bucketed by status",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_exam_queue(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamQueueResponse:
    """Get the submission queue for a specific exam, split by workflow state:

    - ``pending``: submissions where responses exist but not all evaluated
    - ``blocked``: submissions with any unresolved blocking flags
    - ``needs_review``: technically complete submissions awaiting a teacher check
    - ``ready_to_publish``: submissions fully evaluated, no blocking flags,
      not yet published

    Each bucket item includes ``submission_id``, ``student_id``,
    ``response_count``, and a human-readable ``status_summary``.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        # ----- Fetch submissions for this exam (tutor-scoped) -----
        has_visible_exam_doc = await _require_exam_visible_or_legacy_student_scope(
            tenant_db,
            exam_id,
            current_user,
        )

        sub_query: Dict[str, Any] = {"exam_id": exam_id}
        if scoped_ids is not None:
            if not has_visible_exam_doc:
                sub_query["student_id"] = {"$in": scoped_ids}

        submissions_cursor = tenant_db["evalpen_submissions"].find(
            sub_query,
            projection={
                "submission_id": 1,
                "student_id": 1,
                "publication_status": 1,
                "page_count": 1,
                "source": 1,
            },
        )
        submissions = await submissions_cursor.to_list(length=5000)

        if not submissions:
            return ExamQueueResponse()

        # ----- Fetch all responses for these submissions -----
        sub_ids = [str(s.get("submission_id") or "") for s in submissions]
        page_counts_by_submission: Dict[str, int] = {}
        page_count_cursor = tenant_db["evalpen_answer_pages"].aggregate(
            [
                {"$match": {"submission_id": {"$in": sub_ids}}},
                {"$group": {"_id": "$submission_id", "count": {"$sum": 1}}},
            ]
        )
        for row in await page_count_cursor.to_list(length=max(len(sub_ids), 1)):
            row_submission_id = str(row.get("_id") or "")
            if row_submission_id:
                page_counts_by_submission[row_submission_id] = max(
                    0, int(row.get("count") or 0)
                )
        resp_cursor = tenant_db["evalpen_detected_responses"].find(
            {
                "submission_id": {"$in": sub_ids},
                "eval_status": {"$ne": "superseded"},
                "superseded_at": {"$exists": False},
            },
            projection={
                "response_id": 1,
                "submission_id": 1,
                "eval_status": 1,
                "flags": 1,
            },
        )
        all_responses = await resp_cursor.to_list(length=50000)

        # Group responses by submission_id
        responses_by_sub: Dict[str, List[Dict[str, Any]]] = {}
        for resp in all_responses:
            sid = resp.get("submission_id", "")
            responses_by_sub.setdefault(sid, []).append(resp)

        # ----- Surface durable PCR job state before OCR has created responses -----
        # A student copy is canonically submitted before the worker performs OCR
        # and evaluation.  Querying the job here lets the teacher distinguish
        # "queued" / "checking" from a genuinely missing submission.
        jobs_by_sub: Dict[str, Dict[str, Any]] = {}
        if sub_ids:
            jobs_cursor = tenant_db["exampen_processing_jobs"].find(
                {"submission_id": {"$in": sub_ids}},
                projection={
                    "submission_id": 1,
                    "status": 1,
                    "last_error": 1,
                    "updated_at": 1,
                },
            ).sort("updated_at", -1)
            for job in await jobs_cursor.to_list(length=5000):
                job_submission_id = str(job.get("submission_id") or "")
                if job_submission_id and job_submission_id not in jobs_by_sub:
                    jobs_by_sub[job_submission_id] = job

        # ----- Determine if this exam has DCR (metadata-driven) -----
        ak_count = await tenant_db["exampen_answer_keys"].count_documents(
            {"exam_id": exam_id}, limit=1
        )
        exam_has_dcr = ak_count > 0

        # ----- DCR: determine which students have DCR results -----
        dcr_student_ids: set = set()
        if exam_has_dcr:
            dcr_cursor = tenant_db["exampen_dcr_results"].aggregate([
                {"$match": {"exam_id": exam_id}},
                {"$group": {"_id": "$student_id"}},
            ])
            dcr_student_docs = await dcr_cursor.to_list(length=5000)
            dcr_student_ids = {d["_id"] for d in dcr_student_docs}

        pcr_question_count = await tenant_db["evalpen_questions"].count_documents(
            {"exam_id": exam_id}, limit=1
        )
        readiness_by_submission = (
            await assess_submissions_readiness(tenant_db, sub_ids)
            if pcr_question_count > 0
            else {}
        )

        # ----- Bucket each submission -----
        pending_items: List[QueueItem] = []
        blocked_items: List[QueueItem] = []
        review_items: List[QueueItem] = []
        ready_items: List[QueueItem] = []

        for sub in submissions:
            sub_id = str(sub.get("submission_id") or "")
            student_id = str(sub.get("student_id") or "")
            pub_status = sub.get("publication_status")
            sub_responses = responses_by_sub.get(sub_id, [])
            response_count = len(sub_responses)
            page_count = max(
                max(0, int(sub.get("page_count") or 0)),
                page_counts_by_submission.get(sub_id, 0),
            )
            submission_source = str(sub.get("source") or "") or None
            processing_job = jobs_by_sub.get(sub_id)
            processing_status = (
                str(processing_job.get("status") or "")
                if processing_job
                else None
            )
            processing_error = (
                str(processing_job.get("last_error") or "") or None
                if processing_job
                else None
            )

            # DCR completeness for this student
            student_has_dcr = student_id in dcr_student_ids
            dcr_complete = not exam_has_dcr or student_has_dcr

            if pcr_question_count > 0:
                readiness = readiness_by_submission.get(
                    sub_id,
                    {
                        "ready": False,
                        "blockers": [
                            {
                                "code": "readiness_missing",
                                "message": "Submission readiness could not be determined",
                            }
                        ],
                    },
                )
            elif exam_has_dcr:
                readiness = {"ready": True, "blockers": []}
            else:
                readiness = {
                    "ready": False,
                    "blockers": [
                        {
                            "code": "paper_catalog_missing",
                            "message": "PCR paper questions are not available yet",
                        }
                    ],
                }
            pcr_ready = bool(readiness.get("ready"))
            blocker_codes = {
                str(item.get("code") or "")
                for item in (readiness.get("blockers") or [])
                if str(item.get("code") or "")
            }
            reviewable_codes = {
                "document_coverage_requires_review",
                "response_assignment_requires_review",
                "evaluation_requires_review",
            }
            needs_teacher_review = bool(blocker_codes) and blocker_codes.issubset(
                reviewable_codes
            )

            # Determine blocking status
            has_unresolved_blocking = False
            unresolved_count = 0
            for resp in sub_responses:
                flags = resp.get("flags", [])
                for f in flags:
                    if (
                        f.get("severity") == "blocking"
                        and not f.get("resolution", {}).get(
                            "resolved", False
                        )
                    ):
                        has_unresolved_blocking = True
                        unresolved_count += 1

            # Determine evaluation status
            pending_responses = sum(
                1
                for r in sub_responses
                if r.get("eval_status", "pending") == "pending"
            )
            pcr_all_evaluated = pcr_ready

            # Fully evaluated = PCR done AND DCR done (if applicable)
            all_evaluated = pcr_all_evaluated and dcr_complete

            active_result_retained = bool(
                readiness.get("active_result_retained")
            )
            job_needs_attention = (
                processing_status
                in {
                    "failed",
                    "retryable_error",
                    "enqueue_failed",
                }
                and not active_result_retained
            )

            # Bucket logic
            if (
                needs_teacher_review
                and not has_unresolved_blocking
                and not job_needs_attention
                and pub_status != "published"
            ):
                review_items.append(
                    QueueItem(
                        submission_id=sub_id,
                        student_id=student_id,
                        response_count=response_count,
                        page_count=page_count,
                        source=submission_source,
                        status_summary=readiness_message(readiness),
                        has_dcr_results=student_has_dcr,
                        processing_status=processing_status,
                        processing_error=processing_error,
                    )
                )
            elif has_unresolved_blocking or job_needs_attention or (
                not pcr_ready
                and processing_status not in {"queued", "processing", None}
            ):
                if has_unresolved_blocking:
                    blocked_summary = f"{unresolved_count} unresolved blocking flag(s)"
                elif processing_status == "blocked_for_review":
                    blocked_summary = "AI checking needs teacher review"
                elif processing_status == "failed":
                    blocked_summary = "AI checking failed"
                elif not pcr_ready:
                    blocked_summary = readiness_message(readiness)
                else:
                    blocked_summary = "AI checking needs a retry"
                blocked_items.append(
                    QueueItem(
                        submission_id=sub_id,
                        student_id=student_id,
                        response_count=response_count,
                        page_count=page_count,
                        source=submission_source,
                        status_summary=blocked_summary,
                        has_dcr_results=student_has_dcr,
                        processing_status=processing_status,
                        processing_error=processing_error,
                    )
                )
            elif not all_evaluated:
                # Build a descriptive status summary
                parts: List[str] = []
                if pending_responses > 0:
                    parts.append(
                        f"{pending_responses}/{response_count} PCR pending"
                    )
                if response_count == 0:
                    if processing_status == "queued":
                        parts.append("AI checking queued")
                    elif processing_status == "processing":
                        parts.append("AI checking in progress")
                    elif processing_status == "completed":
                        parts.append("AI completed without detected responses")
                if exam_has_dcr and not student_has_dcr:
                    parts.append("DCR results missing")
                status_msg = "; ".join(parts) if parts else "Pending"

                pending_items.append(
                    QueueItem(
                        submission_id=sub_id,
                        student_id=student_id,
                        response_count=response_count,
                        page_count=page_count,
                        source=submission_source,
                        status_summary=status_msg,
                        has_dcr_results=student_has_dcr,
                        processing_status=processing_status,
                        processing_error=processing_error,
                    )
                )
            elif all_evaluated and pub_status != "published":
                ready_summary = (
                    "Verified result retained; latest reprocess failed"
                    if active_result_retained
                    else "Fully evaluated, ready to publish"
                )
                ready_items.append(
                    QueueItem(
                        submission_id=sub_id,
                        student_id=student_id,
                        response_count=response_count,
                        page_count=page_count,
                        source=submission_source,
                        status_summary=ready_summary,
                        has_dcr_results=student_has_dcr,
                        processing_status=processing_status,
                        processing_error=processing_error,
                    )
                )
            # else: already published — not in any queue bucket

        return ExamQueueResponse(
            pending=pending_items,
            blocked=blocked_items,
            needs_review=review_items,
            ready_to_publish=ready_items,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get exam queue for %s: %s",
            exam_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exam queue",
        )
