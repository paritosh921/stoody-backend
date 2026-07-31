"""
EvalPen Student BFF (Backend-for-Frontend) API — Read-only published
exam results for students.

Provides two endpoints:
  1. GET /exams — List published exams the student has been evaluated on
  2. GET /exams/{exam_id}/scores — Per-question score breakdown for a
     published exam

Architecture:
    Read-only aggregation over evalpen_submissions,
    evalpen_evaluations, exampen_dcr_results, and evalpen_questions.
    No writes — this module is a pure reader.

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  NONE
    - Reads from: evalpen_submissions, evalpen_evaluations,
                  exampen_dcr_results, evalpen_questions
    - Never writes to: any collection

Hard constraints:
    - C1: MongoDB only
    - C5: Ownership boundaries — BFF endpoints are pure readers
    - Read-only: no $set, $push, insert, or update operations
    - Student can ONLY see their own results (student_id from JWT)
    - Only published results are visible (publication_status = "published")
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_student(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency: restrict student BFF endpoints to student users only.

    Admins and tutors should use the teacher BFF instead.
    """
    allowed = {"student", "b2c_user"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student access required for this endpoint",
        )
    return current_user


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class StudentExamItem(BaseModel):
    """Per-exam summary row for the student exam list."""

    exam_id: str
    title: str
    exam_type: Optional[str] = None
    total_score: float = 0.0
    max_score: float = 0.0
    published_at: Optional[str] = None
    recheck_available: bool = True
    recheck_count: int = 0
    conversation_count: int = 0


class StudentExamListResponse(BaseModel):
    """Response for GET /exams."""

    items: List[StudentExamItem] = Field(default_factory=list)


class StudentMarkBreakdownItem(BaseModel):
    """A student-safe, mark-by-mark explanation for one evaluated response."""

    description: str
    marks_awarded: float = 0.0
    max_marks: float = 0.0
    feedback: Optional[str] = None


class StudentRecheckStatusItem(BaseModel):
    request_id: str
    exam_id: str
    student_id: str
    question_id: str
    submission_id: str
    status: str
    reason: str
    teacher_response: Optional[str] = None
    original_score: float = 0.0
    original_max_score: float = 0.0
    updated_score: Optional[float] = None
    updated_max_score: Optional[float] = None
    created_at: str
    updated_at: Optional[str] = None
    resolved_at: Optional[str] = None


class QuestionScoreItem(BaseModel):
    """Per-question score entry within an exam score breakdown."""

    question_id: str
    question_number: Optional[int] = None
    score: float = 0.0
    max_score: float = 0.0
    feedback: Optional[str] = None
    eval_type: str = "pcr"  # "pcr" or "dcr"
    answer_state: Optional[str] = None
    mark_breakdown: List[StudentMarkBreakdownItem] = Field(default_factory=list)
    reference_answer: Optional[str] = None
    teacher_feedback: Optional[str] = None
    recheck_status: Optional[StudentRecheckStatusItem] = None


class StudentExamScoresResponse(BaseModel):
    """Response for GET /exams/{exam_id}/scores."""

    exam_id: str
    student_id: str
    total_score: float = 0.0
    max_score: float = 0.0
    questions: List[QuestionScoreItem] = Field(default_factory=list)
    recheck_available: bool = True


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
# Helper: extract student_id from JWT claims
# ---------------------------------------------------------------------------

def _clean_identity(value: Any) -> str:
    return str(value or "").strip()


def _get_student_id(current_user: Dict[str, Any]) -> str:
    """Return the Student DB identifier, falling back for legacy sessions.

    Conducted-exam rosters are built from ``students.student_id``.  Login
    sessions historically only carried the Mongo account ``user_id``, so the
    fallback keeps older score records readable while new sessions use the
    stable Student DB identifier.
    """
    student_id = _clean_identity(current_user.get("student_id")) or _clean_identity(
        current_user.get("user_id")
    )
    if not student_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity could not be determined from token",
        ).sort("published_at", -1)
    return student_id


async def _get_student_identity_ids(
    tenant_db: Any,
    current_user: Dict[str, Any],
) -> List[str]:
    """Return the canonical Student DB id followed by safe legacy aliases.

    A currently logged-in student can hold a token issued before the
    ``student_id`` claim existed.  Resolve that token back to its own student
    profile so a roster selected in Student DB immediately works, without
    requiring every student to log out first.  The account ``user_id`` stays
    as a read-only alias so already-published legacy records remain visible.
    """
    account_id = _clean_identity(current_user.get("user_id"))
    roster_student_id = _clean_identity(current_user.get("student_id"))

    if not roster_student_id and tenant_db is not None:
        clauses: List[Dict[str, Any]] = []
        if ObjectId.is_valid(account_id):
            clauses.append({"_id": ObjectId(account_id)})
        if account_id:
            clauses.append({"student_id": account_id})
        username = _clean_identity(current_user.get("username"))
        if username:
            clauses.extend(({"username": username}, {"username_lower": username.lower()}))

        if clauses:
            profile_query: Dict[str, Any] = clauses[0] if len(clauses) == 1 else {"$or": clauses}
            profile = await tenant_db["students"].find_one(
                profile_query,
                projection={"student_id": 1},
            )
            roster_student_id = _clean_identity((profile or {}).get("student_id"))

    canonical_id = roster_student_id or account_id
    if not canonical_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity could not be determined from token",
        )

    return list(dict.fromkeys(identity for identity in (canonical_id, account_id) if identity))


# ---------------------------------------------------------------------------
# Helper: safe datetime to ISO string
# ---------------------------------------------------------------------------

def _dt_to_iso(val: Any) -> Optional[str]:
    """Convert a datetime or string to ISO format string."""
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _safe_marks(value: Any) -> float:
    """Return a non-negative numeric mark without making score pages fragile."""
    try:
        numeric = float(value or 0.0)
        return max(0.0, numeric) if math.isfinite(numeric) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_score(value: Any) -> float:
    """Return a finite score while preserving objective negative marking."""

    try:
        numeric = float(value or 0.0)
        return numeric if math.isfinite(numeric) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_question_number(value: Any) -> Optional[int]:
    """Return a positive question number when legacy data provides one."""
    try:
        number = int(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def _student_safe_text(value: Any, *, limit: int = 1600) -> Optional[str]:
    """Return short presentation text without exposing internal evaluation data."""
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] or None


def _student_mark_breakdown(evaluation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Project the rubric result into fields students are allowed to see.

    PCR stores both structured criterion rows and a legacy step representation.
    Prefer criteria because they directly reflect the teacher-approved marking
    plan; fall back to steps for older evaluations.  Evidence, model metadata,
    internal flags, and raw provider output must never leave the student BFF.
    """
    raw_rows = evaluation.get("criterion_marks") or evaluation.get("step_marks") or []
    if not isinstance(raw_rows, list):
        return []

    rows: List[Dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, Mapping):
            continue
        description = _student_safe_text(
            raw_row.get("description") or raw_row.get("step") or f"Mark {index}",
            limit=500,
        )
        max_marks = _safe_marks(raw_row.get("max_marks", raw_row.get("marks_possible")))
        marks_awarded = _safe_marks(
            raw_row.get("marks_awarded", raw_row.get("score", raw_row.get("awarded")))
        )
        if max_marks > 0:
            marks_awarded = min(marks_awarded, max_marks)
        if not description:
            continue
        rows.append(
            {
                "description": description,
                "marks_awarded": marks_awarded,
                "max_marks": max_marks,
                "feedback": _student_safe_text(
                    raw_row.get("rationale")
                    or raw_row.get("justification")
                    or raw_row.get("feedback")
                ),
            }
        )
    return rows


async def _get_pcr_question_catalog(
    tenant_db: Any,
    exam_id: str,
) -> List[Dict[str, Any]]:
    """Load the immutable paper question list used as the PCR denominator."""
    if not exam_id:
        return []
    cursor = tenant_db["evalpen_questions"].find(
        {"exam_id": exam_id},
        projection={"question_id": 1, "question_number": 1, "max_marks": 1},
    ).sort([("question_number", 1), ("question_id", 1)])
    docs = await cursor.to_list(length=1000)
    catalog: List[Dict[str, Any]] = []
    for doc in docs:
        question_id = str(doc.get("question_id") or "").strip()
        if not question_id:
            continue
        number = doc.get("question_number")
        catalog.append(
            {
                "question_id": question_id,
                "question_number": _safe_question_number(number),
                "max_marks": _safe_marks(doc.get("max_marks")),
            }
        )
    return catalog


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/exams",
    response_model=StudentExamListResponse,
    summary="List published exams with scores for the authenticated student",
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Student access required"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def list_student_exams(
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
) -> StudentExamListResponse:
    """List all published exams where the authenticated student has results.

    Queries ``evalpen_submissions`` for the student's own submissions
    with ``publication_status = "published"``.  For each published
    submission, aggregates PCR scores from ``evalpen_evaluations`` and
    DCR scores from ``exampen_dcr_results``.

    Exam title and type are resolved from ``evalpen_questions``
    (first question's exam metadata) with a human-readable fallback.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    student_ids = await _get_student_identity_ids(tenant_db, current_user)
    student_id = student_ids[0]

    try:
        # ----- Fetch published submissions for this student -----
        submissions_cursor = tenant_db["evalpen_submissions"].find(
            {
                "student_id": {"$in": student_ids},
                "publication_status": "published",
            },
            projection={
                "submission_id": 1,
                "exam_id": 1,
                "published_at": 1,
                "publication_snapshot": 1,
            },
        )
        submissions = await submissions_cursor.to_list(length=1000)

        if not submissions:
            return StudentExamListResponse(items=[])

        # ----- Collect exam_ids and submission_ids -----
        exam_ids: List[str] = []
        sub_ids: List[str] = []
        # Map exam_id -> submission metadata
        exam_sub_map: Dict[str, Dict[str, Any]] = {}
        for sub in submissions:
            eid = sub.get("exam_id", "")
            sid = sub.get("submission_id", "")
            if eid:
                exam_ids.append(eid)
                sub_ids.append(sid)
                exam_sub_map.setdefault(eid, {
                    "submission_id": sid,
                    "published_at": sub.get("published_at"),
                    "publication_snapshot": sub.get("publication_snapshot"),
                })

        # ----- PCR: aggregate evaluations per exam -----
        from api.v1._exampen_imports import load_exampen

        pcr_scores: Dict[str, Dict[str, float]] = {}
        try:
            _pcr_storage = load_exampen("pcr.storage")
            DetectedResponseRepository = _pcr_storage.DetectedResponseRepository
            EvaluationRepository = _pcr_storage.EvaluationRepository

            resp_repo = DetectedResponseRepository(tenant_db)
            eval_repo = EvaluationRepository(tenant_db)

            for eid, sub_info in exam_sub_map.items():
                sid = sub_info["submission_id"]
                question_catalog = await _get_pcr_question_catalog(tenant_db, eid)
                if question_catalog:
                    from services.exampen_submission_readiness import (
                        validate_publication_snapshot,
                    )

                    snapshot = sub_info.get("publication_snapshot")
                    if not validate_publication_snapshot(
                        snapshot,
                        submission_id=sid,
                        exam_id=eid,
                    ):
                        logger.error(
                            "Withholding published PCR result with invalid snapshot: %s",
                            sid,
                        )
                        pcr_scores[eid] = {"invalid": 1.0}
                        continue
                    pcr_scores[eid] = {
                        "total_score": _safe_score(snapshot.get("total_score")),
                        "max_score": _safe_marks(snapshot.get("total_max_score")),
                    }
        except ImportError:
            logger.debug("PCR storage not available for student BFF aggregation")

        # ----- DCR: aggregate scores per exam -----
        dcr_scores: Dict[str, Dict[str, float]] = {}
        try:
            dcr_cursor = tenant_db["exampen_dcr_results"].find(
                {
                    "exam_id": {"$in": exam_ids},
                    "student_id": {"$in": student_ids},
                },
                projection={"exam_id": 1, "score": 1, "max_score": 1},
            )
            dcr_docs = await dcr_cursor.to_list(length=5000)
            for doc in dcr_docs:
                eid = doc.get("exam_id", "")
                if eid not in dcr_scores:
                    dcr_scores[eid] = {"total_score": 0.0, "max_score": 0.0}
                dcr_scores[eid]["total_score"] += doc.get("score", 0.0)
                dcr_scores[eid]["max_score"] += doc.get("max_score", 0.0)
        except Exception as exc:
            logger.debug("DCR scores not available for student BFF: %s", exc)

        # ----- Resolve exam titles from canonical sessions and papers -----
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
        exam_documents = await tenant_db["exampen_exams"].find(
            {"exam_id": {"$in": exam_ids}},
            {
                "_id": 0,
                "exam_id": 1,
                "title": 1,
                "exam_type": 1,
                "prepared_document_id": 1,
            },
        ).to_list(length=5000)
        exam_document_map = {
            str(item.get("exam_id") or ""): item for item in exam_documents
        }
        prepared_ids = [
            str(item.get("prepared_document_id") or "")
            for item in exam_documents
            if str(item.get("prepared_document_id") or "")
        ]
        prepared_documents = (
            await tenant_db["documents"].find(
                {"document_id": {"$in": prepared_ids}},
                {"_id": 0, "document_id": 1, "title": 1, "exam_mode": 1},
            ).to_list(length=5000)
            if prepared_ids
            else []
        )
        prepared_document_map = {
            str(item.get("document_id") or ""): item for item in prepared_documents
        }
        recheck_documents = await tenant_db["evalpen_recheck_requests"].find(
            {
                "exam_id": {"$in": exam_ids},
                "student_id": {"$in": student_ids},
            },
            {"_id": 0, "exam_id": 1},
        ).to_list(length=5000)
        recheck_counts: Dict[str, int] = {}
        for item in recheck_documents:
            recheck_exam_id = str(item.get("exam_id") or "")
            recheck_counts[recheck_exam_id] = recheck_counts.get(recheck_exam_id, 0) + 1

        # ----- Build response items -----
        items: List[StudentExamItem] = []
        for eid in dict.fromkeys(exam_ids):
            pcr = pcr_scores.get(eid, {"total_score": 0.0, "max_score": 0.0})
            if pcr.get("invalid"):
                continue
            dcr = dcr_scores.get(eid, {"total_score": 0.0, "max_score": 0.0})
            combined_total = pcr["total_score"] + dcr["total_score"]
            combined_max = pcr["max_score"] + dcr["max_score"]

            title_info = title_map.get(eid, {})
            exam_document = exam_document_map.get(eid, {})
            prepared_document = prepared_document_map.get(
                str(exam_document.get("prepared_document_id") or ""),
                {},
            )
            exam_type = (
                exam_document.get("exam_type")
                or prepared_document.get("exam_mode")
                or title_info.get("exam_type")
            )
            title = (
                exam_document.get("title")
                or prepared_document.get("title")
                or title_info.get("title")
                or (f"{str(exam_type).upper()} Exam" if exam_type else "Exam Result")
            )

            sub_info = exam_sub_map.get(eid, {})
            published_at = _dt_to_iso(sub_info.get("published_at"))

            items.append(
                StudentExamItem(
                    exam_id=eid,
                    title=title,
                    exam_type=exam_type,
                    total_score=combined_total,
                    max_score=combined_max,
                    published_at=published_at,
                    recheck_available=True,
                    recheck_count=recheck_counts.get(eid, 0),
                    conversation_count=0,
                )
            )

        items.sort(key=lambda x: x.published_at or "", reverse=True)

        return StudentExamListResponse(items=items)

    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("Exam-conductor module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exam evaluation engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to list exams for student %s: %s",
            student_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exam list",
        )


@router.get(
    "/exams/{exam_id}/scores",
    response_model=StudentExamScoresResponse,
    summary="Get per-question score breakdown for a published exam",
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Student access required"},
        404: {"description": "No published submission found for this exam"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_student_exam_scores(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
) -> StudentExamScoresResponse:
    """Get a per-question score breakdown for the authenticated student
    on a specific exam.

    Published-only guard: returns 404 if the student has no submission
    with ``publication_status = "published"`` for the given exam.

    Combines PCR evaluations (via ``evalpen_evaluations``) and DCR
    results (via ``exampen_dcr_results``) into a unified question list.
    Each question entry indicates its ``eval_type`` ("pcr" or "dcr").
    """
    tenant_db = await _get_tenant_db(db, current_user)
    student_ids = await _get_student_identity_ids(tenant_db, current_user)
    student_id = student_ids[0]

    try:
        # ----- Published-only guard -----
        submission = await tenant_db["evalpen_submissions"].find_one(
            {
                "exam_id": exam_id,
                "student_id": {"$in": student_ids},
                "publication_status": "published",
            },
            projection={"submission_id": 1, "publication_snapshot": 1},
        )
        if submission is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No published results found for this exam",
            )

        submission_id = submission.get("submission_id", "")
        recheck_documents = await tenant_db["evalpen_recheck_requests"].find(
            {"submission_id": submission_id},
            {"_id": 0, "active_key": 0, "resolution_lock": 0, "evaluation_id": 0},
        ).sort("created_at", -1).to_list(length=5000)
        recheck_by_question: Dict[str, StudentRecheckStatusItem] = {}
        for item in recheck_documents:
            recheck_question_id = str(item.get("question_id") or "")
            if not recheck_question_id or recheck_question_id in recheck_by_question:
                continue
            recheck_by_question[recheck_question_id] = StudentRecheckStatusItem(
                request_id=str(item.get("request_id") or ""),
                exam_id=str(item.get("exam_id") or ""),
                student_id=str(item.get("student_id") or ""),
                question_id=recheck_question_id,
                submission_id=str(item.get("submission_id") or ""),
                status=str(item.get("status") or "open"),
                reason=str(item.get("reason") or ""),
                teacher_response=item.get("teacher_response"),
                original_score=_safe_score(item.get("original_score")),
                original_max_score=_safe_marks(item.get("original_max_score")),
                updated_score=item.get("updated_score"),
                updated_max_score=item.get("updated_max_score"),
                created_at=_dt_to_iso(item.get("created_at")) or "",
                updated_at=_dt_to_iso(item.get("updated_at")),
                resolved_at=_dt_to_iso(item.get("resolved_at")),
            )

        question_catalog_for_integrity = await _get_pcr_question_catalog(
            tenant_db, exam_id
        )
        publication_snapshot = submission.get("publication_snapshot")
        if question_catalog_for_integrity:
            from services.exampen_submission_readiness import (
                validate_publication_snapshot,
            )

            if not validate_publication_snapshot(
                publication_snapshot,
                submission_id=submission_id,
                exam_id=exam_id,
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "Published PCR results failed the integrity check and require "
                        "teacher review before they can be shown"
                    ),
                )

        questions: List[QuestionScoreItem] = []
        total_score = 0.0
        total_max = 0.0

        # ----- PCR evaluations -----
        from api.v1._exampen_imports import load_exampen

        try:
            _pcr_storage = load_exampen("pcr.storage")
            DetectedResponseRepository = _pcr_storage.DetectedResponseRepository
            EvaluationRepository = _pcr_storage.EvaluationRepository

            resp_repo = DetectedResponseRepository(tenant_db)
            eval_repo = EvaluationRepository(tenant_db)

            responses = await resp_repo.get_responses_by_submission(
                submission_id
            )
            question_catalog = await _get_pcr_question_catalog(tenant_db, exam_id)
            catalog_question_ids = {
                question["question_id"] for question in question_catalog
            }
            evaluated_by_question: Dict[str, Dict[str, Any]] = {}
            legacy_evaluations: List[Dict[str, Any]] = []

            for resp in responses:
                response_id = resp.get("response_id", "")
                question_id = str(resp.get("question_id") or "")
                ev = await eval_repo.get_evaluation_by_response(response_id)
                if not ev:
                    continue
                entry = {
                    "score": _safe_score(ev.get("total_score")),
                    "max_score": _safe_marks(ev.get("max_score")),
                    "feedback": _student_safe_text(ev.get("overall_feedback")),
                    "answer_state": resp.get("answer_state")
                    or ("not_attempted" if resp.get("is_missing_response") else "detected"),
                    "mark_breakdown": _student_mark_breakdown(ev),
                    "reference_answer": _student_safe_text(
                        ev.get("reference_solution"), limit=5000
                    ),
                    "teacher_feedback": _student_safe_text(
                        ev.get("teacher_feedback") or ev.get("teacher_note")
                    ),
                    "question_number": _safe_question_number(resp.get("question_number")),
                }
                if question_id and question_id in catalog_question_ids:
                    # One score per paper question.  Duplicate OCR segments
                    # are reviewable evidence, never a second mark allocation.
                    evaluated_by_question.setdefault(question_id, entry)
                else:
                    legacy_evaluations.append(
                        {"question_id": question_id, **entry}
                    )

            if question_catalog:
                # Published PCR scores are read from the immutable release
                # snapshot.  Mutable evaluation documents remain useful for
                # staff audit, but cannot silently change a student's result
                # after publication.
                evaluated_by_question = {
                    str(row.get("question_id") or ""): {
                        "score": _safe_score(row.get("score")),
                        "max_score": _safe_marks(row.get("max_score")),
                        "feedback": _student_safe_text(row.get("overall_feedback")),
                        "answer_state": row.get("answer_state") or "detected",
                        "mark_breakdown": _student_mark_breakdown(row),
                        "reference_answer": _student_safe_text(
                            row.get("reference_solution"), limit=5000
                        ),
                        "teacher_feedback": _student_safe_text(
                            row.get("teacher_feedback")
                        ),
                        "question_number": _safe_question_number(
                            row.get("question_number")
                        ),
                    }
                    for row in publication_snapshot.get("score_rows", [])
                    if isinstance(row, dict) and row.get("question_id")
                }
                missing_question_ids: List[str] = []
                for question in question_catalog:
                    question_id = question["question_id"]
                    result = evaluated_by_question.get(question_id)
                    if result is None:
                        missing_question_ids.append(question_id)
                        continue

                    total_score += result["score"]
                    # The immutable paper, not a response document, fixes the
                    # question maximum.  This also repairs already-published
                    # partial submissions created before answer slots existed.
                    total_max += question["max_marks"]
                    questions.append(
                        QuestionScoreItem(
                            question_id=question_id,
                            question_number=question["question_number"],
                            score=result["score"],
                            max_score=question["max_marks"],
                            feedback=result["feedback"],
                            eval_type="pcr",
                            answer_state=result["answer_state"],
                            mark_breakdown=result["mark_breakdown"],
                            reference_answer=result["reference_answer"],
                            teacher_feedback=result["teacher_feedback"],
                            recheck_status=recheck_by_question.get(question_id),
                        )
                    )
                if missing_question_ids:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=(
                            "Published PCR results are incomplete and require teacher review"
                        ),
                    )
            else:
                # Fallback for historical records where immutable question
                # metadata is unavailable.  Preserve their previous result.
                for result in legacy_evaluations:
                    total_score += result["score"]
                    total_max += result["max_score"]
                    questions.append(
                        QuestionScoreItem(
                            question_id=result["question_id"],
                            question_number=result.get("question_number"),
                            score=result["score"],
                            max_score=result["max_score"],
                            feedback=result["feedback"],
                            eval_type="pcr",
                            answer_state=result["answer_state"],
                            mark_breakdown=result["mark_breakdown"],
                            reference_answer=result["reference_answer"],
                            teacher_feedback=result["teacher_feedback"],
                            recheck_status=recheck_by_question.get(
                                result["question_id"]
                            ),
                        )
                    )
        except ImportError:
            logger.debug(
                "PCR storage not available for student score breakdown"
            )

        # ----- DCR results -----
        try:
            dcr_cursor = tenant_db["exampen_dcr_results"].find(
                {
                    "exam_id": exam_id,
                    "student_id": {"$in": student_ids},
                },
                projection={
                    "question_id": 1,
                    "question_number": 1,
                    "score": 1,
                    "max_score": 1,
                },
            )
            dcr_docs = await dcr_cursor.to_list(length=5000)
            for doc in dcr_docs:
                q_score = doc.get("score", 0.0)
                q_max = doc.get("max_score", 0.0)
                total_score += q_score
                total_max += q_max
                questions.append(
                    QuestionScoreItem(
                        question_id=doc.get("question_id", ""),
                        question_number=_safe_question_number(doc.get("question_number")),
                        score=q_score,
                        max_score=q_max,
                        feedback=None,
                        eval_type="dcr",
                        recheck_status=recheck_by_question.get(
                            str(doc.get("question_id") or "")
                        ),
                    )
                )
        except Exception as exc:
            logger.debug(
                "DCR results not available for student score breakdown: %s",
                exc,
            )

        return StudentExamScoresResponse(
            exam_id=exam_id,
            student_id=student_id,
            total_score=total_score,
            max_score=total_max,
            questions=questions,
            recheck_available=True,
        )

    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("Exam-conductor module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exam evaluation engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to get exam scores for student %s exam %s: %s",
            student_id,
            exam_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exam scores",
        )
