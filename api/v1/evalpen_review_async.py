"""
EvalPen Review & Publication API — Teacher-facing review of evaluation
results, score overrides with audit trail, and grade publication.

Provides endpoints for teachers (admin/tutor) to:
  1. View evaluation summaries per submission (all responses + scores + flags)
  2. View aggregated exam results across all students
  3. Override scores with mandatory audit trail
  4. Publish/finalize submission results

Architecture:
    PCR_EVAL_ENGINE_SPEC {7}, DUAL_MODE_ARCHITECTURE.md {7}

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  score overrides via EvaluationRepository.override_score(),
               publication status on evalpen_submissions
    - Reads from: evalpen_submissions, evalpen_detected_responses,
                  evalpen_evaluations, exampen_dcr_results
    - Never writes to: canonical artifact content, detected_text,
                       practice persistence

Hard constraints:
    - C1: MongoDB only
    - C5: Ownership boundaries — review endpoints are readers of
          PCR/DCR result collections
    - TAMPER_PROOF_SPEC Layer 3: Every score override must have
      actor_id, reason (min 5 chars), before/after, timestamp.
      Append-only audit trail.

API authority:
    new-docs/api/review.openapi.yaml
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from utils.tutor_scoping import get_tutor_scoped_students
from utils.s3_storage import PrivateObjectStorageError, create_private_download_url

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency: require admin or tutor role for review endpoints."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for review operations",
        )
    return current_user


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ScoreOverrideRequest(BaseModel):
    """Request body for overriding an evaluation score.

    TAMPER_PROOF_SPEC Layer 3 / Section 6, Rule 4:
    Score overrides require actor_id and reason.
    """

    new_score: float = Field(
        ...,
        ge=0,
        description="New total score to assign",
    )
    reason: str = Field(
        ...,
        min_length=5,
        description="Human-readable justification (min 5 chars, per TAMPER_PROOF_SPEC)",
    )

    @field_validator("reason")
    @classmethod
    def reason_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Reason must be at least 5 characters")
        return v.strip()


class CriterionMarkOverrideItem(BaseModel):
    """One teacher-entered score for an already-locked rubric criterion."""

    criterion_id: str = Field(..., min_length=1, max_length=64)
    marks_awarded: float = Field(..., ge=0)


class CriterionMarksOverrideRequest(BaseModel):
    """Teacher correction of every criterion; total is recomputed server-side."""

    criteria: List[CriterionMarkOverrideItem] = Field(..., min_length=1)
    reason: str = Field(..., min_length=5)

    @field_validator("reason")
    @classmethod
    def reason_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Reason must be at least 5 characters")
        return v.strip()


class PublishRequest(BaseModel):
    """Request body for publishing submission results."""

    note: Optional[str] = Field(
        default=None,
        description="Optional note attached to the publication action",
    )


class AuditEntryAPI(BaseModel):
    """Audit trail entry in API responses."""

    actor_id: str
    timestamp: Optional[str] = None
    action: str
    before: Optional[Any] = None
    after: Optional[Any] = None
    reason: Optional[str] = None


class ResponseSummaryAPI(BaseModel):
    """Summary of a detected response within a submission review."""

    response_id: str
    evaluation_id: Optional[str] = None
    question_id: Optional[str] = None
    question_number: Optional[int] = None
    content_type: str = "TEXT_ONLY"
    eval_status: str = "pending"
    # The answer text and AI-generated evaluation artefacts are deliberately
    # returned only by this staff-only review endpoint.  They let the teacher
    # audit what was read, the worked solution used for marking, and the
    # per-step rationale before publishing a score.
    detected_text: Optional[str] = None
    total_score: Optional[float] = None
    max_score: Optional[float] = None
    overall_feedback: Optional[str] = None
    reference_solution: Optional[str] = None
    step_marks: Optional[List[Dict[str, Any]]] = None
    criterion_marks: Optional[List[Dict[str, Any]]] = None
    marking_policy: Optional[Dict[str, Any]] = None
    manual_review_required: bool = False
    flags: Optional[List[Dict[str, Any]]] = None
    has_blocking_flags: bool = False
    # Every PCR paper is represented as a complete question matrix.  These
    # fields distinguish a true blank answer from an OCR/AI failure.
    is_missing_response: bool = False
    answer_state: Optional[str] = None


class SubmissionSummaryReviewAPI(BaseModel):
    """Full evaluation summary for a single submission."""

    submission_id: str
    exam_id: str
    student_id: str
    source: str = "camera"
    segmentation_status: str = "pending"
    processing_status: Optional[str] = None
    # Staff-only diagnostic.  Students continue to receive the safe status
    # surface from their BFF and are never shown worker/storage internals.
    processing_error: Optional[str] = None
    # ``available`` means OCR/mapping and every detected answer's evaluation
    # completed (a genuinely blank paper can score zero); ``processing`` and
    # ``unavailable`` must never be rendered as a 0-mark attempt.
    score_state: str = "processing"
    publication_status: Optional[str] = None
    responses: List[ResponseSummaryAPI] = Field(default_factory=list)
    total_score: float = 0.0
    total_max_score: float = 0.0
    evaluated_count: int = 0
    blocked_count: int = 0
    pending_count: int = 0


class SubmissionPageThumbnailAPI(BaseModel):
    """A staff-authorized, short-lived preview of one private answer page."""

    page_id: str
    page_index: int
    image_url: Optional[str] = None
    width: int = 0
    height: int = 0
    regions: List[Dict[str, Any]] = Field(default_factory=list)


class SubmissionPagesAPI(BaseModel):
    submission_id: str
    total_pages: int = 0
    pages: List[SubmissionPageThumbnailAPI] = Field(default_factory=list)


class ExamResultStudentAPI(BaseModel):
    """Per-student result row for exam-level results."""

    student_id: str
    submission_id: Optional[str] = None
    pcr_total_score: float = 0.0
    pcr_max_score: float = 0.0
    dcr_total_score: float = 0.0
    dcr_max_score: float = 0.0
    combined_total: float = 0.0
    combined_max: float = 0.0
    publication_status: Optional[str] = None
    blocked_responses: int = 0


class ExamResultsAPI(BaseModel):
    """Aggregated results for an entire exam."""

    exam_id: str
    students: List[ExamResultStudentAPI] = Field(default_factory=list)
    total_students: int = 0


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
    """Return a non-negative numeric mark without letting malformed data break totals."""
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


async def _get_pcr_question_catalog(
    tenant_db: Any,
    exam_id: str,
) -> List[Dict[str, Any]]:
    """Return the immutable session questions in paper order.

    The catalog is the denominator for every PCR score.  Evaluation rows are
    evidence about an answer; they must never decide how many marks a paper is
    out of.
    """
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
        question_number = doc.get("question_number")
        catalog.append(
            {
                "question_id": question_id,
                "question_number": (
                    int(question_number)
                    if isinstance(question_number, (int, float))
                    else None
                ),
                "max_marks": _safe_marks(doc.get("max_marks")),
            }
        )
    return catalog


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


def _check_student_in_scope(
    student_id: str,
    scoped_ids: Optional[List[str]],
) -> None:
    """Raise 403 if *student_id* is not in the tutor's visible scope.

    When *scoped_ids* is ``None`` the caller is an admin and the check
    is a no-op.
    """
    if scoped_ids is None:
        return
    if student_id not in scoped_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this student's data",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/submissions/{submission_id}/summary",
    response_model=SubmissionSummaryReviewAPI,
    summary="Get evaluation summary for a submission",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Submission not found"},
        503: {"description": "Tenant database or exam-conductor unavailable"},
    },
)
async def get_submission_summary(
    submission_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> SubmissionSummaryReviewAPI:
    """Get a complete evaluation summary for a submission.

    Returns all detected responses with their eval_status, scores, flags,
    and aggregate totals. This is the primary teacher review view.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        from api.v1._exampen_imports import load_exampen

        _pcr_storage = load_exampen("pcr.storage")
        SubmissionRepository = _pcr_storage.SubmissionRepository
        DetectedResponseRepository = _pcr_storage.DetectedResponseRepository
        EvaluationRepository = _pcr_storage.EvaluationRepository

        # Fetch submission
        sub_repo = SubmissionRepository(tenant_db)
        submission = await sub_repo.get_submission(submission_id)
        if submission is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Submission {submission_id} not found",
            )

        # Tutor scoping: verify this submission's student is visible
        _sub_dict_for_scope = (
            submission
            if isinstance(submission, dict)
            else submission.__dict__
            if hasattr(submission, "__dict__")
            else {}
        )
        _check_student_in_scope(
            _sub_dict_for_scope.get("student_id", ""), scoped_ids
        )

        sub_dict = _sub_dict_for_scope
        segmentation_status = str(sub_dict.get("segmentation_status") or "pending")
        processing_job = await tenant_db["exampen_processing_jobs"].find_one(
            {"submission_id": submission_id}
        )
        processing_status = (
            str(processing_job.get("status"))
            if isinstance(processing_job, dict) and processing_job.get("status")
            else None
        )
        processing_error = (
            str(processing_job.get("last_error"))[:500]
            if isinstance(processing_job, dict) and processing_job.get("last_error")
            else None
        )
        processing_failed = segmentation_status == "failed" or processing_status in {
            "failed",
            "retryable_error",
            "enqueue_failed",
        }
        question_catalog = await _get_pcr_question_catalog(
            tenant_db,
            str(sub_dict.get("exam_id") or ""),
        )
        catalog_by_id = {
            item["question_id"]: item for item in question_catalog
        }

        # Fetch all detected responses
        resp_repo = DetectedResponseRepository(tenant_db)
        response_docs = await resp_repo.get_responses_by_submission(
            submission_id
        )

        # Fetch evaluations for each response
        eval_repo = EvaluationRepository(tenant_db)

        response_summaries: List[ResponseSummaryAPI] = []
        total_score = 0.0
        evaluated_max = 0.0
        evaluated_count = 0
        blocked_count = 0
        pending_count = 0
        scored_question_ids: set[str] = set()
        completed_question_keys: set[str] = set()
        observed_question_ids: set[str] = set()

        for resp_doc in response_docs:
            response_id = resp_doc.get("response_id", "")
            eval_status = str(resp_doc.get("eval_status") or "pending").lower()
            question_id = str(resp_doc.get("question_id") or "")
            catalog_question = catalog_by_id.get(question_id)
            if question_id:
                observed_question_ids.add(question_id)

            # Check for blocking flags
            flags = resp_doc.get("flags", [])
            has_blocking = any(
                f.get("severity") == "blocking"
                and not f.get("resolution", {}).get("resolved", False)
                for f in flags
            )

            # Try to get evaluation for this response
            evaluation = await eval_repo.get_evaluation_by_response(
                response_id
            )

            resp_score = None
            resp_max = None
            feedback = None
            reference_solution = None
            step_marks = None
            criterion_marks = None
            marking_policy = None
            manual_review_required = False

            # A completed evaluation document is the source of truth for a
            # final award.  OCR/evaluation status updates happen in separate
            # writes, so a response can legitimately still say ``ready`` or
            # ``ready_with_warnings`` after its evaluation was saved.
            if evaluation:
                resp_score = evaluation.get("total_score", 0.0)
                resp_max = evaluation.get("max_score", 0.0)
                feedback = evaluation.get("overall_feedback")
                reference_solution = evaluation.get("reference_solution")
                raw_criterion_marks = evaluation.get("criterion_marks")
                if isinstance(raw_criterion_marks, list):
                    criterion_marks = [
                        mark for mark in raw_criterion_marks if isinstance(mark, dict)
                    ] or None
                marking_policy = (
                    evaluation.get("marking_policy")
                    if isinstance(evaluation.get("marking_policy"), dict)
                    else None
                )
                manual_review_required = bool(
                    evaluation.get("manual_review_required")
                )
                raw_step_marks = evaluation.get("step_marks")
                if isinstance(raw_step_marks, list):
                    # Evaluation records are persisted as plain Mongo
                    # documents, but keep the API robust if an older record
                    # contains an unexpected value.
                    step_marks = [
                        mark for mark in raw_step_marks if isinstance(mark, dict)
                    ] or None
                if has_blocking or eval_status == "blocked":
                    # Never treat a score as publishable while a blocking
                    # flag remains unresolved, even if an earlier evaluator
                    # pass wrote a preliminary record.
                    blocked_count += 1
                elif manual_review_required or eval_status == "manual_review":
                    pending_count += 1
                else:
                    # A malformed legacy submission can contain two OCR
                    # segments for the same question.  Count a question once;
                    # current submissions block this situation before eval.
                    completion_key = question_id or response_id
                    if not question_id or question_id not in scored_question_ids:
                        total_score += _safe_marks(resp_score)
                        evaluated_max += _safe_marks(resp_max)
                        if question_id:
                            scored_question_ids.add(question_id)
                    if completion_key not in completed_question_keys:
                        completed_question_keys.add(completion_key)
                        evaluated_count += 1
            elif has_blocking or eval_status == "blocked":
                blocked_count += 1
            elif eval_status == "not_attempted":
                # Older completed submissions can contain an explicit blank
                # answer slot without its zero-mark evaluation document.
                # It is still a final, deliberately blank question.
                resp_score = 0.0
                completion_key = question_id or response_id
                if completion_key not in completed_question_keys:
                    completed_question_keys.add(completion_key)
                    evaluated_count += 1
            else:
                # ``ready``, ``ready_with_warnings`` and a bare ``evaluated``
                # response are not final by themselves.  Wait until the
                # immutable evaluation record exists so the workspace cannot
                # show a transient 0/40 while the Results view has the score.
                pending_count += 1

            # A pending response still belongs to a fixed paper question.  It
            # has no award yet, but the UI can show the question maximum.
            if resp_max is None and catalog_question is not None:
                resp_max = catalog_question["max_marks"]

            # Serialize flags for API response
            api_flags = [
                {
                    "flag_id": f.get("flag_id", ""),
                    "source": f.get("source", ""),
                    "flag_type": f.get("flag_type", ""),
                    "severity": f.get("severity", ""),
                    "reason": f.get("reason", ""),
                    "suggested_action": f.get("suggested_action"),
                    "resolved": f.get("resolution", {}).get(
                        "resolved", False
                    ),
                }
                for f in flags
            ]

            response_summaries.append(
                ResponseSummaryAPI(
                    response_id=response_id,
                    evaluation_id=evaluation.get("evaluation_id") if evaluation else None,
                    question_id=question_id or None,
                    question_number=(
                        resp_doc.get("question_number")
                        if resp_doc.get("question_number") is not None
                        else catalog_question.get("question_number")
                        if catalog_question is not None
                        else None
                    ),
                    content_type=resp_doc.get("content_type", "TEXT_ONLY"),
                    eval_status=eval_status,
                    detected_text=resp_doc.get("detected_text"),
                    total_score=resp_score,
                    max_score=resp_max,
                    overall_feedback=feedback,
                    reference_solution=reference_solution,
                    step_marks=step_marks,
                    criterion_marks=criterion_marks,
                    marking_policy=marking_policy,
                    manual_review_required=manual_review_required,
                    flags=api_flags if api_flags else None,
                    has_blocking_flags=has_blocking,
                    is_missing_response=bool(resp_doc.get("is_missing_response")),
                    answer_state=resp_doc.get("answer_state"),
                )
            )

        # Older submissions may have been processed before answer slots were
        # introduced.  Once segmentation has completed, every catalog question
        # absent from the detected-response collection is genuinely blank.
        # Add those final zero rows even when another response is still being
        # evaluated; the submission as a whole remains ``processing`` below.
        # A failed job must never be turned into a false 0-mark paper.
        if segmentation_status == "complete" and not processing_failed:
            for question in question_catalog:
                question_id = question["question_id"]
                if question_id in observed_question_ids:
                    continue
                completed_question_keys.add(question_id)
                response_summaries.append(
                    ResponseSummaryAPI(
                        response_id=f"UNANSWERED-{submission_id}-{question_id}",
                        question_id=question_id,
                        question_number=question["question_number"],
                        content_type="TEXT_ONLY",
                        eval_status="not_attempted",
                        total_score=0.0,
                        max_score=question["max_marks"],
                        overall_feedback=(
                            "No answer was detected for this question, so 0 marks were awarded."
                        ),
                        has_blocking_flags=False,
                        is_missing_response=True,
                        answer_state="not_attempted",
                    )
                )
                evaluated_count += 1

        # A score is only available when OCR/mapping has completed and every
        # detected answer is terminally evaluated or explicitly blank.  This
        # prevents the review workspace from publishing an OCR-era 0/40 that
        # disagrees with the immutable evaluation records shown in Results.
        if processing_failed:
            score_state = "unavailable"
        elif segmentation_status != "complete" or pending_count > 0 or blocked_count > 0:
            score_state = "processing"
        else:
            score_state = "available"

        response_summaries.sort(
            key=lambda item: (
                item.question_number is None,
                item.question_number if item.question_number is not None else 10**9,
                item.is_missing_response,
                item.response_id,
            )
        )
        total_max = (
            sum(question["max_marks"] for question in question_catalog)
            if question_catalog
            else evaluated_max
        )

        return SubmissionSummaryReviewAPI(
            submission_id=submission_id,
            exam_id=sub_dict.get("exam_id", ""),
            student_id=sub_dict.get("student_id", ""),
            source=sub_dict.get("source", "camera"),
            segmentation_status=segmentation_status,
            processing_status=processing_status,
            processing_error=processing_error,
            score_state=score_state,
            publication_status=sub_dict.get("publication_status"),
            responses=response_summaries,
            total_score=total_score,
            total_max_score=total_max,
            evaluated_count=evaluated_count,
            blocked_count=blocked_count,
            pending_count=pending_count,
        )

    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("PCR storage module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to get submission summary for %s: %s",
            submission_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve submission summary",
        )


@router.get(
    "/submissions/{submission_id}/pages",
    response_model=SubmissionPagesAPI,
    summary="Get staff-authorized private answer-page previews",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Submission not found"},
    },
)
async def get_submission_pages(
    submission_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> SubmissionPagesAPI:
    """Return temporary S3 previews only after role and tutor-scope checks.

    Raw ``s3://`` references are never returned to the browser.  The client
    receives a five-minute presigned URL for each page, generated only after
    the same staff authorization used by the score review endpoint.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id},
        projection={"student_id": 1},
    )
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Submission {submission_id} not found",
        )
    _check_student_in_scope(str(submission.get("student_id") or ""), scoped_ids)

    page_docs = await tenant_db["evalpen_answer_pages"].find(
        {"submission_id": submission_id}
    ).sort("page_number", 1).to_list(length=100)
    pages: List[SubmissionPageThumbnailAPI] = []
    for index, page_doc in enumerate(page_docs):
        raw_image_ref = str(page_doc.get("raw_image_ref") or "")
        image_url: Optional[str] = None
        if raw_image_ref.startswith("s3://"):
            try:
                image_url = create_private_download_url(
                    raw_image_ref,
                    allowed_key_prefix="private/exampen/",
                    expires_in=300,
                )
            except PrivateObjectStorageError:
                logger.warning(
                    "Could not create private answer-page preview: submission=%s page=%s",
                    submission_id,
                    page_doc.get("page_id"),
                )

        pages.append(
            SubmissionPageThumbnailAPI(
                page_id=str(page_doc.get("page_id") or f"{submission_id}-{index + 1}"),
                page_index=max(0, int(page_doc.get("page_number") or index + 1) - 1),
                image_url=image_url,
                width=int(page_doc.get("image_width_px") or 0),
                height=int(page_doc.get("image_height_px") or 0),
            )
        )
    return SubmissionPagesAPI(
        submission_id=submission_id,
        total_pages=len(pages),
        pages=pages,
    )


@router.get(
    "/exams/{exam_id}/results",
    response_model=ExamResultsAPI,
    summary="Get aggregated results for an entire exam",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database or exam-conductor unavailable"},
    },
)
async def get_exam_results(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamResultsAPI:
    """Get aggregated results for an entire exam across all students.

    Combines PCR and DCR results to provide a unified exam results view.
    Returns per-student score summaries.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        from api.v1._exampen_imports import load_exampen

        # --- PCR data ---
        pcr_available = True
        try:
            _pcr_storage = load_exampen("pcr.storage")
            SubmissionRepository = _pcr_storage.SubmissionRepository
            EvaluationRepository = _pcr_storage.EvaluationRepository
            DetectedResponseRepository = _pcr_storage.DetectedResponseRepository
        except ImportError:
            pcr_available = False

        # --- DCR data ---
        dcr_available = True
        try:
            DCRRepository = load_exampen("dcr.repository").DCRRepository
        except ImportError:
            dcr_available = False

        # Collect per-student data
        student_results: Dict[str, ExamResultStudentAPI] = {}

        # PCR: find submissions for this exam, then aggregate evaluations
        if pcr_available:
            sub_repo = SubmissionRepository(tenant_db)
            resp_repo = DetectedResponseRepository(tenant_db)
            eval_repo = EvaluationRepository(tenant_db)
            question_catalog = await _get_pcr_question_catalog(tenant_db, exam_id)
            paper_max_score = sum(
                question["max_marks"] for question in question_catalog
            )
            catalog_question_ids = {
                question["question_id"] for question in question_catalog
            }

            # Find submissions for this exam (tutor-scoped)
            sub_query: Dict[str, Any] = {"exam_id": exam_id}
            if scoped_ids is not None:
                sub_query["student_id"] = {"$in": scoped_ids}

            submissions_cursor = tenant_db["evalpen_submissions"].find(
                sub_query,
                projection={
                    "submission_id": 1,
                    "student_id": 1,
                    "publication_status": 1,
                },
            )
            submissions = await submissions_cursor.to_list(length=1000)

            for sub in submissions:
                student_id = sub.get("student_id", "")
                submission_id = sub.get("submission_id", "")

                # Get responses for this submission
                responses = await resp_repo.get_responses_by_submission(
                    submission_id
                )

                pcr_total = 0.0
                evaluated_max = 0.0
                blocked_count = 0
                scored_question_ids: set[str] = set()

                for resp in responses:
                    response_id = resp.get("response_id", "")

                    # Count blocked
                    if resp.get("eval_status") == "blocked":
                        blocked_count += 1

                    # Get evaluation
                    ev = await eval_repo.get_evaluation_by_response(
                        response_id
                    )
                    if ev:
                        question_id = str(resp.get("question_id") or "")
                        # Never let duplicate OCR segments make one paper
                        # question count twice.  New jobs block duplicates;
                        # this also repairs historical result aggregation.
                        if (
                            question_id
                            and question_id in catalog_question_ids
                            and question_id in scored_question_ids
                        ):
                            continue
                        pcr_total += _safe_marks(ev.get("total_score"))
                        evaluated_max += _safe_marks(ev.get("max_score"))
                        if question_id and question_id in catalog_question_ids:
                            scored_question_ids.add(question_id)

                entry = student_results.get(student_id)
                if entry is None:
                    entry = ExamResultStudentAPI(
                        student_id=student_id,
                        submission_id=submission_id,
                        publication_status=sub.get("publication_status"),
                    )
                    student_results[student_id] = entry

                entry.pcr_total_score = pcr_total
                entry.pcr_max_score = paper_max_score or evaluated_max
                entry.blocked_responses = blocked_count

        # DCR: aggregate results per student
        if dcr_available:
            dcr_repo = DCRRepository(tenant_db)

            # Find distinct students with DCR results for this exam (tutor-scoped)
            dcr_query: Dict[str, Any] = {"exam_id": exam_id}
            if scoped_ids is not None:
                dcr_query["student_id"] = {"$in": scoped_ids}

            dcr_cursor = tenant_db["exampen_dcr_results"].find(
                dcr_query,
                projection={
                    "student_id": 1,
                    "score": 1,
                    "max_score": 1,
                },
            )
            dcr_docs = await dcr_cursor.to_list(length=5000)

            # Group by student
            dcr_by_student: Dict[str, Dict[str, float]] = {}
            for doc in dcr_docs:
                sid = doc.get("student_id", "")
                if sid not in dcr_by_student:
                    dcr_by_student[sid] = {
                        "total_score": 0.0,
                        "max_score": 0.0,
                    }
                dcr_by_student[sid]["total_score"] += doc.get("score", 0.0)
                dcr_by_student[sid]["max_score"] += doc.get(
                    "max_score", 0.0
                )

            for sid, dcr_scores in dcr_by_student.items():
                entry = student_results.get(sid)
                if entry is None:
                    entry = ExamResultStudentAPI(student_id=sid)
                    student_results[sid] = entry
                entry.dcr_total_score = dcr_scores["total_score"]
                entry.dcr_max_score = dcr_scores["max_score"]

        # Compute combined totals
        for entry in student_results.values():
            entry.combined_total = (
                entry.pcr_total_score + entry.dcr_total_score
            )
            entry.combined_max = entry.pcr_max_score + entry.dcr_max_score

        students_list = sorted(
            student_results.values(),
            key=lambda s: s.student_id,
        )

        return ExamResultsAPI(
            exam_id=exam_id,
            students=students_list,
            total_students=len(students_list),
        )

    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("Exam-conductor module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exam-conductor engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to get exam results for %s: %s",
            exam_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exam results",
        )


@router.post(
    "/evaluations/{evaluation_id}/override",
    summary="Override a PCR evaluation score with audit trail",
    responses={
        400: {"description": "Invalid request (reason too short, score invalid)"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Evaluation not found"},
        503: {"description": "Tenant database or exam-conductor unavailable"},
    },
)
async def override_evaluation_score(
    evaluation_id: str,
    body: ScoreOverrideRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Override the score on a PCR evaluation.

    TAMPER_PROOF_SPEC Layer 3, Section 6 Rule 4:
    Every score override must have actor_id, reason (min 5 chars),
    before/after state, and timestamp. The override is append-only --
    previous scores are preserved in the audit trail.

    The score update and audit entry are applied atomically via
    ``EvaluationRepository.override_score()``.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        from api.v1._exampen_imports import load_exampen

        EvaluationRepository = load_exampen(
            "pcr.storage"
        ).EvaluationRepository

        eval_repo = EvaluationRepository(tenant_db)

        # Verify evaluation exists and get current score
        existing = await eval_repo.get_evaluation(evaluation_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {evaluation_id} not found",
            )
        if isinstance(existing.get("criterion_marks"), list) and existing.get("criterion_marks"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This evaluation uses a locked criterion rubric. "
                    "Adjust its individual criterion marks instead of overriding the total."
                ),
            )

        # Tutor scoping: verify this evaluation's student is visible
        eval_student_id = existing.get("student_id")
        if not eval_student_id:
            # Fallback: look up student_id via response -> submission
            _resp_id = existing.get("response_id", "")
            if _resp_id:
                _resp_doc = await tenant_db[
                    "evalpen_detected_responses"
                ].find_one(
                    {"response_id": _resp_id},
                    projection={"submission_id": 1},
                )
                if _resp_doc:
                    _sub_doc = await tenant_db[
                        "evalpen_submissions"
                    ].find_one(
                        {"submission_id": _resp_doc.get("submission_id", "")},
                        projection={"student_id": 1},
                    )
                    eval_student_id = (
                        _sub_doc.get("student_id") if _sub_doc else None
                    )
        if eval_student_id:
            _check_student_in_scope(eval_student_id, scoped_ids)

        # Validate new_score against max_score
        max_score = existing.get("max_score", 0.0)
        if body.new_score > max_score:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"New score ({body.new_score}) cannot exceed "
                    f"max_score ({max_score})"
                ),
            )

        actor_id = current_user.get("user_id", "unknown")
        old_score = existing.get("total_score", 0.0)

        # Perform the atomic score override with audit trail
        success = await eval_repo.override_score(
            evaluation_id,
            new_total_score=body.new_score,
            actor_id=actor_id,
            reason=body.reason,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Score override operation failed",
            )

        logger.info(
            "Score override on evaluation %s: %.2f -> %.2f by %s (%s)",
            evaluation_id,
            old_score,
            body.new_score,
            actor_id,
            body.reason,
        )

        return {
            "evaluation_id": evaluation_id,
            "previous_score": old_score,
            "new_score": body.new_score,
            "actor_id": actor_id,
            "overridden_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("PCR storage module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Score override failed for evaluation %s: %s",
            evaluation_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Score override encountered an internal error",
        )


@router.post(
    "/evaluations/{evaluation_id}/criterion-marks",
    summary="Teacher-review locked PCR criterion marks with audit trail",
    responses={
        400: {"description": "Invalid criterion score or reason"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Evaluation not found"},
        409: {"description": "Evaluation does not use a criterion rubric"},
    },
)
async def override_criterion_marks(
    evaluation_id: str,
    body: CriterionMarksOverrideRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Record a teacher's criterion-level correction and recompute the total.

    The API never accepts a browser-supplied aggregate score for structured
    papers.  It validates the exact frozen criterion IDs/maxima and makes the
    repository update plus audit append atomically.
    """

    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    try:
        from api.v1._exampen_imports import load_exampen

        EvaluationRepository = load_exampen("pcr.storage").EvaluationRepository
        eval_repo = EvaluationRepository(tenant_db)
        existing = await eval_repo.get_evaluation(evaluation_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {evaluation_id} not found",
            )

        raw_criteria = existing.get("criterion_marks")
        if not isinstance(raw_criteria, list) or not raw_criteria:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This evaluation does not use a locked criterion rubric",
            )

        eval_student_id = existing.get("student_id")
        if not eval_student_id:
            response_doc = await tenant_db["evalpen_detected_responses"].find_one(
                {"response_id": existing.get("response_id", "")},
                projection={"submission_id": 1, "student_id": 1},
            )
            eval_student_id = (response_doc or {}).get("student_id")
            if not eval_student_id and response_doc:
                submission_doc = await tenant_db["evalpen_submissions"].find_one(
                    {"submission_id": response_doc.get("submission_id", "")},
                    projection={"student_id": 1},
                )
                eval_student_id = (submission_doc or {}).get("student_id")
        if eval_student_id:
            _check_student_in_scope(str(eval_student_id), scoped_ids)

        expected: Dict[str, float] = {}
        for criterion in raw_criteria:
            if not isinstance(criterion, dict):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Stored criterion rubric is malformed; contact an administrator",
                )
            criterion_id = str(criterion.get("criterion_id") or "").strip()
            if not criterion_id or criterion_id in expected:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Stored criterion rubric is malformed; contact an administrator",
                )
            expected[criterion_id] = float(criterion.get("max_marks") or 0.0)

        submitted: Dict[str, float] = {}
        for item in body.criteria:
            criterion_id = item.criterion_id.strip()
            if criterion_id in submitted:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Criterion {criterion_id} was submitted more than once",
                )
            submitted[criterion_id] = float(item.marks_awarded)
        if set(submitted) != set(expected):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Submit one mark for every locked criterion",
            )
        for criterion_id, awarded in submitted.items():
            if (
                not math.isfinite(awarded)
                or not math.isfinite(expected[criterion_id])
                or awarded < 0
                or awarded > expected[criterion_id]
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Criterion {criterion_id} must be between 0 and "
                        f"{expected[criterion_id]:g}"
                    ),
                )

        actor_id = current_user.get("user_id", "unknown")
        updated = await eval_repo.override_criterion_marks(
            evaluation_id,
            marks_by_criterion=submitted,
            actor_id=actor_id,
            reason=body.reason,
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Criterion marks could not be updated; reload and try again",
            )

        await tenant_db["evalpen_detected_responses"].update_one(
            {"response_id": existing.get("response_id", "")},
            {
                "$set": {
                    "eval_status": "evaluated_teacher_reviewed",
                    "teacher_reviewed_at": datetime.now(timezone.utc),
                    "teacher_reviewed_by": actor_id,
                }
            },
        )
        return {
            "evaluation_id": evaluation_id,
            "total_score": updated["total_score"],
            "max_score": updated.get("max_score"),
            "criterion_marks": updated["criterion_marks"],
            "actor_id": actor_id,
        }
    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("PCR storage module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR engine is not available in this deployment",
        ) from exc
    except Exception as exc:
        logger.error(
            "Criterion mark override failed for %s: %s",
            evaluation_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Criterion mark override encountered an internal error",
        ) from exc


@router.post(
    "/submissions/{submission_id}/publish",
    summary="Publish/finalize submission results",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Submission not found"},
        409: {"description": "Submission has unresolved blocking flags"},
        503: {"description": "Tenant database or exam-conductor unavailable"},
    },
)
async def publish_submission(
    submission_id: str,
    body: PublishRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Mark submission results as published/finalized.

    Checks that all blocking flags have been resolved before allowing
    publication. Records the publication event in the submission's
    metadata with actor_id and timestamp.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        from api.v1._exampen_imports import load_exampen

        _pcr_storage = load_exampen("pcr.storage")
        DetectedResponseRepository = _pcr_storage.DetectedResponseRepository

        submissions_col = tenant_db["evalpen_submissions"]

        # Verify submission exists
        submission = await submissions_col.find_one(
            {"submission_id": submission_id}
        )
        if submission is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Submission {submission_id} not found",
            )

        # Tutor scoping: verify this submission's student is visible
        _check_student_in_scope(
            submission.get("student_id", ""), scoped_ids
        )

        # Check for unresolved blocking flags
        resp_repo = DetectedResponseRepository(tenant_db)
        blocked_responses = await resp_repo.get_responses_with_blocking_flags(
            submission_id=submission_id
        )

        # Filter to only truly unresolved blocking flags
        unresolved_blocking = []
        for resp in blocked_responses:
            flags = resp.get("flags", [])
            has_unresolved = any(
                f.get("severity") == "blocking"
                and not f.get("resolution", {}).get("resolved", False)
                for f in flags
            )
            if has_unresolved:
                unresolved_blocking.append(resp.get("response_id", ""))

        if unresolved_blocking:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Cannot publish: {len(unresolved_blocking)} response(s) "
                    f"have unresolved blocking flags. Resolve all blocking "
                    f"flags before publishing."
                ),
            )

        manual_review_count = await tenant_db["evalpen_detected_responses"].count_documents(
            {"submission_id": submission_id, "eval_status": "manual_review"}
        )
        if manual_review_count:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Cannot publish: {manual_review_count} response(s) still require "
                    "teacher criterion review."
                ),
            )

        actor_id = current_user.get("user_id", "unknown")
        now = datetime.now(timezone.utc)

        # Update publication status (non-immutable metadata field)
        await submissions_col.update_one(
            {"submission_id": submission_id},
            {
                "$set": {
                    "publication_status": "published",
                    "published_at": now,
                    "published_by": actor_id,
                    "publication_note": body.note,
                },
            },
        )

        logger.info(
            "Submission %s published by %s at %s",
            submission_id,
            actor_id,
            now.isoformat(),
        )

        return {
            "submission_id": submission_id,
            "publication_status": "published",
            "published_at": now.isoformat(),
            "published_by": actor_id,
        }

    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("PCR storage module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Publication failed for submission %s: %s",
            submission_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Publication encountered an internal error",
        )
