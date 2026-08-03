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
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from services.answer_mapping_contract import normalize_answer_label
from services.evalpen_flag_utils import is_flag_resolved, resolve_flag
from services.objective_scoring_service import is_integer_question
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
    amend_published: bool = Field(
        default=False,
        description=(
            "Explicit confirmation that a published result may be revised. "
            "The corrected result remains published and receives a new audited snapshot."
        ),
    )

    @field_validator("reason")
    @classmethod
    def reason_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Reason must be at least 5 characters")
        return v.strip()


class EvaluationApprovalRequest(BaseModel):
    """Teacher confirmation of a nonblocking AI review."""

    award_full_marks: bool = Field(
        default=False,
        description="Award every locked criterion its maximum before approving.",
    )
    reason: str = Field(
        default="Teacher verified the answer against the original answer copy",
        min_length=5,
        max_length=500,
    )

    @field_validator("reason")
    @classmethod
    def approval_reason_min_length(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Reason must be at least 5 characters")
        return value


class ResponseManualResolutionRequest(BaseModel):
    """Teacher-owned final score when automated grading could not materialize."""

    awarded_marks: float = Field(..., ge=0)
    reason: str = Field(..., min_length=5, max_length=500)

    @field_validator("reason")
    @classmethod
    def manual_resolution_reason(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Reason must be at least 5 characters")
        return value


class QuestionTaxonomyUpdateRequest(BaseModel):
    """Persisted teacher-owned topic classification for one frozen question."""

    topic: str = Field(..., min_length=1, max_length=120)
    sub_topic: Optional[str] = Field(default=None, max_length=120)

    @field_validator("topic")
    @classmethod
    def normalized_topic(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Topic is required")
        return value

    @field_validator("sub_topic")
    @classmethod
    def normalized_sub_topic(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None


class AutoTagQuestionsRequest(BaseModel):
    """Batch topic generation while preserving teacher-owned tags by default."""

    replace_existing: bool = False


class CriterionMarkOverrideItem(BaseModel):
    """One teacher-entered score for an already-locked rubric criterion."""

    criterion_id: str = Field(..., min_length=1, max_length=64)
    marks_awarded: float = Field(..., ge=0)


class CriterionMarksOverrideRequest(BaseModel):
    """Teacher correction of every criterion; total is recomputed server-side."""

    criteria: List[CriterionMarkOverrideItem] = Field(..., min_length=1)
    reason: str = Field(..., min_length=5)
    amend_published: bool = Field(
        default=False,
        description=(
            "Explicit confirmation that a published result may be revised. "
            "The corrected result remains published and receives a new audited snapshot."
        ),
    )

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


class DocumentCoverageReviewRequest(BaseModel):
    """Teacher confirmation that every submitted page was visually reviewed."""

    grading_run_id: str = Field(..., min_length=1, max_length=128)
    note: str = Field(..., min_length=5, max_length=1000)

    @field_validator("grading_run_id", "note")
    @classmethod
    def strip_document_review_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be blank")
        return value


class ResponseRegionCorrection(BaseModel):
    page_number: int = Field(..., ge=1)
    x_start: float = Field(0.0, ge=0)
    y_start: float = Field(..., ge=0)
    x_end: Optional[float] = Field(None, gt=0)
    y_end: float = Field(..., gt=0)
    region_id: Optional[str] = None
    evidence_kind: Optional[str] = None
    continuation_group: Optional[str] = None
    evidence: Optional[str] = None
    mapping_confidence: Optional[float] = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_range(self) -> "ResponseRegionCorrection":
        if self.y_end <= self.y_start:
            raise ValueError("y_end must be greater than y_start")
        if self.x_end is not None and self.x_end <= self.x_start:
            raise ValueError("x_end must be greater than x_start")
        return self


class ResponseSplitPart(BaseModel):
    question_id: str = Field(..., min_length=1)
    detected_text: str = ""
    source_pages: List[ResponseRegionCorrection] = Field(default_factory=list)


class ResponseAssignmentCorrectionRequest(BaseModel):
    """Teacher correction for response ownership or proven absence."""

    action: str = Field(
        ...,
        pattern=(
            "^(assign|split|merge|confirm_not_attempted|"
            "discard_non_answer|set_objective_answer)$"
        ),
    )
    reason: str = Field(..., min_length=5, max_length=1000)
    response_id: Optional[str] = None
    response_ids: List[str] = Field(default_factory=list)
    question_id: Optional[str] = None
    selected_answer: Optional[str] = None
    parts: List[ResponseSplitPart] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def correction_reason(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Reason must be at least 5 characters")
        return value


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
    method_policy: Optional[Dict[str, Any]] = None
    method_analysis: Optional[Dict[str, Any]] = None
    eval_path: Optional[str] = None
    manual_review_required: bool = False
    flags: Optional[List[Dict[str, Any]]] = None
    has_blocking_flags: bool = False
    # Every PCR paper is represented as a complete question matrix.  These
    # fields distinguish a true blank answer from an OCR/AI failure.
    is_missing_response: bool = False
    answer_state: Optional[str] = None
    source_pages: List[Dict[str, Any]] = Field(default_factory=list)
    question_assignment: Optional[Dict[str, Any]] = None
    manual_review_reason: Optional[str] = None
    grading_mode: Optional[str] = None


class QuestionCatalogItemAPI(BaseModel):
    """One immutable paper question, independent of OCR response cardinality."""

    question_id: str
    question_number: Optional[int] = None
    question_text: str = ""
    max_marks: float = 0.0
    reference_solution: Optional[str] = None
    source_page_number: Optional[int] = None
    source_region_id: Optional[str] = None
    source_bbox_percent: Optional[Dict[str, Any]] = None
    grading_mode: Optional[str] = None
    option_labels: List[str] = Field(default_factory=list)


class SubmissionSummaryReviewAPI(BaseModel):
    """Full evaluation summary for a single submission."""

    submission_id: str
    exam_id: str
    student_id: str
    source: str = "camera"
    segmentation_status: str = "pending"
    processing_status: Optional[str] = None
    processing_job_id: Optional[str] = None
    can_reprocess: bool = False
    reprocess_block_reason: Optional[str] = None
    # Staff-only diagnostic.  Students continue to receive the safe status
    # surface from their BFF and are never shown worker/storage internals.
    processing_error: Optional[str] = None
    processing_failure_code: Optional[str] = None
    processing_retry_at: Optional[str] = None
    processing_attempts: int = 0
    # ``available`` means all materialized evaluations can be displayed. A
    # separate review_state/readiness gate still prevents unsafe publication;
    # ``processing`` and ``unavailable`` must never render as a final zero.
    score_state: str = "processing"
    publication_status: Optional[str] = None
    question_catalog: List[QuestionCatalogItemAPI] = Field(default_factory=list)
    responses: List[ResponseSummaryAPI] = Field(default_factory=list)
    unassigned_responses: List[ResponseSummaryAPI] = Field(default_factory=list)
    page_count: int = 0
    total_score: float = 0.0
    total_max_score: float = 0.0
    evaluated_count: int = 0
    blocked_count: int = 0
    pending_count: int = 0
    review_count: int = 0
    review_state: str = "processing"
    document_review: Optional[Dict[str, Any]] = None


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


def _safe_score(value: Any) -> float:
    """Return a finite score while preserving Objective negative marking."""

    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _catalog_grading_mode(question: Dict[str, Any]) -> str:
    grading_mode = str(question.get("grading_mode") or "").strip().lower()
    question_type = str(question.get("question_type") or "").strip().lower()
    if (
        grading_mode in {"objective", "mcq"}
        or question_type in {"objective", "mcq"}
    ) and not is_integer_question(question):
        return "objective"
    return "subjective"


def _catalog_option_labels(question: Dict[str, Any]) -> List[str]:
    """Return the immutable option labels without exposing the answer key."""

    options = question.get("options") or question.get("enhanced_options") or []
    if not isinstance(options, list):
        return []
    labels: List[str] = []
    for index, option in enumerate(options):
        raw_label = (
            option.get("label") or option.get("key") or option.get("id")
            if isinstance(option, dict)
            else ""
        )
        label = normalize_answer_label(raw_label) or (
            chr(ord("A") + index) if index < 6 else ""
        )
        if label and label not in labels:
            labels.append(label)
    return labels


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
        projection={
            "question_id": 1,
            "question_number": 1,
            "question_text": 1,
            "max_marks": 1,
            "reference_solution": 1,
            "source_page_number": 1,
            "source_region_id": 1,
            "source_bbox_percent": 1,
            "grading_mode": 1,
            "question_type": 1,
            "options": 1,
            "enhanced_options": 1,
        },
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
                "question_text": str(doc.get("question_text") or "").strip(),
                "reference_solution": (
                    str(doc.get("reference_solution") or "").strip() or None
                ),
                "source_page_number": doc.get("source_page_number"),
                "source_region_id": doc.get("source_region_id"),
                "source_bbox_percent": doc.get("source_bbox_percent"),
                "grading_mode": _catalog_grading_mode(doc),
                "option_labels": _catalog_option_labels(doc),
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


async def _amend_published_submission_snapshot(
    tenant_db: Any,
    submission: Dict[str, Any],
    *,
    evaluation_id: str,
    actor_id: str,
    reason: str,
    review_lease_token: str,
) -> Dict[str, Any]:
    """Replace a published score snapshot while preserving its prior release.

    Published results stay visible to the student.  The previous immutable
    snapshot is retained in ``publication_history`` and the live snapshot is
    replaced under the submission review lease, so a correction cannot race
    publishing, reprocessing, or another teacher edit.
    """

    from services.exampen_submission_readiness import build_publication_snapshot

    submission_id = str(submission.get("submission_id") or "")
    previous_snapshot = submission.get("publication_snapshot")
    previous_snapshot_hash = str(
        submission.get("publication_snapshot_hash")
        or (previous_snapshot or {}).get("snapshot_hash")
        or ""
    )
    previous_total = (
        (previous_snapshot or {}).get("total_score")
        if isinstance(previous_snapshot, dict)
        else None
    )

    next_snapshot = await build_publication_snapshot(
        tenant_db,
        submission_id,
        actor_id=actor_id,
    )
    amended_at = next_snapshot.pop("published_at_dt")
    next_revision = int(submission.get("result_revision") or 0) + 1

    update_filter: Dict[str, Any] = {
        "submission_id": submission_id,
        "publication_status": "published",
        "review_mutation_lease_token": review_lease_token,
    }
    if "publication_snapshot_hash" in submission:
        update_filter["publication_snapshot_hash"] = submission.get(
            "publication_snapshot_hash"
        )
    else:
        update_filter["publication_snapshot_hash"] = {"$exists": False}

    amended = await tenant_db["evalpen_submissions"].update_one(
        update_filter,
        {
            "$set": {
                "publication_snapshot": next_snapshot,
                "publication_snapshot_hash": next_snapshot["snapshot_hash"],
                "result_revision": next_revision,
                "last_amended_at": amended_at,
                "last_amended_by": actor_id,
                "last_amendment_reason": reason,
                "updated_at": amended_at,
            },
            "$push": {
                "publication_history": {
                    "action": "published_score_amended",
                    "amended_at": amended_at,
                    "amended_by": actor_id,
                    "reason": reason,
                    "evaluation_id": evaluation_id,
                    "revision": next_revision,
                    "previous_snapshot_hash": previous_snapshot_hash or None,
                    "snapshot_hash": next_snapshot["snapshot_hash"],
                    "previous_total_score": previous_total,
                    "total_score": next_snapshot.get("total_score"),
                    # Preserve exactly what the student saw before this
                    # amendment.  The new release remains in
                    # ``publication_snapshot``.
                    "previous_snapshot": previous_snapshot,
                }
            },
        },
    )
    if amended.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The published result changed while this amendment was being "
                "saved. Refresh the paper and try again."
            ),
        )

    return {
        "publication_status": "published",
        "result_revision": next_revision,
        "amended_at": amended_at.isoformat(),
        "snapshot_hash": next_snapshot["snapshot_hash"],
    }


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
            {"submission_id": submission_id},
            sort=[("created_at", -1)],
        )
        processing_job_id = (
            str(processing_job.get("job_id"))
            if isinstance(processing_job, dict) and processing_job.get("job_id")
            else None
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
        processing_failure_code = (
            str(processing_job.get("failure_code"))[:120]
            if isinstance(processing_job, dict) and processing_job.get("failure_code")
            else None
        )
        processing_retry_at = (
            _dt_to_iso(processing_job.get("next_retry_at"))
            if isinstance(processing_job, dict)
            and processing_job.get("next_retry_at") is not None
            else None
        )
        try:
            processing_attempts = max(
                0, int((processing_job or {}).get("attempts") or 0)
            )
        except (TypeError, ValueError):
            processing_attempts = 0
        retry_scheduled = (
            processing_status == "retryable_error" and bool(processing_retry_at)
        )
        terminal_reprocess_statuses = {
            "completed",
            "blocked_for_review",
            "failed",
            "retryable_error",
            "enqueue_failed",
        }
        publication_status = str(sub_dict.get("publication_status") or "pending")
        contract_migration_required = (
            processing_failure_code == "UnsupportedGradingContractError"
        )
        can_reprocess = bool(
            processing_job_id
            and publication_status != "published"
            and not retry_scheduled
            and not contract_migration_required
            and processing_status in terminal_reprocess_statuses
        )
        reprocess_block_reason = None
        if publication_status == "published":
            reprocess_block_reason = "Published results must use the recheck workflow"
        elif contract_migration_required:
            reprocess_block_reason = (
                "This exam uses an unsupported frozen grading contract. "
                "Migrate and reprocess the complete cohort."
            )
        elif retry_scheduled:
            reprocess_block_reason = "An automatic retry is already scheduled"
        elif not processing_job_id:
            reprocess_block_reason = "No canonical processing job is available yet"
        elif processing_status not in terminal_reprocess_statuses:
            reprocess_block_reason = "The current processing job is still active"
        processing_active = processing_status in {"queued", "processing"} or retry_scheduled
        processing_failed = (
            (segmentation_status == "failed" and not processing_active)
            or processing_status in {"failed", "enqueue_failed"}
            or (processing_status == "retryable_error" and not retry_scheduled)
        )
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

        # Fetch all evaluations in one query. The previous per-response lookup
        # made a 100-question review perform 100 sequential database calls.
        response_ids = [
            str(item.get("response_id") or "")
            for item in response_docs
            if str(item.get("response_id") or "")
        ]
        evaluation_docs = (
            await tenant_db["evalpen_evaluations"]
            .find({"response_id": {"$in": response_ids}})
            .to_list(length=5000)
            if response_ids
            else []
        )
        evaluations_by_response = {
            str(item.get("response_id") or ""): item for item in evaluation_docs
        }

        response_summaries: List[ResponseSummaryAPI] = []
        unassigned_summaries: List[ResponseSummaryAPI] = []
        total_score = 0.0
        evaluated_max = 0.0
        evaluated_count = 0
        blocked_count = 0
        pending_count = 0
        review_count = 0
        scored_question_ids: set[str] = set()
        completed_question_keys: set[str] = set()
        observed_question_ids: set[str] = set()

        for resp_doc in response_docs:
            response_id = resp_doc.get("response_id", "")
            eval_status = str(resp_doc.get("eval_status") or "pending").lower()
            question_id = str(resp_doc.get("question_id") or "")
            catalog_question = catalog_by_id.get(question_id)
            is_unassigned = bool(question_catalog) and catalog_question is None
            if question_id and not is_unassigned:
                observed_question_ids.add(question_id)

            # Check for blocking flags
            flags = resp_doc.get("flags", [])
            has_blocking = any(
                f.get("severity") == "blocking"
                and not is_flag_resolved(f)
                for f in flags
            )

            evaluation = evaluations_by_response.get(response_id)

            resp_score = None
            resp_max = None
            feedback = None
            reference_solution = None
            step_marks = None
            criterion_marks = None
            marking_policy = None
            method_policy = None
            method_analysis = None
            assignment = resp_doc.get("question_assignment")
            manual_review_required = bool(
                resp_doc.get("manual_review_required")
                or (
                    isinstance(assignment, dict)
                    and assignment.get("manual_review_required")
                )
            )

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
                method_policy = (
                    evaluation.get("method_policy")
                    if isinstance(evaluation.get("method_policy"), dict)
                    else None
                )
                method_analysis = (
                    evaluation.get("method_analysis")
                    if isinstance(evaluation.get("method_analysis"), dict)
                    else None
                )
                manual_review_required = bool(
                    evaluation.get("manual_review_required")
                    or manual_review_required
                )
                raw_step_marks = evaluation.get("step_marks")
                if isinstance(raw_step_marks, list):
                    # Evaluation records are persisted as plain Mongo
                    # documents, but keep the API robust if an older record
                    # contains an unexpected value.
                    step_marks = [
                        mark for mark in raw_step_marks if isinstance(mark, dict)
                    ] or None
                if is_unassigned:
                    blocked_count += 1
                elif has_blocking or eval_status == "blocked":
                    # Never treat a score as publishable while a blocking
                    # flag remains unresolved, even if an earlier evaluator
                    # pass wrote a preliminary record.
                    blocked_count += 1
                else:
                    if manual_review_required or eval_status == "manual_review":
                        review_count += 1
                    # A malformed legacy submission can contain two OCR
                    # segments for the same question.  Count a question once;
                    # current submissions block this situation before eval.
                    completion_key = question_id or response_id
                    if not question_id or question_id not in scored_question_ids:
                        total_score += _safe_score(resp_score)
                        evaluated_max += _safe_marks(resp_max)
                        if question_id:
                            scored_question_ids.add(question_id)
                    if completion_key not in completed_question_keys:
                        completed_question_keys.add(completion_key)
                        evaluated_count += 1
            elif is_unassigned:
                blocked_count += 1
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

            if method_analysis is None and isinstance(assignment, dict):
                method_analysis = (
                    assignment.get("method_analysis")
                    if isinstance(assignment.get("method_analysis"), dict)
                    else None
                )

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
                    "resolved": is_flag_resolved(f),
                }
                for f in flags
            ]

            response_summary = ResponseSummaryAPI(
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
                    method_policy=method_policy,
                    method_analysis=method_analysis,
                    eval_path=(evaluation.get("eval_path") if evaluation else None),
                    manual_review_required=manual_review_required,
                    flags=api_flags if api_flags else None,
                    has_blocking_flags=has_blocking,
                    is_missing_response=bool(resp_doc.get("is_missing_response")),
                    answer_state=resp_doc.get("answer_state"),
                    source_pages=[
                        item
                        for item in (resp_doc.get("source_pages") or [])
                        if isinstance(item, dict)
                    ],
                    question_assignment=(
                        resp_doc.get("question_assignment")
                        if isinstance(resp_doc.get("question_assignment"), dict)
                        else None
                    ),
                    manual_review_reason=resp_doc.get("manual_review_reason"),
                    grading_mode=(
                        str(resp_doc.get("grading_mode") or "").strip()
                        or str((evaluation or {}).get("grading_mode") or "").strip()
                        or str((catalog_question or {}).get("grading_mode") or "").strip()
                        or None
                    ),
                )
            if is_unassigned:
                unassigned_summaries.append(response_summary)
            else:
                response_summaries.append(response_summary)

        # A missing response row is not proof that the student skipped the
        # question.  Keep the missing state explicit and blocking until the
        # document mapper or a teacher records evidence-backed absence.
        if question_catalog and not processing_failed and not processing_active:
            for question in question_catalog:
                question_id = question["question_id"]
                if question_id in observed_question_ids:
                    continue
                response_summaries.append(
                    ResponseSummaryAPI(
                        response_id=f"UNRESOLVED-{submission_id}-{question_id}",
                        question_id=question_id,
                        question_number=question["question_number"],
                        content_type="TEXT_ONLY",
                        eval_status="unresolved",
                        total_score=None,
                        max_score=question["max_marks"],
                        overall_feedback=(
                            "No verified answer state exists for this question. Review the copy before publishing."
                        ),
                        has_blocking_flags=True,
                        is_missing_response=False,
                        answer_state="unresolved",
                        grading_mode=question.get("grading_mode"),
                    )
                )
                blocked_count += 1

        # Display every persisted award once evaluation has settled. Review
        # and blocking states are reported separately and still prevent
        # publication; they must not hide already-materialized marks.
        if processing_failed:
            score_state = "unavailable"
        elif processing_active or segmentation_status != "complete" or pending_count > 0:
            score_state = "processing"
        else:
            score_state = "available"

        document_review = (
            sub_dict.get("document_review")
            if isinstance(sub_dict.get("document_review"), dict)
            else None
        )
        # Derive the displayed state from canonical rows on every read. The
        # stored field is an index hint and can briefly lag behind a review or
        # reprocess write; it must never hide a current blocker.
        review_state = (
            "blocked"
            if processing_failed or (blocked_count and score_state != "processing")
            else "processing"
            if score_state == "processing"
            else "needs_review"
            if review_count or (document_review and document_review.get("required"))
            else "ready"
        )

        response_summaries.sort(
            key=lambda item: (
                item.question_number is None,
                item.question_number if item.question_number is not None else 10**9,
                item.is_missing_response,
                item.response_id,
            )
        )
        unassigned_summaries.sort(key=lambda item: item.response_id)
        total_max = (
            sum(question["max_marks"] for question in question_catalog)
            if question_catalog
            else evaluated_max
        )
        page_count = await tenant_db["evalpen_answer_pages"].count_documents(
            {"submission_id": submission_id}
        )

        return SubmissionSummaryReviewAPI(
            submission_id=submission_id,
            exam_id=sub_dict.get("exam_id", ""),
            student_id=sub_dict.get("student_id", ""),
            source=sub_dict.get("source", "camera"),
            segmentation_status=segmentation_status,
            processing_status=processing_status,
            processing_job_id=processing_job_id,
            can_reprocess=can_reprocess,
            reprocess_block_reason=reprocess_block_reason,
            processing_error=processing_error,
            processing_failure_code=processing_failure_code,
            processing_retry_at=processing_retry_at,
            processing_attempts=processing_attempts,
            score_state=score_state,
            publication_status=sub_dict.get("publication_status"),
            question_catalog=question_catalog,
            responses=response_summaries,
            unassigned_responses=unassigned_summaries,
            page_count=page_count,
            total_score=total_score,
            total_max_score=total_max,
            evaluated_count=evaluated_count,
            blocked_count=blocked_count,
            pending_count=pending_count,
            review_count=review_count,
            review_state=review_state,
            document_review=document_review,
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


_REVIEWABLE_READINESS_CODES = frozenset(
    {
        "document_coverage_requires_review",
        "response_assignment_requires_review",
        "evaluation_requires_review",
    }
)


def _review_transition_from_readiness(
    readiness: Dict[str, Any],
    *,
    now: datetime,
) -> tuple[str, Dict[str, Any]]:
    """Move a reviewed submission toward publication without publishing it."""

    blocker_codes = {
        str(item.get("code") or "")
        for item in (readiness.get("blockers") or [])
        if str(item.get("code") or "")
    }
    review_state = (
        "ready"
        if readiness.get("ready")
        else "needs_review"
        if blocker_codes and blocker_codes.issubset(_REVIEWABLE_READINESS_CODES)
        else "blocked"
    )
    state_update: Dict[str, Any] = {
        "review_state": review_state,
        "publication_status": "ready" if readiness.get("ready") else "pending",
        "updated_at": now,
    }
    return review_state, state_update


def _resolve_review_flags(
    raw_flags: Any,
    *,
    actor_id: str,
    resolved_at: datetime,
    reason: str,
) -> List[Dict[str, Any]]:
    """Resolve review flags consistently for every teacher override path."""

    resolved: List[Dict[str, Any]] = []
    for raw_flag in raw_flags or []:
        if not isinstance(raw_flag, dict):
            continue
        resolved.append(
            dict(raw_flag)
            if is_flag_resolved(raw_flag)
            else resolve_flag(
                raw_flag,
                actor_id=actor_id,
                resolved_at=resolved_at,
                reason=reason,
            )
        )
    return resolved


def _teacher_reviewed_response_fields(
    response: Dict[str, Any],
    *,
    actor_id: str,
    reviewed_at: datetime,
    reason: str,
) -> Dict[str, Any]:
    """Build a review update that is safe for legacy response documents."""

    fields: Dict[str, Any] = {
        "eval_status": "evaluated_teacher_reviewed",
        "manual_review_required": False,
        "manual_review_reason": None,
        "flags": _resolve_review_flags(
            response.get("flags"),
            actor_id=actor_id,
            resolved_at=reviewed_at,
            reason=reason,
        ),
        "teacher_review_status": "approved",
        "teacher_reviewed_by": actor_id,
        "teacher_reviewed_at": reviewed_at,
        "updated_at": reviewed_at,
    }
    if isinstance(response.get("question_assignment"), dict):
        fields["question_assignment.manual_review_required"] = False
    return fields


async def _refresh_unpublished_review_state(
    tenant_db: Any,
    submission_id: str,
    *,
    now: datetime,
) -> tuple[str, Dict[str, Any], str]:
    """Recompute readiness after a teacher edit without publishing the result."""

    from services.exampen_submission_readiness import assess_submission_readiness

    readiness = await assess_submission_readiness(tenant_db, submission_id)
    review_state, state_update = _review_transition_from_readiness(
        readiness,
        now=now,
    )
    changed = await tenant_db["evalpen_submissions"].update_one(
        {
            "submission_id": submission_id,
            "publication_status": {"$ne": "published"},
        },
        {"$set": state_update},
    )
    if changed.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The result was published while the review was being saved",
        )
    return review_state, readiness, str(state_update["publication_status"])


@router.post(
    "/submissions/{submission_id}/document-review",
    summary="Confirm full answer-copy coverage review",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Submission not found"},
        409: {"description": "Review is stale or the result is already published"},
    },
)
async def confirm_document_coverage_review(
    submission_id: str,
    body: DocumentCoverageReviewRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Clear one submission-level coverage gate without changing marks.

    The run id fences this acknowledgement against a simultaneous reprocess:
    a teacher can only confirm the exact visual ledger currently on screen.
    """

    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id}
    )
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Submission {submission_id} not found",
        )
    _check_student_in_scope(str(submission.get("student_id") or ""), scoped_ids)
    if str(submission.get("publication_status") or "").lower() == "published":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Published results are immutable",
        )

    current_review = submission.get("document_review")
    if not isinstance(current_review, dict):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This submission has no document-coverage review to confirm",
        )
    if str(current_review.get("grading_run_id") or "") != body.grading_run_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The answer copy was reprocessed. Refresh before confirming it.",
        )

    now = datetime.now(timezone.utc)
    actor_id = str(current_user.get("user_id") or "unknown")
    accepted_review = {
        **current_review,
        "status": "accepted",
        "required": False,
        "accepted_at": now,
        "accepted_by": actor_id,
        "acceptance_note": body.note,
        "updated_at": now,
    }
    updated = await tenant_db["evalpen_submissions"].update_one(
        {
            "submission_id": submission_id,
            "publication_status": {"$ne": "published"},
            "document_review.grading_run_id": body.grading_run_id,
        },
        {
            "$set": {
                "document_review": accepted_review,
                "updated_at": now,
            },
            "$push": {
                "document_review_history": {
                    "action": "coverage_confirmed",
                    "grading_run_id": body.grading_run_id,
                    "actor_id": actor_id,
                    "note": body.note,
                    "timestamp": now,
                    "before": current_review,
                    "after": accepted_review,
                }
            },
        },
    )
    if updated.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The review changed while it was being confirmed. Refresh and retry.",
        )

    from services.exampen_submission_readiness import assess_submission_readiness

    readiness = await assess_submission_readiness(tenant_db, submission_id)
    review_state, state_update = _review_transition_from_readiness(
        readiness,
        now=now,
    )
    state_changed = await tenant_db["evalpen_submissions"].update_one(
        {
            "submission_id": submission_id,
            "publication_status": {"$ne": "published"},
        },
        {"$set": state_update},
    )
    if state_changed.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The result was published while this review was being confirmed",
        )
    publication_status = str(
        state_update.get("publication_status")
        or submission.get("publication_status")
        or "pending"
    )

    return {
        "submission_id": submission_id,
        "review_state": review_state,
        "document_review": accepted_review,
        "readiness": readiness,
        "publication": None,
        "publication_status": publication_status,
    }


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
    active_responses = await tenant_db["evalpen_detected_responses"].find(
        {
            "submission_id": submission_id,
            "superseded_at": {"$exists": False},
            "eval_status": {"$ne": "superseded"},
        },
        {
            "response_id": 1,
            "question_id": 1,
            "question_number": 1,
            "source_pages": 1,
            "answer_state": 1,
        },
    ).to_list(length=5000)
    regions_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for response in active_responses:
        for source_region in response.get("source_pages") or []:
            if not isinstance(source_region, dict):
                continue
            try:
                source_page_number = int(source_region.get("page_number"))
            except (TypeError, ValueError):
                continue
            regions_by_page.setdefault(source_page_number, []).append(
                {
                    **source_region,
                    "response_id": str(response.get("response_id") or ""),
                    "question_id": response.get("question_id"),
                    "question_number": response.get("question_number"),
                    "answer_state": response.get("answer_state"),
                }
            )
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
                regions=regions_by_page.get(
                    int(page_doc.get("page_number") or index + 1), []
                ),
            )
        )
    return SubmissionPagesAPI(
        submission_id=submission_id,
        total_pages=len(pages),
        pages=pages,
    )


class CollectionRosterItemAPI(BaseModel):
    """One expected student row for the collection / workspace roster."""

    student_id: str
    student_name: Optional[str] = None
    submission_id: Optional[str] = None
    status: str = "expected"
    source: Optional[str] = None
    last_activity: Optional[str] = None


class ExamRosterAPI(BaseModel):
    """Roster + collection progress for a conducted exam sitting."""

    exam_id: str
    expected_students: List[CollectionRosterItemAPI]
    total_expected: int = 0
    total_submitted: int = 0
    total_blocked: int = 0
    total_needs_review: int = 0
    total_ready: int = 0
    total_published: int = 0


@router.get(
    "/exams/{exam_id}/roster",
    response_model=ExamRosterAPI,
    summary="Get expected-student roster and collection status for an exam",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_exam_roster(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamRosterAPI:
    """Return the planned roster with live submission/processing status.

    Used by the ExamPen workspace collection monitor and student explorer.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    roster_ids = [
        str(student_id).strip()
        for student_id in (exam.get("roster") or [])
        if str(student_id).strip()
    ]
    if scoped_ids is not None:
        allowed = {str(item) for item in scoped_ids}
        roster_ids = [student_id for student_id in roster_ids if student_id in allowed]

    # Student display names (best-effort).
    name_by_id: Dict[str, str] = {}
    if roster_ids:
        try:
            cursor = tenant_db["students"].find(
                {"student_id": {"$in": roster_ids}},
                {"student_id": 1, "name": 1, "full_name": 1, "display_name": 1},
            )
            for doc in await cursor.to_list(length=max(len(roster_ids), 1)):
                sid = str(doc.get("student_id") or "").strip()
                if not sid:
                    continue
                name = (
                    doc.get("display_name")
                    or doc.get("full_name")
                    or doc.get("name")
                    or ""
                )
                if name:
                    name_by_id[sid] = str(name)
        except Exception:
            logger.debug("Student name lookup failed for exam %s roster", exam_id)

    submissions = await tenant_db["evalpen_submissions"].find(
        {"exam_id": exam_id},
        projection={
            "submission_id": 1,
            "student_id": 1,
            "source": 1,
            "publication_status": 1,
            "segmentation_status": 1,
            "review_state": 1,
            "updated_at": 1,
            "created_at": 1,
        },
    ).to_list(length=5000)
    if scoped_ids is not None:
        allowed = {str(item) for item in scoped_ids}
        submissions = [
            doc
            for doc in submissions
            if str(doc.get("student_id") or "") in allowed
        ]

    jobs = await tenant_db["exampen_processing_jobs"].find(
        {"exam_id": exam_id},
        projection={
            "submission_id": 1,
            "student_id": 1,
            "status": 1,
            "updated_at": 1,
        },
    ).to_list(length=5000)
    job_by_submission = {
        str(job.get("submission_id") or ""): job
        for job in jobs
        if job.get("submission_id")
    }

    # Prefer the latest submission per student.
    sub_by_student: Dict[str, Dict[str, Any]] = {}
    for doc in submissions:
        student_id = str(doc.get("student_id") or "").strip()
        if not student_id:
            continue
        existing = sub_by_student.get(student_id)
        if existing is None:
            sub_by_student[student_id] = doc
            continue
        existing_ts = existing.get("updated_at") or existing.get("created_at")
        new_ts = doc.get("updated_at") or doc.get("created_at")
        if new_ts and (not existing_ts or new_ts > existing_ts):
            sub_by_student[student_id] = doc

    # Include unexpected submitters who are not on the formal roster.
    all_student_ids = list(dict.fromkeys([*roster_ids, *sub_by_student.keys()]))

    # Response/eval status for submitted copies.
    submission_ids = [
        str(doc.get("submission_id"))
        for doc in sub_by_student.values()
        if doc.get("submission_id")
    ]
    blocked_submissions: set[str] = set()
    review_submissions: set[str] = set()
    ready_submissions: set[str] = set()
    if submission_ids:
        try:
            cursor = tenant_db["evalpen_detected_responses"].aggregate(
                [
                    {
                        "$match": {
                            "submission_id": {"$in": submission_ids},
                            "superseded_at": {"$exists": False},
                        }
                    },
                    {
                        "$group": {
                            "_id": "$submission_id",
                            "statuses": {"$addToSet": "$eval_status"},
                        }
                    },
                ]
            )
            for row in await cursor.to_list(length=max(len(submission_ids), 1)):
                sid = str(row.get("_id") or "")
                statuses = {str(s or "").lower() for s in (row.get("statuses") or [])}
                if "blocked" in statuses:
                    blocked_submissions.add(sid)
                elif "manual_review" in statuses:
                    review_submissions.add(sid)
                elif statuses and statuses.issubset(
                    {
                        "evaluated",
                        "evaluated_with_warnings",
                        "not_attempted",
                    }
                ):
                    ready_submissions.add(sid)
                elif any(s.startswith("evaluated") for s in statuses):
                    ready_submissions.add(sid)
        except Exception:
            logger.debug("Roster response status aggregation failed for exam %s", exam_id)

    items: List[CollectionRosterItemAPI] = []
    total_submitted = 0
    total_blocked = 0
    total_needs_review = 0
    total_ready = 0
    total_published = 0

    for student_id in all_student_ids:
        submission = sub_by_student.get(student_id)
        submission_id = (
            str(submission.get("submission_id")) if submission and submission.get("submission_id") else None
        )
        job = job_by_submission.get(submission_id or "")
        publication = str((submission or {}).get("publication_status") or "").lower()
        job_status = str((job or {}).get("status") or "").lower()
        review_state = str((submission or {}).get("review_state") or "").lower()
        source = str((submission or {}).get("source") or "").lower() or None
        if source and source not in {"pen", "camera", "mixed"}:
            source = "camera" if source in {"upload", "pdf", "image"} else source

        if not submission_id:
            status_value = "expected"
        elif publication == "published":
            status_value = "published"
            total_published += 1
            total_submitted += 1
            total_ready += 1
        elif submission_id in blocked_submissions or review_state == "blocked" or job_status in {
            "failed",
            "retryable_error",
            "enqueue_failed",
        }:
            status_value = "blocked"
            total_blocked += 1
            total_submitted += 1
        elif review_state == "needs_review" or submission_id in review_submissions or (
            job_status == "blocked_for_review"
            and submission_id not in blocked_submissions
        ):
            status_value = "review"
            total_needs_review += 1
            total_submitted += 1
        elif (
            submission_id in ready_submissions
            or job_status == "completed"
            or publication in {"ready", "unpublished"}
        ):
            status_value = "ready"
            total_ready += 1
            total_submitted += 1
        elif job_status in {
            "queued",
            "processing",
            "retryable_error",
            "enqueue_failed",
            "not_enqueued",
        } or str((submission or {}).get("segmentation_status") or "") in {
            "pending",
            "processing",
        }:
            status_value = "evaluating"
            total_submitted += 1
        else:
            status_value = "submitted"
            total_submitted += 1

        last_activity = _dt_to_iso(
            (job or {}).get("updated_at")
            or (submission or {}).get("updated_at")
            or (submission or {}).get("created_at")
        )
        items.append(
            CollectionRosterItemAPI(
                student_id=student_id,
                student_name=name_by_id.get(student_id),
                submission_id=submission_id,
                status=status_value,
                source=source if source in {"pen", "camera", "mixed"} else None,
                last_activity=last_activity,
            )
        )

    return ExamRosterAPI(
        exam_id=exam_id,
        expected_students=items,
        total_expected=len(items),
        total_submitted=total_submitted,
        total_blocked=total_blocked,
        total_needs_review=total_needs_review,
        total_ready=total_ready,
        total_published=total_published,
    )


@router.get(
    "/exams/{exam_id}/analytics",
    summary="Get PCR analytics at class, question, topic, and student level",
)
async def get_exam_analytics(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Build stable analytics from persisted question evaluations.

    Topic labels are teacher-owned metadata. Scores are never regenerated for
    analytics, so this endpoint is fast and produces the same result as the
    review and publication surfaces.
    """

    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    question_catalog = await _get_pcr_question_catalog(tenant_db, exam_id)
    if not question_catalog:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No finalized PCR questions found for exam {exam_id}",
        )

    taxonomy_docs = await tenant_db["evalpen_question_taxonomy"].find(
        {"exam_id": exam_id},
        {"_id": 0, "question_id": 1, "topic": 1, "sub_topic": 1},
    ).to_list(length=max(len(question_catalog), 100))
    taxonomy_by_question = {
        str(item.get("question_id") or ""): item
        for item in taxonomy_docs
        if str(item.get("question_id") or "")
    }

    questions: Dict[str, Dict[str, Any]] = {}
    for index, raw_question in enumerate(question_catalog):
        question_id = str(raw_question.get("question_id") or "")
        if not question_id:
            continue
        taxonomy = taxonomy_by_question.get(question_id, {})
        topic = str(
            taxonomy.get("topic")
            or raw_question.get("topic")
            or "Uncategorized"
        ).strip()
        sub_topic = str(
            taxonomy.get("sub_topic")
            or raw_question.get("sub_topic")
            or ""
        ).strip() or None
        questions[question_id] = {
            "question_id": question_id,
            "question_number": raw_question.get("question_number") or index + 1,
            "question_text": str(raw_question.get("question_text") or ""),
            "max_marks": float(raw_question.get("max_marks") or 0.0),
            "topic": topic,
            "sub_topic": sub_topic,
            "assessed_count": 0,
            "attempted_count": 0,
            "full_count": 0,
            "partial_count": 0,
            "zero_count": 0,
            "review_count": 0,
            "total_score": 0.0,
            "total_max": 0.0,
        }

    submission_query: Dict[str, Any] = {"exam_id": exam_id}
    if scoped_ids is not None:
        submission_query["student_id"] = {"$in": scoped_ids}
    submissions = await tenant_db["evalpen_submissions"].find(
        submission_query,
        {
            "_id": 0,
            "submission_id": 1,
            "student_id": 1,
            "publication_status": 1,
        },
    ).to_list(length=5000)
    submission_ids = [
        str(item.get("submission_id") or "")
        for item in submissions
        if str(item.get("submission_id") or "")
    ]

    responses = (
        await tenant_db["evalpen_detected_responses"]
        .find(
            {
                "submission_id": {"$in": submission_ids},
                "question_id": {"$in": list(questions)},
                "eval_status": {"$ne": "superseded"},
                "superseded_at": {"$exists": False},
            },
            {
                "_id": 0,
                "response_id": 1,
                "submission_id": 1,
                "question_id": 1,
                "is_missing_response": 1,
                "manual_review_required": 1,
                "flags": 1,
                "updated_at": 1,
                "created_at": 1,
            },
        )
        .sort([("updated_at", -1), ("created_at", -1)])
        .to_list(length=max(len(submission_ids) * max(len(questions), 1) * 2, 5000))
        if submission_ids
        else []
    )
    response_ids = [
        str(item.get("response_id") or "")
        for item in responses
        if str(item.get("response_id") or "")
    ]
    evaluations = (
        await tenant_db["evalpen_evaluations"]
        .find(
            {"response_id": {"$in": response_ids}},
            {
                "_id": 0,
                "evaluation_id": 1,
                "response_id": 1,
                "total_score": 1,
                "max_score": 1,
                "manual_review_required": 1,
                "flags": 1,
                "updated_at": 1,
                "created_at": 1,
            },
        )
        .sort([("updated_at", -1), ("created_at", -1)])
        .to_list(length=max(len(response_ids) * 2, 5000))
        if response_ids
        else []
    )
    evaluation_by_response: Dict[str, Dict[str, Any]] = {}
    for evaluation in evaluations:
        response_id = str(evaluation.get("response_id") or "")
        if response_id and response_id not in evaluation_by_response:
            evaluation_by_response[response_id] = evaluation

    def _has_unresolved_flags(document: Dict[str, Any]) -> bool:
        return any(
            isinstance(item, dict) and not is_flag_resolved(item)
            for item in document.get("flags") or []
        )

    student_rows: Dict[str, Dict[str, Any]] = {}
    for submission in submissions:
        submission_id = str(submission.get("submission_id") or "")
        student_id = str(submission.get("student_id") or "")
        if not submission_id or not student_id:
            continue
        student_rows[submission_id] = {
            "student_id": student_id,
            "submission_id": submission_id,
            "publication_status": submission.get("publication_status"),
            "total_score": 0.0,
            "total_max": 0.0,
            "review_count": 0,
            "assessed_count": 0,
            "topics": {},
        }

    seen_slots: set[tuple[str, str]] = set()
    for response in responses:
        response_id = str(response.get("response_id") or "")
        submission_id = str(response.get("submission_id") or "")
        question_id = str(response.get("question_id") or "")
        evaluation = evaluation_by_response.get(response_id)
        question = questions.get(question_id)
        student = student_rows.get(submission_id)
        slot = (submission_id, question_id)
        if not evaluation or not question or not student or slot in seen_slots:
            continue
        seen_slots.add(slot)

        maximum = float(evaluation.get("max_score") or question["max_marks"] or 0.0)
        score = max(0.0, min(float(evaluation.get("total_score") or 0.0), maximum))
        needs_review = bool(
            response.get("manual_review_required")
            or evaluation.get("manual_review_required")
            or _has_unresolved_flags(response)
            or _has_unresolved_flags(evaluation)
        )
        attempted = not bool(response.get("is_missing_response"))

        question["assessed_count"] += 1
        question["attempted_count"] += int(attempted)
        question["review_count"] += int(needs_review)
        question["total_score"] += score
        question["total_max"] += maximum
        if maximum > 0 and score >= maximum - 1e-6:
            question["full_count"] += 1
        elif score > 0:
            question["partial_count"] += 1
        else:
            question["zero_count"] += 1

        student["total_score"] += score
        student["total_max"] += maximum
        student["review_count"] += int(needs_review)
        student["assessed_count"] += 1
        topic_row = student["topics"].setdefault(
            question["topic"],
            {"score": 0.0, "max_score": 0.0},
        )
        topic_row["score"] += score
        topic_row["max_score"] += maximum

    question_rows: List[Dict[str, Any]] = []
    topic_rows_by_name: Dict[str, Dict[str, Any]] = {}
    for question in sorted(
        questions.values(),
        key=lambda item: (int(item.get("question_number") or 0), item["question_id"]),
    ):
        total_max = float(question.pop("total_max"))
        total_score = float(question.pop("total_score"))
        question["average_percent"] = (
            round((total_score / total_max) * 100, 1) if total_max > 0 else 0.0
        )
        question["average_score"] = (
            round(total_score / question["assessed_count"], 2)
            if question["assessed_count"]
            else 0.0
        )
        question_rows.append(question)
        topic_row = topic_rows_by_name.setdefault(
            question["topic"],
            {
                "topic": question["topic"],
                "question_count": 0,
                "assessed_answers": 0,
                "review_count": 0,
                "total_score": 0.0,
                "total_max": 0.0,
                "sub_topics": set(),
            },
        )
        topic_row["question_count"] += 1
        topic_row["assessed_answers"] += question["assessed_count"]
        topic_row["review_count"] += question["review_count"]
        topic_row["total_score"] += question["average_score"] * question["assessed_count"]
        topic_row["total_max"] += question["max_marks"] * question["assessed_count"]
        if question.get("sub_topic"):
            topic_row["sub_topics"].add(question["sub_topic"])

    topic_rows: List[Dict[str, Any]] = []
    for topic_row in topic_rows_by_name.values():
        total_score = float(topic_row.pop("total_score"))
        total_max = float(topic_row.pop("total_max"))
        topic_row["average_percent"] = (
            round((total_score / total_max) * 100, 1) if total_max > 0 else 0.0
        )
        topic_row["sub_topics"] = sorted(topic_row["sub_topics"])
        topic_rows.append(topic_row)
    topic_rows.sort(key=lambda item: (item["average_percent"], item["topic"]))

    student_results: List[Dict[str, Any]] = []
    for student in student_rows.values():
        topic_performance = []
        for topic, values in student.pop("topics").items():
            maximum = float(values["max_score"])
            topic_performance.append(
                {
                    "topic": topic,
                    "score": round(float(values["score"]), 2),
                    "max_score": round(maximum, 2),
                    "percent": (
                        round((float(values["score"]) / maximum) * 100, 1)
                        if maximum > 0
                        else 0.0
                    ),
                }
            )
        topic_performance.sort(key=lambda item: (-item["percent"], item["topic"]))
        strengths = [item for item in topic_performance if item["percent"] >= 70][:2]
        focus = sorted(topic_performance, key=lambda item: (item["percent"], item["topic"]))[:2]
        total_max = float(student["total_max"])
        percent = (
            round((float(student["total_score"]) / total_max) * 100, 1)
            if total_max > 0
            else 0.0
        )
        if not topic_performance:
            feedback = "No evaluated answers are available yet."
        else:
            strength_text = (
                ", ".join(f"{item['topic']} ({item['percent']:.0f}%)" for item in strengths)
                if strengths
                else "no topic is consistently secure yet"
            )
            focus_text = ", ".join(
                f"{item['topic']} ({item['percent']:.0f}%)" for item in focus
            )
            feedback = (
                f"Strongest area: {strength_text}. "
                f"Next focus: {focus_text}. "
                f"{student['review_count']} answer(s) still need teacher confirmation."
                if student["review_count"]
                else f"Strongest area: {strength_text}. Next focus: {focus_text}."
            )
        student.update(
            {
                "total_score": round(float(student["total_score"]), 2),
                "total_max": round(total_max, 2),
                "percent": percent,
                "topic_performance": topic_performance,
                "feedback": feedback,
            }
        )
        student_results.append(student)
    student_results.sort(key=lambda item: (-item["percent"], item["student_id"]))

    percentages = [item["percent"] for item in student_results if item["total_max"] > 0]
    return {
        "exam_id": exam_id,
        "scope": "PCR",
        "class_summary": {
            "submitted_students": len(submissions),
            "evaluated_students": len(percentages),
            "average_percent": round(sum(percentages) / len(percentages), 1) if percentages else 0.0,
            "highest_percent": max(percentages) if percentages else 0.0,
            "lowest_percent": min(percentages) if percentages else 0.0,
            "questions": len(question_rows),
            "review_answers": sum(item["review_count"] for item in question_rows),
        },
        "questions": question_rows,
        "topics": topic_rows,
        "students": student_results,
    }


@router.put(
    "/exams/{exam_id}/questions/{question_id}/taxonomy",
    summary="Assign a persisted topic and sub-topic to a frozen PCR question",
)
async def update_question_taxonomy(
    exam_id: str,
    question_id: str,
    body: QuestionTaxonomyUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    tenant_db = await _get_tenant_db(db, current_user)
    catalog = await _get_pcr_question_catalog(tenant_db, exam_id)
    if question_id not in {
        str(question.get("question_id") or "") for question in catalog
    }:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question is not part of this finalized exam",
        )
    actor_id = str(current_user.get("user_id") or current_user.get("_id") or "unknown")
    now = datetime.now(timezone.utc)
    await tenant_db["evalpen_question_taxonomy"].update_one(
        {"exam_id": exam_id, "question_id": question_id},
        {
            "$set": {
                "topic": body.topic,
                "sub_topic": body.sub_topic,
                "updated_at": now,
                "updated_by": actor_id,
            },
            "$setOnInsert": {
                "created_at": now,
                "created_by": actor_id,
            },
        },
        upsert=True,
    )
    return {
        "exam_id": exam_id,
        "question_id": question_id,
        "topic": body.topic,
        "sub_topic": body.sub_topic,
        "updated_at": now.isoformat(),
    }


@router.post(
    "/exams/{exam_id}/taxonomy/auto-tag",
    summary="Classify all untagged PCR questions in one persisted AI batch",
)
async def auto_tag_exam_questions(
    exam_id: str,
    body: AutoTagQuestionsRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    tenant_db = await _get_tenant_db(db, current_user)
    catalog = await _get_pcr_question_catalog(tenant_db, exam_id)
    if not catalog:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No finalized PCR questions found for exam {exam_id}",
        )

    existing_docs = await tenant_db["evalpen_question_taxonomy"].find(
        {"exam_id": exam_id},
        {"_id": 0, "question_id": 1, "topic": 1, "sub_topic": 1, "source": 1},
    ).to_list(length=max(len(catalog), 100))
    existing_by_question = {
        str(item.get("question_id") or ""): item
        for item in existing_docs
        if str(item.get("question_id") or "")
    }
    candidates = [
        question
        for question in catalog
        if body.replace_existing
        or not str(
            (existing_by_question.get(str(question.get("question_id") or "")) or {}).get("topic")
            or ""
        ).strip()
    ]
    if not candidates:
        return {
            "exam_id": exam_id,
            "tagged_count": 0,
            "preserved_count": len(existing_by_question),
            "message": "Every question already has a persisted topic",
        }

    exam = await tenant_db["exampen_exams"].find_one(
        {"exam_id": exam_id},
        {
            "_id": 0,
            "title": 1,
            "exam_title": 1,
            "subject": 1,
            "exam_subject": 1,
            "standard": 1,
            "class_name": 1,
            "grade": 1,
        },
    ) or {}
    classifier_questions = [
        {
            "id": str(question.get("question_id") or ""),
            "question_id": str(question.get("question_id") or ""),
            "question_number": question.get("question_number") or index + 1,
            "question_text": str(question.get("question_text") or ""),
            "text": str(question.get("question_text") or ""),
            "points": float(question.get("max_marks") or 0.0),
        }
        for index, question in enumerate(candidates)
    ]

    try:
        from services.tally_question_map_service import build_tally_question_map

        classification = await build_tally_question_map(
            tally_document_id=f"evalpen-taxonomy:{exam_id}",
            source_document_id=exam_id,
            questions=classifier_questions,
            subject=str(
                exam.get("subject")
                or exam.get("exam_subject")
                or exam.get("title")
                or exam.get("exam_title")
                or ""
            ).strip()
            or None,
            standard=str(
                exam.get("standard")
                or exam.get("class_name")
                or exam.get("grade")
                or ""
            ).strip()
            or None,
            generated_by=str(
                current_user.get("user_id") or current_user.get("_id") or "unknown"
            ),
        )
    except Exception as exc:
        logger.error(
            "Automatic question taxonomy failed for exam %s: %s",
            exam_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Automatic topic classification is temporarily unavailable",
        ) from exc

    valid_question_ids = {
        str(question.get("question_id") or "") for question in candidates
    }
    actor_id = str(current_user.get("user_id") or current_user.get("_id") or "unknown")
    now = datetime.now(timezone.utc)
    tagged: List[Dict[str, Any]] = []
    for item in classification.get("items") or []:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or "")
        topic = " ".join(str(item.get("topic") or "").split())[:120]
        sub_topic = " ".join(str(item.get("sub_topic") or "").split())[:120]
        source = str(item.get("source") or "")
        if (
            question_id not in valid_question_ids
            or not topic
            or topic.lower() in {"unmapped", "uncategorized"}
            or source != "ai"
        ):
            continue
        document = {
            "exam_id": exam_id,
            "question_id": question_id,
            "topic": topic,
            "sub_topic": sub_topic or None,
            "source": "ai_batch",
            "confidence": float(item.get("confidence") or 0.0),
            "updated_at": now,
            "updated_by": actor_id,
        }
        await tenant_db["evalpen_question_taxonomy"].update_one(
            {"exam_id": exam_id, "question_id": question_id},
            {
                "$set": document,
                "$setOnInsert": {
                    "created_at": now,
                    "created_by": actor_id,
                },
            },
            upsert=True,
        )
        tagged.append(
            {
                "question_id": question_id,
                "topic": topic,
                "sub_topic": sub_topic or None,
                "confidence": document["confidence"],
            }
        )

    if not tagged:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The topic classifier returned no reliable labels. Check the "
                "configured AI provider and try again; no placeholder tags were saved."
            ),
        )

    await tenant_db["evalpen_question_taxonomy_audit"].insert_one(
        {
            "audit_id": f"QTA-{uuid.uuid4().hex[:20]}",
            "exam_id": exam_id,
            "action": "replace_ai_taxonomy" if body.replace_existing else "tag_missing_questions",
            "question_ids": [item["question_id"] for item in tagged],
            "tagged_count": len(tagged),
            "preserved_count": len(catalog) - len(candidates),
            "actor_id": actor_id,
            "created_at": now,
        }
    )
    return {
        "exam_id": exam_id,
        "tagged_count": len(tagged),
        "preserved_count": len(catalog) - len(candidates),
        "items": tagged,
        "generated_at": now.isoformat(),
    }


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
            load_exampen("pcr.storage")
        except ImportError:
            pcr_available = False

        # --- DCR data ---
        dcr_available = True
        try:
            load_exampen("dcr.repository")
        except ImportError:
            dcr_available = False

        # Collect per-student data
        student_results: Dict[str, ExamResultStudentAPI] = {}

        # PCR: find submissions for this exam, then aggregate evaluations
        if pcr_available:
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
            submissions = await submissions_cursor.to_list(length=5000)

            # Read responses/evaluations in two bounded batch queries. The
            # former nested repository loop performed one response query per
            # student and then one evaluation query per question, which made
            # Results latency grow as students x questions.
            submission_ids = [
                str(item.get("submission_id") or "")
                for item in submissions
                if str(item.get("submission_id") or "")
            ]
            responses = (
                await tenant_db["evalpen_detected_responses"]
                .find(
                    {
                        "submission_id": {"$in": submission_ids},
                        "eval_status": {"$ne": "superseded"},
                        "superseded_at": {"$exists": False},
                    },
                    {
                        "response_id": 1,
                        "submission_id": 1,
                        "question_id": 1,
                        "eval_status": 1,
                    },
                )
                .to_list(length=max(len(submission_ids) * 1000, 5000))
                if submission_ids
                else []
            )
            responses_by_submission: Dict[str, List[Dict[str, Any]]] = {}
            response_ids: List[str] = []
            for response in responses:
                owner_submission_id = str(response.get("submission_id") or "")
                response_id = str(response.get("response_id") or "")
                responses_by_submission.setdefault(owner_submission_id, []).append(
                    response
                )
                if response_id:
                    response_ids.append(response_id)

            evaluation_docs = (
                await tenant_db["evalpen_evaluations"]
                .find(
                    {"response_id": {"$in": response_ids}},
                    {
                        "response_id": 1,
                        "total_score": 1,
                        "max_score": 1,
                    },
                )
                .to_list(length=max(len(response_ids) * 2, 5000))
                if response_ids
                else []
            )
            evaluations_by_response = {
                str(item.get("response_id") or ""): item
                for item in evaluation_docs
                if str(item.get("response_id") or "")
            }

            for sub in submissions:
                student_id = sub.get("student_id", "")
                submission_id = sub.get("submission_id", "")
                submission_responses = responses_by_submission.get(
                    str(submission_id), []
                )

                pcr_total = 0.0
                evaluated_max = 0.0
                blocked_count = 0
                scored_question_ids: set[str] = set()

                for resp in submission_responses:
                    response_id = resp.get("response_id", "")

                    # Count blocked
                    if resp.get("eval_status") == "blocked":
                        blocked_count += 1

                    ev = evaluations_by_response.get(str(response_id))
                    if ev:
                        question_id = str(resp.get("question_id") or "")
                        if (
                            catalog_question_ids
                            and question_id not in catalog_question_ids
                        ):
                            # Stray/unassigned OCR evidence is review work,
                            # never an extra marks-bearing paper question.
                            if resp.get("eval_status") != "blocked":
                                blocked_count += 1
                            continue
                        # Never let duplicate OCR segments make one paper
                        # question count twice.  New jobs block duplicates;
                        # this also repairs historical result aggregation.
                        if (
                            question_id
                            and question_id in catalog_question_ids
                            and question_id in scored_question_ids
                        ):
                            continue
                        pcr_total += _safe_score(ev.get("total_score"))
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
    review_lease_token: Optional[str] = None
    review_submission_id = ""
    release_review_lease = None

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

        # Tutor scoping and publication freeze are resolved through the
        # response's canonical submission, even when the evaluation already
        # carries a student_id.
        eval_student_id = existing.get("student_id")
        _resp_doc = await tenant_db["evalpen_detected_responses"].find_one(
            {"response_id": existing.get("response_id", "")},
            projection={
                "submission_id": 1,
                "student_id": 1,
                "flags": 1,
                "question_assignment": 1,
            },
        )
        _sub_doc = None
        if _resp_doc:
            review_submission_id = str(_resp_doc.get("submission_id") or "")
            _sub_doc = await tenant_db["evalpen_submissions"].find_one(
                {"submission_id": review_submission_id},
            )
            eval_student_id = eval_student_id or (_sub_doc or {}).get("student_id")
        if (
            (_sub_doc or {}).get("publication_status") == "published"
            and not body.amend_published
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This result is already published. Confirm that you want "
                    "to save an audited published-score amendment."
                ),
            )
        if eval_student_id:
            _check_student_in_scope(eval_student_id, scoped_ids)

        if review_submission_id:
            from services.exampen_review_lease import (
                SubmissionReviewBusyError,
                acquire_submission_review_lease,
                release_submission_review_lease,
            )

            release_review_lease = release_submission_review_lease
            try:
                review_lease_token = await acquire_submission_review_lease(
                    tenant_db,
                    review_submission_id,
                    actor_id=str(current_user.get("user_id") or "unknown"),
                    operation="score_amendment",
                )
            except SubmissionReviewBusyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(exc),
                ) from exc

            # Publication may have completed after the first read but before
            # this mutation acquired ownership.
            _sub_doc = await tenant_db["evalpen_submissions"].find_one(
                {"submission_id": review_submission_id}
            )
            if (
                (_sub_doc or {}).get("publication_status") == "published"
                and not body.amend_published
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "This result was published while you were editing it. "
                        "Refresh and confirm a published-score amendment."
                    ),
                )

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

        teacher_reviewed_at = datetime.now(timezone.utc)
        await tenant_db["evalpen_evaluations"].update_one(
            {"evaluation_id": evaluation_id},
            {
                "$set": {
                    "manual_review_required": False,
                    "manual_review_reason": None,
                    "flags": _resolve_review_flags(
                        existing.get("flags"),
                        actor_id=str(actor_id),
                        resolved_at=teacher_reviewed_at,
                        reason=body.reason,
                    ),
                    "teacher_review_status": "approved",
                    "teacher_reviewed": True,
                    "teacher_reviewed_by": actor_id,
                    "teacher_reviewed_at": teacher_reviewed_at,
                    "updated_at": teacher_reviewed_at,
                }
            },
        )
        if _resp_doc:
            await tenant_db["evalpen_detected_responses"].update_one(
                {"response_id": existing.get("response_id", "")},
                {
                    "$set": _teacher_reviewed_response_fields(
                        _resp_doc,
                        actor_id=str(actor_id),
                        reviewed_at=teacher_reviewed_at,
                        reason=body.reason,
                    )
                },
            )

        logger.info(
            "Score override on evaluation %s: %.2f -> %.2f by %s (%s)",
            evaluation_id,
            old_score,
            body.new_score,
            actor_id,
            body.reason,
        )

        amendment: Dict[str, Any] = {}
        if (
            (_sub_doc or {}).get("publication_status") == "published"
            and review_lease_token
        ):
            try:
                amendment = await _amend_published_submission_snapshot(
                    tenant_db,
                    _sub_doc,
                    evaluation_id=evaluation_id,
                    actor_id=str(actor_id),
                    reason=body.reason,
                    review_lease_token=review_lease_token,
                )
            except Exception:
                # Mongo deployments used by schools do not always support
                # cross-collection transactions.  Keep the released snapshot
                # and evaluation consistent if the second write cannot land.
                rolled_back = await eval_repo.override_score(
                    evaluation_id,
                    new_total_score=old_score,
                    actor_id=str(actor_id),
                    reason=(
                        "Automatic rollback: published result snapshot "
                        "amendment did not complete"
                    ),
                )
                if not rolled_back:
                    logger.critical(
                        "Published score amendment rollback failed for %s",
                        evaluation_id,
                    )
                raise

        review_state = None
        readiness: Dict[str, Any] = {}
        publication_status = str((_sub_doc or {}).get("publication_status") or "pending")
        if publication_status != "published" and review_submission_id:
            review_state, readiness, publication_status = (
                await _refresh_unpublished_review_state(
                    tenant_db,
                    review_submission_id,
                    now=teacher_reviewed_at,
                )
            )

        return {
            "evaluation_id": evaluation_id,
            "previous_score": old_score,
            "new_score": body.new_score,
            "actor_id": actor_id,
            "overridden_at": datetime.now(timezone.utc).isoformat(),
            "review_state": review_state,
            "readiness": readiness,
            "publication_status": publication_status,
            **amendment,
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
    finally:
        if (
            review_lease_token
            and review_submission_id
            and release_review_lease is not None
        ):
            try:
                await release_review_lease(
                    tenant_db,
                    review_submission_id,
                    review_lease_token,
                )
            except Exception as exc:
                logger.error(
                    "Failed to release score-amendment lease for %s: %s",
                    review_submission_id,
                    exc,
                    exc_info=True,
                )


@router.post(
    "/responses/{response_id}/manual-resolution",
    summary="Resolve a mapped response that automated grading could not score",
)
async def resolve_response_manually(
    response_id: str,
    body: ResponseManualResolutionRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Persist an audited teacher score, clear review flags, and recalculate readiness."""

    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    response = await tenant_db["evalpen_detected_responses"].find_one(
        {"response_id": response_id}
    )
    if response is None:
        raise HTTPException(status_code=404, detail=f"Response {response_id} not found")

    submission_id = str(response.get("submission_id") or "")
    student_id = str(response.get("student_id") or "")
    if student_id:
        _check_student_in_scope(student_id, scoped_ids)
    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id}
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="The answer-copy submission no longer exists")
    if str(submission.get("publication_status") or "").lower() == "published":
        raise HTTPException(
            status_code=409,
            detail="Published results must be changed through the audited amendment workflow",
        )

    exam_id = str(response.get("exam_id") or submission.get("exam_id") or "")
    question_id = str(response.get("question_id") or "")
    catalog = await _get_pcr_question_catalog(tenant_db, exam_id)
    question = next(
        (item for item in catalog if str(item.get("question_id") or "") == question_id),
        None,
    )
    if question is None:
        raise HTTPException(
            status_code=409,
            detail="This response is not assigned to a finalized exam question",
        )
    max_score = float(question.get("max_marks") or 0.0)
    awarded_marks = float(body.awarded_marks)
    if max_score <= 0:
        raise HTTPException(status_code=409, detail="The finalized question has no valid maximum mark")
    if not math.isfinite(awarded_marks) or awarded_marks > max_score:
        raise HTTPException(
            status_code=400,
            detail=f"Awarded marks must be between 0 and {max_score:g}",
        )

    from services.exampen_review_lease import (
        SubmissionReviewBusyError,
        acquire_submission_review_lease,
        release_submission_review_lease,
    )

    actor_id = str(current_user.get("user_id") or current_user.get("_id") or "unknown")
    try:
        lease_token = await acquire_submission_review_lease(
            tenant_db,
            submission_id,
            actor_id=actor_id,
            operation="manual_response_resolution",
        )
    except SubmissionReviewBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        current_submission = await tenant_db["evalpen_submissions"].find_one(
            {"submission_id": submission_id}
        )
        if str((current_submission or {}).get("publication_status") or "").lower() == "published":
            raise HTTPException(
                status_code=409,
                detail="This result was published while the review was being saved",
            )

        existing = await tenant_db["evalpen_evaluations"].find_one(
            {"response_id": response_id}
        )
        if existing and existing.get("criterion_marks"):
            raise HTTPException(
                status_code=409,
                detail="This response has criterion marks. Use Edit marks to preserve the locked marking plan.",
            )

        now = datetime.now(timezone.utc)
        evaluation_id = str((existing or {}).get("evaluation_id") or f"eval-{uuid.uuid4().hex}")
        previous_score = (existing or {}).get("total_score")
        evaluation_fields: Dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "response_id": response_id,
            "submission_id": submission_id,
            "exam_id": exam_id,
            "student_id": student_id,
            "question_id": question_id,
            "eval_path": "teacher_manual_resolution",
            "model_used": "teacher",
            "total_score": awarded_marks,
            "max_score": max_score,
            "scoreable_max": max_score,
            "step_marks": [],
            "criterion_marks": [],
            "overall_feedback": body.reason,
            "teacher_feedback": body.reason,
            "reference_solution": question.get("reference_solution"),
            "manual_review_required": False,
            "manual_review_reason": None,
            "flags": _resolve_review_flags(
                (existing or {}).get("flags"),
                actor_id=actor_id,
                resolved_at=now,
                reason=body.reason,
            ),
            "teacher_review_status": "approved",
            "teacher_reviewed": True,
            "teacher_reviewed_by": actor_id,
            "teacher_reviewed_at": now,
            "updated_at": now,
        }
        audit_entry = {
            "actor_id": actor_id,
            "timestamp": now,
            "action": "manual_response_resolution",
            "before": {"total_score": previous_score},
            "after": {"total_score": awarded_marks, "max_score": max_score},
            "reason": body.reason,
        }
        if existing:
            await tenant_db["evalpen_evaluations"].update_one(
                {"_id": existing["_id"]},
                {"$set": evaluation_fields, "$push": {"audit_trail": audit_entry}},
            )
        else:
            await tenant_db["evalpen_evaluations"].insert_one(
                {
                    **evaluation_fields,
                    "created_at": now,
                    "audit_trail": [audit_entry],
                    "token_usage": {},
                    "raw_llm_response": None,
                }
            )

        await tenant_db["evalpen_detected_responses"].update_one(
            {"_id": response["_id"]},
            {
                "$set": {
                    "eval_status": "evaluated",
                    "manual_review_required": False,
                    "manual_review_reason": None,
                    "question_assignment.manual_review_required": False,
                    "flags": _resolve_review_flags(
                        response.get("flags"),
                        actor_id=actor_id,
                        resolved_at=now,
                        reason=body.reason,
                    ),
                    "teacher_review_status": "approved",
                    "teacher_reviewed_by": actor_id,
                    "teacher_reviewed_at": now,
                    "updated_at": now,
                }
            },
        )
        await tenant_db["evalpen_teacher_review_audit"].insert_one(
            {
                "audit_id": f"TRA-{uuid.uuid4().hex[:20]}",
                "evaluation_id": evaluation_id,
                "response_id": response_id,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "student_id": student_id,
                "action": "manual_response_resolution",
                "before": {"total_score": previous_score},
                "after": {"total_score": awarded_marks, "max_score": max_score},
                "reason": body.reason,
                "actor_id": actor_id,
                "created_at": now,
            }
        )
        review_state, readiness, publication_status = await _refresh_unpublished_review_state(
            tenant_db,
            submission_id,
            now=now,
        )
        return {
            "response_id": response_id,
            "evaluation_id": evaluation_id,
            "total_score": awarded_marks,
            "max_score": max_score,
            "review_state": review_state,
            "publication_status": publication_status,
            "readiness": readiness,
        }
    finally:
        await release_submission_review_lease(tenant_db, submission_id, lease_token)


@router.post(
    "/evaluations/{evaluation_id}/approve-review",
    summary="Approve a nonblocking AI review or mark it fully correct",
)
async def approve_evaluation_review(
    evaluation_id: str,
    body: EvaluationApprovalRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Resolve a visual/manual warning without pretending a blocker is safe."""

    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    evaluation = await tenant_db["evalpen_evaluations"].find_one(
        {"evaluation_id": evaluation_id}
    )
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation {evaluation_id} not found",
        )
    response = await tenant_db["evalpen_detected_responses"].find_one(
        {"response_id": str(evaluation.get("response_id") or "")}
    )
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The evaluated response no longer exists",
        )
    student_id = str(evaluation.get("student_id") or response.get("student_id") or "")
    if student_id:
        _check_student_in_scope(student_id, scoped_ids)
    submission_id = str(response.get("submission_id") or "")
    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id}
    )
    if (submission or {}).get("publication_status") == "published":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Published results must be amended through the audited marks editor",
        )

    def _unresolved_blockers(document: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            item
            for item in document.get("flags") or []
            if isinstance(item, dict)
            and not is_flag_resolved(item)
            and str(item.get("severity") or "").lower() == "blocking"
        ]

    blockers = _unresolved_blockers(response) + _unresolved_blockers(evaluation)
    if blockers:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This answer has an unresolved evidence blocker. Correct the "
                "answer assignment or use the marks editor before approving it."
            ),
        )

    actor_id = str(current_user.get("user_id") or current_user.get("_id") or "unknown")
    now = datetime.now(timezone.utc)

    def _resolve_nonblocking_flags(raw_flags: Any) -> List[Dict[str, Any]]:
        resolved: List[Dict[str, Any]] = []
        for raw_flag in raw_flags or []:
            if not isinstance(raw_flag, dict):
                continue
            flag = (
                dict(raw_flag)
                if is_flag_resolved(raw_flag)
                else resolve_flag(
                    raw_flag,
                    actor_id=actor_id,
                    resolved_at=now,
                    reason=body.reason,
                )
            )
            resolved.append(flag)
        return resolved

    previous_score = float(evaluation.get("total_score") or 0.0)
    max_score = float(evaluation.get("max_score") or 0.0)
    criterion_marks = [
        dict(item)
        for item in evaluation.get("criterion_marks") or []
        if isinstance(item, dict)
    ]
    step_marks = [
        dict(item)
        for item in evaluation.get("step_marks") or []
        if isinstance(item, dict)
    ]
    if body.award_full_marks:
        for criterion in criterion_marks:
            criterion["marks_awarded"] = float(criterion.get("max_marks") or 0.0)
            criterion["credit_basis"] = "direct_evidence"
            criterion["teacher_confirmed"] = True
            criterion["rationale"] = (
                "Teacher verified this criterion against the original answer copy "
                "and confirmed it as correct."
            )
        criterion_maxima = {
            str(item.get("criterion_id") or ""): float(item.get("max_marks") or 0.0)
            for item in criterion_marks
            if str(item.get("criterion_id") or "")
        }
        for step in step_marks:
            criterion_id = str(step.get("criterion_id") or "")
            if criterion_id in criterion_maxima:
                step["marks_awarded"] = criterion_maxima[criterion_id]
                step["credit_basis"] = "direct_evidence"
                step["teacher_confirmed"] = True
                step["rationale"] = (
                    "Teacher verified this step against the original answer copy "
                    "and confirmed it as correct."
                )
        new_score = max_score
    else:
        new_score = previous_score

    evaluation_update: Dict[str, Any] = {
        "total_score": new_score,
        "manual_review_required": False,
        "manual_review_reason": None,
        "flags": _resolve_nonblocking_flags(evaluation.get("flags")),
        "teacher_review_status": "approved",
        "teacher_reviewed": True,
        "teacher_reviewed_by": actor_id,
        "teacher_reviewed_at": now,
        "updated_at": now,
    }
    if body.award_full_marks:
        evaluation_update["overall_feedback"] = (
            "Your teacher reviewed the original answer and confirmed this answer as correct."
        )
        evaluation_update["teacher_feedback"] = body.reason
    if criterion_marks:
        evaluation_update["criterion_marks"] = criterion_marks
    if step_marks:
        evaluation_update["step_marks"] = step_marks

    await tenant_db["evalpen_evaluations"].update_one(
        {"_id": evaluation["_id"]},
        {"$set": evaluation_update},
    )
    await tenant_db["evalpen_detected_responses"].update_one(
        {"_id": response["_id"]},
        {
            "$set": {
                "eval_status": "evaluated",
                "manual_review_required": False,
                "manual_review_reason": None,
                "question_assignment.manual_review_required": False,
                "flags": _resolve_nonblocking_flags(response.get("flags")),
                "teacher_review_status": "approved",
                "teacher_reviewed_by": actor_id,
                "teacher_reviewed_at": now,
                "updated_at": now,
            }
        },
    )
    await tenant_db["evalpen_teacher_review_audit"].insert_one(
        {
            "audit_id": f"TRA-{uuid.uuid4().hex[:20]}",
            "evaluation_id": evaluation_id,
            "response_id": response.get("response_id"),
            "submission_id": submission_id,
            "exam_id": response.get("exam_id"),
            "student_id": student_id,
            "action": "award_full_marks" if body.award_full_marks else "approve_current_score",
            "before": {
                "total_score": previous_score,
                "manual_review_required": bool(
                    evaluation.get("manual_review_required")
                    or response.get("manual_review_required")
                ),
                "overall_feedback": evaluation.get("overall_feedback"),
                "criterion_marks": evaluation.get("criterion_marks"),
            },
            "after": {
                "total_score": new_score,
                "manual_review_required": False,
                "overall_feedback": evaluation_update.get(
                    "overall_feedback",
                    evaluation.get("overall_feedback"),
                ),
                "criterion_marks": criterion_marks,
            },
            "reason": body.reason,
            "actor_id": actor_id,
            "created_at": now,
        }
    )

    from services.exampen_submission_readiness import assess_submission_readiness

    readiness = await assess_submission_readiness(tenant_db, submission_id)
    review_state, state_update = _review_transition_from_readiness(
        readiness,
        now=now,
    )
    state_changed = await tenant_db["evalpen_submissions"].update_one(
        {
            "submission_id": submission_id,
            "publication_status": {"$ne": "published"},
        },
        {"$set": state_update},
    )
    if state_changed.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The result was published while this review was being approved",
        )
    publication_status = str(
        state_update.get("publication_status")
        or (submission or {}).get("publication_status")
        or "pending"
    )

    return {
        "evaluation_id": evaluation_id,
        "submission_id": submission_id,
        "previous_score": previous_score,
        "new_score": new_score,
        "review_status": "approved",
        "review_state": review_state,
        "readiness": readiness,
        "publication": None,
        "publication_status": publication_status,
        "awarded_full_marks": body.award_full_marks,
        "approved_at": now.isoformat(),
    }


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
    review_lease_token: Optional[str] = None
    review_submission_id = ""
    release_review_lease = None
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
        response_doc = await tenant_db["evalpen_detected_responses"].find_one(
            {"response_id": existing.get("response_id", "")},
            projection={
                "submission_id": 1,
                "student_id": 1,
                "flags": 1,
                "question_assignment": 1,
            },
        )
        eval_student_id = eval_student_id or (response_doc or {}).get("student_id")
        submission_doc = None
        if response_doc:
            review_submission_id = str(response_doc.get("submission_id") or "")
            submission_doc = await tenant_db["evalpen_submissions"].find_one(
                {"submission_id": review_submission_id},
            )
            eval_student_id = eval_student_id or (submission_doc or {}).get("student_id")
        if (
            (submission_doc or {}).get("publication_status") == "published"
            and not body.amend_published
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This result is already published. Confirm that you want "
                    "to save an audited published-score amendment."
                ),
            )
        if eval_student_id:
            _check_student_in_scope(str(eval_student_id), scoped_ids)

        if review_submission_id:
            from services.exampen_review_lease import (
                SubmissionReviewBusyError,
                acquire_submission_review_lease,
                release_submission_review_lease,
            )

            release_review_lease = release_submission_review_lease
            try:
                review_lease_token = await acquire_submission_review_lease(
                    tenant_db,
                    review_submission_id,
                    actor_id=str(current_user.get("user_id") or "unknown"),
                    operation="criterion_score_amendment",
                )
            except SubmissionReviewBusyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(exc),
                ) from exc

            submission_doc = await tenant_db["evalpen_submissions"].find_one(
                {"submission_id": review_submission_id}
            )
            if (
                (submission_doc or {}).get("publication_status") == "published"
                and not body.amend_published
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "This result was published while you were editing it. "
                        "Refresh and confirm a published-score amendment."
                    ),
                )

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

        teacher_reviewed_at = datetime.now(timezone.utc)
        await tenant_db["evalpen_evaluations"].update_one(
            {"evaluation_id": evaluation_id},
            {
                "$set": {
                    "manual_review_required": False,
                    "manual_review_reason": None,
                    "flags": _resolve_review_flags(
                        existing.get("flags"),
                        actor_id=str(actor_id),
                        resolved_at=teacher_reviewed_at,
                        reason=body.reason,
                    ),
                    "teacher_review_status": "approved",
                    "teacher_reviewed": True,
                    "teacher_reviewed_by": actor_id,
                    "teacher_reviewed_at": teacher_reviewed_at,
                    "updated_at": teacher_reviewed_at,
                }
            },
        )
        await tenant_db["evalpen_detected_responses"].update_one(
            {"response_id": existing.get("response_id", "")},
            {
                "$set": _teacher_reviewed_response_fields(
                    response_doc or {},
                    actor_id=str(actor_id),
                    reviewed_at=teacher_reviewed_at,
                    reason=body.reason,
                )
            },
        )

        amendment: Dict[str, Any] = {}
        if (
            (submission_doc or {}).get("publication_status") == "published"
            and review_lease_token
        ):
            try:
                amendment = await _amend_published_submission_snapshot(
                    tenant_db,
                    submission_doc,
                    evaluation_id=evaluation_id,
                    actor_id=str(actor_id),
                    reason=body.reason,
                    review_lease_token=review_lease_token,
                )
            except Exception:
                previous_marks = {
                    str(item.get("criterion_id") or ""): float(
                        item.get("marks_awarded") or 0.0
                    )
                    for item in raw_criteria
                    if isinstance(item, dict) and item.get("criterion_id")
                }
                rolled_back = await eval_repo.override_criterion_marks(
                    evaluation_id,
                    marks_by_criterion=previous_marks,
                    actor_id=str(actor_id),
                    reason=(
                        "Automatic rollback: published result snapshot "
                        "amendment did not complete"
                    ),
                )
                if rolled_back is None:
                    logger.critical(
                        "Published criterion amendment rollback failed for %s",
                        evaluation_id,
                    )
                raise

        review_state = None
        readiness: Dict[str, Any] = {}
        publication_status = str(
            (submission_doc or {}).get("publication_status") or "pending"
        )
        if publication_status != "published" and review_submission_id:
            review_state, readiness, publication_status = (
                await _refresh_unpublished_review_state(
                    tenant_db,
                    review_submission_id,
                    now=teacher_reviewed_at,
                )
            )

        return {
            "evaluation_id": evaluation_id,
            "total_score": updated["total_score"],
            "max_score": updated.get("max_score"),
            "criterion_marks": updated["criterion_marks"],
            "actor_id": actor_id,
            "review_state": review_state,
            "readiness": readiness,
            "publication_status": publication_status,
            **amendment,
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
    finally:
        if (
            review_lease_token
            and review_submission_id
            and release_review_lease is not None
        ):
            try:
                await release_review_lease(
                    tenant_db,
                    review_submission_id,
                    review_lease_token,
                )
            except Exception as exc:
                logger.error(
                    "Failed to release criterion-amendment lease for %s: %s",
                    review_submission_id,
                    exc,
                    exc_info=True,
                )


async def _correct_response_assignment_impl(
    submission_id: str,
    body: ResponseAssignmentCorrectionRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id}
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    _check_student_in_scope(str(submission.get("student_id") or ""), scoped_ids)
    if submission.get("publication_status") == "published":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Published results are immutable; create a formal result revision instead",
        )
    active_processing_job = await tenant_db["exampen_processing_jobs"].find_one(
        {
            "submission_id": submission_id,
            "status": {"$in": ["queued", "processing"]},
        },
        {"status": 1},
    )
    if active_processing_job is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Wait for answer-copy processing to finish before correcting response ownership"
            ),
        )

    exam_id = str(submission.get("exam_id") or "")
    student_id = str(submission.get("student_id") or "")
    actor_id = str(current_user.get("user_id") or "unknown")
    now = datetime.now(timezone.utc)
    questions = await tenant_db["evalpen_questions"].find(
        {"exam_id": exam_id}
    ).to_list(length=1000)
    catalog = {
        str(question.get("question_id") or ""): question
        for question in questions
        if str(question.get("question_id") or "")
    }

    requested_question_ids = {
        value
        for value in [body.question_id, *(part.question_id for part in body.parts)]
        if value
    }
    unknown = sorted(requested_question_ids - set(catalog))
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown immutable question id(s): {', '.join(unknown)}",
        )

    async def _active_response(response_id: str) -> Dict[str, Any]:
        response = await tenant_db["evalpen_detected_responses"].find_one(
            {
                "submission_id": submission_id,
                "response_id": response_id,
                "superseded_at": {"$exists": False},
            }
        )
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Active response {response_id} was not found",
            )
        return response

    async def _clear_target_slot(
        question_id: str,
        *,
        replacing_response_ids: Optional[set[str]] = None,
    ) -> None:
        owners = await tenant_db["evalpen_detected_responses"].find(
            {
                "submission_id": submission_id,
                "question_id": question_id,
                "superseded_at": {"$exists": False},
            }
        ).to_list(length=20)
        replacing_response_ids = replacing_response_ids or set()
        real_owners = [
            item
            for item in owners
            if not item.get("is_missing_response")
            and str(item.get("response_id") or "") not in replacing_response_ids
        ]
        if real_owners:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "The target question already has answer evidence. Split or merge "
                    "the responses instead of overwriting it."
                ),
            )
        for owner in owners:
            if str(owner.get("response_id") or "") in replacing_response_ids:
                continue
            await tenant_db["evalpen_detected_responses"].update_one(
                {"_id": owner["_id"]},
                {
                    "$set": {
                        "eval_status": "superseded",
                        "superseded_at": now,
                        "superseded_by": actor_id,
                        "superseded_reason": "Teacher replaced not-attempted state with answer evidence",
                    }
                },
            )

    async def _assert_target_slot_available(
        question_id: str,
        *,
        replacing_response_ids: Optional[set[str]] = None,
    ) -> None:
        owners = await tenant_db["evalpen_detected_responses"].find(
            {
                "submission_id": submission_id,
                "question_id": question_id,
                "superseded_at": {"$exists": False},
            },
            {"response_id": 1, "is_missing_response": 1},
        ).to_list(length=20)
        replacing_response_ids = replacing_response_ids or set()
        if any(
            not owner.get("is_missing_response")
            and str(owner.get("response_id") or "") not in replacing_response_ids
            for owner in owners
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "The target question already has answer evidence. Split or merge "
                    "the responses instead of overwriting it."
                ),
            )

    def _resolved_flags(flags: Any) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for raw in flags or []:
            if not isinstance(raw, dict):
                continue
            flag = dict(raw)
            if flag.get("severity") == "blocking":
                flag["resolution"] = {
                    "resolved": True,
                    "resolution": "response_ownership_corrected_by_teacher",
                    "note": body.reason,
                    "resolved_by": actor_id,
                    "resolved_at": now,
                }
            result.append(flag)
        return result

    def _region_evidence_atoms(regions: List[Dict[str, Any]]) -> List[str]:
        atoms: List[str] = []
        for region in regions:
            payload = (
                f"teacher-region-v1|{submission_id}|{int(region['page_number'])}|"
                f"{float(region.get('x_start') or 0):.4f}|"
                f"{float(region['y_start']):.4f}|"
                f"{float(region.get('x_end') or 210):.4f}|"
                f"{float(region['y_end']):.4f}"
            )
            atoms.append(
                "teacher-region:"
                + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
            )
        return sorted(set(atoms))

    async def _create_assigned_response(
        source: Dict[str, Any],
        *,
        question_id: str,
        detected_text: Optional[str] = None,
        source_pages: Optional[List[Dict[str, Any]]] = None,
        operation: str,
        replacing_response_ids: Optional[set[str]] = None,
        evidence_atom_ids: Optional[List[str]] = None,
        clear_flags: bool = False,
    ) -> Dict[str, Any]:
        await _clear_target_slot(
            question_id,
            replacing_response_ids=replacing_response_ids,
        )
        question = catalog[question_id]
        response_id = f"RESP-TEACH-{uuid.uuid4().hex[:16]}"
        response = {
            key: value
            for key, value in source.items()
            if key not in {
                "_id",
                "response_id",
                "question_id",
                "question_number",
                "created_at",
                "evidence_atom_ids",
                "evidence_version",
                "evidence_source",
                "objective_result",
            }
        }
        text_source = source.get("detected_text") if detected_text is None else detected_text
        text = str(text_source or "").strip()
        pages = source.get("source_pages") if source_pages is None else source_pages
        resolved_atoms = (
            list(evidence_atom_ids)
            if evidence_atom_ids is not None
            else _region_evidence_atoms(list(pages or []))
            if source_pages is not None
            else [str(item) for item in (source.get("evidence_atom_ids") or []) if str(item)]
        )
        response.update(
            {
                "response_id": response_id,
                "submission_id": submission_id,
                "exam_id": exam_id,
                "student_id": student_id,
                "question_id": question_id,
                "question_number": question.get("question_number"),
                "detected_text": text,
                "source_pages": pages or [],
                "evidence_version": 1,
                "evidence_atom_ids": sorted(set(resolved_atoms)),
                "evidence_source": f"teacher_{operation}",
                "flags": (
                    []
                    if clear_flags
                    else _resolved_flags(source.get("flags"))
                ),
                "question_assignment": {
                    "method": f"teacher_{operation}",
                    "confidence": 1.0,
                    "manual_review_required": False,
                    "reason": body.reason,
                    "assigned_by": actor_id,
                    "assigned_at": now,
                    "source_response_id": str(source.get("response_id") or ""),
                },
                "manual_review_required": False,
                "manual_review_reason": None,
                "is_missing_response": False,
                "absence_proven": False,
                "answer_state": "detected",
                "eval_status": "ready",
                "grading_mode": _catalog_grading_mode(question),
                "word_count": len(text.split()),
                "created_at": now,
                "updated_at": now,
                "_immutable": True,
            }
        )
        await tenant_db["evalpen_detected_responses"].insert_one(response)
        return response

    async def _supersede(source: Dict[str, Any], operation: str) -> None:
        await tenant_db["evalpen_detected_responses"].update_one(
            {"_id": source["_id"], "superseded_at": {"$exists": False}},
            {
                "$set": {
                    "eval_status": "superseded",
                    "superseded_at": now,
                    "superseded_by": actor_id,
                    "superseded_reason": body.reason,
                    "superseded_operation": operation,
                }
            },
        )

    created: List[Dict[str, Any]] = []
    source_ids: List[str] = []

    if body.action == "set_objective_answer":
        if not body.response_id or not body.question_id or not body.selected_answer:
            raise HTTPException(
                status_code=400,
                detail=(
                    "set_objective_answer requires response_id, question_id, "
                    "and selected_answer"
                ),
            )
        source = await _active_response(body.response_id)
        if str(source.get("question_id") or "") != body.question_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This action only corrects the selected option. Move the "
                    "evidence to the correct question first."
                ),
            )
        question = catalog[body.question_id]
        if _catalog_grading_mode(question) != "objective":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Selected-option correction is available only for "
                    "Objective PCR questions"
                ),
            )
        option_labels = _catalog_option_labels(question)
        selected_answer = normalize_answer_label(body.selected_answer)
        if (
            len(option_labels) < 2
            or not selected_answer
            or selected_answer not in option_labels
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Choose one of the immutable question options: "
                    + ", ".join(option_labels)
                    if option_labels
                    else "This Objective question has no valid option catalog"
                ),
            )
        await _assert_target_slot_available(
            body.question_id,
            replacing_response_ids={body.response_id},
        )
        created.append(
            await _create_assigned_response(
                source,
                question_id=body.question_id,
                detected_text=selected_answer,
                operation="objective_answer",
                replacing_response_ids={body.response_id},
                clear_flags=True,
            )
        )
        await _supersede(source, "set_objective_answer")
        source_ids.append(body.response_id)

    elif body.action == "assign":
        if not body.response_id or not body.question_id:
            raise HTTPException(status_code=400, detail="assign requires response_id and question_id")
        source = await _active_response(body.response_id)
        await _assert_target_slot_available(
            body.question_id,
            replacing_response_ids={body.response_id},
        )
        created.append(
            await _create_assigned_response(
                source,
                question_id=body.question_id,
                operation="assignment",
                replacing_response_ids={body.response_id},
            )
        )
        await _supersede(source, "assign")
        source_ids.append(body.response_id)

    elif body.action == "split":
        if not body.response_id or len(body.parts) < 2:
            raise HTTPException(status_code=400, detail="split requires response_id and at least two parts")
        if len({part.question_id for part in body.parts}) != len(body.parts):
            raise HTTPException(status_code=400, detail="Each split part must target a different question")
        source = await _active_response(body.response_id)
        original_regions = [
            region
            for region in (source.get("source_pages") or [])
            if isinstance(region, dict)
            and region.get("page_number") is not None
            and region.get("y_start") is not None
            and region.get("y_end") is not None
        ]
        if not original_regions:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This response has no page region and cannot be split safely",
            )

        split_regions: List[Dict[str, Any]] = []
        for part in body.parts:
            pages = [region.model_dump() for region in part.source_pages]
            if not part.detected_text.strip() or not pages:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Each split part requires corrected text and a source-page region",
                )
            for region in pages:
                contained = any(
                    int(original.get("page_number")) == int(region["page_number"])
                    and float(region.get("x_start") or 0)
                    >= float(original.get("x_start") or 0) - 0.01
                    and float(region.get("x_end") or 210)
                    <= float(original.get("x_end") or 210) + 0.01
                    and float(region["y_start"]) >= float(original["y_start"]) - 0.01
                    and float(region["y_end"]) <= float(original["y_end"]) + 0.01
                    for original in original_regions
                )
                if not contained:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Every split region must stay inside the original response evidence",
                    )
                split_regions.append(region)

        for original in original_regions:
            page_number = int(original["page_number"])
            start = float(original["y_start"])
            end = float(original["y_end"])
            ranges = sorted(
                (
                    max(start, float(region["y_start"])),
                    min(end, float(region["y_end"])),
                )
                for region in split_regions
                if int(region["page_number"]) == page_number
                and float(region["y_end"]) > start
                and float(region["y_start"]) < end
            )
            cursor = start
            for range_start, range_end in ranges:
                if range_start > cursor + 0.5:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Split regions must cover the complete original response without gaps",
                    )
                if range_start < cursor - 0.01:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Split regions must not overlap",
                    )
                cursor = max(cursor, range_end)
            if cursor < end - 0.5:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Split regions must cover the complete original response without gaps",
                )

        for part in body.parts:
            await _assert_target_slot_available(
                part.question_id,
                replacing_response_ids={body.response_id},
            )
        for part in body.parts:
            pages = [
                {
                    **region.model_dump(),
                    # A teacher split creates new evidence regions. Reusing the
                    # model mapper's identifier for two different questions
                    # would make the evidence graph internally ambiguous.
                    "region_id": f"teacher-split-{uuid.uuid4().hex[:16]}",
                    "continuation_group": None,
                    "mapping_confidence": 1.0,
                }
                for region in part.source_pages
            ]
            created.append(
                await _create_assigned_response(
                    source,
                    question_id=part.question_id,
                    detected_text=part.detected_text,
                    source_pages=pages,
                    operation="split",
                    replacing_response_ids={body.response_id},
                )
            )
        await _supersede(source, "split")
        source_ids.append(body.response_id)

    elif body.action == "merge":
        merge_ids = list(dict.fromkeys(body.response_ids))
        if len(merge_ids) < 2 or not body.question_id:
            raise HTTPException(status_code=400, detail="merge requires at least two response_ids and question_id")
        sources = [await _active_response(response_id) for response_id in merge_ids]
        await _assert_target_slot_available(
            body.question_id,
            replacing_response_ids=set(merge_ids),
        )
        merged_text = "\n\n".join(
            str(source.get("detected_text") or "").strip()
            for source in sources
            if str(source.get("detected_text") or "").strip()
        )
        merged_pages: List[Dict[str, Any]] = []
        seen_regions = set()
        for source in sources:
            for region in source.get("source_pages") or []:
                if not isinstance(region, dict):
                    continue
                key = (
                    region.get("page_number"),
                    region.get("x_start"),
                    region.get("y_start"),
                    region.get("x_end"),
                    region.get("y_end"),
                )
                if key not in seen_regions:
                    seen_regions.add(key)
                    merged_pages.append(region)
        created.append(
            await _create_assigned_response(
                sources[0],
                question_id=body.question_id,
                detected_text=merged_text,
                source_pages=merged_pages,
                operation="merge",
                replacing_response_ids=set(merge_ids),
                evidence_atom_ids=sorted(
                    {
                        str(atom_id)
                        for source in sources
                        for atom_id in (source.get("evidence_atom_ids") or [])
                        if str(atom_id)
                    }
                ),
            )
        )
        for source in sources:
            await _supersede(source, "merge")
        source_ids.extend(merge_ids)

    elif body.action == "confirm_not_attempted":
        if not body.question_id:
            raise HTTPException(status_code=400, detail="confirm_not_attempted requires question_id")
        await _clear_target_slot(body.question_id)
        question = catalog[body.question_id]
        response_id = f"RESP-TEACH-BLANK-{uuid.uuid4().hex[:12]}"
        blank = {
            "response_id": response_id,
            "submission_id": submission_id,
            "exam_id": exam_id,
            "student_id": student_id,
            "question_id": body.question_id,
            "question_number": question.get("question_number"),
            "detected_text": "",
            "source_pages": [],
            "content_type": "TEXT_ONLY",
            "text_coverage_ratio": 0.0,
            "segmentation_confidence": 1.0,
            "ocr_confidence": 1.0,
            "flags": [],
            "word_count": 0,
            "is_continuation": False,
            "is_missing_response": True,
            "absence_proven": True,
            "answer_state": "not_attempted",
            "question_assignment": {
                "method": "not_attempted",
                "confidence": 1.0,
                "reason": body.reason,
                "absence_proof": {
                    "verified": True,
                    "method": "teacher_visual_confirmation",
                    "verified_by": actor_id,
                    "verified_at": now,
                    "reason": body.reason,
                },
            },
            "manual_review_required": False,
            "eval_status": "ready",
            "created_at": now,
            "updated_at": now,
            "_immutable": True,
        }
        await tenant_db["evalpen_detected_responses"].insert_one(blank)
        created.append(blank)

    elif body.action == "discard_non_answer":
        if not body.response_id:
            raise HTTPException(status_code=400, detail="discard_non_answer requires response_id")
        source = await _active_response(body.response_id)
        await _supersede(source, "discard_non_answer")
        source_ids.append(body.response_id)

    await tenant_db["evalpen_response_assignment_audit"].insert_one(
        {
            "audit_id": f"RAUD-{uuid.uuid4().hex[:16]}",
            "submission_id": submission_id,
            "exam_id": exam_id,
            "student_id": student_id,
            "action": body.action,
            "source_response_ids": source_ids,
            "created_response_ids": [str(item.get("response_id") or "") for item in created],
            "question_ids": sorted(requested_question_ids),
            "selected_answer": (
                normalize_answer_label(body.selected_answer)
                if body.action == "set_objective_answer"
                else None
            ),
            "reason": body.reason,
            "actor_id": actor_id,
            "created_at": now,
        }
    )

    evaluation_errors: List[str] = []
    if created:
        from api.v1.evalpen_evaluate_async import _build_eval_core

        eval_core = await _build_eval_core(tenant_db)
        for response in created:
            try:
                result = await eval_core.evaluate_response(
                    str(response["response_id"]),
                    question_id=str(response["question_id"]),
                )
                if result.error:
                    evaluation_errors.append(str(result.error))
            except Exception as exc:
                logger.exception(
                    "Teacher-corrected response %s could not be reevaluated",
                    response.get("response_id"),
                )
                evaluation_errors.append(str(exc)[:500])

    from services.exampen_submission_readiness import (
        assess_submission_readiness,
    )

    # A teacher correction changes review/publication eligibility, never the
    # fact that the durable processing job finished.
    await tenant_db["exampen_processing_jobs"].update_one(
        {"submission_id": submission_id},
        {
            "$set": {
                "status": "completed",
                "last_error": None,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {
                "job_id": f"pcr-job-{submission_id}",
                "exam_id": exam_id,
                "student_id": student_id,
                "created_at": now,
            },
        },
        upsert=True,
    )
    readiness = await assess_submission_readiness(tenant_db, submission_id)
    blocker_codes = {
        str(item.get("code") or "") for item in (readiness.get("blockers") or [])
    }
    reviewable_codes = {
        "document_coverage_requires_review",
        "response_assignment_requires_review",
        "evaluation_requires_review",
    }
    review_state = (
        "ready"
        if readiness.get("ready")
        else "needs_review"
        if blocker_codes and blocker_codes.issubset(reviewable_codes)
        else "blocked"
    )
    await tenant_db["evalpen_submissions"].update_one(
        {"submission_id": submission_id},
        {"$set": {"review_state": review_state, "updated_at": now}},
    )
    return {
        "submission_id": submission_id,
        "action": body.action,
        "created_response_ids": [str(item.get("response_id") or "") for item in created],
        "evaluation_errors": evaluation_errors,
        "review_state": review_state,
        "readiness": readiness,
    }


@router.post(
    "/submissions/{submission_id}/response-assignment",
    summary="Correct PCR response ownership with audit and reevaluation",
)
async def correct_response_assignment(
    submission_id: str,
    body: ResponseAssignmentCorrectionRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Serialize evidence corrections for one answer copy.

    A correction can supersede several immutable response rows and create
    several replacements.  Mongo deployments used by some institutions do
    not provide cross-document transactions, so a short, fenced lease keeps
    two teachers (or two browser retries) from racing those writes.
    """

    tenant_db = await _get_tenant_db(db, current_user)
    from services.exampen_review_lease import (
        SubmissionReviewBusyError,
        acquire_submission_review_lease,
        release_submission_review_lease,
    )

    try:
        lease_token = await acquire_submission_review_lease(
            tenant_db,
            submission_id,
            actor_id=str(current_user.get("user_id") or "unknown"),
            operation="response_assignment",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        ) from exc
    except SubmissionReviewBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    try:
        return await _correct_response_assignment_impl(
            submission_id,
            body,
            current_user=current_user,
            db=db,
        )
    finally:
        await release_submission_review_lease(
            tenant_db,
            submission_id,
            lease_token,
        )


async def _publish_submission_impl(
    submission_id: str,
    body: PublishRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
    *,
    review_lease_token: Optional[str] = None,
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
        if submission.get("publication_status") == "published":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This result has already been published and is immutable",
            )

        # Tutor scoping: verify this submission's student is visible
        _check_student_in_scope(
            submission.get("student_id", ""), scoped_ids
        )

        from services.exampen_submission_readiness import (
            assess_submission_readiness,
            build_publication_snapshot,
            readiness_message,
        )

        readiness = await assess_submission_readiness(tenant_db, submission_id)
        if not readiness.get("ready"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": f"Cannot publish: {readiness_message(readiness)}",
                    "readiness": readiness,
                },
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
                and not is_flag_resolved(f)
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
        publication_snapshot = await build_publication_snapshot(
            tenant_db,
            submission_id,
            actor_id=actor_id,
        )
        now = publication_snapshot.pop("published_at_dt")

        # Update publication status (non-immutable metadata field)
        publish_filter: Dict[str, Any] = {
            "submission_id": submission_id,
            "publication_status": {"$ne": "published"},
        }
        if review_lease_token:
            publish_filter["review_mutation_lease_token"] = review_lease_token
        published = await submissions_col.update_one(
            publish_filter,
            {
                "$set": {
                    "publication_status": "published",
                    "review_state": "published",
                    "published_at": now,
                    "published_by": actor_id,
                    "publication_note": body.note,
                    "publication_snapshot": publication_snapshot,
                    "publication_snapshot_hash": publication_snapshot["snapshot_hash"],
                },
                "$push": {
                    "publication_history": {
                        "action": "published",
                        "published_at": now,
                        "published_by": actor_id,
                        "publication_note": body.note,
                        "snapshot_hash": publication_snapshot["snapshot_hash"],
                    }
                },
            },
        )
        if published.matched_count != 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Publication ownership changed while the result snapshot was "
                    "being created. Refresh before trying again."
                ),
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
    """Publish under the same fence used by correction and reprocessing."""

    tenant_db = await _get_tenant_db(db, current_user)
    from services.exampen_review_lease import (
        SubmissionReviewBusyError,
        acquire_submission_review_lease,
        release_submission_review_lease,
    )

    try:
        lease_token = await acquire_submission_review_lease(
            tenant_db,
            submission_id,
            actor_id=str(current_user.get("user_id") or "unknown"),
            operation="publish",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found",
        ) from exc
    except SubmissionReviewBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    try:
        return await _publish_submission_impl(
            submission_id,
            body,
            current_user=current_user,
            db=db,
            review_lease_token=lease_token,
        )
    finally:
        await release_submission_review_lease(
            tenant_db,
            submission_id,
            lease_token,
        )
