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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from utils.tutor_scoping import get_tutor_scoped_students

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
    content_type: str = "TEXT_ONLY"
    eval_status: str = "pending"
    total_score: Optional[float] = None
    max_score: Optional[float] = None
    overall_feedback: Optional[str] = None
    flags: Optional[List[Dict[str, Any]]] = None
    has_blocking_flags: bool = False


class SubmissionSummaryReviewAPI(BaseModel):
    """Full evaluation summary for a single submission."""

    submission_id: str
    exam_id: str
    student_id: str
    source: str = "camera"
    segmentation_status: str = "pending"
    publication_status: Optional[str] = None
    responses: List[ResponseSummaryAPI] = Field(default_factory=list)
    total_score: float = 0.0
    total_max_score: float = 0.0
    evaluated_count: int = 0
    blocked_count: int = 0
    pending_count: int = 0


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

        # Fetch all detected responses
        resp_repo = DetectedResponseRepository(tenant_db)
        response_docs = await resp_repo.get_responses_by_submission(
            submission_id
        )

        # Fetch evaluations for each response
        eval_repo = EvaluationRepository(tenant_db)

        response_summaries: List[ResponseSummaryAPI] = []
        total_score = 0.0
        total_max = 0.0
        evaluated_count = 0
        blocked_count = 0
        pending_count = 0

        for resp_doc in response_docs:
            response_id = resp_doc.get("response_id", "")
            eval_status = resp_doc.get("eval_status", "pending")

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

            if evaluation:
                resp_score = evaluation.get("total_score", 0.0)
                resp_max = evaluation.get("max_score", 0.0)
                feedback = evaluation.get("overall_feedback")
                total_score += resp_score or 0.0
                total_max += resp_max or 0.0
                evaluated_count += 1
            elif eval_status == "blocked":
                blocked_count += 1
            elif eval_status == "pending":
                pending_count += 1

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
                    question_id=resp_doc.get("question_id"),
                    content_type=resp_doc.get("content_type", "TEXT_ONLY"),
                    eval_status=eval_status,
                    total_score=resp_score,
                    max_score=resp_max,
                    overall_feedback=feedback,
                    flags=api_flags if api_flags else None,
                    has_blocking_flags=has_blocking,
                )
            )

        # Get submission metadata
        sub_dict = (
            submission
            if isinstance(submission, dict)
            else submission.__dict__
            if hasattr(submission, "__dict__")
            else {}
        )

        return SubmissionSummaryReviewAPI(
            submission_id=submission_id,
            exam_id=sub_dict.get("exam_id", ""),
            student_id=sub_dict.get("student_id", ""),
            source=sub_dict.get("source", "camera"),
            segmentation_status=sub_dict.get(
                "segmentation_status", "pending"
            ),
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
                pcr_max = 0.0
                blocked_count = 0

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
                        pcr_total += ev.get("total_score", 0.0)
                        pcr_max += ev.get("max_score", 0.0)

                entry = student_results.get(student_id)
                if entry is None:
                    entry = ExamResultStudentAPI(
                        student_id=student_id,
                        submission_id=submission_id,
                        publication_status=sub.get("publication_status"),
                    )
                    student_results[student_id] = entry

                entry.pcr_total_score = pcr_total
                entry.pcr_max_score = pcr_max
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
