"""
EvalPen Flagged Response Queue API — Teacher-facing flagged response
management for reviewing, accepting, rejecting, or manually scoring
blocked PCR responses.

Provides endpoints for teachers (admin/tutor) to:
  1. View the full queue of responses with unresolved blocking flags
  2. View flagged responses for a specific submission
  3. Review individual flagged responses (accept, reject, or manually score)
  4. Get flag resolution statistics

Architecture:
    PCR_EVAL_ENGINE_SPEC {6} (Unified Flag System)

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  flag resolution metadata on evalpen_detected_responses,
               eval_status transitions on detected responses,
               manual evaluations via EvaluationRepository
    - Reads from: evalpen_detected_responses, evalpen_evaluations
    - Never writes to: canonical artifact content, detected_text,
                       practice persistence

Hard constraints:
    - C1: MongoDB only
    - C5: Ownership boundaries
    - TAMPER_PROOF_SPEC Layer 3: Flag resolution records must preserve
      who resolved, when, what action, and before/after state.
      Append-only.

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
    """Dependency: require admin or tutor role for flagged queue endpoints."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for flagged response operations",
        )
    return current_user


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class FlagDetailAPI(BaseModel):
    """Detailed flag information for queue display."""

    flag_id: str
    source: str = ""
    flag_type: str = ""
    severity: str = ""
    reason: str = ""
    suggested_action: Optional[str] = None
    resolved: bool = False
    resolution: Optional[Dict[str, Any]] = None


class FlaggedResponseAPI(BaseModel):
    """A flagged response in the review queue."""

    response_id: str
    submission_id: str
    question_id: Optional[str] = None
    detected_text: Optional[str] = None
    content_type: str = "TEXT_ONLY"
    eval_status: str = "blocked"
    segmentation_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    flags: List[FlagDetailAPI] = Field(default_factory=list)
    blocking_flag_count: int = 0
    unresolved_blocking_count: int = 0


class ReviewFlaggedRequest(BaseModel):
    """Request body for reviewing a flagged response.

    Actions:
      - ``accept``: Remove blocking status; allow auto-evaluation to proceed.
        Resolves all blocking flags on the response.
      - ``reject``: Mark the response as invalid/unusable. Sets eval_status
        to ``rejected``.
      - ``manual_score``: Teacher provides a manual score directly.
        Requires ``manual_score`` and ``manual_max_score`` fields.
    """

    action: str = Field(
        ...,
        description="Review action: accept, reject, or manual_score",
    )
    reason: str = Field(
        ...,
        min_length=5,
        description="Justification for the review action (min 5 chars)",
    )
    manual_score: Optional[float] = Field(
        default=None,
        ge=0,
        description="Manual score (required when action=manual_score)",
    )
    manual_max_score: Optional[float] = Field(
        default=None,
        ge=0,
        description="Max score for manual scoring (required when action=manual_score)",
    )

    @field_validator("action")
    @classmethod
    def valid_action(cls, v: str) -> str:
        allowed = {"accept", "reject", "manual_score"}
        if v not in allowed:
            raise ValueError(
                f"Invalid action '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            )
        return v

    @field_validator("reason")
    @classmethod
    def reason_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Reason must be at least 5 characters")
        return v.strip()


class FlagStatsAPI(BaseModel):
    """Flag resolution statistics."""

    total_flagged_responses: int = 0
    total_blocking_flags: int = 0
    resolved_blocking_flags: int = 0
    unresolved_blocking_flags: int = 0
    by_flag_type: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)


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
# Helper: convert a response doc to FlaggedResponseAPI
# ---------------------------------------------------------------------------

def _doc_to_flagged_response(doc: Dict[str, Any]) -> FlaggedResponseAPI:
    """Convert a detected response MongoDB document to FlaggedResponseAPI."""
    flags_raw = doc.get("flags", [])
    flag_details: List[FlagDetailAPI] = []
    blocking_count = 0
    unresolved_blocking = 0

    for f in flags_raw:
        is_blocking = f.get("severity") == "blocking"
        resolution = f.get("resolution")
        is_resolved = (
            resolution.get("resolved", False) if resolution else False
        )

        if is_blocking:
            blocking_count += 1
            if not is_resolved:
                unresolved_blocking += 1

        flag_details.append(
            FlagDetailAPI(
                flag_id=f.get("flag_id", ""),
                source=f.get("source", ""),
                flag_type=f.get("flag_type", ""),
                severity=f.get("severity", ""),
                reason=f.get("reason", ""),
                suggested_action=f.get("suggested_action"),
                resolved=is_resolved,
                resolution=resolution,
            )
        )

    return FlaggedResponseAPI(
        response_id=doc.get("response_id", ""),
        submission_id=doc.get("submission_id", ""),
        question_id=doc.get("question_id"),
        detected_text=doc.get("detected_text"),
        content_type=doc.get("content_type", "TEXT_ONLY"),
        eval_status=doc.get("eval_status", "blocked"),
        segmentation_confidence=doc.get("segmentation_confidence"),
        ocr_confidence=doc.get("ocr_confidence"),
        flags=flag_details,
        blocking_flag_count=blocking_count,
        unresolved_blocking_count=unresolved_blocking,
    )


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
    "/queue",
    summary="Get all responses with unresolved blocking flags",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database or exam-conductor unavailable"},
    },
)
async def get_flagged_queue(
    exam_id: Optional[str] = Query(
        default=None,
        description="Filter by exam_id (requires joining through submissions)",
    ),
    limit: int = Query(default=100, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Get the teacher review queue: all responses with unresolved blocking flags.

    Returns responses where ``flags.severity == "blocking"`` and the flag
    has not been resolved. This is the primary queue teachers use to review
    flagged student work before evaluation can proceed.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        from api.v1._exampen_imports import load_exampen

        DetectedResponseRepository = load_exampen(
            "pcr.storage"
        ).DetectedResponseRepository

        resp_repo = DetectedResponseRepository(tenant_db)

        # Build submission_id filter from exam_id and/or tutor scoping
        sub_query: Dict[str, Any] = {}
        if exam_id:
            sub_query["exam_id"] = exam_id
        if scoped_ids is not None:
            sub_query["student_id"] = {"$in": scoped_ids}

        # If we need to restrict by submission, resolve submission_ids
        query: Dict[str, Any] = {"flags.severity": "blocking"}
        if sub_query:
            sub_cursor = tenant_db["evalpen_submissions"].find(
                sub_query,
                projection={"submission_id": 1},
            )
            sub_docs = await sub_cursor.to_list(length=1000)
            sub_ids = [s["submission_id"] for s in sub_docs]
            if not sub_ids:
                return {"items": [], "total": 0}
            query["submission_id"] = {"$in": sub_ids}

        # Fetch flagged responses
        cursor = (
            tenant_db["evalpen_detected_responses"]
            .find(query)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)

        # Filter to only those with truly unresolved blocking flags
        flagged_items = []
        for doc in docs:
            api_item = _doc_to_flagged_response(doc)
            if api_item.unresolved_blocking_count > 0:
                flagged_items.append(api_item)

        # Get total count for pagination
        total_count = await tenant_db[
            "evalpen_detected_responses"
        ].count_documents(query)

        return {
            "items": [item.model_dump() for item in flagged_items],
            "total": total_count,
            "limit": limit,
            "skip": skip,
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
            "Failed to get flagged queue: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve flagged response queue",
        )


@router.get(
    "/queue/{submission_id}",
    summary="Get flagged responses for a specific submission",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Submission not found"},
        503: {"description": "Tenant database or exam-conductor unavailable"},
    },
)
async def get_flagged_for_submission(
    submission_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Get all flagged responses for a specific submission.

    Returns responses that have any blocking flags (resolved or unresolved)
    for teacher review context.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        from api.v1._exampen_imports import load_exampen

        _pcr_storage = load_exampen("pcr.storage")
        SubmissionRepository = _pcr_storage.SubmissionRepository
        DetectedResponseRepository = _pcr_storage.DetectedResponseRepository

        # Verify submission exists
        sub_repo = SubmissionRepository(tenant_db)
        submission = await sub_repo.get_submission(submission_id)
        if submission is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Submission {submission_id} not found",
            )

        # Tutor scoping: verify this submission's student is visible
        _sub_dict = (
            submission
            if isinstance(submission, dict)
            else submission.__dict__
            if hasattr(submission, "__dict__")
            else {}
        )
        _check_student_in_scope(
            _sub_dict.get("student_id", ""), scoped_ids
        )

        # Fetch responses with blocking flags
        resp_repo = DetectedResponseRepository(tenant_db)
        docs = await resp_repo.get_responses_with_blocking_flags(
            submission_id=submission_id
        )

        flagged_items = [_doc_to_flagged_response(d) for d in docs]

        return {
            "submission_id": submission_id,
            "items": [item.model_dump() for item in flagged_items],
            "total": len(flagged_items),
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
            "Failed to get flagged responses for submission %s: %s",
            submission_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve flagged responses",
        )


@router.post(
    "/{response_id}/review",
    summary="Teacher reviews a flagged response",
    responses={
        400: {"description": "Invalid request"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Response not found"},
        409: {"description": "Response has no blocking flags to review"},
        503: {"description": "Tenant database or exam-conductor unavailable"},
    },
)
async def review_flagged_response(
    response_id: str,
    body: ReviewFlaggedRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Teacher reviews a flagged response.

    Actions:
      - ``accept``: Resolves all blocking flags and transitions eval_status
        to ``ready`` or ``ready_with_warnings``, allowing auto-evaluation.
      - ``reject``: Marks the response as rejected. No evaluation will occur.
      - ``manual_score``: Teacher provides a manual score. Creates an
        evaluation record with the manual score and audit trail.

    TAMPER_PROOF_SPEC Layer 3:
    All review actions record actor_id, reason, before/after state,
    and timestamp in the audit trail. Flag resolutions preserve the
    original flag data (Section 6, Rule 3/5).
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        from api.v1._exampen_imports import load_exampen

        _pcr_storage = load_exampen("pcr.storage")
        DetectedResponseRepository = _pcr_storage.DetectedResponseRepository
        EvaluationRepository = _pcr_storage.EvaluationRepository

        resp_repo = DetectedResponseRepository(tenant_db)
        eval_repo = EvaluationRepository(tenant_db)

        # Fetch the response
        response_doc = await resp_repo.get_response(response_id)
        if response_doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Response {response_id} not found",
            )
        if response_doc.get("eval_status") == "superseded":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Response {response_id} has been superseded",
            )

        # Tutor scoping: verify this response's student is visible.
        # Try student_id on the response first, fall back to submission lookup.
        _resp_student_id = response_doc.get("student_id")
        if not _resp_student_id:
            _sub_lookup = await tenant_db["evalpen_submissions"].find_one(
                {"submission_id": response_doc.get("submission_id", "")},
                projection={"student_id": 1},
            )
            _resp_student_id = (
                _sub_lookup.get("student_id") if _sub_lookup else None
            )
        if _resp_student_id:
            _check_student_in_scope(_resp_student_id, scoped_ids)

        # Verify there are blocking flags to review
        flags = response_doc.get("flags", [])
        unresolved_blocking = [
            f
            for f in flags
            if f.get("severity") == "blocking"
            and not f.get("resolution", {}).get("resolved", False)
        ]

        if not unresolved_blocking and body.action != "manual_score":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Response {response_id} has no unresolved blocking "
                    f"flags to review"
                ),
            )

        actor_id = current_user.get("user_id", "unknown")
        now = datetime.now(timezone.utc)
        old_eval_status = response_doc.get("eval_status", "blocked")

        if body.action == "accept":
            # Resolve all unresolved blocking flags
            for flag in unresolved_blocking:
                flag_id = flag.get("flag_id", "")
                resolution_entry = {
                    "resolved": True,
                    "resolution": "accepted_by_teacher",
                    "note": body.reason,
                    "resolved_by": actor_id,
                    "resolved_at": now,
                }
                await tenant_db["evalpen_detected_responses"].update_one(
                    {
                        "response_id": response_id,
                        "flags.flag_id": flag_id,
                    },
                    {"$set": {"flags.$.resolution": resolution_entry}},
                )

            # Determine new status
            remaining_warnings = any(
                f.get("severity") == "warning"
                and not f.get("resolution", {}).get("resolved", False)
                for f in flags
                if f.get("severity") != "blocking"
            )
            new_status = (
                "ready_with_warnings" if remaining_warnings else "ready"
            )

            await resp_repo.update_eval_status(response_id, new_status)

            logger.info(
                "Flagged response %s accepted by %s -> %s",
                response_id,
                actor_id,
                new_status,
            )

            return {
                "response_id": response_id,
                "action": "accept",
                "previous_eval_status": old_eval_status,
                "new_eval_status": new_status,
                "flags_resolved": len(unresolved_blocking),
                "reviewed_by": actor_id,
                "reviewed_at": now.isoformat(),
            }

        elif body.action == "reject":
            # Resolve all blocking flags as rejected
            for flag in unresolved_blocking:
                flag_id = flag.get("flag_id", "")
                resolution_entry = {
                    "resolved": True,
                    "resolution": "rejected_by_teacher",
                    "note": body.reason,
                    "resolved_by": actor_id,
                    "resolved_at": now,
                }
                await tenant_db["evalpen_detected_responses"].update_one(
                    {
                        "response_id": response_id,
                        "flags.flag_id": flag_id,
                    },
                    {"$set": {"flags.$.resolution": resolution_entry}},
                )

            # Set eval_status to rejected
            await resp_repo.update_eval_status(response_id, "rejected")

            logger.info(
                "Flagged response %s rejected by %s",
                response_id,
                actor_id,
            )

            return {
                "response_id": response_id,
                "action": "reject",
                "previous_eval_status": old_eval_status,
                "new_eval_status": "rejected",
                "flags_resolved": len(unresolved_blocking),
                "reviewed_by": actor_id,
                "reviewed_at": now.isoformat(),
            }

        elif body.action == "manual_score":
            # Validate required fields for manual scoring
            if body.manual_score is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="manual_score is required when action=manual_score",
                )
            if body.manual_max_score is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="manual_max_score is required when action=manual_score",
                )
            if body.manual_score > body.manual_max_score:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"manual_score ({body.manual_score}) cannot exceed "
                        f"manual_max_score ({body.manual_max_score})"
                    ),
                )

            # Resolve all blocking flags as manually scored
            for flag in unresolved_blocking:
                flag_id = flag.get("flag_id", "")
                resolution_entry = {
                    "resolved": True,
                    "resolution": "manually_scored_by_teacher",
                    "note": body.reason,
                    "resolved_by": actor_id,
                    "resolved_at": now,
                }
                await tenant_db["evalpen_detected_responses"].update_one(
                    {
                        "response_id": response_id,
                        "flags.flag_id": flag_id,
                    },
                    {"$set": {"flags.$.resolution": resolution_entry}},
                )

            # Check if evaluation already exists
            existing_eval = await eval_repo.get_evaluation_by_response(
                response_id
            )

            if existing_eval:
                # Override existing evaluation score
                evaluation_id = existing_eval.get("evaluation_id", "")
                old_score = existing_eval.get("total_score", 0.0)

                await eval_repo.override_score(
                    evaluation_id,
                    new_total_score=body.manual_score,
                    actor_id=actor_id,
                    reason=f"Manual score from flagged review: {body.reason}",
                )

                await resp_repo.update_eval_status(
                    response_id, "manual_review"
                )

                return {
                    "response_id": response_id,
                    "action": "manual_score",
                    "evaluation_id": evaluation_id,
                    "previous_score": old_score,
                    "new_score": body.manual_score,
                    "max_score": body.manual_max_score,
                    "previous_eval_status": old_eval_status,
                    "new_eval_status": "manual_review",
                    "reviewed_by": actor_id,
                    "reviewed_at": now.isoformat(),
                }
            else:
                # Create a new manual evaluation
                import uuid

                evaluation_id = f"EVAL-MAN-{uuid.uuid4().hex[:12]}"
                eval_doc = {
                    "evaluation_id": evaluation_id,
                    "response_id": response_id,
                    "question_id": response_doc.get("question_id", ""),
                    "student_id": response_doc.get("student_id", ""),
                    "eval_path": "manual_teacher_review",
                    "model_used": None,
                    "total_score": body.manual_score,
                    "max_score": body.manual_max_score,
                    "scoreable_max": body.manual_max_score,
                    "step_marks": [],
                    "overall_feedback": f"Manually scored by teacher: {body.reason}",
                    "reference_solution": None,
                    "token_usage": None,
                    "raw_llm_response": None,
                    "audit_trail": [
                        {
                            "actor_id": actor_id,
                            "timestamp": now,
                            "action": "manual_evaluation_created",
                            "before": None,
                            "after": {
                                "total_score": body.manual_score,
                                "max_score": body.manual_max_score,
                            },
                            "reason": body.reason,
                        }
                    ],
                    "created_at": now,
                }

                await eval_repo.insert_evaluation(eval_doc)
                await resp_repo.update_eval_status(
                    response_id, "manual_review"
                )

                logger.info(
                    "Manual evaluation %s created for response %s by %s "
                    "(score: %.2f/%.2f)",
                    evaluation_id,
                    response_id,
                    actor_id,
                    body.manual_score,
                    body.manual_max_score,
                )

                return {
                    "response_id": response_id,
                    "action": "manual_score",
                    "evaluation_id": evaluation_id,
                    "previous_score": None,
                    "new_score": body.manual_score,
                    "max_score": body.manual_max_score,
                    "previous_eval_status": old_eval_status,
                    "new_eval_status": "manual_review",
                    "reviewed_by": actor_id,
                    "reviewed_at": now.isoformat(),
                }

        # Should not reach here due to validator, but guard anyway
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown action: {body.action}",
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
            "Flagged review failed for response %s: %s",
            response_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Flagged response review encountered an internal error",
        )


@router.get(
    "/stats",
    response_model=FlagStatsAPI,
    summary="Get flag resolution statistics",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database or exam-conductor unavailable"},
    },
)
async def get_flag_stats(
    exam_id: Optional[str] = Query(
        default=None,
        description="Filter stats by exam_id",
    ),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> FlagStatsAPI:
    """Get flag resolution statistics.

    Returns counts of total flagged responses, blocking vs warning flags,
    resolved vs unresolved, grouped by flag type and severity.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)

    try:
        # Build base query
        base_query: Dict[str, Any] = {
            "flags": {"$exists": True, "$ne": []},
            "eval_status": {"$ne": "superseded"},
        }

        # Build submission filter from exam_id and/or tutor scoping
        sub_query: Dict[str, Any] = {}
        if exam_id:
            sub_query["exam_id"] = exam_id
        if scoped_ids is not None:
            sub_query["student_id"] = {"$in": scoped_ids}

        if sub_query:
            # Get submission_ids matching the filters
            sub_cursor = tenant_db["evalpen_submissions"].find(
                sub_query,
                projection={"submission_id": 1},
            )
            sub_docs = await sub_cursor.to_list(length=1000)
            sub_ids = [s["submission_id"] for s in sub_docs]
            if not sub_ids:
                return FlagStatsAPI()
            base_query["submission_id"] = {"$in": sub_ids}

        # Fetch all responses with flags
        cursor = tenant_db["evalpen_detected_responses"].find(base_query)
        docs = await cursor.to_list(length=5000)

        total_flagged = len(docs)
        total_blocking = 0
        resolved_blocking = 0
        unresolved_blocking = 0
        by_flag_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for doc in docs:
            flags = doc.get("flags", [])
            for f in flags:
                severity = f.get("severity", "info")
                flag_type = f.get("flag_type", "unknown")

                by_flag_type[flag_type] = by_flag_type.get(flag_type, 0) + 1
                by_severity[severity] = by_severity.get(severity, 0) + 1

                if severity == "blocking":
                    total_blocking += 1
                    resolution = f.get("resolution", {})
                    if resolution.get("resolved", False):
                        resolved_blocking += 1
                    else:
                        unresolved_blocking += 1

        return FlagStatsAPI(
            total_flagged_responses=total_flagged,
            total_blocking_flags=total_blocking,
            resolved_blocking_flags=resolved_blocking,
            unresolved_blocking_flags=unresolved_blocking,
            by_flag_type=by_flag_type,
            by_severity=by_severity,
        )

    except ImportError as exc:
        logger.error("PCR storage module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to get flag stats: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve flag statistics",
        )
