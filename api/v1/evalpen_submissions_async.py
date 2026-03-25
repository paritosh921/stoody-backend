"""
EvalPen PCR Submission API — Conducted-exam submission ingestion,
detected response retrieval, and flag resolution.

Exposes the PCR submission endpoints for conducted exams via REST.
All endpoints delegate to repository and service classes — no engine
logic lives in this module.

Architecture:
    PCR_EVAL_ENGINE_SPEC §3, §7

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  evalpen_submissions (via IngestService), flag resolution
               on evalpen_detected_responses
    - Reads from: evalpen_submissions, evalpen_detected_responses
    - Never writes to: canonical artifact content, practice persistence

Hard constraints:
    - C1: MongoDB only
    - C3: Practice persistence untouched
    - C5: Ownership boundaries — server-side artifact references only

API authority:
    new-docs/api/eval-submissions.openapi.yaml

Test IDs:
    - I-PCR-01: conducted artifact -> PageOCR -> detected responses
    - I-TAMP-02: conducted PCR eval fetches server-side artifact
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

# Two routers: ``router`` for /submissions endpoints, ``flags_router``
# for /flags endpoints.  The integrator should mount them at separate
# prefixes to match the OpenAPI paths:
#   router       -> /api/v1/evalpen/submissions
#   flags_router -> /api/v1/evalpen/flags
router = APIRouter()
flags_router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency: require admin or tutor role for PCR submission endpoints."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for PCR submission operations",
        )
    return current_user


# ---------------------------------------------------------------------------
# Request / Response models (match eval-submissions.openapi.yaml exactly)
# ---------------------------------------------------------------------------

class PageRefAPI(BaseModel):
    """Page reference within a submission."""

    page_num: Optional[int] = None
    raw_asset_ref: Optional[str] = None


class CreateSubmissionRequest(BaseModel):
    """API request body for creating a PCR submission.

    Matches CreateSubmissionRequest in eval-submissions.openapi.yaml.
    """

    exam_id: str
    student_id: str
    admin_id: Optional[str] = None
    source: str = Field(
        ...,
        description="Artifact source: ble_pen or camera",
    )
    page_count: Optional[int] = None
    page_refs: Optional[List[PageRefAPI]] = None


class SubmissionAcceptedAPI(BaseModel):
    """API response for accepted submission.

    Matches SubmissionAccepted in eval-submissions.openapi.yaml.
    """

    submission_id: str
    segmentation_status: str


class SubmissionSummaryAPI(BaseModel):
    """Submission summary for listing.

    Matches SubmissionSummary in eval-submissions.openapi.yaml.
    """

    submission_id: str
    exam_id: str
    student_id: str
    source: str
    segmentation_status: str
    submitted_at: Optional[str] = None


class FlagAPI(BaseModel):
    """Flag on a detected response.

    Matches Flag in eval-submissions.openapi.yaml.
    """

    flag_id: str
    source: str
    flag_type: str
    severity: str
    reason: str
    suggested_action: Optional[str] = None


class DetectedResponseAPI(BaseModel):
    """Detected response for a submission.

    Matches DetectedResponse in eval-submissions.openapi.yaml.
    """

    response_id: str
    question_id: Optional[str] = None
    source_pages: Optional[List[int]] = None
    detected_text: Optional[str] = None
    content_type: str
    eval_status: str
    segmentation_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    flags: Optional[List[FlagAPI]] = None


class ResolveFlagRequest(BaseModel):
    """API request for resolving a flag.

    Matches ResolveFlagRequest in eval-submissions.openapi.yaml.
    """

    resolution: str
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: resolve tenant DB
# ---------------------------------------------------------------------------

async def _get_tenant_db_for_user(
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
# Helper: convert MongoDB doc to API model
# ---------------------------------------------------------------------------

def _doc_to_submission_summary(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a submission MongoDB document to SubmissionSummaryAPI dict."""
    submitted_at = doc.get("submitted_at")
    if hasattr(submitted_at, "isoformat"):
        submitted_at = submitted_at.isoformat()
    elif submitted_at is not None:
        submitted_at = str(submitted_at)

    return {
        "submission_id": doc.get("submission_id", ""),
        "exam_id": doc.get("exam_id", ""),
        "student_id": doc.get("student_id", ""),
        "source": doc.get("source", "camera"),
        "segmentation_status": doc.get("segmentation_status", "pending"),
        "submitted_at": submitted_at,
    }


def _doc_to_detected_response(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a detected response MongoDB document to DetectedResponseAPI dict."""
    # Extract page numbers from source_pages (may be list of dicts or ints)
    source_pages_raw = doc.get("source_pages", [])
    source_pages: List[int] = []
    for sp in source_pages_raw:
        if isinstance(sp, dict):
            source_pages.append(sp.get("page_number", 0))
        elif isinstance(sp, int):
            source_pages.append(sp)

    # Extract flags
    flags_raw = doc.get("flags", [])
    flags = [
        {
            "flag_id": f.get("flag_id", ""),
            "source": f.get("source", ""),
            "flag_type": f.get("flag_type", ""),
            "severity": f.get("severity", ""),
            "reason": f.get("reason", ""),
            "suggested_action": f.get("suggested_action"),
        }
        for f in flags_raw
    ]

    return {
        "response_id": doc.get("response_id", ""),
        "question_id": doc.get("question_id"),
        "source_pages": source_pages if source_pages else None,
        "detected_text": doc.get("detected_text"),
        "content_type": doc.get("content_type", "TEXT_ONLY"),
        "eval_status": doc.get("eval_status", "pending"),
        "segmentation_confidence": doc.get("segmentation_confidence"),
        "ocr_confidence": doc.get("ocr_confidence"),
        "flags": flags if flags else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Register or ingest a PCR conducted-exam submission",
    responses={
        400: {"description": "Invalid request"},
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database or ingest engine unavailable"},
    },
)
async def create_submission(
    request: Request,
    body: CreateSubmissionRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> SubmissionAcceptedAPI:
    """Register or ingest a PCR conducted-exam submission.

    Delegates to ``IngestService.ingest_submission()`` which:
      1. Validates and normalizes artifact data.
      2. Computes content hashes (TAMPER_PROOF_SPEC Layer 1).
      3. Persists the submission with write-once semantics.

    Does NOT accept client-submitted answer text (C5).
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        from api.v1._exampen_imports import load_exampen
        IngestService = load_exampen("ingest.service").IngestService

        ingest_service = IngestService(tenant_db)
        await ingest_service.initialize()

        # Resolve admin_id: use request body or fall back to current user
        admin_id = body.admin_id or current_user.get("user_id", "")

        # Convert page_refs to the format IngestService expects
        page_refs = None
        if body.page_refs:
            page_refs = [
                {
                    "page_number": pr.page_num or (i + 1),
                    "raw_asset_ref": pr.raw_asset_ref,
                }
                for i, pr in enumerate(body.page_refs)
            ]

        result = await ingest_service.ingest_submission(
            exam_id=body.exam_id,
            student_id=body.student_id,
            admin_id=admin_id,
            source=body.source,
            page_refs=page_refs,
        )

        return SubmissionAcceptedAPI(
            submission_id=result.submission_id,
            segmentation_status=result.segmentation_status.value
            if hasattr(result.segmentation_status, "value")
            else str(result.segmentation_status),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except ImportError as exc:
        logger.error("Ingest module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingest engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Submission ingest failed for exam=%s student=%s: %s",
            body.exam_id,
            body.student_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Submission ingest encountered an internal error",
        )


@router.get(
    "",
    summary="List PCR submissions",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def list_submissions(
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """List PCR submissions for the current tenant.

    Returns ``{ items: [...] }`` matching the OpenAPI schema.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        from api.v1._exampen_imports import load_exampen
        SubmissionRepository = load_exampen("pcr.storage").SubmissionRepository

        repo = SubmissionRepository(tenant_db)
        admin_id = current_user.get("user_id")
        docs = await repo.list_submissions(admin_id=admin_id)

        return {
            "items": [_doc_to_submission_summary(d) for d in docs],
        }

    except ImportError as exc:
        logger.error("PCR storage module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error("Failed to list submissions: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list submissions",
        )


@router.get(
    "/{submission_id}/responses",
    summary="Get detected responses for one submission",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Submission not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_submission_responses(
    submission_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Get all detected responses for a specific submission.

    Returns ``{ items: [...] }`` matching the OpenAPI schema.
    Server-side fetch of detected responses (TAMPER_PROOF_SPEC Layer 2).
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        from api.v1._exampen_imports import load_exampen
        SubmissionRepository = load_exampen("pcr.storage").SubmissionRepository
        DetectedResponseRepository = load_exampen("pcr.storage").DetectedResponseRepository

        # Verify submission exists
        sub_repo = SubmissionRepository(tenant_db)
        submission = await sub_repo.get_submission(submission_id)
        if submission is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Submission {submission_id} not found",
            )

        # Fetch detected responses
        resp_repo = DetectedResponseRepository(tenant_db)
        docs = await resp_repo.get_responses_by_submission(submission_id)

        return {
            "items": [_doc_to_detected_response(d) for d in docs],
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
            "Failed to get responses for submission=%s: %s",
            submission_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detected responses",
        )


@flags_router.patch(
    "/{flag_id}/resolve",
    summary="Resolve one PCR flag",
    responses={
        400: {"description": "Invalid resolution request"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Flag not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def resolve_flag(
    flag_id: str,
    body: ResolveFlagRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Resolve a flag on a detected response.

    The resolution is recorded in the flag's metadata with an audit trail
    (TAMPER_PROOF_SPEC Layer 3). The original flag data is preserved —
    resolution appends state, it does not overwrite (Section 6, Rule 3).

    If the resolved flag was the only blocking flag, the response's
    ``eval_status`` may transition from ``blocked`` to ``ready``.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        DetectedResponseRepository = load_exampen("pcr.storage").DetectedResponseRepository

        resp_repo = DetectedResponseRepository(tenant_db)

        # Find the response containing the flag
        response_doc = await tenant_db["evalpen_detected_responses"].find_one(
            {"flags.flag_id": flag_id}
        )
        if response_doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flag {flag_id} not found",
            )

        response_id = response_doc.get("response_id", "")
        actor_id = current_user.get("user_id", "unknown")

        # Update the specific flag with resolution metadata (append, not overwrite)
        resolution_entry = {
            "resolved": True,
            "resolution": body.resolution,
            "note": body.note,
            "resolved_by": actor_id,
            "resolved_at": datetime.now(timezone.utc),
        }

        await tenant_db["evalpen_detected_responses"].update_one(
            {"response_id": response_id, "flags.flag_id": flag_id},
            {
                "$set": {
                    "flags.$.resolution": resolution_entry,
                },
            },
        )

        # Check if all blocking flags are now resolved; if so, transition
        # eval_status from blocked -> ready
        updated_doc = await resp_repo.get_response(response_id)
        if updated_doc and updated_doc.get("eval_status") == "blocked":
            flags = updated_doc.get("flags", [])
            still_blocking = any(
                f.get("severity") == "blocking"
                and not f.get("resolution", {}).get("resolved", False)
                for f in flags
            )
            if not still_blocking:
                has_warnings = any(
                    f.get("severity") == "warning"
                    and not f.get("resolution", {}).get("resolved", False)
                    for f in flags
                )
                new_status = "ready_with_warnings" if has_warnings else "ready"
                await resp_repo.update_eval_status(response_id, new_status)
                logger.info(
                    "All blocking flags resolved for response %s — "
                    "transitioned to %s",
                    response_id,
                    new_status,
                )

        logger.info(
            "Resolved flag %s on response %s by %s: %s",
            flag_id,
            response_id,
            actor_id,
            body.resolution,
        )

        return {
            "flag_id": flag_id,
            "response_id": response_id,
            "resolved": True,
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
            "Failed to resolve flag %s: %s",
            flag_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve flag",
        )
