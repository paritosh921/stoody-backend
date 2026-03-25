"""
EvalPen DCR API — Conducted-exam evaluation and result retrieval endpoints.

Exposes the DCR (Direct Character Recognition) engine for conducted exams
via REST endpoints.  All endpoints delegate to ``DCRService`` — no engine
logic lives in this module.

Architecture:
    DUAL_MODE_ARCHITECTURE.md §4

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  (delegated to DCRService -> DCRRepository -> exampen_dcr_results)
    - Reads from: evalpen_submissions (via DCRRepository, read-only)
    - Never writes to: canonical conducted-exam artifacts, practice persistence

Hard constraints:
    - C1: MongoDB only
    - C2: archiveDCR is backup only — never referenced
    - C3: No practice persistence created
    - C5: DCR uses canonical artifact references only; does not accept
           client-submitted answer text as authoritative

Failure modes:
    - DCR-01: HWR confidence too low → gate fallback (handled by DCRService)
    - DCR-02: Numeric template mismatch → tolerance rules (handled by TemplateMatcher)
    - DCR-03: DCR accidentally depends on deep PCR semantics → scope guard in DCRService

Test IDs:
    - I-DCR-01: canonical artifact -> DCR result commit
    - I-TAMP-01: conducted DCR eval fetches server-side artifact
    - U-TAMP-01: rejects client-submitted authoritative answer text
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

# Lazy imports for exam-conductor modules are done inside endpoint functions
# to avoid import errors when the exam-conductor package is not fully
# installed or when dependencies (ONNX, etc.) are missing in the
# development environment.

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency: require admin or tutor role for all DCR endpoints.

    DCR evaluation and result retrieval are admin/tutor-only operations.
    Students access results through separate read-only views (if exposed).
    """
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for DCR operations",
        )
    return current_user


# ---------------------------------------------------------------------------
# Request / Response models (API-facing — thin wrappers around service models)
# ---------------------------------------------------------------------------

class DCREvaluateAPIRequest(BaseModel):
    """API request body for triggering DCR evaluation.

    Only accepts canonical artifact references (submission_id) — does NOT
    accept client-submitted answer text (C5 / U-TAMP-01).
    """

    submission_id: str = Field(
        ...,
        description="Canonical submission ID from the ingest substrate",
    )
    exam_id: str = Field(
        ...,
        description="Conducted exam identifier",
    )
    student_id: str = Field(
        ...,
        description="Student identity",
    )
    question_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional subset of question IDs to evaluate. "
            "When None, all questions for the exam are evaluated."
        ),
    )


class DCRQuestionResultAPI(BaseModel):
    """Single question result in the API response."""

    question_id: str
    recognized_text: str
    confidence: float
    match_type: str
    score: float
    max_score: float
    used_gate_fallback: bool = False


class DCREvaluateAPIResponse(BaseModel):
    """API response for DCR evaluation."""

    submission_id: str
    exam_id: str
    student_id: str
    results: List[DCRQuestionResultAPI] = Field(default_factory=list)
    total_score: float = 0.0
    total_max_score: float = 0.0
    evaluated_at: str  # ISO 8601
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class DCRResultAPI(BaseModel):
    """Stored DCR result for a single question (retrieval response)."""

    exam_id: str
    student_id: str
    question_id: str
    recognized_text: str
    confidence: float
    match_type: str
    score: float
    max_score: float
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DCRScoreAPI(BaseModel):
    """Aggregate score response."""

    exam_id: str
    student_id: str
    total_score: float
    total_max_score: float


# ---------------------------------------------------------------------------
# Helper: build DCRService from tenant DB
# ---------------------------------------------------------------------------

async def _build_dcr_service(tenant_db: Any) -> Any:
    """Instantiate a DCRService backed by the given tenant database.

    Imports are deferred to avoid top-level dependency on the
    exam-conductor package (which may pull in ONNX / heavy deps).
    """
    from api.v1._exampen_imports import load_exampen
    _dcr = load_exampen("dcr")
    _dcr_repo = load_exampen("dcr.repository")
    _dcr_recog = load_exampen("dcr.recognizer")
    _dcr_match = load_exampen("dcr.matcher")
    _dcr_svc = load_exampen("dcr.service")
    DCRRepository = _dcr_repo.DCRRepository
    HWRRecognizer = _dcr_recog.HWRRecognizer
    TemplateMatcher = _dcr_match.TemplateMatcher
    DCRService = _dcr_svc.DCRService

    repo = DCRRepository(tenant_db)
    recognizer = HWRRecognizer()
    matcher = TemplateMatcher()
    service = DCRService(repo, recognizer, matcher)
    return service


async def _get_tenant_db_for_user(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Any:
    """Resolve the tenant database from the authenticated user's JWT claims.

    Raises HTTP 503 if the database is unavailable.
    """
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
# Stub answer-key / region loaders
# ---------------------------------------------------------------------------
# The DCRService.evaluate() method requires callback loaders for answer keys
# and question spatial regions.  In a full deployment these would query the
# exam metadata collections.  Here we provide thin async stubs that query
# the tenant DB directly.  If the exam metadata structure changes, only
# these loaders need updating — the DCRService contract remains stable.

async def _make_answer_key_loader(tenant_db: Any):
    """Return an async loader function: (exam_id, question_ids?) -> list[AnswerKey]."""
    from api.v1._exampen_imports import load_exampen
    AnswerKey = load_exampen("dcr.models").AnswerKey

    async def _load(
        exam_id: str,
        question_ids: Optional[List[str]] = None,
    ) -> List[Any]:
        """Load answer keys from exam metadata in the tenant DB."""
        query: Dict[str, Any] = {"exam_id": exam_id}
        if question_ids:
            query["question_id"] = {"$in": question_ids}

        cursor = tenant_db["exampen_answer_keys"].find(query)
        docs = await cursor.to_list(length=1000)
        keys = []
        for doc in docs:
            keys.append(
                AnswerKey(
                    question_id=doc["question_id"],
                    expected_text=doc["expected_text"],
                    max_score=doc.get("max_score", 1.0),
                    match_mode=doc.get("match_mode"),
                    numeric_tolerance=doc.get("numeric_tolerance"),
                    page_number=doc.get("page_number"),
                )
            )
        return keys

    return _load


async def _make_region_loader(tenant_db: Any):
    """Return an async loader: (exam_id) -> {page_number: [region_dicts]}.

    When ``exampen_question_regions`` is empty (not yet populated), falls
    back to generating whole-page regions from answer keys so that the
    DCR recognizer processes every page instead of skipping all of them.
    """

    async def _load(exam_id: str) -> Dict[int, List[Dict[str, Any]]]:
        """Load question bounding-box regions for exam pages."""
        cursor = tenant_db["exampen_question_regions"].find({"exam_id": exam_id})
        docs = await cursor.to_list(length=5000)

        regions_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for doc in docs:
            page_num = doc.get("page_number", 1)
            region_entry = {
                "question_id": doc.get("question_id"),
                "bbox": doc.get("bbox"),
                "region_type": doc.get("region_type"),
            }
            regions_by_page.setdefault(page_num, []).append(region_entry)

        if not regions_by_page:
            # No explicit regions stored yet. Generate whole-page fallback
            # regions from answer keys so DCR can still process pages.
            # Each answer key with a page_number gets a null-bbox region
            # (meaning "whole page"), which the recognizer handles.
            ak_cursor = tenant_db["exampen_answer_keys"].find({"exam_id": exam_id})
            ak_docs = await ak_cursor.to_list(length=1000)
            for ak in ak_docs:
                page_num = ak.get("page_number", 1)
                regions_by_page.setdefault(page_num, []).append({
                    "question_id": ak["question_id"],
                    "bbox": None,  # whole-page — recognizer processes all strokes
                    "region_type": "whole_page_fallback",
                })
            if regions_by_page:
                logger.info(
                    "exampen_question_regions empty for exam %s; "
                    "generated %d whole-page fallback regions from answer keys",
                    exam_id,
                    sum(len(v) for v in regions_by_page.values()),
                )

        return regions_by_page

    return _load


# ---------------------------------------------------------------------------
# Helper: convert DCRResult model to API dict
# ---------------------------------------------------------------------------

def _result_to_api(result: Any) -> Dict[str, Any]:
    """Convert a DCRResult model instance to a DCRResultAPI-compatible dict."""
    audit_trail = []
    for entry in getattr(result, "audit_trail", []):
        audit_trail.append(entry.model_dump(mode="json") if hasattr(entry, "model_dump") else {})

    return {
        "exam_id": result.exam_id,
        "student_id": result.student_id,
        "question_id": result.question_id,
        "recognized_text": result.recognized_text,
        "confidence": result.confidence,
        "match_type": result.match_type.value if hasattr(result.match_type, "value") else str(result.match_type),
        "score": result.score,
        "max_score": result.max_score,
        "audit_trail": audit_trail,
        "created_at": result.created_at.isoformat() if hasattr(result.created_at, "isoformat") else str(result.created_at),
        "updated_at": result.updated_at.isoformat() if hasattr(result.updated_at, "isoformat") else str(result.updated_at),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/evaluate",
    response_model=DCREvaluateAPIResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger DCR evaluation for a conducted-exam submission",
    responses={
        400: {"description": "Invalid request or submission/exam mismatch"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Submission not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def evaluate_dcr(
    request: Request,
    body: DCREvaluateAPIRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> DCREvaluateAPIResponse:
    """Trigger DCR evaluation for a conducted-exam submission.

    This endpoint:
      1. Resolves the tenant database from the authenticated user.
      2. Instantiates ``DCRService`` with the tenant DB.
      3. Loads answer keys and question regions from exam metadata.
      4. Calls ``service.evaluate()`` which fetches canonical artifacts
         from the ingest substrate (I-TAMP-01 / C5).
      5. Returns per-question recognition, matching, and scoring results.

    The endpoint does NOT accept client-supplied answer text — only
    canonical artifact references (C5 / U-TAMP-01).

    Failure modes handled:
      - DCR-01: Low HWR confidence → gate fallback (if configured)
      - DCR-02: Numeric tolerance in template matching
      - DCR-03: Scope guard (no deep PCR semantics)
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        DCREvaluateRequest = load_exampen("dcr.models").DCREvaluateRequest

        service = await _build_dcr_service(tenant_db)

        # Build the service-level request
        eval_request = DCREvaluateRequest(
            submission_id=body.submission_id,
            exam_id=body.exam_id,
            student_id=body.student_id,
            question_ids=body.question_ids,
        )

        # Build loaders for answer keys and question regions
        answer_key_loader = await _make_answer_key_loader(tenant_db)
        region_loader = await _make_region_loader(tenant_db)

        # Execute evaluation (I-DCR-01: canonical artifact -> DCR result commit)
        response = await service.evaluate(
            eval_request,
            answer_key_loader,
            region_loader,
        )

        # Map service response to API response
        api_results = []
        for qr in response.results:
            api_results.append(
                DCRQuestionResultAPI(
                    question_id=qr.question_id,
                    recognized_text=qr.recognized_text,
                    confidence=qr.confidence,
                    match_type=qr.match_type.value if hasattr(qr.match_type, "value") else str(qr.match_type),
                    score=qr.score,
                    max_score=qr.max_score,
                    used_gate_fallback=qr.used_gate_fallback,
                )
            )

        return DCREvaluateAPIResponse(
            submission_id=response.submission_id,
            exam_id=response.exam_id,
            student_id=response.student_id,
            results=api_results,
            total_score=response.total_score,
            total_max_score=response.total_max_score,
            evaluated_at=response.evaluated_at.isoformat(),
            errors=response.errors,
        )

    except ValueError as exc:
        # DCRService raises ValueError for not-found or identity mismatch
        detail = str(exc)
        if "not found" in detail.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )
    except ImportError as exc:
        logger.error("DCR module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "DCR evaluation failed for submission=%s exam=%s student=%s: %s",
            body.submission_id,
            body.exam_id,
            body.student_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DCR evaluation encountered an internal error",
        )


@router.get(
    "/results/{exam_id}/{student_id}",
    response_model=List[DCRResultAPI],
    summary="Get all DCR results for an exam + student",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_dcr_results(
    exam_id: str,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> List[DCRResultAPI]:
    """Retrieve all stored DCR results for an exam + student pair.

    Returns an empty list if no results have been computed yet.
    Results are read-only views of ``exampen_dcr_results`` documents.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        service = await _build_dcr_service(tenant_db)
        results = await service.get_results(exam_id, student_id)

        return [
            DCRResultAPI(**_result_to_api(r))
            for r in results
        ]
    except ImportError as exc:
        logger.error("DCR module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to fetch DCR results for exam=%s student=%s: %s",
            exam_id,
            student_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve DCR results",
        )


@router.get(
    "/results/{exam_id}/{student_id}/{question_id}",
    response_model=DCRResultAPI,
    summary="Get single DCR result for a specific question",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Result not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_dcr_result_by_question(
    exam_id: str,
    student_id: str,
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> DCRResultAPI:
    """Retrieve a single DCR result for a specific exam + student + question.

    Returns 404 if the result does not exist.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        DCRRepository = load_exampen("dcr.repository").DCRRepository

        repo = DCRRepository(tenant_db)
        result = await repo.get_result(exam_id, student_id, question_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No DCR result found for exam={exam_id}, "
                    f"student={student_id}, question={question_id}"
                ),
            )

        return DCRResultAPI(**_result_to_api(result))

    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("DCR module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to fetch DCR result for exam=%s student=%s question=%s: %s",
            exam_id,
            student_id,
            question_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve DCR result",
        )


@router.get(
    "/score/{exam_id}/{student_id}",
    response_model=DCRScoreAPI,
    summary="Get aggregate DCR score for an exam + student",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_dcr_score(
    exam_id: str,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> DCRScoreAPI:
    """Get the aggregate total score for a conducted DCR exam.

    Uses a MongoDB aggregation pipeline to sum ``score`` and ``max_score``
    across all question results for the specified exam + student.

    Returns ``{total_score: 0.0, total_max_score: 0.0}`` when no results
    exist yet.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        service = await _build_dcr_service(tenant_db)
        score_data = await service.get_total_score(exam_id, student_id)

        return DCRScoreAPI(
            exam_id=exam_id,
            student_id=student_id,
            total_score=score_data["total_score"],
            total_max_score=score_data["total_max_score"],
        )
    except ImportError as exc:
        logger.error("DCR module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to fetch DCR score for exam=%s student=%s: %s",
            exam_id,
            student_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve DCR score",
        )
