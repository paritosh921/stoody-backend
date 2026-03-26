"""
EvalPen PCR Evaluation API — Trigger single/batch evaluation of conducted-exam
responses and retrieve evaluation results.

All evaluation endpoints delegate to ``EvalCore`` — no engine logic lives in
this module.  Evaluation uses server-side response fetch (TAMPER_PROOF_SPEC
Layer 2): the client provides ``response_id`` and ``question_id``, not answer
text.  The engine fetches canonical ``detected_text`` from storage.

Architecture:
    PCR_EVAL_ENGINE_SPEC §5

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  (delegated to EvalCore -> EvaluationRepository -> evalpen_evaluations)
    - Reads from: evalpen_detected_responses (via EvalCore, server-side fetch),
                  evalpen_evaluations (retrieval)
    - Never writes to: evalpen_submissions, canonical artifacts, practice persistence

Hard constraints:
    - C1: MongoDB only
    - C3: Practice persistence untouched (this file is conducted-exam only)
    - C4: All LLM calls through the gate (delegated to EvalCore)
    - C5: Ownership boundaries — response_id references only, no client answer text

API authority:
    new-docs/api/eval-evaluate.openapi.yaml

Test IDs:
    - I-TAMP-01: conducted eval fetches server-side artifact
    - U-TAMP-01: rejects client-submitted authoritative answer text
    - I-PCR-02: blocking flags prevent auto-eval
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

# Two routers: ``router`` for /evaluate endpoints, ``evaluations_router``
# for /evaluations retrieval endpoints.  The integrator should mount them
# at separate prefixes to match the OpenAPI paths:
#   router             -> /api/v1/evalpen/evaluate
#   evaluations_router -> /api/v1/evalpen/evaluations
router = APIRouter()
evaluations_router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency: require admin or tutor role for PCR evaluation endpoints."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for PCR evaluation operations",
        )
    return current_user


# ---------------------------------------------------------------------------
# Request / Response models (match eval-evaluate.openapi.yaml exactly)
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """API request for evaluating a single server-side PCR response.

    Only accepts ``response_id`` and ``question_id`` — does NOT accept
    client-submitted answer text (C5 / TAMPER_PROOF_SPEC Layer 2).
    """

    response_id: str
    question_id: str


class EvaluateAcceptedAPI(BaseModel):
    """API response for accepted evaluation request."""

    evaluation_id: str
    status: str


class BatchEvaluateRequest(BaseModel):
    """API request for batch evaluation of server-side PCR responses."""

    items: List[EvaluateRequest]


class StepMarkAPI(BaseModel):
    """Step-wise marking breakdown entry."""

    step: Optional[str] = None
    marks_awarded: Optional[float] = None
    marks_possible: Optional[float] = None
    justification: Optional[str] = None


class ReferenceSolutionAPI(BaseModel):
    """Reference solution summary."""

    text: Optional[str] = None
    source: Optional[str] = None
    version: Optional[int] = None


class TokenUsageAPI(BaseModel):
    """Token usage summary from gate call."""

    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class EvaluationDetailAPI(BaseModel):
    """Full evaluation detail.

    Matches EvaluationDetail in eval-evaluate.openapi.yaml.
    """

    evaluation_id: str
    response_id: str
    question_id: str
    eval_path: Optional[str] = None
    model_used: Optional[str] = None
    content_type: Optional[str] = None
    total_score: float
    max_score: float
    scoreable_max: Optional[float] = None
    step_marks: Optional[List[StepMarkAPI]] = None
    overall_feedback: Optional[str] = None
    reference_solution: Optional[ReferenceSolutionAPI] = None
    token_usage: Optional[TokenUsageAPI] = None


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
# Helper: build EvalCore from tenant DB
# ---------------------------------------------------------------------------

async def _build_eval_core(tenant_db: Any) -> Any:
    """Instantiate an EvalCore backed by the given tenant database.

    Imports are deferred to avoid top-level dependency on the
    exam-conductor package.
    """
    import importlib
    _pcr_storage = importlib.import_module("exam-conductor.pcr.storage")
    _pcr_services = importlib.import_module("exam-conductor.pcr.services")
    _llm_gate = importlib.import_module("exam-conductor.llm_gate")
    DetectedResponseRepository = _pcr_storage.DetectedResponseRepository
    EvaluationRepository = _pcr_storage.EvaluationRepository
    QuestionRepository = _pcr_storage.QuestionRepository
    SolutionRepository = _pcr_storage.SolutionRepository
    SolutionCache = _pcr_services.SolutionCache
    EvalCore = _pcr_services.EvalCore
    LLMGate = _llm_gate.LLMGate

    response_repo = DetectedResponseRepository(tenant_db)
    eval_repo = EvaluationRepository(tenant_db)
    question_repo = QuestionRepository(tenant_db)
    solution_repo = SolutionRepository(tenant_db)
    gate = LLMGate(tenant_db)

    solution_cache = SolutionCache(
        solution_repo=solution_repo,
        gate=gate,
    )

    return EvalCore(
        response_repo=response_repo,
        eval_repo=eval_repo,
        question_repo=question_repo,
        solution_cache=solution_cache,
        gate=gate,
    )


# ---------------------------------------------------------------------------
# Helper: convert EvalResult to API response
# ---------------------------------------------------------------------------

def _eval_result_to_api(result: Any) -> Dict[str, Any]:
    """Convert an EvalResult dataclass to EvaluationDetailAPI-compatible dict."""
    step_marks = None
    if result.step_marks:
        step_marks = [
            {
                "step": sm.step,
                "marks_awarded": sm.marks_awarded,
                "marks_possible": sm.max_marks,
                "justification": sm.rationale,
            }
            for sm in result.step_marks
        ]

    reference_solution = None
    if result.reference_solution:
        reference_solution = {
            "text": result.reference_solution,
            "source": "cache",
            "version": None,
        }

    token_usage = None
    if result.token_usage:
        token_usage = {
            "model": result.token_usage.get("model"),
            "input_tokens": result.token_usage.get("input_tokens"),
            "output_tokens": result.token_usage.get("output_tokens"),
            "total_tokens": result.token_usage.get("total_tokens"),
        }

    return {
        "evaluation_id": result.evaluation_id,
        "response_id": result.response_id,
        "question_id": result.question_id or "",
        "eval_path": result.eval_path or None,
        "model_used": result.model_used or None,
        "total_score": result.total_score,
        "max_score": result.max_score,
        "scoreable_max": result.scoreable_max,
        "step_marks": step_marks,
        "overall_feedback": result.overall_feedback or None,
        "reference_solution": reference_solution,
        "token_usage": token_usage,
    }


def _doc_to_evaluation_detail(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a MongoDB evaluation document to EvaluationDetailAPI dict."""
    step_marks = None
    step_marks_raw = doc.get("step_marks", [])
    if step_marks_raw:
        step_marks = [
            {
                "step": sm.get("step"),
                "marks_awarded": sm.get("marks_awarded"),
                "marks_possible": sm.get("max_marks"),
                "justification": sm.get("rationale"),
            }
            for sm in step_marks_raw
        ]

    reference_solution = None
    ref_sol = doc.get("reference_solution")
    if ref_sol:
        if isinstance(ref_sol, str):
            reference_solution = {"text": ref_sol, "source": None, "version": None}
        elif isinstance(ref_sol, dict):
            reference_solution = {
                "text": ref_sol.get("text") or ref_sol.get("reference_solution"),
                "source": ref_sol.get("source"),
                "version": ref_sol.get("version"),
            }

    token_usage = None
    tu = doc.get("token_usage")
    if tu:
        token_usage = {
            "model": tu.get("model"),
            "input_tokens": tu.get("input_tokens"),
            "output_tokens": tu.get("output_tokens"),
            "total_tokens": tu.get("total_tokens"),
        }

    return {
        "evaluation_id": doc.get("evaluation_id", ""),
        "response_id": doc.get("response_id", ""),
        "question_id": doc.get("question_id", ""),
        "eval_path": doc.get("eval_path"),
        "model_used": doc.get("model_used"),
        "total_score": doc.get("total_score", 0.0),
        "max_score": doc.get("max_score", 0.0),
        "scoreable_max": doc.get("scoreable_max"),
        "step_marks": step_marks,
        "overall_feedback": doc.get("overall_feedback"),
        "reference_solution": reference_solution,
        "token_usage": token_usage,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Evaluate one server-side PCR response",
    responses={
        400: {"description": "Invalid request or response not found"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Response not found"},
        503: {"description": "Evaluation engine unavailable"},
    },
)
async def evaluate_single(
    request: Request,
    body: EvaluateRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> EvaluateAcceptedAPI:
    """Trigger evaluation for a single server-side PCR response.

    The endpoint accepts ``response_id`` and ``question_id`` only (C5) —
    the evaluation engine fetches the canonical ``detected_text`` from
    server-side storage (TAMPER_PROOF_SPEC Layer 2).
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        eval_core = await _build_eval_core(tenant_db)
        result = await eval_core.evaluate_response(
            body.response_id,
            question_id=body.question_id,
        )

        if result.error:
            error_lower = result.error.lower()
            if "not found" in error_lower:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result.error,
                )
            if "mismatch" in error_lower:
                # question_id from wire doesn't match stored response —
                # this is a client validation error, not a server failure.
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=result.error,
                )
            # Other errors are internal failures
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Evaluation failed: {result.error}",
            )

        # Blocked by flags — reject rather than returning out-of-schema status.
        # The OpenAPI spec only allows "queued" or "evaluating" in 202 responses.
        if result.skipped:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Response {body.response_id} has blocking flags and "
                    f"cannot be auto-evaluated. Resolve flags first."
                ),
            )

        return EvaluateAcceptedAPI(
            evaluation_id=result.evaluation_id,
            status="evaluating",
        )

    except HTTPException:
        raise
    except ImportError as exc:
        logger.error("PCR evaluation module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR evaluation engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Evaluation failed for response=%s: %s",
            body.response_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation encountered an internal error",
        )


@router.post(
    "/batch",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Evaluate a batch of server-side PCR responses",
    responses={
        400: {"description": "Invalid request"},
        403: {"description": "Insufficient permissions"},
        503: {"description": "Evaluation engine unavailable"},
    },
)
async def evaluate_batch(
    request: Request,
    body: BatchEvaluateRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Trigger batch evaluation for multiple server-side PCR responses.

    Each item in the batch specifies ``response_id`` and ``question_id``
    (C5). The engine fetches canonical text for each response from
    server-side storage.

    Budget exhaustion mid-batch is handled gracefully — already-evaluated
    responses are preserved and the batch returns with partial results
    (LLM_GATE_SPEC §9.1).
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    if not body.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch must contain at least one evaluation request",
        )

    try:
        eval_core = await _build_eval_core(tenant_db)

        results = []
        for item in body.items:
            try:
                result = await eval_core.evaluate_response(
                    item.response_id,
                    question_id=item.question_id,
                )
                if result.skipped:
                    results.append({
                        "response_id": item.response_id,
                        "status": "queued",
                        "note": "Blocked by flags — requires manual review",
                    })
                elif result.error:
                    results.append({
                        "response_id": item.response_id,
                        "status": "queued",
                        "note": f"Error: {result.error}",
                    })
                else:
                    results.append({
                        "evaluation_id": result.evaluation_id,
                        "response_id": result.response_id,
                        "status": "evaluating",
                    })
            except Exception as exc:
                logger.error(
                    "Batch eval failed for response=%s: %s",
                    item.response_id,
                    exc,
                    exc_info=True,
                )
                results.append({
                    "response_id": item.response_id,
                    "status": "queued",
                    "note": f"Error: {str(exc)}",
                })

        return {
            "total": len(body.items),
            "results": results,
        }

    except ImportError as exc:
        logger.error("PCR evaluation module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR evaluation engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Batch evaluation failed: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch evaluation encountered an internal error",
        )


@evaluations_router.get(
    "/{evaluation_id}",
    response_model=EvaluationDetailAPI,
    summary="Get one PCR evaluation result",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Evaluation not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_evaluation(
    evaluation_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> EvaluationDetailAPI:
    """Retrieve a single PCR evaluation result by evaluation_id.

    Returns the full evaluation detail including step marks, feedback,
    reference solution, and token usage.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        import importlib
        _pcr_storage = importlib.import_module("exam-conductor.pcr.storage")
        EvaluationRepository = _pcr_storage.EvaluationRepository

        eval_repo = EvaluationRepository(tenant_db)
        doc = await eval_repo.get_evaluation(evaluation_id)

        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {evaluation_id} not found",
            )

        return EvaluationDetailAPI(**_doc_to_evaluation_detail(doc))

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
            "Failed to get evaluation %s: %s",
            evaluation_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evaluation",
        )
