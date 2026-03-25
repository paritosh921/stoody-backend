"""
EvalPen PCR Practice API — Stateless live practice evaluation.

This endpoint evaluates a practice response synchronously and returns
the result.  It does NOT write to ``evalpen_submissions`` or any new
ExamPen collections (C3).  It does NOT create immutable practice
artifacts inside ExamPen (PCR_EVAL_ENGINE_SPEC §8.2).

Practice mode accepts student-supplied text or image references directly
(unlike conducted-exam endpoints which enforce server-side fetch).
Token logging may still occur through the LLM gate.

Architecture:
    PCR_EVAL_ENGINE_SPEC §2.2, §8.2

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  none (stateless — token logging is gate-owned)
    - Reads from: evalpen_questions (question metadata),
                  evalpen_solutions (solution cache, read-only)
    - Never writes to: evalpen_submissions, evalpen_detected_responses,
      evalpen_evaluations, practice persistence

Hard constraints:
    - C1: MongoDB only
    - C3: Practice persistence untouched — POST /practice/evaluate is stateless
    - C4: All LLM calls through the gate with pcr_practice caller_id

API authority:
    new-docs/api/eval-practice.openapi.yaml

Test IDs:
    - I-PCR-03: practice stateless, no new persistence
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models (match eval-practice.openapi.yaml exactly)
# ---------------------------------------------------------------------------

class PracticeEvaluateRequest(BaseModel):
    """API request for stateless practice evaluation.

    Matches PracticeEvaluateRequest in eval-practice.openapi.yaml.

    Practice mode accepts student-supplied content directly — it is
    NOT subject to the conducted-exam tamper-proofing requirements.
    """

    question_id: str
    source_type: str = Field(
        ...,
        description="Source type: canvas or camera",
    )
    text: Optional[str] = None
    image_ref: Optional[str] = None


class PracticeTokenUsageAPI(BaseModel):
    """Token usage in practice response."""

    caller: Optional[str] = None
    total_tokens: Optional[int] = None


class PracticeEvaluateResponse(BaseModel):
    """API response for practice evaluation.

    Matches PracticeEvaluateResponse in eval-practice.openapi.yaml.
    """

    question_id: str
    total_score: float
    max_score: float
    feedback: Optional[str] = None
    token_usage: Optional[PracticeTokenUsageAPI] = None


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
# Helper: build EvalCore for practice (lightweight — no eval_repo needed
# since practice does not persist evaluations)
# ---------------------------------------------------------------------------

async def _build_practice_eval_core(tenant_db: Any) -> Any:
    """Instantiate an EvalCore for practice evaluation.

    Imports are deferred to avoid top-level dependency on the
    exam-conductor package.

    Practice mode still needs:
    - QuestionRepository (to look up question metadata)
    - SolutionRepository (to look up cached reference solutions)
    - LLMGate (to call the model)

    It does NOT need EvaluationRepository for persistence (C3).
    However, EvalCore requires all protocol dependencies, so we
    provide a full instance.  The practice path (evaluate_practice)
    simply never calls the evaluation writer.
    """
    from api.v1._exampen_imports import load_exampen
    _pcr_storage = load_exampen("pcr.storage")
    _pcr_services = load_exampen("pcr.services")
    _llm_gate = load_exampen("llm_gate")
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
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/evaluate",
    response_model=PracticeEvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate one live practice response",
    responses={
        400: {"description": "Invalid request — no text or image provided"},
        403: {"description": "Insufficient permissions"},
        422: {"description": "Image OCR produced no extractable text"},
        503: {"description": "Evaluation engine or LLM gate unavailable"},
    },
)
async def evaluate_practice(
    request: Request,
    body: PracticeEvaluateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
) -> PracticeEvaluateResponse:
    """Evaluate a live practice response statelessly.

    This endpoint:
      1. Accepts student-supplied text or image reference.
      2. Looks up question metadata and cached reference solution.
      3. Calls the LLM gate with ``pcr_practice`` caller_id (C4).
      4. Returns the evaluation result synchronously.

    It does NOT:
      - Write to ``evalpen_submissions`` (C3)
      - Write to ``evalpen_evaluations`` (C3)
      - Create any new ExamPen persistence (PCR_EVAL_ENGINE_SPEC §8.2)

    Token logging through the gate is the gate's responsibility and
    does NOT constitute new practice persistence (LLM_GATE_SPEC §11.3).
    """
    # Validate that we have evaluable content
    if not body.text and not body.image_ref:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or 'image_ref' must be provided for evaluation",
        )

    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        from api.v1._exampen_imports import load_exampen

        eval_core = await _build_practice_eval_core(tenant_db)

        # If image_ref is provided without text, run camera OCR to extract it.
        student_text = body.text
        if not student_text and body.image_ref:
            try:
                _ocr_mod = load_exampen("pcr.services.ocr_service")
                _llm_gate = load_exampen("llm_gate")
                LLMVisionCameraAdapter = _ocr_mod.LLMVisionCameraAdapter
                LLMGate = _llm_gate.LLMGate
            except ImportError as ocr_exc:
                logger.error("Camera OCR module import failed: %s", ocr_exc)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Camera OCR pipeline is not available in this deployment",
                )

            gate = LLMGate(tenant_db)
            adapter = LLMVisionCameraAdapter(gate=gate)
            ocr_result = await adapter.recognize_pages(
                [{"page_number": 1, "raw_image_ref": body.image_ref}],
                source="camera",
            )

            # Extract text from all OCR text blocks across pages
            extracted_parts: list[str] = []
            for page_ocr in ocr_result.pages:
                for block in page_ocr.text_blocks:
                    if block.text and block.text.strip():
                        extracted_parts.append(block.text.strip())
            student_text = "\n".join(extracted_parts)

            if not student_text:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Camera OCR could not extract any text from the provided image",
                )

        # Delegate to EvalCore.evaluate_practice() — stateless, no persistence (C3)
        result = await eval_core.evaluate_practice(
            student_response=student_text,
            question_id=body.question_id,
        )

        if result.error:
            logger.warning(
                "Practice evaluation returned error for question=%s: %s",
                body.question_id,
                result.error,
            )

        # Build token usage for response
        token_usage = None
        if result.token_usage:
            token_usage = PracticeTokenUsageAPI(
                caller=result.token_usage.get("caller", "pcr_practice"),
                total_tokens=result.token_usage.get("total_tokens"),
            )

        return PracticeEvaluateResponse(
            question_id=body.question_id,
            total_score=result.total_score,
            max_score=result.max_score,
            feedback=result.overall_feedback or None,
            token_usage=token_usage,
        )

    except ImportError as exc:
        logger.error("PCR evaluation module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR practice evaluation engine is not available in this deployment",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Practice evaluation failed for question=%s: %s",
            body.question_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Practice evaluation encountered an internal error",
        )
