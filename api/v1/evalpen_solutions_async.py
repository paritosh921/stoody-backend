"""
EvalPen PCR Solutions API — Reference solution and question metadata management.

Exposes endpoints for retrieving and upserting versioned reference solutions,
and for registering question metadata (complexity, eval template, max marks).

Architecture:
    PCR_EVAL_ENGINE_SPEC §5.1 (Solution Cache Strategy), §7.4, §7.5

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
    - Writes:  evalpen_solutions (via SolutionRepository),
               evalpen_questions (via QuestionRepository)
    - Reads from: evalpen_solutions, evalpen_questions
    - Never writes to: evalpen_submissions, evalpen_detected_responses,
      practice persistence

Hard constraints:
    - C1: MongoDB only
    - C3: Practice persistence untouched

API authority:
    new-docs/api/eval-solutions.openapi.yaml

Test IDs:
    - U-EVAL-01: eval result parsing and scoring envelope
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

# Two routers: ``router`` for /solutions endpoints, ``questions_router``
# for /questions endpoint.  The integrator should mount them at separate
# prefixes to match the OpenAPI paths:
#   router           -> /api/v1/evalpen/solutions
#   questions_router -> /api/v1/evalpen/questions
router = APIRouter()
questions_router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency: require admin or tutor role for solution management."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for solution management",
        )
    return current_user


# ---------------------------------------------------------------------------
# Request / Response models (match eval-solutions.openapi.yaml exactly)
# ---------------------------------------------------------------------------

class SolutionDetailAPI(BaseModel):
    """Solution detail response.

    Matches SolutionDetail in eval-solutions.openapi.yaml.
    """

    question_id: str
    version: int
    reference_solution: str
    solution_source: str
    model_used: Optional[str] = None


class UpsertSolutionRequest(BaseModel):
    """API request for upserting a solution.

    Matches UpsertSolutionRequest in eval-solutions.openapi.yaml.
    """

    reference_solution: str
    solution_source: str = Field(
        ...,
        description="Solution source: teacher or llm",
    )
    model_used: Optional[str] = None


class QuestionMetadataRequest(BaseModel):
    """API request for registering question metadata.

    Matches QuestionMetadata in eval-solutions.openapi.yaml.
    """

    question_id: str
    exam_id: str
    subject: Optional[str] = None
    question_type: str
    complexity: str = Field(
        ...,
        description="Complexity tier: L1, L2, or L3",
    )
    eval_template: str
    max_marks: int
    expects_diagram: Optional[bool] = None
    diagram_weight: Optional[float] = None


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
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{question_id}",
    response_model=SolutionDetailAPI,
    summary="Get active solution for one question",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "No solution found for question"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_solution(
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> SolutionDetailAPI:
    """Get the active (latest version) reference solution for a question.

    The solution cache key is question-centric (PCR_EVAL_ENGINE_SPEC §5.1).
    Returns the highest-version solution document.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        from api.v1._exampen_imports import load_exampen
        SolutionRepository = load_exampen("pcr.storage").SolutionRepository

        repo = SolutionRepository(tenant_db)
        doc = await repo.get_latest_solution(question_id)

        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No solution found for question {question_id}",
            )

        return SolutionDetailAPI(
            question_id=doc.get("question_id", question_id),
            version=doc.get("version", 1),
            reference_solution=doc.get("reference_solution", ""),
            solution_source=doc.get("solution_source", "teacher"),
            model_used=doc.get("model_used"),
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
            "Failed to get solution for question=%s: %s",
            question_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve solution",
        )


@router.put(
    "/{question_id}",
    summary="Upsert versioned solution for one question",
    responses={
        400: {"description": "Invalid solution data"},
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def upsert_solution(
    question_id: str,
    body: UpsertSolutionRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> SolutionDetailAPI:
    """Upsert a versioned reference solution for a question.

    Solutions are versioned and append-only at the version level
    (PCR_EVAL_ENGINE_SPEC §7.5).  The version number is auto-incremented.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        from api.v1._exampen_imports import load_exampen
        SolutionRepository = load_exampen("pcr.storage").SolutionRepository

        repo = SolutionRepository(tenant_db)
        await repo.ensure_indexes()

        doc = {
            "question_id": question_id,
            "reference_solution": body.reference_solution,
            "solution_source": body.solution_source,
        }
        if body.model_used:
            doc["model_used"] = body.model_used

        result = await repo.upsert_solution(doc)

        return SolutionDetailAPI(
            question_id=result.get("question_id", question_id),
            version=result.get("version", 1),
            reference_solution=result.get("reference_solution", ""),
            solution_source=result.get("solution_source", body.solution_source),
            model_used=result.get("model_used"),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except ImportError as exc:
        logger.error("PCR storage module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to upsert solution for question=%s: %s",
            question_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upsert solution",
        )


@questions_router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Register or update PCR question metadata",
    responses={
        400: {"description": "Invalid question metadata"},
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def register_question(
    body: QuestionMetadataRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Register or update PCR question metadata.

    Questions are mutable (PCR_EVAL_ENGINE_SPEC §7.4) — teachers can
    update complexity, eval template, rubric, and diagram expectations.
    This uses upsert semantics so the latest metadata always wins.
    """
    tenant_db = await _get_tenant_db_for_user(db, current_user)

    try:
        from api.v1._exampen_imports import load_exampen
        QuestionRepository = load_exampen("pcr.storage").QuestionRepository

        repo = QuestionRepository(tenant_db)
        await repo.ensure_indexes()

        doc: Dict[str, Any] = {
            "question_id": body.question_id,
            "exam_id": body.exam_id,
            "question_type": body.question_type,
            "complexity": body.complexity,
            "eval_template": body.eval_template,
            "max_marks": body.max_marks,
        }
        if body.subject is not None:
            doc["subject"] = body.subject
        if body.expects_diagram is not None:
            doc["expects_diagram"] = body.expects_diagram
        if body.diagram_weight is not None:
            doc["diagram_weight"] = body.diagram_weight

        result, was_update = await repo.upsert_question(doc)

        return {
            "question_id": body.question_id,
            "status": "updated" if was_update else "created",
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except ImportError as exc:
        logger.error("PCR storage module import failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PCR engine is not available in this deployment",
        )
    except Exception as exc:
        logger.error(
            "Failed to register question=%s: %s",
            body.question_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register question metadata",
        )
