"""
Question Attempt API

QLock (Question Lock) sessions for SmartBoard.
When a teacher locks a question, all connected pens are tracked and their
strokes are tagged with the question_attempt_id for later collection.

Auth: Accepts SmartBoard cloud JWT via Authorization: Bearer header.
Tenant isolation enforced via JWT claims.
Feature gating: smartboard_cloud_access required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .dashboard import dashboard_ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/question-attempts", tags=["question-attempts"])

ATTEMPTS_DIR = Path(__file__).resolve().parents[2] / "data" / "question_attempts"

EVAL_CONCURRENCY = 4


def _verify_tenant(attempt: QuestionAttempt, user: dict):
    """Raise 403 if the authenticated user's tenant does not own this attempt."""
    attempt_tenant = attempt.tenant_id
    user_tenant = user.get("tenant_id") or user.get("db_name")
    if attempt_tenant and user_tenant and attempt_tenant != user_tenant:
        raise HTTPException(status_code=403, detail="Tenant mismatch — access denied")


def ensure_attempts_dir():
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)


def get_app_state():
    from main_async import app
    return app.state


async def _require_smartboard_auth(request: Request) -> dict:
    from api.v1.auth_async import get_current_user
    from core.tenant_features import is_feature_enabled
    from fastapi.security import HTTPBearer

    security = HTTPBearer()
    try:
        credentials = await security(request)
        user = await get_current_user(request, credentials, get_app_state().auth)
    except Exception:
        raise HTTPException(status_code=401, detail="SmartBoard JWT required")

    if not is_feature_enabled(
        user.get("enabled_features"),
        "smartboard_cloud_access",
        user.get("enabled_features_v2"),
    ):
        raise HTTPException(
            status_code=403,
            detail="Smartboard cloud access not enabled for this institution",
        )

    user_type = (user.get("user_type") or user.get("role") or "").lower()
    if user_type not in ("tutor", "teacher", "admin"):
        raise HTTPException(status_code=403, detail="Only tutors/teachers can use QLock")

    return user


async def _resolve_tenant_db(user: dict):
    """Resolve the tenant database for LLM gate routing."""
    db_name = user.get("db_name") or user.get("tenant_id")
    if not db_name:
        return None

    try:
        db_manager = getattr(get_app_state(), "db", None)
        if db_manager:
            return await db_manager.get_tenant_db(db_name)
    except Exception as exc:
        logger.warning("[QLock] Could not resolve tenant DB for gate routing: %s", exc)

    return None


# --- Data Models ---

class QuestionBounds(BaseModel):
    x: float
    y: float
    width: float
    height: float


class CreateQuestionAttemptRequest(BaseModel):
    question_text: str
    question_image_b64: Optional[str] = None
    bounds: Optional[QuestionBounds] = None
    duration: int = Field(default=120, ge=30, le=600)
    pen_ids: List[str] = Field(default_factory=list)


class EndQuestionAttemptRequest(BaseModel):
    reason: Literal['teacher_closed', 'timer_expired', 'all_submitted'] = 'teacher_closed'


class SubmitResponseRequest(BaseModel):
    pen_id: str


class PenResponseStatus(BaseModel):
    pen_id: str
    status: Literal['writing', 'submitted', 'timeout'] = 'writing'
    submit_ts: Optional[float] = None
    pages_written: List[int] = Field(default_factory=list)
    stroke_count: int = 0


class QuestionAttempt(BaseModel):
    id: str
    question_text: str
    question_image_b64: Optional[str] = None
    bounds: Optional[QuestionBounds] = None
    lock_ts: float
    end_ts: Optional[float] = None
    duration: int = 120
    status: Literal['active', 'collecting', 'completed'] = 'active'
    connected_pens: List[str] = Field(default_factory=list)
    pen_responses: Dict[str, PenResponseStatus] = Field(default_factory=dict)
    ai_solution: Optional[str] = None
    ai_final_answer: Optional[str] = None
    tenant_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuestionAttemptResponse(BaseModel):
    id: str
    question_text: str
    lock_ts: float
    end_ts: Optional[float] = None
    duration: int
    status: str
    connected_pens: List[str]
    pen_responses: Dict[str, PenResponseStatus]


class StrokePoint(BaseModel):
    x: float
    y: float
    pressure: Optional[float] = None


class TaggedStroke(BaseModel):
    id: str
    pen_id: str
    question_attempt_id: str
    points: List[StrokePoint]
    color: Optional[str] = None
    strokeWidth: Optional[float] = None
    timestamp: float
    page_no: int
    book_type: Optional[str] = None


class EvaluationResult(BaseModel):
    success: bool
    pen_id: str
    score: Optional[str] = None
    extracted_answer: Optional[str] = None
    correct_answer: Optional[str] = None
    feedback: Optional[str] = None
    ai_solution: Optional[str] = None
    error: Optional[str] = None


class EvaluateRequest(BaseModel):
    pen_id: str
    answer_image_b64: str


class EvaluateAllRequest(BaseModel):
    pen_images: Dict[str, str]


class EvaluateAllResponse(BaseModel):
    success: bool
    attempt_id: str
    results: Dict[str, EvaluationResult]
    errors: Optional[List[str]] = None


class GenerateSolutionResponse(BaseModel):
    success: bool
    attempt_id: str
    solution: Optional[str] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None


# --- In-Memory Storage ---

_active_attempts: Dict[str, QuestionAttempt] = {}
_pen_to_attempt: Dict[str, str] = {}
_attempt_strokes: Dict[str, List[Dict[str, Any]]] = {}


def _save_attempt(attempt: QuestionAttempt):
    ensure_attempts_dir()
    filepath = ATTEMPTS_DIR / f"{attempt.id}.json"
    with open(filepath, "w") as f:
        json.dump(attempt.model_dump(), f, indent=2, default=str)


def _load_attempt(attempt_id: str) -> Optional[QuestionAttempt]:
    filepath = ATTEMPTS_DIR / f"{attempt_id}.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            data = json.load(f)
            return QuestionAttempt(**data)
    return None


# --- Public API for stroke tagging ---

def get_active_attempt_for_pen(pen_id: str) -> Optional[str]:
    return _pen_to_attempt.get(pen_id)


def add_stroke_to_attempt(attempt_id: str, stroke: Dict[str, Any], pen_id: str, page_no: int):
    if attempt_id not in _attempt_strokes:
        _attempt_strokes[attempt_id] = []

    stroke_with_meta = {
        **stroke,
        "question_attempt_id": attempt_id,
        "pen_id": pen_id,
        "page_no": page_no,
    }
    _attempt_strokes[attempt_id].append(stroke_with_meta)

    if attempt_id in _active_attempts:
        attempt = _active_attempts[attempt_id]
        if pen_id in attempt.pen_responses:
            response = attempt.pen_responses[pen_id]
            response.stroke_count += 1
            if page_no not in response.pages_written:
                response.pages_written.append(page_no)


def get_strokes_for_attempt(attempt_id: str, pen_id: Optional[str] = None) -> List[Dict[str, Any]]:
    strokes = _attempt_strokes.get(attempt_id, [])
    if pen_id:
        strokes = [s for s in strokes if s.get("pen_id") == pen_id]
    return strokes


# --- API Endpoints ---

@router.post("", response_model=QuestionAttemptResponse)
async def create_question_attempt(
    request: Request,
    payload: CreateQuestionAttemptRequest,
):
    user = await _require_smartboard_auth(request)

    connected_pens = payload.pen_ids
    if not connected_pens:
        try:
            state = get_app_state()
            pens_data = await state.dashboard_registry.list_pen_states()
            connected_pens = [p["pen_id"] for p in pens_data if p.get("connected")]
        except Exception:
            connected_pens = []

    attempt_id = f"qa-{uuid4().hex[:12]}"
    now = time.time() * 1000

    attempt = QuestionAttempt(
        id=attempt_id,
        question_text=payload.question_text,
        question_image_b64=payload.question_image_b64,
        bounds=payload.bounds,
        lock_ts=now,
        duration=payload.duration,
        status='active',
        connected_pens=connected_pens,
        pen_responses={
            pen_id: PenResponseStatus(pen_id=pen_id)
            for pen_id in connected_pens
        },
        tenant_id=user.get("tenant_id") or user.get("db_name"),
    )

    _active_attempts[attempt_id] = attempt
    _attempt_strokes[attempt_id] = []

    for pen_id in connected_pens:
        _pen_to_attempt[pen_id] = attempt_id

    _save_attempt(attempt)

    await dashboard_ws_manager.broadcast({
        "type": "question_lock",
        "attempt_id": attempt_id,
        "question_text": payload.question_text,
        "lock_ts": now,
        "duration": payload.duration,
        "connected_pens": connected_pens,
    })

    logger.info(f"[QLock] Created attempt {attempt_id} with {len(connected_pens)} pens")

    return QuestionAttemptResponse(
        id=attempt.id,
        question_text=attempt.question_text,
        lock_ts=attempt.lock_ts,
        end_ts=attempt.end_ts,
        duration=attempt.duration,
        status=attempt.status,
        connected_pens=attempt.connected_pens,
        pen_responses=attempt.pen_responses,
    )


@router.get("/{attempt_id}", response_model=QuestionAttemptResponse)
async def get_question_attempt(request: Request, attempt_id: str):
    user = await _require_smartboard_auth(request)

    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")

    _verify_tenant(attempt, user)

    return QuestionAttemptResponse(
        id=attempt.id,
        question_text=attempt.question_text,
        lock_ts=attempt.lock_ts,
        end_ts=attempt.end_ts,
        duration=attempt.duration,
        status=attempt.status,
        connected_pens=attempt.connected_pens,
        pen_responses=attempt.pen_responses,
    )


@router.post("/{attempt_id}/end")
async def end_question_attempt(
    request: Request,
    attempt_id: str,
    payload: EndQuestionAttemptRequest,
):
    user = await _require_smartboard_auth(request)

    attempt = _active_attempts.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")

    _verify_tenant(attempt, user)

    if attempt.status != 'active':
        raise HTTPException(status_code=400, detail="Question attempt is not active")

    now = time.time() * 1000
    attempt.end_ts = now
    attempt.status = 'completed'

    for pen_id, response in attempt.pen_responses.items():
        if response.status == 'writing':
            response.status = 'timeout'
            response.submit_ts = now

    for pen_id in attempt.connected_pens:
        if _pen_to_attempt.get(pen_id) == attempt_id:
            del _pen_to_attempt[pen_id]

    _save_attempt(attempt)

    await dashboard_ws_manager.broadcast({
        "type": "question_end",
        "attempt_id": attempt_id,
        "reason": payload.reason,
        "end_ts": now,
    })

    strokes = get_strokes_for_attempt(attempt_id)
    logger.info(f"[QLock] Ended attempt {attempt_id}: {len(strokes)} strokes collected")

    return {
        "status": "ok",
        "attempt_id": attempt_id,
        "reason": payload.reason,
        "stroke_count": len(strokes),
        "pen_count": len(attempt.connected_pens),
    }


@router.post("/{attempt_id}/submit")
async def submit_response(
    request: Request,
    attempt_id: str,
    payload: SubmitResponseRequest,
):
    user = await _require_smartboard_auth(request)

    attempt = _active_attempts.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")

    _verify_tenant(attempt, user)

    if attempt.status != 'active':
        raise HTTPException(status_code=400, detail="Question attempt is not active")

    if payload.pen_id not in attempt.pen_responses:
        raise HTTPException(status_code=400, detail="Pen not part of this question attempt")

    now = time.time() * 1000
    response = attempt.pen_responses[payload.pen_id]
    response.status = 'submitted'
    response.submit_ts = now

    if _pen_to_attempt.get(payload.pen_id) == attempt_id:
        del _pen_to_attempt[payload.pen_id]

    _save_attempt(attempt)

    await dashboard_ws_manager.broadcast({
        "type": "student_submit",
        "attempt_id": attempt_id,
        "pen_id": payload.pen_id,
        "submit_ts": now,
    })

    strokes = get_strokes_for_attempt(attempt_id, payload.pen_id)
    return {
        "status": "ok",
        "pen_id": payload.pen_id,
        "submit_ts": now,
        "stroke_count": len(strokes),
    }


@router.get("/{attempt_id}/strokes")
async def get_attempt_strokes(request: Request, attempt_id: str, pen_id: Optional[str] = None):
    user = await _require_smartboard_auth(request)

    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")

    _verify_tenant(attempt, user)

    strokes = get_strokes_for_attempt(attempt_id, pen_id)

    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for stroke in strokes:
        page = stroke.get("page_no", 0)
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(stroke)

    return {
        "attempt_id": attempt_id,
        "pen_id": pen_id,
        "total_strokes": len(strokes),
        "strokes": strokes,
        "by_page": by_page,
    }


@router.post("/{attempt_id}/generate-solution", response_model=GenerateSolutionResponse)
async def generate_solution(request: Request, attempt_id: str):
    user = await _require_smartboard_auth(request)

    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")

    _verify_tenant(attempt, user)

    if attempt.ai_solution:
        return GenerateSolutionResponse(
            success=True,
            attempt_id=attempt_id,
            solution=attempt.ai_solution,
            final_answer=attempt.ai_final_answer,
        )

    from core.ocr_service import get_ocr_service
    ocr_service = get_ocr_service()
    tenant_db = await _resolve_tenant_db(user)

    try:
        result = await ocr_service.generate_solution(
            question_text=attempt.question_text,
            question_image_b64=attempt.question_image_b64,
            tenant_db=tenant_db,
        )
    except Exception as e:
        logger.error(f"[QLock] Solution generation failed: {e}")
        return GenerateSolutionResponse(
            success=False,
            attempt_id=attempt_id,
            error=str(e),
        )

    if result.get("success"):
        attempt.ai_solution = result.get("solution")
        attempt.ai_final_answer = result.get("final_answer")
        _save_attempt(attempt)

    return GenerateSolutionResponse(
        success=result.get("success", False),
        attempt_id=attempt_id,
        solution=result.get("solution"),
        final_answer=result.get("final_answer"),
        error=result.get("error"),
    )


@router.post("/{attempt_id}/evaluate", response_model=EvaluationResult)
async def evaluate_response(request: Request, attempt_id: str, payload: EvaluateRequest):
    user = await _require_smartboard_auth(request)

    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")

    _verify_tenant(attempt, user)

    question_text = attempt.question_text
    correct_answer = attempt.ai_final_answer

    from core.ocr_service import get_ocr_service
    ocr_service = get_ocr_service()
    tenant_db = await _resolve_tenant_db(user)

    result = await ocr_service.evaluate_answer(
        question_text=question_text,
        answer_image_b64=payload.answer_image_b64,
        tenant_db=tenant_db,
        correct_answer=correct_answer,
    )

    if not result.get("success"):
        return EvaluationResult(
            success=False,
            pen_id=payload.pen_id,
            error=result.get("error", "Evaluation failed"),
        )

    score = result.get("score", "inconclusive")
    if score not in ("correct", "incorrect", "partial", "inconclusive"):
        score = "inconclusive"

    logger.info(f"[QLock] Evaluated pen {payload.pen_id}: {score}")

    return EvaluationResult(
        success=True,
        pen_id=payload.pen_id,
        score=score,
        extracted_answer=result.get("extracted_answer"),
        correct_answer=correct_answer or result.get("correct_answer"),
        feedback=result.get("feedback"),
        ai_solution=attempt.ai_solution if score != "correct" else None,
    )


@router.post("/{attempt_id}/evaluate-all", response_model=EvaluateAllResponse)
async def evaluate_all_responses(request: Request, attempt_id: str, payload: EvaluateAllRequest):
    user = await _require_smartboard_auth(request)

    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")

    _verify_tenant(attempt, user)

    question_text = attempt.question_text
    correct_answer = attempt.ai_final_answer

    from core.ocr_service import get_ocr_service
    ocr_service = get_ocr_service()
    tenant_db = await _resolve_tenant_db(user)

    sem = asyncio.Semaphore(EVAL_CONCURRENCY)

    async def evaluate_single(pen_id: str, image_b64: str) -> tuple:
        async with sem:
            try:
                result = await ocr_service.evaluate_answer(
                    question_text=question_text,
                    answer_image_b64=image_b64,
                    tenant_db=tenant_db,
                    correct_answer=correct_answer,
                )
                score = result.get("score", "inconclusive")
                if score not in ("correct", "incorrect", "partial", "inconclusive"):
                    score = "inconclusive"

                return (pen_id, EvaluationResult(
                    success=result.get("success", False),
                    pen_id=pen_id,
                    score=score,
                    extracted_answer=result.get("extracted_answer"),
                    correct_answer=correct_answer or result.get("correct_answer"),
                    feedback=result.get("feedback"),
                    ai_solution=attempt.ai_solution if score != "correct" else None,
                    error=result.get("error"),
                ))
            except Exception as e:
                logger.error(f"[QLock] Evaluation failed for pen {pen_id}: {e}")
                return (pen_id, EvaluationResult(
                    success=False,
                    pen_id=pen_id,
                    score="inconclusive",
                    error=str(e),
                ))

    tasks = [
        evaluate_single(pen_id, image_b64)
        for pen_id, image_b64 in payload.pen_images.items()
    ]

    results_list = await asyncio.gather(*tasks)
    results = {pen_id: result for pen_id, result in results_list}

    errors = [
        f"{pen_id}: {r.error}"
        for pen_id, r in results.items()
        if not r.success and r.error
    ]

    logger.info(f"[QLock] Evaluated {len(results)} responses for attempt {attempt_id}")

    return EvaluateAllResponse(
        success=len(errors) == 0,
        attempt_id=attempt_id,
        results=results,
        errors=errors if errors else None,
    )
