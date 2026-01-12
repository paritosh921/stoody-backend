"""
Question Attempt API

This module handles question lock sessions and response collection.
When a teacher locks a question, all connected pens are tracked and their
strokes are tagged with the question_attempt_id for later collection.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .dashboard import dashboard_ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/question-attempts", tags=["question-attempts"])

# Storage path for question attempts
ATTEMPTS_DIR = Path(__file__).resolve().parents[2] / "data" / "question_attempts"


def ensure_attempts_dir():
    """Ensure the attempts directory exists."""
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)


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
    auto_collect_after_ms: int = Field(default=120000, ge=30000, le=600000)  # 30s to 10min


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
    auto_collect_after_ms: int = 120000
    status: Literal['active', 'collecting', 'completed'] = 'active'
    connected_pens: List[str] = Field(default_factory=list)
    pen_responses: Dict[str, PenResponseStatus] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuestionAttemptResponse(BaseModel):
    id: str
    question_text: str
    lock_ts: float
    end_ts: Optional[float] = None
    auto_collect_after_ms: int
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


# --- In-Memory Storage (will be persisted to disk) ---

# Active question attempts
_active_attempts: Dict[str, QuestionAttempt] = {}

# Mapping of pen_id -> active attempt_id (for stroke tagging)
_pen_to_attempt: Dict[str, str] = {}

# Strokes tagged with question_attempt_id
_attempt_strokes: Dict[str, List[Dict[str, Any]]] = {}


def get_app_state():
    from main_async import app
    return app.state


def _save_attempt(attempt: QuestionAttempt):
    """Save attempt to disk."""
    ensure_attempts_dir()
    filepath = ATTEMPTS_DIR / f"{attempt.id}.json"
    with open(filepath, "w") as f:
        json.dump(attempt.model_dump(), f, indent=2, default=str)


def _load_attempt(attempt_id: str) -> Optional[QuestionAttempt]:
    """Load attempt from disk."""
    filepath = ATTEMPTS_DIR / f"{attempt_id}.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            data = json.load(f)
            return QuestionAttempt(**data)
    return None


# --- Public API for stroke tagging (used by pen_workers) ---

def get_active_attempt_for_pen(pen_id: str) -> Optional[str]:
    """Get the active question attempt ID for a pen, if any."""
    return _pen_to_attempt.get(pen_id)


def add_stroke_to_attempt(attempt_id: str, stroke: Dict[str, Any], pen_id: str, page_no: int):
    """Add a stroke to an active attempt."""
    if attempt_id not in _attempt_strokes:
        _attempt_strokes[attempt_id] = []
    
    stroke_with_meta = {
        **stroke,
        "question_attempt_id": attempt_id,
        "pen_id": pen_id,
        "page_no": page_no,
    }
    _attempt_strokes[attempt_id].append(stroke_with_meta)
    
    # Update pen response tracking
    if attempt_id in _active_attempts:
        attempt = _active_attempts[attempt_id]
        if pen_id in attempt.pen_responses:
            response = attempt.pen_responses[pen_id]
            response.stroke_count += 1
            if page_no not in response.pages_written:
                response.pages_written.append(page_no)


def get_strokes_for_attempt(attempt_id: str, pen_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all strokes for an attempt, optionally filtered by pen_id."""
    strokes = _attempt_strokes.get(attempt_id, [])
    if pen_id:
        strokes = [s for s in strokes if s.get("pen_id") == pen_id]
    return strokes


# --- API Endpoints ---

@router.post("", response_model=QuestionAttemptResponse)
async def create_question_attempt(
    payload: CreateQuestionAttemptRequest,
    token: Optional[str] = None,
    state=Depends(get_app_state)
):
    """
    Create a new question attempt (lock a question).
    All connected pens will be tracked for this question.
    """
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    # Get all connected pens (may be empty if no pens connected yet)
    pens_data = await state.dashboard_registry.list_pen_states()
    connected_pens = [p["pen_id"] for p in pens_data if p.get("connected")]
    
    # Note: We allow creating attempts with 0 pens - pens can join later
    # This supports the "teacher prep" workflow where question is locked before students start
    
    # Create new attempt
    attempt_id = f"qa-{uuid4().hex[:12]}"
    now = time.time() * 1000  # milliseconds
    
    attempt = QuestionAttempt(
        id=attempt_id,
        question_text=payload.question_text,
        question_image_b64=payload.question_image_b64,
        bounds=payload.bounds,
        lock_ts=now,
        auto_collect_after_ms=payload.auto_collect_after_ms,
        status='active',
        connected_pens=connected_pens,
        pen_responses={
            pen_id: PenResponseStatus(pen_id=pen_id)
            for pen_id in connected_pens
        },
    )
    
    # Store in memory
    _active_attempts[attempt_id] = attempt
    _attempt_strokes[attempt_id] = []
    
    # Map all connected pens to this attempt (for stroke tagging)
    for pen_id in connected_pens:
        _pen_to_attempt[pen_id] = attempt_id
    
    # Save to disk
    _save_attempt(attempt)
    
    # Broadcast to all connected frontends
    await dashboard_ws_manager.broadcast({
        "type": "question_lock",
        "attempt_id": attempt_id,
        "question_text": payload.question_text,
        "lock_ts": now,
        "auto_collect_after_ms": payload.auto_collect_after_ms,
        "connected_pens": connected_pens,
    })
    
    logger.info(f"Created question attempt {attempt_id} with {len(connected_pens)} pens")
    
    return QuestionAttemptResponse(
        id=attempt.id,
        question_text=attempt.question_text,
        lock_ts=attempt.lock_ts,
        end_ts=attempt.end_ts,
        auto_collect_after_ms=attempt.auto_collect_after_ms,
        status=attempt.status,
        connected_pens=attempt.connected_pens,
        pen_responses=attempt.pen_responses,
    )


@router.get("/{attempt_id}", response_model=QuestionAttemptResponse)
async def get_question_attempt(
    attempt_id: str,
    token: Optional[str] = None,
    state=Depends(get_app_state)
):
    """Get the status of a question attempt."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")
    
    return QuestionAttemptResponse(
        id=attempt.id,
        question_text=attempt.question_text,
        lock_ts=attempt.lock_ts,
        end_ts=attempt.end_ts,
        auto_collect_after_ms=attempt.auto_collect_after_ms,
        status=attempt.status,
        connected_pens=attempt.connected_pens,
        pen_responses=attempt.pen_responses,
    )


@router.post("/{attempt_id}/end")
async def end_question_attempt(
    attempt_id: str,
    payload: EndQuestionAttemptRequest,
    token: Optional[str] = None,
    state=Depends(get_app_state)
):
    """
    End a question attempt (stop collecting responses).
    This marks any unsubmitted pens as 'timeout'.
    """
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    attempt = _active_attempts.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")
    
    if attempt.status != 'active':
        raise HTTPException(status_code=400, detail="Question attempt is not active")
    
    # Mark as completed
    now = time.time() * 1000
    attempt.end_ts = now
    attempt.status = 'completed'
    
    # Mark unsubmitted pens as timeout
    for pen_id, response in attempt.pen_responses.items():
        if response.status == 'writing':
            response.status = 'timeout'
            response.submit_ts = now
    
    # Remove pen-to-attempt mappings
    for pen_id in attempt.connected_pens:
        if _pen_to_attempt.get(pen_id) == attempt_id:
            del _pen_to_attempt[pen_id]
    
    # Save to disk
    _save_attempt(attempt)
    
    # Broadcast end
    await dashboard_ws_manager.broadcast({
        "type": "question_end",
        "attempt_id": attempt_id,
        "reason": payload.reason,
        "end_ts": now,
    })
    
    strokes = get_strokes_for_attempt(attempt_id)
    logger.info(f"Ended question attempt {attempt_id}: {len(strokes)} strokes collected")
    
    return {
        "status": "ok",
        "attempt_id": attempt_id,
        "reason": payload.reason,
        "stroke_count": len(strokes),
        "pen_count": len(attempt.connected_pens),
    }


@router.post("/{attempt_id}/submit")
async def submit_response(
    attempt_id: str,
    payload: SubmitResponseRequest,
    token: Optional[str] = None,
    state=Depends(get_app_state)
):
    """
    Mark a student's response as submitted.
    Called when student taps submit button or double-tap is detected.
    """
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    attempt = _active_attempts.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")
    
    if attempt.status != 'active':
        raise HTTPException(status_code=400, detail="Question attempt is not active")
    
    if payload.pen_id not in attempt.pen_responses:
        raise HTTPException(status_code=400, detail="Pen not part of this question attempt")
    
    # Mark as submitted
    now = time.time() * 1000
    response = attempt.pen_responses[payload.pen_id]
    response.status = 'submitted'
    response.submit_ts = now
    
    # Remove pen from active tracking
    if _pen_to_attempt.get(payload.pen_id) == attempt_id:
        del _pen_to_attempt[payload.pen_id]
    
    # Save to disk
    _save_attempt(attempt)
    
    # Broadcast submit
    await dashboard_ws_manager.broadcast({
        "type": "student_submit",
        "attempt_id": attempt_id,
        "pen_id": payload.pen_id,
        "submit_ts": now,
    })
    
    strokes = get_strokes_for_attempt(attempt_id, payload.pen_id)
    logger.info(f"Pen {payload.pen_id} submitted for attempt {attempt_id}: {len(strokes)} strokes")
    
    return {
        "status": "ok",
        "pen_id": payload.pen_id,
        "submit_ts": now,
        "stroke_count": len(strokes),
    }


@router.get("/{attempt_id}/strokes")
async def get_attempt_strokes(
    attempt_id: str,
    pen_id: Optional[str] = None,
    token: Optional[str] = None,
    state=Depends(get_app_state)
):
    """
    Get all strokes collected for a question attempt.
    Optionally filter by pen_id.
    Returns strokes grouped by page number.
    """
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")
    
    strokes = get_strokes_for_attempt(attempt_id, pen_id)
    
    # Group by page
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


class EvaluateRequest(BaseModel):
    pen_id: str
    answer_image_b64: str  # Base64 PNG of rendered strokes


class EvaluationResult(BaseModel):
    success: bool
    pen_id: str
    score: Optional[Literal['correct', 'incorrect', 'partial']] = None
    extracted_answer: Optional[str] = None
    feedback: Optional[str] = None
    error: Optional[str] = None


@router.post("/{attempt_id}/evaluate", response_model=EvaluationResult)
async def evaluate_response(
    attempt_id: str,
    payload: EvaluateRequest,
    token: Optional[str] = None,
    state=Depends(get_app_state)
):
    """
    Evaluate a student's response using AI.
    
    Expects:
    - pen_id: Which student's response to evaluate
    - answer_image_b64: Base64 PNG of the student's rendered strokes
    
    Returns:
    - score: 'correct', 'incorrect', or 'partial'
    - extracted_answer: What the AI read from the handwriting
    - feedback: Brief feedback for the student
    """
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    # Get the attempt
    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")
    
    if payload.pen_id not in attempt.pen_responses:
        raise HTTPException(status_code=400, detail="Pen not part of this question attempt")
    
    # Get the question text
    question_text = attempt.question_text
    
    # Import OCR service and evaluate
    from core.ocr_service import get_ocr_service
    ocr_service = get_ocr_service()
    
    result = await ocr_service.evaluate_answer(
        question_text=question_text,
        answer_image_b64=payload.answer_image_b64,
    )
    
    if not result.get("success"):
        return EvaluationResult(
            success=False,
            pen_id=payload.pen_id,
            error=result.get("error", "Evaluation failed"),
        )
    
    # Update the pen response with evaluation
    if attempt_id in _active_attempts:
        response = _active_attempts[attempt_id].pen_responses.get(payload.pen_id)
        if response:
            # Store evaluation in the response (would need to extend PenResponseStatus)
            pass
    
    logger.info(f"Evaluated response for pen {payload.pen_id}: {result.get('score')}")
    
    return EvaluationResult(
        success=True,
        pen_id=payload.pen_id,
        score=result.get("score"),
        extracted_answer=result.get("extracted_answer"),
        feedback=result.get("feedback"),
    )


class EvaluateAllRequest(BaseModel):
    """Request to evaluate all pens in a question attempt."""
    pen_images: Dict[str, str]  # Map of pen_id -> base64 PNG of rendered strokes


class EvaluateAllResponse(BaseModel):
    """Response with evaluation results for all pens."""
    success: bool
    attempt_id: str
    results: Dict[str, EvaluationResult]
    errors: Optional[List[str]] = None


@router.post("/{attempt_id}/evaluate-all", response_model=EvaluateAllResponse)
async def evaluate_all_responses(
    attempt_id: str,
    payload: EvaluateAllRequest,
    token: Optional[str] = None,
    state=Depends(get_app_state)
):
    """
    Evaluate all students' responses for a question attempt.
    
    This endpoint processes multiple pens in parallel for efficiency.
    
    Expects:
    - pen_images: Dict mapping pen_id to base64 PNG of rendered strokes
    
    Returns:
    - results: Dict mapping pen_id to EvaluationResult
    """
    import asyncio
    
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    # Get the attempt
    attempt = _active_attempts.get(attempt_id) or _load_attempt(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")
    
    question_text = attempt.question_text
    
    # Import OCR service
    from core.ocr_service import get_ocr_service
    ocr_service = get_ocr_service()
    
    # Evaluate all pens in parallel
    async def evaluate_single(pen_id: str, image_b64: str) -> tuple:
        try:
            result = await ocr_service.evaluate_answer(
                question_text=question_text,
                answer_image_b64=image_b64,
            )
            return (pen_id, EvaluationResult(
                success=result.get("success", False),
                pen_id=pen_id,
                score=result.get("score"),
                extracted_answer=result.get("extracted_answer"),
                feedback=result.get("feedback"),
                error=result.get("error"),
            ))
        except Exception as e:
            logger.error(f"Evaluation failed for pen {pen_id}: {e}")
            return (pen_id, EvaluationResult(
                success=False,
                pen_id=pen_id,
                error=str(e),
            ))
    
    # Create tasks for all pens
    tasks = [
        evaluate_single(pen_id, image_b64)
        for pen_id, image_b64 in payload.pen_images.items()
    ]
    
    # Run all evaluations in parallel
    results_list = await asyncio.gather(*tasks)
    results = {pen_id: result for pen_id, result in results_list}
    
    # Collect errors
    errors = [
        f"{pen_id}: {r.error}"
        for pen_id, r in results.items()
        if not r.success and r.error
    ]
    
    logger.info(f"Evaluated {len(results)} responses for attempt {attempt_id}")
    
    return EvaluateAllResponse(
        success=len(errors) == 0,
        attempt_id=attempt_id,
        results=results,
        errors=errors if errors else None,
    )
