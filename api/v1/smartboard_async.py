"""
SmartBoard API for real-time pen monitoring and teaching sessions.

Provides:
- Session management for SmartBoard teaching sessions
- WebSocket for real-time pen data streaming
- Question attempts and AI evaluation
- Student pen tracking
"""

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Auth Dependencies
# =============================================================================

def require_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require tutor access"""
    if current_user.get("user_type") != "tutor":
        raise HTTPException(status_code=403, detail="Tutor access required")
    return current_user


# =============================================================================
# Models
# =============================================================================

class StrokePoint(BaseModel):
    x: float
    y: float
    pressure: Optional[float] = None


class DashboardStroke(BaseModel):
    id: Optional[str] = None
    points: List[Dict[str, Any]]
    color: Optional[str] = None
    strokeWidth: Optional[float] = None
    timestamp: Optional[float] = None


class PenSummary(BaseModel):
    pen_id: str
    pen_mac: str
    student_name: Optional[str] = None
    student_id: Optional[str] = None
    connected: bool = False
    battery: Optional[int] = None
    page_no: Optional[int] = None
    book_type: Optional[str] = None
    last_frame_ts: Optional[float] = None
    strokes: List[DashboardStroke] = []
    color: Optional[str] = None


class PensResponse(BaseModel):
    pens: List[PenSummary]
    maxPens: int = 30


class CreateSessionRequest(BaseModel):
    standard: str
    section: str
    subject: str
    topic: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    tutor_id: str
    standard: str
    section: str
    subject: str
    topic: Optional[str] = None
    status: str
    connected_pens: List[str] = []
    started_at: datetime


class QuestionAttemptRequest(BaseModel):
    question_text: str
    question_image_b64: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    auto_collect_after_ms: Optional[int] = None  # 30000 - 600000


class QuestionAttemptResponse(BaseModel):
    attempt_id: str
    session_id: str
    question_text: str
    status: str  # active, collecting, ended
    created_at: datetime
    auto_collect_at: Optional[datetime] = None


class EvaluateRequest(BaseModel):
    pen_id: str
    answer_image_b64: str


class EvaluateResponse(BaseModel):
    success: bool
    score: str  # correct, incorrect, partial
    extracted_answer: str
    feedback: str
    error: Optional[str] = None


# =============================================================================
# In-Memory State (for WebSocket & Session tracking)
# =============================================================================

@dataclass
class SmartBoardSession:
    """Active SmartBoard teaching session"""
    session_id: str
    tutor_id: str
    tutor_ws: Optional[WebSocket] = None
    standard: str = ""
    section: str = ""
    subject: str = ""
    topic: Optional[str] = None
    status: str = "active"
    connected_pens: Set[str] = field(default_factory=set)
    pen_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)


# Global session storage (consider Redis for production)
_sessions: Dict[str, SmartBoardSession] = {}
_tutor_sessions: Dict[str, str] = {}  # tutor_id -> session_id


class SmartBoardWebSocketManager:
    """Manages WebSocket connections for SmartBoard dashboard"""

    def __init__(self):
        self.connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.connections:
                self.connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected dashboards"""
        async with self.lock:
            dead = []
            for ws in self.connections:
                if ws.application_state != WebSocketState.CONNECTED:
                    dead.append(ws)
                    continue
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in self.connections:
                    self.connections.remove(ws)

    async def send_to_session(self, session_id: str, message: dict):
        """Send message to a specific session's tutor"""
        session = _sessions.get(session_id)
        if session and session.tutor_ws:
            try:
                if session.tutor_ws.application_state == WebSocketState.CONNECTED:
                    await session.tutor_ws.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to session {session_id}: {e}")


ws_manager = SmartBoardWebSocketManager()


# =============================================================================
# Helper Functions
# =============================================================================

def get_db(request: Request) -> DatabaseManager:
    """Get database manager from app state"""
    return request.app.state.db


# =============================================================================
# Session Management Endpoints
# =============================================================================

@router.post("/sessions", response_model=SessionResponse)
async def create_smartboard_session(
    data: CreateSessionRequest,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """Create a new SmartBoard teaching session"""
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    # Check if tutor already has an active session
    existing_session_id = _tutor_sessions.get(tutor_id)
    if existing_session_id and existing_session_id in _sessions:
        existing = _sessions[existing_session_id]
        if existing.status == "active":
            raise HTTPException(
                status_code=400,
                detail="You already have an active SmartBoard session"
            )

    session_id = f"SB-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    session = SmartBoardSession(
        session_id=session_id,
        tutor_id=tutor_id,
        standard=data.standard,
        section=data.section,
        subject=data.subject,
        topic=data.topic,
    )

    _sessions[session_id] = session
    _tutor_sessions[tutor_id] = session_id

    # Save to database
    db = get_db(request)
    await db.mongo_insert_one("smartboard_sessions", {
        "session_id": session_id,
        "tutor_id": tutor_id,
        "standard": data.standard,
        "section": data.section,
        "subject": data.subject,
        "topic": data.topic,
        "status": "active",
        "connected_pens": [],
        "started_at": session.started_at,
    })

    logger.info(f"Created SmartBoard session {session_id} for tutor {tutor_id}")

    return SessionResponse(
        session_id=session_id,
        tutor_id=tutor_id,
        standard=data.standard,
        section=data.section,
        subject=data.subject,
        topic=data.topic,
        status="active",
        connected_pens=[],
        started_at=session.started_at,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_smartboard_session(
    session_id: str,
    current_user: dict = Depends(require_tutor)
):
    """Get SmartBoard session details"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        session_id=session.session_id,
        tutor_id=session.tutor_id,
        standard=session.standard,
        section=session.section,
        subject=session.subject,
        topic=session.topic,
        status=session.status,
        connected_pens=list(session.connected_pens),
        started_at=session.started_at,
    )


@router.put("/sessions/{session_id}/end")
async def end_smartboard_session(
    session_id: str,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """End a SmartBoard session"""
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.tutor_id != tutor_id:
        raise HTTPException(status_code=403, detail="Not your session")

    session.status = "ended"

    # Update database
    db = get_db(request)
    await db.mongo_update_one(
        "smartboard_sessions",
        {"session_id": session_id},
        {"$set": {"status": "ended", "ended_at": datetime.utcnow()}}
    )

    # Cleanup
    if tutor_id in _tutor_sessions:
        del _tutor_sessions[tutor_id]

    logger.info(f"Ended SmartBoard session {session_id}")

    return {"status": "ended"}


@router.get("/pens", response_model=PensResponse)
async def get_connected_pens(
    session_id: Optional[str] = None,
    current_user: dict = Depends(require_tutor)
):
    """Get list of connected pens (for a session or all)"""
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    # Get session if specified, otherwise use tutor's active session
    if session_id:
        session = _sessions.get(session_id)
    else:
        session_id = _tutor_sessions.get(tutor_id)
        session = _sessions.get(session_id) if session_id else None

    pens = []
    if session:
        for pen_id, pen_data in session.pen_states.items():
            pens.append(PenSummary(
                pen_id=pen_id,
                pen_mac=pen_data.get("pen_mac", "UNKNOWN"),
                student_name=pen_data.get("student_name"),
                student_id=pen_data.get("student_id"),
                connected=pen_data.get("connected", False),
                battery=pen_data.get("battery"),
                page_no=pen_data.get("page_no"),
                book_type=pen_data.get("book_type"),
                last_frame_ts=pen_data.get("last_frame_ts"),
                color=pen_data.get("color"),
                strokes=[
                    DashboardStroke(**s) for s in pen_data.get("strokes", [])[-10:]
                ],
            ))

    return PensResponse(pens=pens, maxPens=30)


@router.post("/pen/{pen_id}/color")
async def update_pen_color(
    pen_id: str,
    color: str,
    current_user: dict = Depends(require_tutor)
):
    """Update a pen's display color"""
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    session_id = _tutor_sessions.get(tutor_id)

    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=404, detail="No active session")

    session = _sessions[session_id]
    if pen_id in session.pen_states:
        session.pen_states[pen_id]["color"] = color

    return {"status": "updated", "color": color}


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@router.websocket("/ws")
async def smartboard_websocket(
    websocket: WebSocket,
    token: Optional[str] = None,
):
    """
    WebSocket for SmartBoard real-time updates.

    Message types from client:
    - register_tutor: {type: "register_tutor", tutor_id: str, session_id: str}
    - pen_status: {type: "pen_status", pen_id: str, ...}
    - pen_strokes: {type: "pen_strokes", pen_id: str, strokes: [...]}
    - heartbeat: {type: "heartbeat"}

    Message types to client:
    - pen_status: Pen connection/battery updates
    - pen_strokes: Real-time stroke data
    - pen_clear: Canvas clear event
    - session_update: Session state changes
    """
    await ws_manager.connect(websocket)

    tutor_session: Optional[SmartBoardSession] = None

    try:
        async for message in websocket.iter_json():
            msg_type = message.get("type")

            if msg_type == "register_tutor":
                # Tutor registering for a session
                session_id = message.get("session_id")
                if session_id and session_id in _sessions:
                    tutor_session = _sessions[session_id]
                    tutor_session.tutor_ws = websocket
                    logger.info(f"Tutor registered for session {session_id}")

                    # Send current pen states
                    for pen_id, pen_data in tutor_session.pen_states.items():
                        await websocket.send_json({
                            "type": "pen_status",
                            "pen_id": pen_id,
                            "pen_mac": pen_data.get("pen_mac"),
                            "connected": pen_data.get("connected", False),
                            "battery": pen_data.get("battery"),
                            "student_name": pen_data.get("student_name"),
                        })

            elif msg_type == "pen_status":
                # Pen status update (from hub or desktop client)
                pen_id = message.get("pen_id")
                session_id = message.get("session_id")

                if session_id and session_id in _sessions:
                    session = _sessions[session_id]
                    if pen_id not in session.pen_states:
                        session.pen_states[pen_id] = {}
                        session.connected_pens.add(pen_id)

                    session.pen_states[pen_id].update({
                        "pen_mac": message.get("pen_mac"),
                        "connected": message.get("connected", True),
                        "battery": message.get("battery"),
                        "student_name": message.get("student_name"),
                        "student_id": message.get("student_id"),
                        "page_no": message.get("page_no"),
                        "book_type": message.get("book_type"),
                        "last_frame_ts": datetime.utcnow().timestamp(),
                    })

                    # Forward to tutor
                    await ws_manager.send_to_session(session_id, message)

            elif msg_type == "pen_strokes":
                # Real-time strokes from a pen
                pen_id = message.get("pen_id")
                session_id = message.get("session_id")
                strokes = message.get("strokes", [])

                if session_id and session_id in _sessions:
                    session = _sessions[session_id]
                    if pen_id in session.pen_states:
                        existing_strokes = session.pen_states[pen_id].get("strokes", [])
                        existing_strokes.extend(strokes)
                        # Keep only last 100 strokes
                        session.pen_states[pen_id]["strokes"] = existing_strokes[-100:]

                    # Forward to tutor
                    await ws_manager.send_to_session(session_id, message)

            elif msg_type == "pen_clear":
                # Student cleared their canvas
                session_id = message.get("session_id")
                if session_id:
                    await ws_manager.send_to_session(session_id, message)

            elif msg_type == "heartbeat":
                await websocket.send_json({"type": "heartbeat_ack"})

    except WebSocketDisconnect:
        logger.info("SmartBoard WebSocket disconnected")
    except Exception as e:
        logger.error(f"SmartBoard WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)
        if tutor_session:
            tutor_session.tutor_ws = None


# =============================================================================
# Question Attempt Endpoints (simplified)
# =============================================================================

@router.post("/question-attempts", response_model=QuestionAttemptResponse)
async def create_question_attempt(
    data: QuestionAttemptRequest,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """Create a new question attempt (lock question on student devices)"""
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    session_id = _tutor_sessions.get(tutor_id)

    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=400, detail="No active SmartBoard session")

    attempt_id = f"QA-{uuid.uuid4().hex[:8].upper()}"
    created_at = datetime.utcnow()

    auto_collect_at = None
    if data.auto_collect_after_ms:
        from datetime import timedelta
        auto_collect_at = created_at + timedelta(milliseconds=data.auto_collect_after_ms)

    # Broadcast question lock to all connected pens
    await ws_manager.send_to_session(session_id, {
        "type": "question_lock",
        "attempt_id": attempt_id,
        "question_text": data.question_text,
        "bounds": data.bounds,
        "auto_collect_at": auto_collect_at.isoformat() if auto_collect_at else None,
    })

    # Save to database
    db = get_db(request)
    await db.mongo_insert_one("question_attempts", {
        "attempt_id": attempt_id,
        "session_id": session_id,
        "tutor_id": tutor_id,
        "question_text": data.question_text,
        "question_image_b64": data.question_image_b64,
        "bounds": data.bounds,
        "status": "active",
        "created_at": created_at,
        "auto_collect_at": auto_collect_at,
        "submissions": [],
    })

    return QuestionAttemptResponse(
        attempt_id=attempt_id,
        session_id=session_id,
        question_text=data.question_text,
        status="active",
        created_at=created_at,
        auto_collect_at=auto_collect_at,
    )


@router.post("/question-attempts/{attempt_id}/end")
async def end_question_attempt(
    attempt_id: str,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """End a question attempt"""
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    session_id = _tutor_sessions.get(tutor_id)

    if not session_id:
        raise HTTPException(status_code=400, detail="No active session")

    # Broadcast question end
    await ws_manager.send_to_session(session_id, {
        "type": "question_end",
        "attempt_id": attempt_id,
    })

    # Update database
    db = get_db(request)
    await db.mongo_update_one(
        "question_attempts",
        {"attempt_id": attempt_id},
        {"$set": {"status": "ended", "ended_at": datetime.utcnow()}}
    )

    return {"status": "ended"}


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_student_answer(
    data: EvaluateRequest,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """Evaluate a student's handwritten answer using AI"""
    import os
    import httpx

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not OPENAI_API_KEY:
        return EvaluateResponse(
            success=False,
            score="",
            extracted_answer="",
            feedback="",
            error="AI evaluation not configured",
        )

    answer_image = data.answer_image_b64
    if "," in answer_image:
        answer_image = answer_image.split(",", 1)[1]

    eval_prompt = """You are evaluating a student's handwritten answer.
Please analyze the image and provide:
1. What you read from the handwriting
2. Whether it's correct, incorrect, or partially correct
3. Brief feedback

Respond in JSON format:
{
  "score": "correct" or "incorrect" or "partial",
  "extracted_answer": "what you read",
  "feedback": "brief feedback"
}"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": eval_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{answer_image}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_completion_tokens": 512,
                },
            )

            if response.status_code != 200:
                return EvaluateResponse(
                    success=False, score="", extracted_answer="",
                    feedback="", error=f"API error: {response.status_code}"
                )

            result_text = response.json()["choices"][0]["message"]["content"]

            import json
            clean_text = result_text.strip()
            if clean_text.startswith("```"):
                clean_text = clean_text.split("```")[1]
                if clean_text.startswith("json"):
                    clean_text = clean_text[4:]
            clean_text = clean_text.strip()

            result = json.loads(clean_text)
            return EvaluateResponse(
                success=True,
                score=result.get("score", "partial"),
                extracted_answer=result.get("extracted_answer", ""),
                feedback=result.get("feedback", ""),
            )

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return EvaluateResponse(
            success=False, score="", extracted_answer="",
            feedback="", error=str(e)
        )
