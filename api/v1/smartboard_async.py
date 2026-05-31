"""
SmartBoard API - Real-time Pen Monitoring for Teaching
Handles session management, WebSocket connections, and AI evaluation
"""

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import secrets
import json
import asyncio
import logging

from bson import ObjectId

from core.database import DatabaseManager
from core.permissions import has_permission
from core.observability import track_websocket_connection
from core.tenant_features import is_feature_enabled
from utils.tutor_scoping import build_tutor_class_criteria
from api.v1.admin_async import get_tenant_db_or_403
from api.v1.auth_async import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new SmartBoard session"""
    title: Optional[str] = None
    class_id: Optional[str] = None  # Link to online class if applicable
    subject: Optional[str] = None
    standard: Optional[str] = None


class SessionResponse(BaseModel):
    """Response model for SmartBoard session"""
    session_id: str
    title: Optional[str] = None
    tutor_id: str
    tutor_name: Optional[str] = None
    class_id: Optional[str] = None
    status: str  # active, ended
    connected_pens: int = 0
    created_at: datetime
    ended_at: Optional[datetime] = None


class QuestionAttemptRequest(BaseModel):
    """Request to lock a question for students"""
    question_text: str
    question_id: Optional[str] = None
    time_limit_seconds: int = Field(default=300, ge=30, le=3600)
    points: int = Field(default=4, ge=1)
    negative_marks: float = Field(default=1, ge=0)


class QuestionAttemptResponse(BaseModel):
    """Response model for question attempt"""
    attempt_id: str
    session_id: str
    question_text: str
    question_id: Optional[str] = None
    time_limit_seconds: int
    points: int
    negative_marks: float
    status: str  # active, ended
    started_at: datetime
    ended_at: Optional[datetime] = None
    submissions_count: int = 0


class EvaluateRequest(BaseModel):
    """Request to trigger AI evaluation"""
    attempt_id: str
    correct_answer: Optional[str] = None


class SmartboardPenName(BaseModel):
    """Minimal pen display-name row exposed to paired smartboards."""
    pen_mac: str
    pen_id: Optional[str] = None
    name: Optional[str] = None
    pen_name: Optional[str] = None
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    last_registered_at: Optional[datetime] = None


class SmartboardPenNamesResponse(BaseModel):
    pens: List[SmartboardPenName]
    count: int
    tenant_id: Optional[str] = None
    refreshed_at: datetime


# =============================================================================
# In-Memory Session Storage (Consider Redis for production)
# =============================================================================

class SmartBoardSession:
    """In-memory representation of an active SmartBoard session"""
    def __init__(self, session_id: str, tutor_id: str, tutor_name: str = None):
        self.session_id = session_id
        self.tutor_id = tutor_id
        self.tutor_name = tutor_name
        self.websockets: List[WebSocket] = []
        self.connected_pens: Dict[str, Dict] = {}  # pen_id -> pen data
        self.active_question: Optional[Dict] = None
        self.created_at = datetime.utcnow()

    def add_websocket(self, ws: WebSocket):
        if ws not in self.websockets:
            self.websockets.append(ws)

    def remove_websocket(self, ws: WebSocket):
        if ws in self.websockets:
            self.websockets.remove(ws)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected websockets"""
        dead_sockets = []
        for ws in self.websockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            self.remove_websocket(ws)


# Global session storage (consider Redis for production)
_sessions: Dict[str, SmartBoardSession] = {}


# =============================================================================
# Helper Functions
# =============================================================================

async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


def require_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require tutor access"""
    user_type = current_user.get("user_type")
    if user_type not in ["tutor", "admin", "b2c_admin"]:
        raise HTTPException(status_code=403, detail="Tutor access required")
    if user_type in ["admin", "b2c_admin"] and not has_permission(current_user, "manage_smartboard"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return current_user


def require_smartboard_cloud_user(current_user: Dict[str, Any] = Depends(require_tutor)):
    """Require a tutor/admin token that is allowed to use SmartBoard cloud APIs."""
    if not is_feature_enabled(
        current_user.get("enabled_features"),
        "smartboard_cloud_access",
        current_user.get("enabled_features_v2"),
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Smartboard cloud access not enabled for this institution",
        )
    return current_user


def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"SB{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{secrets.token_hex(4).upper()}"


def _actor_is_admin(current_user: Dict[str, Any]) -> bool:
    return (current_user.get("user_type") or "").lower() in {"admin", "b2c_admin"}


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _display_name(*, student_name: Optional[str], pen_name: Optional[str]) -> Optional[str]:
    return _safe_str(student_name) or _safe_str(pen_name)


async def _smartboard_visible_student_usernames(
    tenant_db,
    current_user: Dict[str, Any],
) -> set[str]:
    """
    Return student usernames visible to this tutor for SmartBoard display.

    Unlike admin pen-management endpoints, this read-only display endpoint does
    not require `can_edit_students`; it follows assigned-student, teacher_ids,
    and class/section tutor scoping.
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    if not tutor_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor identity missing from token",
        )

    tutor_doc = await tenant_db["tutors"].find_one({"tutor_id": tutor_id})
    if tutor_doc is None:
        try:
            tutor_doc = await tenant_db["tutors"].find_one({"_id": ObjectId(str(tutor_id))})
        except Exception:
            tutor_doc = None
    if tutor_doc is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor not found in this tenant",
        )

    usernames: set[str] = set()
    assigned_student_ids = [s for s in (tutor_doc.get("assigned_student_ids") or []) if s]
    if assigned_student_ids:
        cursor = tenant_db["students"].find(
            {"student_id": {"$in": assigned_student_ids}},
            {"username": 1},
        )
        async for doc in cursor:
            username = doc.get("username")
            if username:
                usernames.add(username)

    cursor = tenant_db["students"].find(
        {"teacher_ids": {"$in": [tutor_id]}},
        {"username": 1},
    )
    async for doc in cursor:
        username = doc.get("username")
        if username:
            usernames.add(username)

    admin_oid = None
    raw_admin_id = current_user.get("admin_id")
    if raw_admin_id:
        try:
            admin_oid = ObjectId(str(raw_admin_id))
        except Exception:
            admin_oid = None

    class_query = build_tutor_class_criteria(tutor_doc, admin_oid)
    if class_query:
        cursor = tenant_db["students"].find(class_query, {"username": 1})
        async for doc in cursor:
            username = doc.get("username")
            if username:
                usernames.add(username)

    return usernames


async def _load_smartboard_pen_names(
    tenant_db,
    current_user: Dict[str, Any],
) -> List[SmartboardPenName]:
    query: Dict[str, Any] = {"status": "active"}
    if not _actor_is_admin(current_user):
        visible_usernames = await _smartboard_visible_student_usernames(tenant_db, current_user)
        if not visible_usernames:
            return []
        query["user_id"] = {"$in": sorted(visible_usernames)}

    pen_docs = await tenant_db["pens"].find(query).sort("last_registered_at", -1).to_list(length=5000)
    usernames = sorted({p.get("user_id") for p in pen_docs if p.get("user_id")})
    student_lookup: Dict[str, Dict[str, Any]] = {}
    if usernames:
        cursor = tenant_db["students"].find(
            {"username": {"$in": usernames}},
            {"username": 1, "student_id": 1, "name": 1, "full_name": 1},
        )
        async for doc in cursor:
            username = doc.get("username")
            if username:
                student_lookup[username] = doc

    rows: List[SmartboardPenName] = []
    for pen in pen_docs:
        pen_mac = _safe_str(pen.get("pen_mac"))
        if not pen_mac:
            continue
        student_doc = student_lookup.get(pen.get("user_id")) or {}
        student_name = _safe_str(student_doc.get("full_name")) or _safe_str(student_doc.get("name"))
        pen_name = _safe_str(pen.get("pen_name"))
        rows.append(
            SmartboardPenName(
                pen_mac=pen_mac.upper(),
                pen_id=_safe_str(pen.get("pen_id")),
                name=_display_name(student_name=student_name, pen_name=pen_name),
                pen_name=pen_name,
                student_id=_safe_str(student_doc.get("student_id")),
                student_name=student_name,
                last_registered_at=pen.get("last_registered_at"),
            )
        )
    return rows


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/pen-names", response_model=SmartboardPenNamesResponse)
async def get_pen_names(
    current_user: Dict[str, Any] = Depends(require_smartboard_cloud_user),
    db: DatabaseManager = Depends(get_database),
):
    """
    Return a safe `pen_mac -> display name` map for paired smartboards.

    This endpoint is intentionally narrower than `/admin/pens`: the tablet only
    receives active pen IDs, MACs, and display labels needed to overlay student
    names on locally streamed hub strokes.
    """
    tenant_db = await get_tenant_db_or_403(db, current_user)
    pens = await _load_smartboard_pen_names(tenant_db, current_user)
    return SmartboardPenNamesResponse(
        pens=pens,
        count=len(pens),
        tenant_id=current_user.get("db_name") or current_user.get("tenant_id"),
        refreshed_at=datetime.utcnow(),
    )


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    request: Request,
    session_data: CreateSessionRequest,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Create a new SmartBoard teaching session
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    tutor_name = current_user.get("name") or current_user.get("username")

    # Get admin_id for tenant isolation
    admin_id = current_user.get("admin_id")
    if current_user.get("user_type") == "admin":
        admin_id = admin_id or current_user.get("user_id")
    if not admin_id:
        raise HTTPException(status_code=403, detail="Tenant context required")

    session_id = generate_session_id()

    # Create in-memory session
    session = SmartBoardSession(session_id, tutor_id, tutor_name)
    _sessions[session_id] = session

    # Persist to database
    session_doc = {
        "session_id": session_id,
        "admin_id": ObjectId(admin_id),  # Tenant isolation
        "title": session_data.title or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        "tutor_id": tutor_id,
        "tutor_name": tutor_name,
        "class_id": session_data.class_id,
        "subject": session_data.subject,
        "standard": session_data.standard,
        "status": "active",
        "connected_pens": [],
        "question_attempts": [],
        "created_at": datetime.utcnow(),
        "ended_at": None
    }
    
    await db.mongo_insert_one("smartboard_sessions", session_doc)
    
    return SessionResponse(
        session_id=session_id,
        title=session_doc["title"],
        tutor_id=tutor_id,
        tutor_name=tutor_name,
        class_id=session_data.class_id,
        status="active",
        connected_pens=0,
        created_at=session_doc["created_at"]
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    request: Request,
    session_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get session details
    """
    # Get admin_id for tenant isolation
    admin_id = current_user.get("admin_id")
    if current_user.get("user_type") == "admin":
        admin_id = admin_id or current_user.get("user_id")
    if not admin_id:
        raise HTTPException(status_code=403, detail="Tenant context required")

    session_doc = await db.mongo_find_one(
        "smartboard_sessions",
        {"session_id": session_id, "admin_id": ObjectId(admin_id)}
    )
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get live pen count from memory if session is active
    connected_pens = 0
    if session_id in _sessions:
        connected_pens = len(_sessions[session_id].connected_pens)
    else:
        connected_pens = len(session_doc.get("connected_pens", []))
    
    return SessionResponse(
        session_id=session_id,
        title=session_doc.get("title"),
        tutor_id=session_doc.get("tutor_id"),
        tutor_name=session_doc.get("tutor_name"),
        class_id=session_doc.get("class_id"),
        status=session_doc.get("status", "active"),
        connected_pens=connected_pens,
        created_at=session_doc.get("created_at"),
        ended_at=session_doc.get("ended_at")
    )


@router.post("/sessions/{session_id}/end")
async def end_session(
    request: Request,
    session_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    End a SmartBoard session
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    # Get admin_id for tenant isolation
    admin_id = current_user.get("admin_id")
    if current_user.get("user_type") == "admin":
        admin_id = admin_id or current_user.get("user_id")
    if not admin_id:
        raise HTTPException(status_code=403, detail="Tenant context required")

    session_doc = await db.mongo_find_one(
        "smartboard_sessions",
        {"session_id": session_id, "admin_id": ObjectId(admin_id)}
    )
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_doc.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to end this session")
    
    # Broadcast session end to all connected clients
    if session_id in _sessions:
        await _sessions[session_id].broadcast({
            "type": "session_ended",
            "session_id": session_id,
            "message": "Session has been ended by the teacher"
        })
        del _sessions[session_id]
    
    # Update database
    await db.mongo_update_one(
        "smartboard_sessions",
        {"session_id": session_id},
        {"$set": {"status": "ended", "ended_at": datetime.utcnow()}}
    )
    
    return {"message": "Session ended successfully"}


@router.post("/sessions/{session_id}/question-attempts", response_model=QuestionAttemptResponse)
async def create_question_attempt(
    request: Request,
    session_id: str,
    question_data: QuestionAttemptRequest,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Lock a question for all students in the session
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    # Get admin_id for tenant isolation
    admin_id = current_user.get("admin_id")
    if current_user.get("user_type") == "admin":
        admin_id = admin_id or current_user.get("user_id")
    if not admin_id:
        raise HTTPException(status_code=403, detail="Tenant context required")

    session_doc = await db.mongo_find_one(
        "smartboard_sessions",
        {"session_id": session_id, "admin_id": ObjectId(admin_id)}
    )
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_doc.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    attempt_id = f"QA{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{secrets.token_hex(3).upper()}"

    attempt = {
        "attempt_id": attempt_id,
        "admin_id": ObjectId(admin_id),  # Tenant isolation
        "session_id": session_id,
        "question_text": question_data.question_text,
        "question_id": question_data.question_id,
        "time_limit_seconds": question_data.time_limit_seconds,
        "points": question_data.points,
        "negative_marks": question_data.negative_marks,
        "status": "active",
        "started_at": datetime.utcnow(),
        "ended_at": None,
        "submissions": []
    }
    
    # Store in database
    await db.mongo_insert_one("question_attempts", attempt)
    
    # Update session with active question
    await db.mongo_update_one(
        "smartboard_sessions",
        {"session_id": session_id},
        {"$push": {"question_attempts": attempt_id}}
    )
    
    # Broadcast to all connected clients
    if session_id in _sessions:
        _sessions[session_id].active_question = attempt
        await _sessions[session_id].broadcast({
            "type": "question_lock",
            "attempt_id": attempt_id,
            "question_text": question_data.question_text,
            "time_limit_seconds": question_data.time_limit_seconds,
            "points": question_data.points,
            "started_at": attempt["started_at"].isoformat()
        })
    
    return QuestionAttemptResponse(
        attempt_id=attempt_id,
        session_id=session_id,
        question_text=question_data.question_text,
        question_id=question_data.question_id,
        time_limit_seconds=question_data.time_limit_seconds,
        points=question_data.points,
        negative_marks=question_data.negative_marks,
        status="active",
        started_at=attempt["started_at"],
        submissions_count=0
    )


@router.post("/sessions/{session_id}/question-attempts/{attempt_id}/end")
async def end_question_attempt(
    request: Request,
    session_id: str,
    attempt_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    End a question attempt and collect submissions
    """
    # Get admin_id for tenant isolation
    admin_id = current_user.get("admin_id")
    if current_user.get("user_type") == "admin":
        admin_id = admin_id or current_user.get("user_id")
    if not admin_id:
        raise HTTPException(status_code=403, detail="Tenant context required")

    attempt = await db.mongo_find_one(
        "question_attempts",
        {"attempt_id": attempt_id, "admin_id": ObjectId(admin_id)}
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")
    
    # Update attempt status
    await db.mongo_update_one(
        "question_attempts",
        {"attempt_id": attempt_id},
        {"$set": {"status": "ended", "ended_at": datetime.utcnow()}}
    )
    
    # Broadcast end to all clients
    if session_id in _sessions:
        _sessions[session_id].active_question = None
        await _sessions[session_id].broadcast({
            "type": "question_end",
            "attempt_id": attempt_id,
            "message": "Time's up! Question ended."
        })
    
    return {"message": "Question attempt ended", "attempt_id": attempt_id}


@router.post("/sessions/{session_id}/evaluate")
async def evaluate_submissions(
    request: Request,
    session_id: str,
    eval_data: EvaluateRequest,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Trigger AI evaluation for all submissions in a question attempt
    """
    # Get admin_id for tenant isolation
    admin_id = current_user.get("admin_id")
    if current_user.get("user_type") == "admin":
        admin_id = admin_id or current_user.get("user_id")
    if not admin_id:
        raise HTTPException(status_code=403, detail="Tenant context required")

    attempt = await db.mongo_find_one(
        "question_attempts",
        {"attempt_id": eval_data.attempt_id, "admin_id": ObjectId(admin_id)}
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Question attempt not found")
    
    submissions = attempt.get("submissions", [])
    
    # TODO: Integrate with AI evaluation service
    # For now, return a placeholder response
    
    return {
        "attempt_id": eval_data.attempt_id,
        "submissions_count": len(submissions),
        "message": "Evaluation triggered",
        "status": "processing"
    }


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.websocket("/ws/{session_id}")
async def smartboard_websocket(
    websocket: WebSocket,
    session_id: str
):
    """
    WebSocket endpoint for real-time pen data streaming

    Query params:
    - token: JWT access token for authentication

    Message types:
    - pen_connect: Student pen connects to session
    - pen_stroke: Real-time stroke data from pen
    - pen_clear: Clear canvas for a pen
    - pen_disconnect: Pen disconnects
    - question_lock: Teacher locks a question
    - question_end: Teacher ends question
    - student_submit: Student submits response
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return

    try:
        auth_manager = websocket.app.state.auth
        payload = auth_manager.decode_access_token(token)
        if not payload:
            await websocket.close(code=4001, reason="Invalid or expired token")
            return

        user_type = (payload.get("user_type") or "").lower()
        if user_type not in ("tutor", "teacher", "admin"):
            await websocket.close(code=4003, reason="Insufficient permissions")
            return

        user_id = payload.get("user_id") or payload.get("sub") or payload.get("tutor_id")
        admin_id = payload.get("admin_id")
    except Exception as e:
        logger.warning(f"WebSocket auth rejected: {e}")
        await websocket.close(code=4001, reason="Authentication failed")
        return

    await websocket.accept()
    connection_tracked = False
    
    # Check if session exists
    if session_id not in _sessions:
        await websocket.send_json({
            "type": "error",
            "message": "Session not found or has ended"
        })
        await websocket.close()
        return

    session = _sessions[session_id]

    # Session ownership / tenant isolation check:
    # The session creator (tutor_id) must match the caller's identity,
    # or the caller must be an admin from the same tenant.
    if user_type in ("tutor", "teacher"):
        if user_id and str(user_id) != str(session.tutor_id):
            await websocket.send_json({
                "type": "error",
                "message": "Not authorized for this session"
            })
            await websocket.close()
            return
    elif user_type == "admin":
        db = websocket.app.state.db
        try:
            session_doc = await db.mongo_find_one(
                "smartboard_sessions",
                {"session_id": session_id, "admin_id": ObjectId(admin_id)},
            )
            if not session_doc:
                await websocket.send_json({
                    "type": "error",
                    "message": "Not authorized for this session"
                })
                await websocket.close()
                return
        except Exception:
            await websocket.send_json({
                "type": "error",
                "message": "Authorization check failed"
            })
            await websocket.close()
            return
    session.add_websocket(websocket)
    track_websocket_connection("smartboard", 1)
    connection_tracked = True
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "pen_connect":
                # Student pen connecting to session
                pen_id = data.get("pen_id")
                student_id = data.get("student_id")
                student_name = data.get("student_name")
                
                session.connected_pens[pen_id] = {
                    "pen_id": pen_id,
                    "student_id": student_id,
                    "student_name": student_name,
                    "connected_at": datetime.utcnow().isoformat(),
                    "strokes": []
                }
                
                # Broadcast pen connection to all
                await session.broadcast({
                    "type": "pen_connected",
                    "pen_id": pen_id,
                    "student_id": student_id,
                    "student_name": student_name,
                    "total_pens": len(session.connected_pens)
                })
                
            elif msg_type == "pen_stroke":
                # Real-time stroke data
                pen_id = data.get("pen_id")
                stroke = data.get("stroke")
                
                if pen_id in session.connected_pens:
                    session.connected_pens[pen_id]["strokes"].append(stroke)
                
                # Broadcast stroke to teacher dashboard
                await session.broadcast({
                    "type": "stroke_update",
                    "pen_id": pen_id,
                    "stroke": stroke
                })
                
            elif msg_type == "pen_clear":
                # Clear canvas for a pen
                pen_id = data.get("pen_id")
                
                if pen_id in session.connected_pens:
                    session.connected_pens[pen_id]["strokes"] = []
                
                await session.broadcast({
                    "type": "canvas_cleared",
                    "pen_id": pen_id
                })
                
            elif msg_type == "student_submit":
                # Student submits their response
                pen_id = data.get("pen_id")
                student_id = data.get("student_id")
                response_image = data.get("response_image")  # Base64 encoded
                
                if session.active_question:
                    # Store submission
                    submission = {
                        "pen_id": pen_id,
                        "student_id": student_id,
                        "response_image": response_image,
                        "submitted_at": datetime.utcnow().isoformat()
                    }
                    
                    # Broadcast submission received
                    await session.broadcast({
                        "type": "submission_received",
                        "pen_id": pen_id,
                        "student_id": student_id,
                        "timestamp": submission["submitted_at"]
                    })
                
            elif msg_type == "heartbeat":
                # Keep connection alive
                await websocket.send_json({"type": "heartbeat_ack"})
                
    except WebSocketDisconnect:
        session.remove_websocket(websocket)
        logger.info(f"WebSocket disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
        session.remove_websocket(websocket)
    finally:
        if connection_tracked:
            track_websocket_connection("smartboard", -1)
