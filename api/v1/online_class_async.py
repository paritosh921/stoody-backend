"""
Online Class WebSocket API

Handles real-time communication for online teaching sessions.
Supports:
- Teacher broadcasting strokes to all students (1:N)
- Students sending strokes to teacher (N:1)
- Session management
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from pydantic import BaseModel
from core.observability import track_websocket_connection

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass
class StudentConnection:
    """Represents a connected student."""
    websocket: WebSocket
    student_id: str
    student_name: str
    pen_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OnlineSession:
    """Represents an online class session."""
    session_id: str
    teacher_id: str
    teacher_name: str
    teacher_ws: Optional[WebSocket] = None
    students: Dict[str, StudentConnection] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True


# In-memory session storage (use Redis for production)
_sessions: Dict[str, OnlineSession] = {}


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    teacher_id: str
    teacher_name: str


class CreateSessionResponse(BaseModel):
    """Response with session details."""
    session_id: str
    session_code: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new online class session."""
    import uuid
    import random
    import string
    
    session_id = str(uuid.uuid4())
    # Generate 6-character session code
    session_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    
    session = OnlineSession(
        session_id=session_id,
        teacher_id=request.teacher_id,
        teacher_name=request.teacher_name,
    )
    _sessions[session_id] = session
    _sessions[session_code] = session  # Also index by code
    
    logger.info(f"Created session {session_id} with code {session_code}")
    
    return CreateSessionResponse(
        session_id=session_id,
        session_code=session_code,
    )


@router.get("/sessions/{session_code}")
async def get_session(session_code: str):
    """Get session info by code."""
    session = _sessions.get(session_code) or _sessions.get(session_code.upper())
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "teacher_name": session.teacher_name,
        "student_count": len(session.students),
        "active": session.active,
    }


@router.websocket("/ws/{session_id}/teacher")
async def teacher_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for teacher."""
    await websocket.accept()
    connection_tracked = False
    
    session = _sessions.get(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    session.teacher_ws = websocket
    track_websocket_connection("online_class_teacher", 1)
    connection_tracked = True
    logger.info(f"Teacher connected to session {session_id}")
    
    # Notify students
    await broadcast_to_students(session, {
        "type": "session_update",
        "teacher_connected": True,
        "active": True,
    })
    
    try:
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                await handle_teacher_message(session, data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from teacher: {message}")
    except WebSocketDisconnect:
        logger.info(f"Teacher disconnected from session {session_id}")
    finally:
        session.teacher_ws = None
        await broadcast_to_students(session, {
            "type": "session_update",
            "teacher_connected": False,
        })
        if connection_tracked:
            track_websocket_connection("online_class_teacher", -1)


@router.websocket("/ws/{session_id}/student")
async def student_websocket(
    websocket: WebSocket,
    session_id: str,
    student_id: str = "",
    student_name: str = "Student",
    pen_id: str = "",
):
    """WebSocket endpoint for students."""
    await websocket.accept()
    connection_tracked = False
    
    session = _sessions.get(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    # Register student
    student = StudentConnection(
        websocket=websocket,
        student_id=student_id or f"student_{len(session.students)}",
        student_name=student_name,
        pen_id=pen_id or None,
    )
    session.students[student.student_id] = student
    track_websocket_connection("online_class_student", 1)
    connection_tracked = True
    
    logger.info(f"Student {student.student_name} joined session {session_id}")
    
    # Notify teacher
    if session.teacher_ws:
        try:
            await session.teacher_ws.send_json({
                "type": "student_join",
                "student_id": student.student_id,
                "student_name": student.student_name,
                "pen_id": student.pen_id,
            })
        except:
            pass
    
    # Send session state to student
    await websocket.send_json({
        "type": "session_update",
        "teacher_connected": session.teacher_ws is not None,
        "active": session.active,
    })
    
    try:
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                await handle_student_message(session, student, data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from student: {message}")
    except WebSocketDisconnect:
        logger.info(f"Student {student.student_name} left session {session_id}")
    finally:
        session.students.pop(student.student_id, None)
        # Notify teacher
        if session.teacher_ws:
            try:
                await session.teacher_ws.send_json({
                    "type": "student_leave",
                    "student_id": student.student_id,
                })
            except:
                pass
        if connection_tracked:
            track_websocket_connection("online_class_student", -1)


async def handle_teacher_message(session: OnlineSession, data: dict):
    """Handle message from teacher."""
    msg_type = data.get("type")
    
    if msg_type == "teacher_stroke":
        # Broadcast stroke to all students (1:N)
        await broadcast_to_students(session, {
            "type": "teacher_stroke",
            "stroke": data.get("stroke"),
        })
    
    elif msg_type == "canvas_command":
        # Canvas commands (clear, undo, etc.)
        await broadcast_to_students(session, {
            "type": "canvas_command",
            "command": data.get("command"),
        })
    
    elif msg_type == "request_student_work":
        # Request specific student's work
        student_id = data.get("student_id")
        # This is for the teacher to view a student's current work
        pass
    
    elif msg_type == "pong":
        # Heartbeat response
        pass


async def handle_student_message(session: OnlineSession, student: StudentConnection, data: dict):
    """Handle message from student."""
    msg_type = data.get("type")
    
    if msg_type == "student_stroke":
        # Forward stroke to teacher (N:1)
        if session.teacher_ws:
            try:
                await session.teacher_ws.send_json({
                    "type": "student_stroke",
                    "student_id": student.student_id,
                    "student_name": student.student_name,
                    "pen_id": student.pen_id,
                    "stroke": data.get("stroke"),
                })
            except:
                pass
    
    elif msg_type == "pong":
        # Heartbeat response
        pass


async def broadcast_to_students(session: OnlineSession, message: dict):
    """Broadcast message to all students in session."""
    dead_connections = []
    
    for student_id, student in session.students.items():
        try:
            await student.websocket.send_json(message)
        except:
            dead_connections.append(student_id)
    
    # Clean up dead connections
    for student_id in dead_connections:
        session.students.pop(student_id, None)


@router.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """End a session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.active = False
    
    # Notify all clients
    await broadcast_to_students(session, {
        "type": "session_update",
        "active": False,
        "message": "Session ended by teacher",
    })
    
    # Close all connections
    for student in list(session.students.values()):
        try:
            await student.websocket.close(code=1000, reason="Session ended")
        except:
            pass
    
    if session.teacher_ws:
        try:
            await session.teacher_ws.close(code=1000, reason="Session ended")
        except:
            pass
    
    # Remove session (keep for a while for history)
    # _sessions.pop(session_id, None)
    
    logger.info(f"Session {session_id} ended")
    
    return {"status": "ended"}


class EvaluateRequest(BaseModel):
    """Request to evaluate student answer."""
    question_text: str
    answer_image_b64: str
    student_id: Optional[str] = None


class EvaluateResponse(BaseModel):
    """Response from AI evaluation."""
    success: bool
    score: str  # "correct", "incorrect", "partial"
    extracted_answer: str
    feedback: str
    error: Optional[str] = None


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_student_answer(request: EvaluateRequest):
    """
    Evaluate a student's handwritten answer using AI.
    
    Takes a question and an image of the student's answer, returns
    a score (correct/incorrect/partial) and feedback.
    """
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
            error="AI evaluation not configured (missing OPENAI_API_KEY)",
        )
    
    # Clean up base64 data
    answer_image = request.answer_image_b64
    if "," in answer_image:
        answer_image = answer_image.split(",", 1)[1]
    
    eval_prompt = f"""You are a teacher evaluating a student's handwritten answer.

QUESTION: {request.question_text}

The attached image shows the student's handwritten response to this question.

Please:
1. Read and interpret the student's handwritten answer
2. Evaluate if the answer is correct, incorrect, or partially correct
3. Provide brief, helpful feedback

Respond in this exact JSON format:
{{
  "score": "correct" or "incorrect" or "partial",
  "extracted_answer": "what you read from the handwriting",
  "feedback": "brief feedback for the student (1-2 sentences)"
}}

Only respond with the JSON, nothing else."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            url = f"{OPENAI_BASE_URL}/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            
            payload = {
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
            }
            
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                return EvaluateResponse(
                    success=False,
                    score="",
                    extracted_answer="",
                    feedback="",
                    error=f"OpenAI API error: {response.text}",
                )
            
            data = response.json()
            response_text = data["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            import json as json_module
            try:
                # Clean up response
                clean_text = response_text.strip()
                if clean_text.startswith("```"):
                    clean_text = clean_text.split("```")[1]
                    if clean_text.startswith("json"):
                        clean_text = clean_text[4:]
                clean_text = clean_text.strip()
                
                result = json_module.loads(clean_text)
                return EvaluateResponse(
                    success=True,
                    score=result.get("score", "partial"),
                    extracted_answer=result.get("extracted_answer", ""),
                    feedback=result.get("feedback", ""),
                )
            except json_module.JSONDecodeError:
                # Fallback
                lower_text = response_text.lower()
                if "correct" in lower_text and "incorrect" not in lower_text:
                    score = "correct"
                elif "incorrect" in lower_text:
                    score = "incorrect"
                else:
                    score = "partial"
                
                return EvaluateResponse(
                    success=True,
                    score=score,
                    extracted_answer="",
                    feedback=response_text[:200],
                )
                
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return EvaluateResponse(
            success=False,
            score="",
            extracted_answer="",
            feedback="",
            error=str(e),
        )
