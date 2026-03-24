"""Invigilator console endpoints — HTTP session list + WebSocket real-time feed.

Routes are mounted at ``/api/v1/exampen/invig``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
    require_exampen_role,
)
from exampen.dcr.storage.exam_repo import ExamRepo
from exampen.dcr.storage.binding_repo import BindingRepo
from exampen.dcr.storage.stroke_raw_repo import StrokeRawRepo

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db


# ---------------------------------------------------------------------------
# HTTP: session list
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_active_sessions(
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("principal", "hod", "invigilator")
    ),
    state: str | None = Query(None),
) -> dict[str, Any]:
    """List active / recent exam sessions for the invigilator console.

    Returns exams in non-terminal states by default, or filtered by state.
    """
    db = await _get_tenant_db(request, user)
    repo = ExamRepo(db)

    filters: dict[str, Any] = {}
    if state:
        filters["state"] = state
    else:
        # Non-terminal states
        filters["state"] = {
            "$in": ["armed", "timer_running", "sync_pending", "scoring"]
        }

    exams = await repo.list_exams(user.tenant_id, filters=filters, limit=50)

    # Enrich each exam with binding + upload counts
    binding_repo = BindingRepo(db)
    stroke_repo = StrokeRawRepo(db)

    sessions = []
    for exam in exams:
        exam_id = exam["_id"]
        bindings = await binding_repo.list_by_exam(exam_id, user.tenant_id)
        upload_status = await stroke_repo.get_exam_upload_status(exam_id, user.tenant_id)

        sessions.append({
            "exam_id": exam_id,
            "title": exam.get("title", ""),
            "state": exam.get("state", ""),
            "scheduled_at": exam.get("scheduled_at"),
            "pen_count": len(bindings),
            "pens_uploading": len(upload_status),
            "pens_complete": sum(
                1 for p in upload_status
                if len(p.get("received_chunks", [])) >= (exam.get("total_chunks_per_pen", 1))
            ),
        })

    return {"sessions": sessions}


# ---------------------------------------------------------------------------
# WebSocket: real-time status feed
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def invig_ws(
    websocket: WebSocket,
) -> None:
    """WebSocket endpoint for real-time invigilator status updates.

    Authentication is handled via a token query parameter because
    WebSocket connections cannot send Authorization headers natively.

    The server pushes periodic status snapshots and listens for NATS
    events to push incremental updates.
    """
    await websocket.accept()

    # Authenticate via query parameter
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    # Validate token using the auth bridge
    try:
        from exampen.dcr.core.auth_bridge import ExamPenUser
        from core.auth import AuthManager

        auth: AuthManager = websocket.app.state.auth
        user_data = await auth.verify_token_and_get_user(token)
        if not user_data:
            await websocket.close(code=4001, reason="Invalid token")
            return

        tenant_id = user_data.get("tenant_id", "")
        user_type = (user_data.get("user_type") or "").strip().lower()

        # Only invigilators, hod, or principal can use this
        allowed_types = {"admin", "tutor"}
        if user_type not in allowed_types:
            await websocket.close(code=4003, reason="Insufficient permissions")
            return

    except Exception as exc:
        logger.warning("WS auth failed: %s", exc)
        await websocket.close(code=4001, reason="Authentication failed")
        return

    exam_id = websocket.query_params.get("exam_id")
    if not exam_id:
        await websocket.close(code=4002, reason="Missing exam_id")
        return

    db = await websocket.app.state.db.get_tenant_db(tenant_id)
    if db is None:
        await websocket.close(code=4003, reason="Database unavailable")
        return

    stroke_repo = StrokeRawRepo(db)
    binding_repo = BindingRepo(db)

    logger.info("Invig WS connected: tenant=%s exam=%s", tenant_id, exam_id)

    try:
        while True:
            # Push a status snapshot every 5 seconds
            bindings = await binding_repo.list_by_exam(exam_id, tenant_id)
            upload_status = await stroke_repo.get_exam_upload_status(exam_id, tenant_id)

            snapshot = {
                "type": "status_snapshot",
                "exam_id": exam_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pen_count": len(bindings),
                "pens": upload_status,
            }
            await websocket.send_text(json.dumps(snapshot, default=str))

            # Wait 5 seconds, but also listen for client messages
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(), timeout=5.0,
                )
                # Client can send ping or request immediate refresh
                if msg == "ping":
                    await websocket.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                pass  # Normal: no client message, push next snapshot

    except WebSocketDisconnect:
        logger.info("Invig WS disconnected: exam=%s", exam_id)
    except Exception as exc:
        logger.warning("Invig WS error: %s", exc)
        await websocket.close(code=1011, reason="Internal error")
