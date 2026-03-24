"""WebSocket endpoint for real-time invigilator dashboard updates.

Contract: ``WS /api/v1/invigilator/ws``

Authentication:
- Query param ``?token=<jwt>`` OR
- First message ``{"type": "auth", "token": "<jwt>"}``

After authentication, the client receives 1 Hz snapshot pushes for the
subscribed exam session.  The client sends a subscribe message to
indicate which exam to watch:
``{"type": "subscribe", "exam_id": "<uuid>"}``
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from exampen_common.auth import ExamPenUser, validate_token
from exampen_common.logging import get_logger

from src.adapters.exam_client import ExamOrchClient
from src.config import WS_PUSH_INTERVAL_SEC
from src.domain.status_aggregator import build_snapshot, snapshot_to_dict
from src.events.hub_relay import HubRelay

_log = get_logger(__name__)

router = APIRouter()


async def _authenticate_ws(
    ws: WebSocket,
    token_param: str | None,
) -> ExamPenUser | None:
    """Authenticate via query param or first message.

    Returns the validated user, or None if authentication fails
    (in which case the WebSocket has already been closed with 4001).
    """
    # Try query param first
    if token_param:
        try:
            return await validate_token(token_param)
        except Exception:
            _log.warning("WS auth via query param failed")

    # Fall back to first-message auth
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        msg = json.loads(raw)
        if msg.get("type") != "auth" or "token" not in msg:
            await ws.close(code=4001, reason="Expected auth message")
            return None
        return await validate_token(msg["token"])
    except asyncio.TimeoutError:
        await ws.close(code=4001, reason="Auth timeout")
        return None
    except Exception:
        _log.warning("WS first-message auth failed")
        await ws.close(code=4001, reason="Authentication failed")
        return None


async def _wait_for_subscribe(ws: WebSocket) -> str | None:
    """Wait for the client to send a subscribe message with exam_id.

    Returns the exam_id, or None if the client sends invalid data
    (in which case the WebSocket has already been closed).
    """
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        msg = json.loads(raw)
        if msg.get("type") != "subscribe" or not msg.get("exam_id"):
            await ws.close(
                code=4002, reason="Expected subscribe message with exam_id",
            )
            return None
        return msg["exam_id"]
    except asyncio.TimeoutError:
        await ws.close(code=4002, reason="Subscribe timeout")
        return None
    except Exception:
        await ws.close(code=4002, reason="Invalid subscribe message")
        return None


@router.websocket("/ws")
async def invigilator_ws(
    ws: WebSocket,
    token: str | None = Query(default=None),
) -> None:
    """WebSocket endpoint for live invigilator dashboard updates."""
    await ws.accept()

    # --- Authenticate ---
    user = await _authenticate_ws(ws, token)
    if user is None:
        return

    # --- Subscribe to an exam ---
    exam_id = await _wait_for_subscribe(ws)
    if exam_id is None:
        return

    # Confirm subscription
    await ws.send_json({
        "event_type": "subscribed",
        "payload": {"exam_id": exam_id, "user_id": user.user_id},
    })

    hub_relay: HubRelay = ws.app.state.hub_relay
    exam_client: ExamOrchClient = ws.app.state.exam_client

    # Event used by hub_relay callback to signal new data
    new_data_event = asyncio.Event()

    async def _on_hub_update(eid: str, payload: dict[str, Any]) -> None:
        """Called by HubRelay when new hub status arrives."""
        new_data_event.set()

    await hub_relay.register_listener(exam_id, _on_hub_update)

    try:
        await _push_loop(ws, exam_id, hub_relay, exam_client, new_data_event)
    except WebSocketDisconnect:
        _log.info("WS client disconnected (exam=%s, user=%s)", exam_id, user.user_id)
    except Exception:
        _log.exception("WS error (exam=%s, user=%s)", exam_id, user.user_id)
    finally:
        await hub_relay.unregister_listener(exam_id, _on_hub_update)
        # Attempt graceful close -- ignore errors if already closed
        try:
            await ws.close()
        except Exception:
            pass


async def _push_loop(
    ws: WebSocket,
    exam_id: str,
    hub_relay: HubRelay,
    exam_client: ExamOrchClient,
    new_data_event: asyncio.Event,
) -> None:
    """Push 1 Hz snapshots to the WebSocket client.

    Sends immediately when new hub data arrives, but never faster
    than ``WS_PUSH_INTERVAL_SEC``.
    """
    while True:
        # Build snapshot from latest hub data + exam-orch session
        hub_data = hub_relay.get_latest(exam_id)
        exam_data = await exam_client.get_session(exam_id)

        snapshot = build_snapshot(hub_data, exam_data)
        envelope = {
            "event_type": "session.snapshot",
            "payload": snapshot_to_dict(snapshot),
        }
        await ws.send_json(envelope)

        # Wait for either new data or the push interval
        new_data_event.clear()
        try:
            await asyncio.wait_for(
                new_data_event.wait(),
                timeout=WS_PUSH_INTERVAL_SEC,
            )
        except asyncio.TimeoutError:
            pass  # Regular interval push
