from __future__ import annotations

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio

from core.dashboard_registry import PenDashboardRegistry
from core.pen_workers import PenWorkers

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


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
    hub_id: Optional[str] = None
    connected: bool = False
    battery: Optional[int] = None
    page_no: Optional[int] = None
    book_type: Optional[str] = None
    last_frame_ts: Optional[float] = None
    strokes: Optional[List[DashboardStroke]] = None
    color: Optional[str] = None


class PensResponse(BaseModel):
    pens: List[PenSummary]
    maxPens: int = Field(default=6)


def get_app_state():
    from main_async import app

    return app.state


@router.get("/pens", response_model=PensResponse)
async def get_dashboard_pens(token: Optional[str] = None, state=Depends(get_app_state)):
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid dashboard token")

    # Get pen registry info for student names
    pen_registry = await state.pen_router.get_all_pens()

    pens_data = await state.dashboard_registry.list_pen_states()
    pen_summaries = []
    for pen in pens_data:
        pen_id = pen["pen_id"]
        # Look up student_name from registry
        registry_info = pen_registry.get(pen_id, {})
        student_name = registry_info.get("student_name")

        pen_summaries.append(
            PenSummary(
                pen_id=pen_id,
                pen_mac=pen.get("pen_mac", "UNKNOWN"),
                student_name=student_name,
                hub_id=pen.get("hub_id"),
                connected=pen.get("connected", False),
                battery=pen.get("battery"),
                page_no=pen.get("page_no"),
                book_type=pen.get("book_type"),
                last_frame_ts=pen.get("last_frame_ts"),
                color=pen.get("color"),
                strokes=[
                    DashboardStroke(
                        id=stroke.get("id"),
                        points=stroke.get("points", []),
                        color=stroke.get("color"),
                        strokeWidth=stroke.get("strokeWidth"),
                        timestamp=stroke.get("timestamp"),
                    )
                    for stroke in pen.get("strokes", [])[-10:]
                ],
            )
        )

    max_pens = int(state.pen_workers.max_pens or 6)
    return PensResponse(pens=pen_summaries, maxPens=max_pens)


class DashboardWebSocketManager:
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
        async with self.lock:
            dead_connections = []
            for ws in self.connections:
                if ws.application_state != WebSocketState.CONNECTED:
                    dead_connections.append(ws)
                    continue
                try:
                    await ws.send_json(message)
                except WebSocketDisconnect:
                    dead_connections.append(ws)
            for ws in dead_connections:
                self.connections.remove(ws)


dashboard_ws_manager = DashboardWebSocketManager()


@router.websocket("/ws")
async def dashboard_ws(websocket: WebSocket, state=Depends(get_app_state)):
    token = websocket.query_params.get("token")
    expected = state.dashboard_token
    if expected and token != expected:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await dashboard_ws_manager.connect(websocket)

    # Send current pen states immediately on connect
    try:
        pens_data = await state.dashboard_registry.list_pen_states()
        for pen in pens_data:
            await websocket.send_json({
                "type": "pen_status",
                "pen_id": pen.get("pen_id"),
                "pen_mac": pen.get("pen_mac"),
                "hub_id": pen.get("hub_id"),
                "connected": pen.get("connected", False),
                "battery": pen.get("battery"),
                "color": pen.get("color"),
            })
    except Exception:
        pass  # Don't fail connection if initial send fails

    async def send_heartbeat():
        """Send periodic heartbeat to keep connection alive."""
        while True:
            try:
                await asyncio.sleep(30)  # Send ping every 30 seconds
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "heartbeat", "timestamp": asyncio.get_event_loop().time()})
            except Exception:
                break

    heartbeat_task = asyncio.create_task(send_heartbeat())

    try:
        while True:
            # Wait for messages with timeout to detect dead connections
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                # Handle pong or other client messages if needed
            except asyncio.TimeoutError:
                # No message received, but connection might still be alive
                # The heartbeat task keeps the connection alive
                continue
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        heartbeat_task.cancel()
        await dashboard_ws_manager.disconnect(websocket)


class PenColorRequest(BaseModel):
    color: str


@router.post("/pen/{pen_id}/color")
async def update_pen_color(
    pen_id: str,
    payload: PenColorRequest,
    token: Optional[str] = None,
    state=Depends(get_app_state),
):
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid dashboard token")

    await state.pen_workers.set_pen_color(pen_id, payload.color)
    await state.dashboard_registry.update_status(pen_id, color=payload.color)
    await dashboard_ws_manager.broadcast(
        {
            "type": "pen_status",
            "pen_id": pen_id,
            "connected": True,
            "color": payload.color,
        }
    )
    return {"status": "ok"}

