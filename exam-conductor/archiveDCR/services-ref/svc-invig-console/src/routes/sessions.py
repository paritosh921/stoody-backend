"""REST endpoints for invigilator exam session queries.

All endpoints are read-only -- this service never writes exam state.
Exam metadata is proxied from svc-exam-orch.  Sync progress and dongle
health come from the hub via NATS relay (``hub_relay.py``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from exampen_common.auth import ExamPenUser, get_current_user

from src.adapters.exam_client import ExamOrchClient
from src.events.hub_relay import HubRelay

router = APIRouter(tags=["sessions"])


def _get_exam_client(request: Request) -> ExamOrchClient:
    return request.app.state.exam_client


def _get_hub_relay(request: Request) -> HubRelay:
    return request.app.state.hub_relay


def _get_token(request: Request) -> str:
    """Extract the Bearer token from the incoming request."""
    auth = request.headers.get("Authorization", "")
    return auth.removeprefix("Bearer ").strip()


@router.get("/sessions")
async def list_sessions(
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
    exam_client: ExamOrchClient = Depends(_get_exam_client),
) -> dict[str, Any]:
    """List active exam sessions (proxied from svc-exam-orch)."""
    token = _get_token(request)
    sessions = await exam_client.list_active_sessions(token)
    return {"items": sessions}


@router.get("/sessions/{exam_id}")
async def get_session(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
    exam_client: ExamOrchClient = Depends(_get_exam_client),
    hub_relay: HubRelay = Depends(_get_hub_relay),
) -> dict[str, Any]:
    """Get session detail with exam state, pen count, and sync progress."""
    token = _get_token(request)
    exam_data = await exam_client.get_session(exam_id, token)
    hub_data = hub_relay.get_latest(exam_id)

    # Enrich exam-orch data with live hub info
    pens = hub_data.get("pens", [])
    dongles = hub_data.get("dongles", [])

    return {
        "exam_id": exam_data.get("exam_id", exam_id),
        "state": exam_data.get("state", "unknown"),
        "timer_remaining_sec": hub_data.get(
            "timer", {},
        ).get(
            "remaining_sec",
            exam_data.get("timer_remaining_sec", 0),
        ),
        "upload_status": exam_data.get("upload_status", "pending"),
        "pen_count": len(pens),
        "dongle_count": len(dongles),
        "sync_progress": {
            "total_pens": len(pens),
            "synced_pens": sum(
                1 for p in pens if p.get("sync_status") == "complete"
            ),
        },
    }


@router.get("/sessions/{exam_id}/sync")
async def get_sync_progress(
    exam_id: str,
    user: ExamPenUser = Depends(get_current_user),
    hub_relay: HubRelay = Depends(_get_hub_relay),
) -> dict[str, Any]:
    """Get per-pen sync progress for an exam (from hub via NATS relay)."""
    hub_data = hub_relay.get_latest(exam_id)
    pens = hub_data.get("pens", [])
    return {
        "items": [
            {
                "pen_mac": p.get("pen_mac", ""),
                "sync_status": p.get("sync_status", "unknown"),
                "progress_pct": p.get("progress_pct", 0),
                "bytes_transferred": p.get("bytes_transferred", 0),
                "bytes_total": p.get("bytes_total", 0),
            }
            for p in pens
        ],
    }


@router.get("/sessions/{exam_id}/dongles")
async def get_dongles(
    exam_id: str,
    user: ExamPenUser = Depends(get_current_user),
    hub_relay: HubRelay = Depends(_get_hub_relay),
) -> dict[str, Any]:
    """Get dongle health and capacity state (from hub via NATS relay)."""
    hub_data = hub_relay.get_latest(exam_id)
    dongles = hub_data.get("dongles", [])
    return {
        "items": [
            {
                "dongle_id": d.get("dongle_id", ""),
                "hci": d.get("hci", ""),
                "health": d.get("health", "unknown"),
                "connected_pens": d.get("connected_pens", 0),
                "capacity": d.get("capacity", 8),
            }
            for d in dongles
        ],
    }
