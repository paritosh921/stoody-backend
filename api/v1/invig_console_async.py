"""
ExamPen Invigilator Console API — read-only operational views for exam sessions.

Handles:
  - Session state for a conducted exam
  - Per-hub connectivity and status
  - Connected pens per hub
  - Upload/sync progress per hub and per pen
  - Operational alerts (degraded storage, failed uploads, missing heartbeats)

Architecture:
    IMPLEMENTATION_PLAN.md §UP-002
    new-docs/api/invig-console.openapi.yaml

Ownership Declaration:
    - Writes:  NONE (read-only surface)
    - Reads from: exampen_exams, exampen_hubs, evalpen_submissions, evalpen_answer_pages
    - Never writes to: any collection

Hard constraints:
    - C1: MongoDB only
    - Read-only — no scoring, review, or lifecycle mutation actions
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()

# Heartbeat is considered stale after this many seconds
HEARTBEAT_STALE_SECONDS = 90


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def require_invigilator_or_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Require admin, tutor, or invigilator role."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for invigilator console",
        )
    return current_user


# ---------------------------------------------------------------------------
# Tenant DB helper
# ---------------------------------------------------------------------------

async def _get_tenant_db(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Any:
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context missing from token",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    exam_id: str
    exam_type: str
    lifecycle_state: str
    roster_size: int
    hub_count: int
    hubs_online: int
    total_pens_connected: int
    upload_completion_pct: float
    started_at: Optional[str] = None
    collection_closed_at: Optional[str] = None


class HubStatus(BaseModel):
    hub_id: str
    hub_name: Optional[str] = None
    online: bool
    last_heartbeat: Optional[str] = None
    assigned_exam_id: Optional[str] = None
    connected_pens: int = 0
    storage_health: str = "unknown"  # ok | degraded | unavailable
    upload_received: int = 0
    upload_acknowledged: int = 0
    session_started_at: Optional[str] = None
    session_ended_at: Optional[str] = None


class PenInfo(BaseModel):
    pen_mac: str
    hub_id: str
    student_id: Optional[str] = None
    connected: bool = True
    sync_status: str = "unknown"  # pending | syncing | synced | uploaded | failed
    pages_uploaded: int = 0


class SyncProgress(BaseModel):
    exam_id: str
    total_expected: int
    total_received: int
    total_acknowledged: int
    completion_pct: float
    by_hub: List[Dict[str, Any]]
    by_pen: List[Dict[str, Any]]


class Alert(BaseModel):
    alert_type: str  # hub_offline | heartbeat_stale | storage_degraded | upload_failed | usb_missing
    severity: str    # warning | critical
    hub_id: Optional[str] = None
    message: str
    detected_at: str


# ---------------------------------------------------------------------------
# Helper: format datetime
# ---------------------------------------------------------------------------

def _fmt(v) -> Optional[str]:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if v is not None:
        return str(v)
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{exam_id}",
    summary="Get current exam session state",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_session_state(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_invigilator_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> SessionSummary:
    """Aggregate session state for one conducted exam.

    Combines exam metadata, hub assignment status, and upload progress
    into a single operational snapshot.
    """
    tenant_db = await _get_tenant_db(db, current_user)

    # Fetch exam
    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    hub_assignments = exam.get("hub_assignments", [])
    hub_ids = [ha.get("hub_id") for ha in hub_assignments]
    roster = exam.get("roster", [])

    # Fetch hub statuses
    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(seconds=HEARTBEAT_STALE_SECONDS)
    hubs_online = 0
    total_pens = 0

    if hub_ids:
        cursor = tenant_db["exampen_hubs"].find({"hub_id": {"$in": hub_ids}})
        hub_docs = await cursor.to_list(length=100)
        for hd in hub_docs:
            last_hb = hd.get("last_heartbeat_at")
            if last_hb and last_hb > stale_threshold:
                hubs_online += 1
            total_pens += hd.get("connected_pen_count", 0)

    # Upload progress
    submission_count = await tenant_db["evalpen_submissions"].count_documents(
        {"exam_id": exam_id}
    )
    roster_size = len(roster)
    completion_pct = (submission_count / roster_size * 100) if roster_size > 0 else 0.0

    return SessionSummary(
        exam_id=exam_id,
        exam_type=exam.get("exam_type", ""),
        lifecycle_state=exam.get("lifecycle_state", "draft"),
        roster_size=roster_size,
        hub_count=len(hub_ids),
        hubs_online=hubs_online,
        total_pens_connected=total_pens,
        upload_completion_pct=round(completion_pct, 1),
        started_at=_fmt(exam.get("started_at")),
        collection_closed_at=_fmt(exam.get("collection_closed_at")),
    )


@router.get(
    "/{exam_id}/hubs",
    summary="Get per-hub connectivity and status",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_hub_statuses(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_invigilator_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> List[HubStatus]:
    """Return per-hub operational status for an exam's assigned hubs."""
    tenant_db = await _get_tenant_db(db, current_user)

    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    hub_assignments = exam.get("hub_assignments", [])
    hub_ids = [ha.get("hub_id") for ha in hub_assignments]
    assignment_map = {ha.get("hub_id"): ha for ha in hub_assignments}

    if not hub_ids:
        return []

    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(seconds=HEARTBEAT_STALE_SECONDS)

    cursor = tenant_db["exampen_hubs"].find({"hub_id": {"$in": hub_ids}})
    hub_docs = {hd["hub_id"]: hd for hd in await cursor.to_list(length=100)}

    # Per-hub upload counts (hub_id is written by ingest service)
    pipeline = [
        {"$match": {"exam_id": exam_id, "hub_id": {"$in": hub_ids}}},
        {
            "$group": {
                "_id": "$hub_id",
                "received": {"$sum": 1},
                "acknowledged": {
                    "$sum": {"$cond": [{"$eq": ["$upload_status", "acknowledged"]}, 1, 0]}
                },
            }
        },
    ]
    upload_cursor = tenant_db["evalpen_submissions"].aggregate(pipeline)
    upload_map = {r["_id"]: r for r in await upload_cursor.to_list(length=100)}

    results = []
    for hid in hub_ids:
        hd = hub_docs.get(hid, {})
        ha = assignment_map.get(hid, {})
        up = upload_map.get(hid, {})

        last_hb = hd.get("last_heartbeat_at")
        online = bool(last_hb and last_hb > stale_threshold)

        results.append(HubStatus(
            hub_id=hid,
            hub_name=ha.get("hub_name") or hd.get("hub_name"),
            online=online,
            last_heartbeat=_fmt(last_hb),
            assigned_exam_id=hd.get("assigned_exam_id"),
            connected_pens=hd.get("connected_pen_count", 0),
            storage_health=hd.get("storage_health", "unknown"),
            upload_received=up.get("received", 0),
            upload_acknowledged=up.get("acknowledged", 0),
            session_started_at=_fmt(ha.get("session_started_at")),
            session_ended_at=_fmt(ha.get("session_ended_at")),
        ))

    return results


@router.get(
    "/{exam_id}/pens",
    summary="Get connected pens per hub for an exam",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_connected_pens(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_invigilator_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> List[PenInfo]:
    """Return all pens reported by hubs assigned to this exam."""
    tenant_db = await _get_tenant_db(db, current_user)

    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    hub_ids = [ha.get("hub_id") for ha in exam.get("hub_assignments", [])]
    if not hub_ids:
        return []

    cursor = tenant_db["exampen_hubs"].find({"hub_id": {"$in": hub_ids}})
    hub_docs = await cursor.to_list(length=100)

    pens = []
    for hd in hub_docs:
        for pen in hd.get("registered_pens", []):
            pens.append(PenInfo(
                pen_mac=pen.get("pen_mac", ""),
                hub_id=hd["hub_id"],
                student_id=pen.get("student_id"),
                connected=pen.get("connected", False),
                sync_status=pen.get("sync_status", "unknown"),
                pages_uploaded=pen.get("pages_uploaded", 0),
            ))

    return pens


@router.get(
    "/{exam_id}/sync-progress",
    summary="Get upload completion per hub and pen",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_sync_progress(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_invigilator_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> SyncProgress:
    """Detailed sync/upload progress broken down by hub and pen."""
    tenant_db = await _get_tenant_db(db, current_user)

    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    roster = exam.get("roster", [])
    roster_size = len(roster)

    # By hub (hub_id is written by ingest service)
    hub_pipeline = [
        {"$match": {"exam_id": exam_id}},
        {
            "$group": {
                "_id": {"$ifNull": ["$hub_id", "unknown"]},
                "received": {"$sum": 1},
                "acknowledged": {
                    "$sum": {"$cond": [{"$eq": ["$upload_status", "acknowledged"]}, 1, 0]}
                },
            }
        },
    ]
    hub_cursor = tenant_db["evalpen_submissions"].aggregate(hub_pipeline)
    by_hub = [
        {"hub_id": r["_id"], "received": r["received"], "acknowledged": r["acknowledged"]}
        for r in await hub_cursor.to_list(length=100)
    ]

    # By pen (pen_mac)
    pen_pipeline = [
        {"$match": {"exam_id": exam_id}},
        {
            "$group": {
                "_id": {"$ifNull": ["$pen_mac", "unknown"]},
                "received": {"$sum": 1},
                "acknowledged": {
                    "$sum": {"$cond": [{"$eq": ["$upload_status", "acknowledged"]}, 1, 0]}
                },
            }
        },
    ]
    pen_cursor = tenant_db["evalpen_submissions"].aggregate(pen_pipeline)
    by_pen = [
        {"pen_mac": r["_id"], "received": r["received"], "acknowledged": r["acknowledged"]}
        for r in await pen_cursor.to_list(length=100)
    ]

    total_received = sum(h["received"] for h in by_hub)
    total_acknowledged = sum(h["acknowledged"] for h in by_hub)
    completion_pct = (total_received / roster_size * 100) if roster_size > 0 else 0.0

    return SyncProgress(
        exam_id=exam_id,
        total_expected=roster_size,
        total_received=total_received,
        total_acknowledged=total_acknowledged,
        completion_pct=round(completion_pct, 1),
        by_hub=by_hub,
        by_pen=by_pen,
    )


@router.get(
    "/{exam_id}/alerts",
    summary="Get operational alerts for an exam session",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_alerts(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_invigilator_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> List[Alert]:
    """Return active operational alerts for an exam's hub fleet.

    Checks for:
      - Hubs with stale/missing heartbeats
      - Degraded storage (USB backup missing)
      - Failed uploads
    """
    tenant_db = await _get_tenant_db(db, current_user)

    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    hub_ids = [ha.get("hub_id") for ha in exam.get("hub_assignments", [])]
    if not hub_ids:
        return []

    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(seconds=HEARTBEAT_STALE_SECONDS)
    now_str = now.isoformat()

    cursor = tenant_db["exampen_hubs"].find({"hub_id": {"$in": hub_ids}})
    hub_docs = await cursor.to_list(length=100)

    alerts: List[Alert] = []

    for hd in hub_docs:
        hid = hd.get("hub_id", "")

        # Heartbeat stale / offline
        last_hb = hd.get("last_heartbeat_at")
        if last_hb is None:
            alerts.append(Alert(
                alert_type="hub_offline",
                severity="critical",
                hub_id=hid,
                message=f"Hub {hid} has never sent a heartbeat",
                detected_at=now_str,
            ))
        elif last_hb < stale_threshold:
            alerts.append(Alert(
                alert_type="heartbeat_stale",
                severity="warning",
                hub_id=hid,
                message=f"Hub {hid} last heartbeat at {_fmt(last_hb)} — over {HEARTBEAT_STALE_SECONDS}s ago",
                detected_at=now_str,
            ))

        # Degraded storage
        storage = hd.get("storage_health", "unknown")
        if storage == "degraded":
            alerts.append(Alert(
                alert_type="storage_degraded",
                severity="warning",
                hub_id=hid,
                message=f"Hub {hid} running in degraded storage mode (USB backup unavailable)",
                detected_at=now_str,
            ))
        elif storage == "unavailable":
            alerts.append(Alert(
                alert_type="storage_degraded",
                severity="critical",
                hub_id=hid,
                message=f"Hub {hid} storage unavailable",
                detected_at=now_str,
            ))

        # Failed uploads
        failed = hd.get("failed_upload_count", 0)
        if failed > 0:
            alerts.append(Alert(
                alert_type="upload_failed",
                severity="warning" if failed < 5 else "critical",
                hub_id=hid,
                message=f"Hub {hid} has {failed} failed upload(s)",
                detected_at=now_str,
            ))

    # Check for hubs in assignment that never registered
    registered_ids = {hd.get("hub_id") for hd in hub_docs}
    for hid in hub_ids:
        if hid not in registered_ids:
            alerts.append(Alert(
                alert_type="hub_offline",
                severity="critical",
                hub_id=hid,
                message=f"Hub {hid} is assigned but has not registered with backend",
                detected_at=now_str,
            ))

    return alerts
