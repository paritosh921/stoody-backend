"""
Hub Registration API

This module handles BLE hub registration and pen synchronization.
Hubs (running on Raspberry Pi) register themselves with the backend,
pushing their connected pens list. This works for both local development
and cloud deployment (AWS).

Conducted-exam artifact uploads from hubs are bridged into the shared
ingest substrate (SWM-012) when exam metadata is present.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hub", tags=["hub"])


# ---------------------------------------------------------------------------
# Ingest bridge helper (SWM-012) — graceful degradation if exam-conductor
# is not available.
# ---------------------------------------------------------------------------

async def _bridge_hub_artifacts_to_ingest(
    *,
    exam_id: str,
    student_id: str,
    admin_id: str,
    pen_mac: str,
    pages: List[Dict[str, Any]],
    db_name: str,
) -> Optional[Dict[str, Any]]:
    """Bridge hub-uploaded conducted-exam artifacts into the shared ingest
    substrate.

    Returns the IngestResult dict on success, or None if the ingest
    substrate is unavailable.  Failures are logged but never propagated
    to the caller — the hub upload must succeed regardless.

    Parameters
    ----------
    exam_id : str
        Conducted exam identifier.
    student_id : str
        Student identity.
    admin_id : str
        Tenant admin who owns this exam context.
    pen_mac : str
        BLE pen MAC address.
    pages : list[dict]
        List of page dicts with ``page_number`` and ``raw_strokes``.
    db_name : str
        Tenant database name (e.g. ``skb_<tenant>``).
    """
    try:
        from api.v1._exampen_imports import load_exampen
        ingest_mod = load_exampen("ingest.service")
        IngestService = ingest_mod.IngestService
    except (ImportError, AttributeError):
        logger.debug(
            "exam-conductor ingest not available — skipping hub artifact bridge"
        )
        return None

    try:
        from core.database import DatabaseManager
        db_manager = DatabaseManager()
        tenant_db = await db_manager.get_tenant_db(db_name)
        if tenant_db is None:
            logger.warning(
                "Could not resolve tenant DB %s for hub ingest bridge", db_name
            )
            return None

        service = IngestService(tenant_db)
        await service.initialize()

        result = await service.ingest_submission(
            exam_id=exam_id,
            student_id=student_id,
            admin_id=admin_id,
            source="ble_pen",
            pen_mac=pen_mac,
            pages=pages,
        )

        logger.info(
            "Hub ingest bridge: submission_id=%s exam=%s student=%s "
            "pages=%d already_existed=%s",
            result.submission_id,
            exam_id,
            student_id,
            result.page_count,
            result.already_existed,
        )
        return result.model_dump()

    except Exception:
        logger.exception(
            "Hub ingest bridge failed for exam=%s student=%s pen=%s "
            "(non-fatal, hub upload continues)",
            exam_id,
            student_id,
            pen_mac,
        )
        return None


class PenInfo(BaseModel):
    """Pen information from hub."""
    pen_mac: str
    pen_id: str
    connected: bool = True


class HubRegistration(BaseModel):
    """Hub registration payload."""
    hub_id: str
    hub_ip: Optional[str] = None
    pens: List[PenInfo] = Field(default_factory=list)
    timestamp: Optional[float] = None


class HubStatus(BaseModel):
    """Hub status response."""
    hub_id: str
    registered: bool
    pen_count: int
    last_seen: Optional[str] = None


# In-memory hub registry (could be moved to Redis/DB for production)
_hub_registry: Dict[str, Dict[str, Any]] = {}


def get_app_state():
    from main_async import app
    return app.state


@router.post("/register", response_model=HubStatus)
async def register_hub(
    payload: HubRegistration,
    token: Optional[str] = None,
    state=Depends(get_app_state),
):
    """
    Register a hub and its connected pens with the backend.

    Called by Pi hub on startup and when pen list changes.
    Updates the backend's pen registry with the hub's pens.
    """
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    hub_id = payload.hub_id
    now = datetime.utcnow()

    # Get current connected count
    connected_count = sum(1 for p in payload.pens if p.connected)

    # Check if pen count changed from last registration
    prev_hub_data = _hub_registry.get(hub_id, {})
    prev_connected_count = prev_hub_data.get("connected_count", -1)

    # Only log when pen count changes
    if connected_count != prev_connected_count:
        logger.info(f"Hub {hub_id}: {connected_count} pens connected")

    # Update hub registry
    _hub_registry[hub_id] = {
        "hub_id": hub_id,
        "hub_ip": payload.hub_ip,
        "pens": [p.model_dump() for p in payload.pens],
        "connected_count": connected_count,
        "last_seen": now.isoformat(),
        "registered_at": prev_hub_data.get("registered_at", now.isoformat()),
    }

    # Sync pens to backend registry (always call to detect disconnections)
    await _sync_pens_to_registry(state, payload.pens, hub_id, prev_hub_data)

    return HubStatus(
        hub_id=hub_id,
        registered=True,
        pen_count=len(payload.pens),
        last_seen=now.isoformat(),
    )


async def _sync_pens_to_registry(state, pens: List[PenInfo], hub_id: str, prev_hub_data: Dict[str, Any]):
    """Sync pens from hub to the backend's pen registry and dashboard."""
    import json
    import time
    from pathlib import Path
    from .dashboard import dashboard_ws_manager

    # Get registry path from pen_router
    registry_path = state.pen_router._registry_path

    # Load existing registry
    existing_registry = []
    if registry_path.exists():
        with open(registry_path, "r") as f:
            existing_registry = json.load(f)

    existing_by_mac = {pen["pen_mac"].upper(): pen for pen in existing_registry}

    # Track which pens are currently connected from this hub
    current_pen_macs = {pen.pen_mac.upper() for pen in pens if pen.connected}

    # Get previously connected pens from this hub to detect disconnections
    prev_pen_macs = {
        p["pen_mac"].upper()
        for p in prev_hub_data.get("pens", [])
        if p.get("connected", True)
    }

    # Pens that were connected before but are now missing = disconnected
    disconnected_macs = prev_pen_macs - current_pen_macs

    # Merge pens from hub
    file_updated = False
    for pen in pens:
        mac_upper = pen.pen_mac.upper()

        if mac_upper not in existing_by_mac:
            # New pen - add to registry
            new_pen = {
                "pen_id": pen.pen_id,
                "pen_mac": mac_upper,
                "student_name": f"Student {len(existing_registry) + 1}",
                "school_id": "SCH-01",
                "instance_id": "INST-1",
                "hub_id": hub_id,
            }
            existing_registry.append(new_pen)
            existing_by_mac[mac_upper] = new_pen
            file_updated = True
            logger.debug(f"Added new pen: {mac_upper} -> {new_pen['student_name']}")
        else:
            # Existing pen - update hub_id if changed
            existing_pen = existing_by_mac[mac_upper]
            if existing_pen.get("hub_id") != hub_id:
                existing_pen["hub_id"] = hub_id
                file_updated = True

    if file_updated:
        with open(registry_path, "w") as f:
            json.dump(existing_registry, f, indent=2)
        logger.debug(f"Registry file updated with {len(existing_registry)} pens")

        # Reload pen router cache
        await state.pen_router.reload()

    # Update dashboard registry with current pen status
    for pen in pens:
        mac_upper = pen.pen_mac.upper()
        pen_record = existing_by_mac.get(mac_upper, {})
        pen_id = pen_record.get("pen_id", pen.pen_id)

        # Update dashboard registry
        await state.dashboard_registry.update_status(
            pen_id,
            pen_mac=mac_upper,
            hub_id=hub_id,
            connected=pen.connected,
            last_frame_ts=time.time(),
        )

        # Broadcast pen status to connected dashboards
        await dashboard_ws_manager.broadcast({
            "type": "pen_status",
            "pen_id": pen_id,
            "pen_mac": mac_upper,
            "hub_id": hub_id,
            "connected": pen.connected,
            "timestamp": time.time(),
        })

    # Mark disconnected pens (pens that were in prev list but not in current)
    for mac in disconnected_macs:
        pen_record = existing_by_mac.get(mac)
        if pen_record:
            pen_id = pen_record.get("pen_id")
            logger.info(f"Pen {pen_id} ({mac}) disconnected from hub {hub_id}")

            # Update dashboard registry
            await state.dashboard_registry.update_status(
                pen_id,
                pen_mac=mac,
                hub_id=hub_id,
                connected=False,
            )

            # Broadcast disconnection to dashboards
            await dashboard_ws_manager.broadcast({
                "type": "pen_status",
                "pen_id": pen_id,
                "pen_mac": mac,
                "hub_id": hub_id,
                "connected": False,
                "timestamp": time.time(),
            })


@router.get("/status", response_model=List[HubStatus])
async def get_hubs_status(
    token: Optional[str] = None,
    state=Depends(get_app_state),
):
    """Get status of all registered hubs."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    return [
        HubStatus(
            hub_id=hub["hub_id"],
            registered=True,
            pen_count=len(hub.get("pens", [])),
            last_seen=hub.get("last_seen"),
        )
        for hub in _hub_registry.values()
    ]


@router.get("/{hub_id}", response_model=Dict[str, Any])
async def get_hub_details(
    hub_id: str,
    token: Optional[str] = None,
    state=Depends(get_app_state),
):
    """Get detailed info for a specific hub."""
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    if hub_id not in _hub_registry:
        raise HTTPException(status_code=404, detail=f"Hub {hub_id} not found")

    return _hub_registry[hub_id]


# ============================================================================
# CONDUCTED-EXAM ARTIFACT UPLOAD (SWM-012 ingest bridge)
# ============================================================================

class ExamArtifactPage(BaseModel):
    """A single page of conducted-exam pen data uploaded by the hub."""
    page_number: int = Field(..., ge=1, description="1-based page number")
    raw_strokes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Canonical stroke vectors from BLE pen",
    )


class HubExamUpload(BaseModel):
    """Conducted-exam artifact upload payload from hub.

    The hub sends this after post-exam pen sync is complete.
    Only payloads with ``exam_id`` trigger the ingest bridge;
    regular (non-exam) hub uploads continue through existing paths.
    """
    hub_id: str
    exam_id: str = Field(..., description="Conducted exam identifier")
    student_id: str = Field(..., description="Student identity")
    admin_id: str = Field(..., description="Tenant admin identity")
    pen_mac: str = Field(..., description="BLE pen MAC address")
    db_name: str = Field(..., description="Tenant database name (skb_<tenant>)")
    pages: List[ExamArtifactPage] = Field(
        default_factory=list,
        description="Per-page stroke data from the pen",
    )


class HubExamUploadResponse(BaseModel):
    """Response from the conducted-exam artifact upload."""
    success: bool
    hub_id: str
    exam_id: str
    student_id: str
    submission_id: Optional[str] = None
    page_count: int = 0
    already_existed: bool = False
    ingest_available: bool = True
    message: str = ""


@router.post("/exam-upload", response_model=HubExamUploadResponse)
async def hub_exam_upload(
    payload: HubExamUpload,
    token: Optional[str] = None,
    state=Depends(get_app_state),
):
    """Receive conducted-exam artifacts from a hub and bridge into the
    shared ingest substrate.

    This endpoint is called by the hub's uplink module after post-exam
    pen sync completes.  It persists the canonical pen artifacts through
    ``IngestService.ingest_submission()`` with full provenance.

    Non-exam flows should NOT call this endpoint — they continue using
    the existing ``/hub/register`` and Stoody pen sync paths.
    """
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")

    # Convert pages to the dict format expected by IngestService
    page_dicts = [
        {
            "page_number": p.page_number,
            "raw_strokes": p.raw_strokes,
        }
        for p in payload.pages
    ]

    result = await _bridge_hub_artifacts_to_ingest(
        exam_id=payload.exam_id,
        student_id=payload.student_id,
        admin_id=payload.admin_id,
        pen_mac=payload.pen_mac.upper(),
        pages=page_dicts,
        db_name=payload.db_name,
    )

    if result is None:
        # Ingest substrate not available — log and return success
        # so the hub doesn't retry needlessly.  The artifacts can be
        # ingested later via a backfill process.
        logger.warning(
            "Ingest bridge unavailable for hub exam upload: "
            "hub=%s exam=%s student=%s",
            payload.hub_id,
            payload.exam_id,
            payload.student_id,
        )
        return HubExamUploadResponse(
            success=True,
            hub_id=payload.hub_id,
            exam_id=payload.exam_id,
            student_id=payload.student_id,
            ingest_available=False,
            message="Artifacts received but ingest substrate is not available. "
                    "They will be ingested via backfill.",
        )

    return HubExamUploadResponse(
        success=True,
        hub_id=payload.hub_id,
        exam_id=payload.exam_id,
        student_id=payload.student_id,
        submission_id=result.get("submission_id"),
        page_count=result.get("page_count", 0),
        already_existed=result.get("already_existed", False),
        ingest_available=True,
        message="Conducted-exam artifacts ingested successfully.",
    )
