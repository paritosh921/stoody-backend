"""Pure-logic dashboard snapshot builder.

ZERO I/O -- this module must never import asyncio, aiohttp, nats,
sqlalchemy, or any other I/O library.  All data arrives as plain dicts
or dataclass instances.

Aggregates hub status data and exam orchestrator session data into the
``DashboardSnapshot`` shape consumed by WebSocket clients and REST
endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PenStatus:
    """Per-pen sync and identity state."""

    pen_mac: str
    student_id: str | None = None
    sync_status: str = "pending"
    bytes_received: int = 0
    total_chunks: int = 0


@dataclass(frozen=True, slots=True)
class DongleStatus:
    """Per-dongle health and capacity state."""

    dongle_mac: str
    status: str = "healthy"
    connected_pens: int = 0
    capacity: int = 8


@dataclass(frozen=True, slots=True)
class WifiStatus:
    """Hub WiFi connectivity state."""

    connected: bool = False
    ssid: str = ""
    signal_strength_dbm: int = 0


@dataclass(frozen=True, slots=True)
class SyncProgress:
    """Aggregate sync progress across all pens."""

    total_pens: int = 0
    synced_pens: int = 0
    syncing_pens: int = 0
    failed_pens: int = 0


@dataclass(frozen=True, slots=True)
class UploadProgress:
    """Aggregate upload progress to backend."""

    status: str = "pending"
    total_chunks: int = 0
    acked_chunks: int = 0


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """Complete snapshot pushed to invigilator WebSocket clients."""

    exam_id: str
    exam_state: str = "unknown"
    timer_remaining_sec: int = 0
    wifi: WifiStatus = field(default_factory=WifiStatus)
    dongles: list[DongleStatus] = field(default_factory=list)
    pens: list[PenStatus] = field(default_factory=list)
    sync_progress: SyncProgress = field(default_factory=SyncProgress)
    upload_progress: UploadProgress = field(default_factory=UploadProgress)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_sync_progress(pens: list[PenStatus]) -> SyncProgress:
    """Derive aggregate sync progress from per-pen status list."""
    total = len(pens)
    synced = sum(1 for p in pens if p.sync_status == "complete")
    syncing = sum(1 for p in pens if p.sync_status in ("connecting", "syncing"))
    failed = sum(1 for p in pens if p.sync_status in ("failed", "timeout"))
    return SyncProgress(
        total_pens=total,
        synced_pens=synced,
        syncing_pens=syncing,
        failed_pens=failed,
    )


def _parse_pens(hub_data: dict[str, Any]) -> list[PenStatus]:
    """Extract pen status list from hub status payload."""
    raw_pens: list[dict[str, Any]] = hub_data.get("pens", [])
    result: list[PenStatus] = []
    for p in raw_pens:
        result.append(PenStatus(
            pen_mac=p.get("pen_mac", ""),
            student_id=p.get("student_id"),
            sync_status=p.get("sync_status", p.get("status", "pending")),
            bytes_received=p.get("bytes_received", 0),
            total_chunks=p.get("total_chunks", 0),
        ))
    return result


def _parse_dongles(hub_data: dict[str, Any]) -> list[DongleStatus]:
    """Extract dongle status list from hub status payload."""
    raw_dongles: list[dict[str, Any]] = hub_data.get("dongles", [])
    result: list[DongleStatus] = []
    for d in raw_dongles:
        result.append(DongleStatus(
            dongle_mac=d.get("dongle_mac", ""),
            status=d.get("status", "healthy"),
            connected_pens=d.get("connected_pens", 0),
            capacity=d.get("capacity", 8),
        ))
    return result


def _parse_wifi(hub_data: dict[str, Any]) -> WifiStatus:
    """Extract WiFi status from hub status payload."""
    wifi_raw: dict[str, Any] = hub_data.get("wifi", {})
    return WifiStatus(
        connected=wifi_raw.get("connected", False),
        ssid=wifi_raw.get("ssid", ""),
        signal_strength_dbm=wifi_raw.get("signal_strength_dbm", 0),
    )


def _parse_upload(hub_data: dict[str, Any]) -> UploadProgress:
    """Extract upload progress from hub status payload."""
    upload_raw: dict[str, Any] = hub_data.get("upload", {})
    return UploadProgress(
        status=upload_raw.get("status", "pending"),
        total_chunks=upload_raw.get("total_chunks", 0),
        acked_chunks=upload_raw.get("acked_chunks", 0),
    )


def build_snapshot(
    hub_data: dict[str, Any],
    exam_data: dict[str, Any],
) -> DashboardSnapshot:
    """Build a complete dashboard snapshot from hub and exam-orch data.

    Parameters
    ----------
    hub_data:
        Latest hub status payload relayed via NATS.  May contain keys:
        ``pens``, ``dongles``, ``wifi``, ``upload``, ``timer``.
    exam_data:
        Exam session data from svc-exam-orch.  Expected keys:
        ``exam_id``, ``state``, ``timer_remaining_sec``, ``upload_status``.

    Returns
    -------
    DashboardSnapshot
        The merged, display-ready snapshot.
    """
    exam_id: str = exam_data.get("exam_id", hub_data.get("exam_id", ""))
    exam_state: str = exam_data.get("state", "unknown")

    # Timer: prefer hub real-time value, fall back to exam-orch
    timer_remaining: int = hub_data.get(
        "timer", {},
    ).get(
        "remaining_sec",
        exam_data.get("timer_remaining_sec", 0),
    )

    pens = _parse_pens(hub_data)
    dongles = _parse_dongles(hub_data)
    wifi = _parse_wifi(hub_data)
    upload = _parse_upload(hub_data)
    sync_progress = build_sync_progress(pens)

    return DashboardSnapshot(
        exam_id=exam_id,
        exam_state=exam_state,
        timer_remaining_sec=timer_remaining,
        wifi=wifi,
        dongles=dongles,
        pens=pens,
        sync_progress=sync_progress,
        upload_progress=upload,
    )


# ---------------------------------------------------------------------------
# Serialisation (pure, no I/O)
# ---------------------------------------------------------------------------


def snapshot_to_dict(snapshot: DashboardSnapshot) -> dict[str, Any]:
    """Convert a DashboardSnapshot to a JSON-serialisable dict."""
    return {
        "exam_id": snapshot.exam_id,
        "exam_state": snapshot.exam_state,
        "timer_remaining_sec": snapshot.timer_remaining_sec,
        "wifi": {
            "connected": snapshot.wifi.connected,
            "ssid": snapshot.wifi.ssid,
            "signal_strength_dbm": snapshot.wifi.signal_strength_dbm,
        },
        "dongles": [
            {
                "dongle_mac": d.dongle_mac,
                "status": d.status,
                "connected_pens": d.connected_pens,
                "capacity": d.capacity,
            }
            for d in snapshot.dongles
        ],
        "pens": [
            {
                "pen_mac": p.pen_mac,
                "student_id": p.student_id,
                "sync_status": p.sync_status,
                "bytes_received": p.bytes_received,
                "total_chunks": p.total_chunks,
            }
            for p in snapshot.pens
        ],
        "sync_progress": {
            "total_pens": snapshot.sync_progress.total_pens,
            "synced_pens": snapshot.sync_progress.synced_pens,
            "syncing_pens": snapshot.sync_progress.syncing_pens,
            "failed_pens": snapshot.sync_progress.failed_pens,
        },
        "upload_progress": {
            "status": snapshot.upload_progress.status,
            "total_chunks": snapshot.upload_progress.total_chunks,
            "acked_chunks": snapshot.upload_progress.acked_chunks,
        },
    }
