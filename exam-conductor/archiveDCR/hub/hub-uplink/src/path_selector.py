"""Upload path selection — ZERO I/O, pure domain logic.

Decides whether to upload via WiFi (primary) or mobile BLE relay
(last resort) based on connectivity state passed in by the caller.

FAILURE_MITIGATION_REGISTER.md U1: BLE relay is ~12 min for 40 pens,
so WiFi is always preferred when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.config import MOBILE_BLE_SEC_PER_CHUNK, WIFI_SEC_PER_CHUNK


class UploadPath(str, Enum):
    """Resolved upload path."""

    WIFI = "wifi"
    MOBILE = "mobile"


@dataclass(slots=True, frozen=True)
class PathDecision:
    """Result of path selection with estimated upload time."""

    path: UploadPath
    estimated_sec: float
    reason: str


def select_path(
    wifi_available: bool,
    backend_reachable: bool,
    mobile_connected: bool,
    total_chunks: int = 0,
) -> PathDecision:
    """Select the best upload path given current connectivity.

    This function performs NO I/O — it only evaluates the boolean
    inputs and returns a decision struct.  The caller is responsible
    for probing WiFi and backend status before calling this.

    Parameters
    ----------
    wifi_available:
        True when the hub has a WiFi IP address and link is up.
    backend_reachable:
        True when an HTTP HEAD to the backend health endpoint succeeded.
    mobile_connected:
        True when an invigilator mobile is connected via BLE relay.
    total_chunks:
        Chunk count used for upload time estimation.  Zero when unknown.
    """
    if wifi_available and backend_reachable:
        return PathDecision(
            path=UploadPath.WIFI,
            estimated_sec=estimate_upload_time(UploadPath.WIFI, total_chunks),
            reason="WiFi connected and backend reachable",
        )

    if mobile_connected:
        return PathDecision(
            path=UploadPath.MOBILE,
            estimated_sec=estimate_upload_time(UploadPath.MOBILE, total_chunks),
            reason="WiFi unavailable; using mobile BLE relay",
        )

    # No path available — default to WiFi so the upload loop retries
    # until connectivity is restored.
    return PathDecision(
        path=UploadPath.WIFI,
        estimated_sec=0.0,
        reason="No connectivity; will retry on WiFi when available",
    )


def estimate_upload_time(path: UploadPath, total_chunks: int) -> float:
    """Return estimated seconds to upload *total_chunks* on *path*.

    Uses constants from ``config.py`` derived from field measurements:
    WiFi ~0.5 s/chunk, BLE relay ~6 s/chunk (per U1 in failure register).
    """
    if total_chunks <= 0:
        return 0.0
    rate = WIFI_SEC_PER_CHUNK if path == UploadPath.WIFI else MOBILE_BLE_SEC_PER_CHUNK
    return rate * total_chunks
