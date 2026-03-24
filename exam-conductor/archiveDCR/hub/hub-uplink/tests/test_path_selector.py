"""Tests for upload path selection — pure domain logic, ZERO I/O.

Test IDs: U-UPL-01 .. U-UPL-07
Validation level: L3 (unit — no I/O, no mocks needed)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.path_selector import PathDecision, UploadPath, estimate_upload_time, select_path


# -----------------------------------------------------------------------
# U-UPL-01: WiFi + backend reachable -> WiFi path
# -----------------------------------------------------------------------

def test_wifi_preferred_when_available() -> None:
    """U-UPL-01: WiFi is selected when both WiFi and backend are up."""
    decision = select_path(
        wifi_available=True,
        backend_reachable=True,
        mobile_connected=False,
        total_chunks=10,
    )
    assert decision.path == UploadPath.WIFI
    assert decision.estimated_sec > 0


# -----------------------------------------------------------------------
# U-UPL-02: WiFi + backend reachable + mobile connected -> still WiFi
# -----------------------------------------------------------------------

def test_wifi_preferred_over_mobile() -> None:
    """U-UPL-02: WiFi takes priority even when mobile is connected."""
    decision = select_path(
        wifi_available=True,
        backend_reachable=True,
        mobile_connected=True,
        total_chunks=5,
    )
    assert decision.path == UploadPath.WIFI


# -----------------------------------------------------------------------
# U-UPL-03: No WiFi + mobile connected -> mobile path
# -----------------------------------------------------------------------

def test_mobile_fallback_when_no_wifi() -> None:
    """U-UPL-03: mobile BLE relay used when WiFi is unavailable."""
    decision = select_path(
        wifi_available=False,
        backend_reachable=False,
        mobile_connected=True,
        total_chunks=10,
    )
    assert decision.path == UploadPath.MOBILE
    assert "mobile" in decision.reason.lower() or "ble" in decision.reason.lower()


# -----------------------------------------------------------------------
# U-UPL-04: WiFi up but backend unreachable + mobile -> mobile
# -----------------------------------------------------------------------

def test_mobile_when_backend_unreachable() -> None:
    """U-UPL-04: backend unreachable over WiFi -> fall back to mobile."""
    decision = select_path(
        wifi_available=True,
        backend_reachable=False,
        mobile_connected=True,
        total_chunks=10,
    )
    assert decision.path == UploadPath.MOBILE


# -----------------------------------------------------------------------
# U-UPL-05: No connectivity at all -> default to WiFi for retry
# -----------------------------------------------------------------------

def test_no_connectivity_defaults_to_wifi() -> None:
    """U-UPL-05: when nothing is available, default to WiFi for retry."""
    decision = select_path(
        wifi_available=False,
        backend_reachable=False,
        mobile_connected=False,
        total_chunks=10,
    )
    assert decision.path == UploadPath.WIFI
    assert "retry" in decision.reason.lower()


# -----------------------------------------------------------------------
# U-UPL-06: Upload time estimate — WiFi faster than mobile
# -----------------------------------------------------------------------

def test_wifi_estimate_faster_than_mobile() -> None:
    """U-UPL-06: WiFi estimate is significantly less than mobile."""
    wifi_time = estimate_upload_time(UploadPath.WIFI, 40)
    mobile_time = estimate_upload_time(UploadPath.MOBILE, 40)
    assert wifi_time < mobile_time
    # Per FAILURE_MITIGATION_REGISTER U1: mobile ~12x slower
    assert mobile_time / wifi_time > 5


# -----------------------------------------------------------------------
# U-UPL-07: Zero chunks -> zero estimated time
# -----------------------------------------------------------------------

def test_zero_chunks_zero_estimate() -> None:
    """U-UPL-07: estimate is 0 when chunk count is 0 or negative."""
    assert estimate_upload_time(UploadPath.WIFI, 0) == 0.0
    assert estimate_upload_time(UploadPath.MOBILE, -1) == 0.0


# -----------------------------------------------------------------------
# U-UPL-07b: PathDecision is frozen dataclass
# -----------------------------------------------------------------------

def test_path_decision_immutable() -> None:
    """U-UPL-07b: PathDecision instances are immutable."""
    decision = select_path(True, True, False, 1)
    import dataclasses

    assert dataclasses.is_dataclass(decision)
    try:
        decision.path = UploadPath.MOBILE  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except (AttributeError, dataclasses.FrozenInstanceError):
        pass
