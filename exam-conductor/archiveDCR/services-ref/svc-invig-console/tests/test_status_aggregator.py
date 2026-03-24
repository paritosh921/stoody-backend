"""Unit tests for the domain status aggregator (ZERO I/O).

Test IDs: U-INVIG-01 through U-INVIG-10
Validation level: L3 (unit test verified, domain logic, no I/O)
"""

from __future__ import annotations

import pytest

from src.domain.status_aggregator import (
    DashboardSnapshot,
    DongleStatus,
    PenStatus,
    SyncProgress,
    UploadProgress,
    WifiStatus,
    build_snapshot,
    build_sync_progress,
    snapshot_to_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hub_data(
    exam_id: str = "exam-001",
    pens: list | None = None,
    dongles: list | None = None,
    wifi: dict | None = None,
    upload: dict | None = None,
    timer: dict | None = None,
) -> dict:
    return {
        "exam_id": exam_id,
        "pens": pens or [],
        "dongles": dongles or [],
        "wifi": wifi or {},
        "upload": upload or {},
        "timer": timer or {},
    }


def _exam_data(
    exam_id: str = "exam-001",
    state: str = "in_progress",
    timer_remaining_sec: int = 3600,
    upload_status: str = "pending",
) -> dict:
    return {
        "exam_id": exam_id,
        "state": state,
        "timer_remaining_sec": timer_remaining_sec,
        "upload_status": upload_status,
    }


# ---------------------------------------------------------------------------
# U-INVIG-01: Empty inputs produce a valid default snapshot
# ---------------------------------------------------------------------------


def test_empty_inputs_produce_default_snapshot():
    """U-INVIG-01: build_snapshot with empty dicts returns safe defaults."""
    snapshot = build_snapshot({}, {})
    assert snapshot.exam_id == ""
    assert snapshot.exam_state == "unknown"
    assert snapshot.timer_remaining_sec == 0
    assert snapshot.pens == []
    assert snapshot.dongles == []
    assert snapshot.wifi.connected is False
    assert snapshot.sync_progress.total_pens == 0
    assert snapshot.upload_progress.status == "pending"


# ---------------------------------------------------------------------------
# U-INVIG-02: Exam state flows from exam_data
# ---------------------------------------------------------------------------


def test_exam_state_from_exam_data():
    """U-INVIG-02: exam_state comes from exam_data.state."""
    snapshot = build_snapshot(
        _hub_data(),
        _exam_data(state="collecting"),
    )
    assert snapshot.exam_state == "collecting"


# ---------------------------------------------------------------------------
# U-INVIG-03: Timer prefers hub real-time value
# ---------------------------------------------------------------------------


def test_timer_prefers_hub_value():
    """U-INVIG-03: Timer from hub overrides exam-orch value."""
    snapshot = build_snapshot(
        _hub_data(timer={"remaining_sec": 1800}),
        _exam_data(timer_remaining_sec=3600),
    )
    assert snapshot.timer_remaining_sec == 1800


def test_timer_falls_back_to_exam_orch():
    """U-INVIG-03b: Timer falls back to exam-orch when hub has no timer."""
    snapshot = build_snapshot(
        _hub_data(timer={}),
        _exam_data(timer_remaining_sec=3600),
    )
    assert snapshot.timer_remaining_sec == 3600


# ---------------------------------------------------------------------------
# U-INVIG-04: Pen parsing
# ---------------------------------------------------------------------------


def test_pen_parsing():
    """U-INVIG-04: Pens are correctly parsed from hub data."""
    snapshot = build_snapshot(
        _hub_data(pens=[
            {
                "pen_mac": "AA:BB:CC:DD:EE:01",
                "student_id": "stu-001",
                "sync_status": "syncing",
                "bytes_received": 4096,
                "total_chunks": 10,
            },
            {
                "pen_mac": "AA:BB:CC:DD:EE:02",
                "sync_status": "complete",
                "total_chunks": 5,
            },
        ]),
        _exam_data(),
    )
    assert len(snapshot.pens) == 2
    assert snapshot.pens[0].pen_mac == "AA:BB:CC:DD:EE:01"
    assert snapshot.pens[0].student_id == "stu-001"
    assert snapshot.pens[0].sync_status == "syncing"
    assert snapshot.pens[0].bytes_received == 4096
    assert snapshot.pens[1].student_id is None
    assert snapshot.pens[1].sync_status == "complete"


# ---------------------------------------------------------------------------
# U-INVIG-05: Dongle parsing
# ---------------------------------------------------------------------------


def test_dongle_parsing():
    """U-INVIG-05: Dongles are correctly parsed from hub data."""
    snapshot = build_snapshot(
        _hub_data(dongles=[
            {
                "dongle_mac": "DD:00:00:00:00:01",
                "status": "healthy",
                "connected_pens": 7,
                "capacity": 8,
            },
            {
                "dongle_mac": "DD:00:00:00:00:02",
                "status": "degraded",
                "connected_pens": 0,
            },
        ]),
        _exam_data(),
    )
    assert len(snapshot.dongles) == 2
    assert snapshot.dongles[0].connected_pens == 7
    assert snapshot.dongles[1].status == "degraded"
    assert snapshot.dongles[1].capacity == 8  # default


# ---------------------------------------------------------------------------
# U-INVIG-06: WiFi parsing
# ---------------------------------------------------------------------------


def test_wifi_parsing():
    """U-INVIG-06: WiFi status is correctly parsed from hub data."""
    snapshot = build_snapshot(
        _hub_data(wifi={
            "connected": True,
            "ssid": "ExamRoom-5G",
            "signal_strength_dbm": -45,
        }),
        _exam_data(),
    )
    assert snapshot.wifi.connected is True
    assert snapshot.wifi.ssid == "ExamRoom-5G"
    assert snapshot.wifi.signal_strength_dbm == -45


# ---------------------------------------------------------------------------
# U-INVIG-07: Sync progress aggregation
# ---------------------------------------------------------------------------


def test_sync_progress_aggregation():
    """U-INVIG-07: Sync progress is correctly derived from pen list."""
    pens = [
        PenStatus(pen_mac="A", sync_status="complete"),
        PenStatus(pen_mac="B", sync_status="syncing"),
        PenStatus(pen_mac="C", sync_status="connecting"),
        PenStatus(pen_mac="D", sync_status="failed"),
        PenStatus(pen_mac="E", sync_status="timeout"),
        PenStatus(pen_mac="F", sync_status="pending"),
    ]
    progress = build_sync_progress(pens)
    assert progress.total_pens == 6
    assert progress.synced_pens == 1
    assert progress.syncing_pens == 2  # syncing + connecting
    assert progress.failed_pens == 2  # failed + timeout


# ---------------------------------------------------------------------------
# U-INVIG-08: Upload progress parsing
# ---------------------------------------------------------------------------


def test_upload_progress():
    """U-INVIG-08: Upload progress is correctly parsed."""
    snapshot = build_snapshot(
        _hub_data(upload={
            "status": "in_progress",
            "total_chunks": 100,
            "acked_chunks": 42,
        }),
        _exam_data(),
    )
    assert snapshot.upload_progress.status == "in_progress"
    assert snapshot.upload_progress.total_chunks == 100
    assert snapshot.upload_progress.acked_chunks == 42


# ---------------------------------------------------------------------------
# U-INVIG-09: snapshot_to_dict round-trip
# ---------------------------------------------------------------------------


def test_snapshot_to_dict_roundtrip():
    """U-INVIG-09: snapshot_to_dict produces JSON-serialisable output."""
    snapshot = build_snapshot(
        _hub_data(
            pens=[{"pen_mac": "X", "sync_status": "complete", "total_chunks": 3}],
            dongles=[{"dongle_mac": "Y", "status": "healthy", "connected_pens": 1}],
            wifi={"connected": True, "ssid": "Net", "signal_strength_dbm": -50},
            upload={"status": "complete", "total_chunks": 10, "acked_chunks": 10},
            timer={"remaining_sec": 900},
        ),
        _exam_data(state="collecting"),
    )
    d = snapshot_to_dict(snapshot)

    assert d["exam_state"] == "collecting"
    assert d["timer_remaining_sec"] == 900
    assert len(d["pens"]) == 1
    assert d["pens"][0]["pen_mac"] == "X"
    assert len(d["dongles"]) == 1
    assert d["wifi"]["connected"] is True
    assert d["sync_progress"]["synced_pens"] == 1
    assert d["upload_progress"]["acked_chunks"] == 10

    # Verify all values are JSON-serialisable (no dataclasses left)
    import json
    json.dumps(d)  # Should not raise


# ---------------------------------------------------------------------------
# U-INVIG-10: exam_id comes from exam_data preferentially
# ---------------------------------------------------------------------------


def test_exam_id_from_exam_data():
    """U-INVIG-10: exam_id prefers exam_data, falls back to hub_data."""
    snapshot = build_snapshot(
        _hub_data(exam_id="hub-id"),
        _exam_data(exam_id="orch-id"),
    )
    assert snapshot.exam_id == "orch-id"

    # Fall back to hub_data when exam_data has no exam_id
    snapshot2 = build_snapshot(
        _hub_data(exam_id="hub-id"),
        {},
    )
    assert snapshot2.exam_id == "hub-id"
