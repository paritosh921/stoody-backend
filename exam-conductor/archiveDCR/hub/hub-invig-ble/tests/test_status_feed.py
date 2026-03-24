"""Unit tests for status feed collector and formatter.

Test IDs: U-INVIG-SF-01 through U-INVIG-SF-05.
Validation level: L3 (unit, no I/O).
"""

from __future__ import annotations

import json

import pytest

from src.status_feed import HubSnapshot, StatusFeedCollector


# ---------------------------------------------------------------------------
# U-INVIG-SF-01: Default snapshot produces valid JSON with all required fields
# ---------------------------------------------------------------------------

def test_default_snapshot_has_required_fields():
    """U-INVIG-SF-01: Freshly created collector outputs all required fields."""
    collector = StatusFeedCollector()
    d = collector.to_dict()

    # Required fields per ble-gatt-spec.md Section 5.
    assert "exam_id" in d
    assert "state" in d
    assert "timer_remaining_sec" in d
    assert d["wifi"]["connected"] is not None
    assert "degraded" in d["storage"]
    assert "complete" in d["sync"]
    assert "in_progress" in d["sync"]
    assert "failed" in d["sync"]
    assert "pending" in d["sync"]


def test_default_json_bytes_are_valid_utf8():
    """U-INVIG-SF-01b: to_json_bytes produces valid UTF-8 JSON."""
    collector = StatusFeedCollector()
    raw = collector.to_json_bytes()

    parsed = json.loads(raw.decode("utf-8"))
    assert isinstance(parsed, dict)
    assert parsed["state"] == "idle"


# ---------------------------------------------------------------------------
# U-INVIG-SF-02: Update from HubSnapshot is reflected in output
# ---------------------------------------------------------------------------

def test_update_from_snapshot():
    """U-INVIG-SF-02: Updating with a HubSnapshot changes the output."""
    collector = StatusFeedCollector()

    snap = HubSnapshot(
        exam_id="exam-123",
        state="pen_sync",
        timer_remaining_sec=0,
        wifi_connected=True,
        wifi_band="5GHz",
        wifi_signal_dbm=-42,
        storage_sd_ok=True,
        storage_usb_ok=True,
        storage_degraded=False,
        sync_complete=34,
        sync_in_progress=3,
        sync_failed=0,
        sync_pending=3,
    )
    collector.update(snap)
    d = collector.to_dict()

    assert d["exam_id"] == "exam-123"
    assert d["state"] == "pen_sync"
    assert d["timer_remaining_sec"] == 0
    assert d["wifi"]["connected"] is True
    assert d["wifi"]["band"] == "5GHz"
    assert d["wifi"]["signal_dbm"] == -42
    assert d["storage"]["sd_ok"] is True
    assert d["storage"]["degraded"] is False
    assert d["sync"]["complete"] == 34
    assert d["sync"]["in_progress"] == 3
    assert d["sync"]["failed"] == 0
    assert d["sync"]["pending"] == 3


# ---------------------------------------------------------------------------
# U-INVIG-SF-03: Update from IPC payload dict
# ---------------------------------------------------------------------------

def test_update_from_ipc_payload():
    """U-INVIG-SF-03: update_from_ipc parses a supervisor snapshot payload."""
    collector = StatusFeedCollector()

    ipc_payload = {
        "exam_id": "exam-456",
        "state": "timer_running",
        "timer": {"remaining_sec": 2700},
        "wifi": {"connected": True, "band": "5GHz", "signal_dbm": -38},
        "storage": {"sd_ok": True, "usb_ok": False, "degraded": True},
        "sync": {"complete": 0, "in_progress": 0, "failed": 0, "pending": 40},
    }
    collector.update_from_ipc(ipc_payload)
    d = collector.to_dict()

    assert d["exam_id"] == "exam-456"
    assert d["state"] == "timer_running"
    assert d["timer_remaining_sec"] == 2700
    assert d["storage"]["usb_ok"] is False
    assert d["storage"]["degraded"] is True
    assert d["sync"]["pending"] == 40


def test_update_from_ipc_with_missing_fields():
    """U-INVIG-SF-03b: Missing optional IPC fields get safe defaults."""
    collector = StatusFeedCollector()

    ipc_payload = {
        "exam_id": "exam-789",
        "state": "armed",
    }
    collector.update_from_ipc(ipc_payload)
    d = collector.to_dict()

    assert d["exam_id"] == "exam-789"
    assert d["timer_remaining_sec"] == 0
    assert d["wifi"]["connected"] is False
    assert d["storage"]["degraded"] is False
    assert d["sync"]["complete"] == 0


# ---------------------------------------------------------------------------
# U-INVIG-SF-04: JSON schema matches ble-gatt-spec.md Section 5
# ---------------------------------------------------------------------------

def test_json_schema_structure():
    """U-INVIG-SF-04: Output JSON structure matches the GATT spec exactly."""
    collector = StatusFeedCollector()
    snap = HubSnapshot(
        exam_id="exam-schema",
        state="uploading",
        timer_remaining_sec=0,
        wifi_connected=True,
        wifi_band="2.4GHz",
        wifi_signal_dbm=-55,
        storage_sd_ok=True,
        storage_usb_ok=True,
        storage_degraded=False,
        sync_complete=40,
        sync_in_progress=0,
        sync_failed=0,
        sync_pending=0,
    )
    collector.update(snap)
    d = collector.to_dict()

    # Top-level keys.
    assert set(d.keys()) == {
        "exam_id", "state", "timer_remaining_sec", "wifi", "storage", "sync",
    }

    # Nested keys.
    assert set(d["wifi"].keys()) == {"connected", "band", "signal_dbm"}
    assert set(d["storage"].keys()) == {"sd_ok", "usb_ok", "degraded"}
    assert set(d["sync"].keys()) == {"complete", "in_progress", "failed", "pending"}


# ---------------------------------------------------------------------------
# U-INVIG-SF-05: Serialized JSON is compact (no unnecessary whitespace)
# ---------------------------------------------------------------------------

def test_json_bytes_compact():
    """U-INVIG-SF-05: Serialized bytes use compact separators (no spaces)."""
    collector = StatusFeedCollector()
    raw = collector.to_json_bytes()
    text = raw.decode("utf-8")

    # Compact JSON should have no ": " or ", ".
    assert ": " not in text
    assert ", " not in text
    # But should still be valid JSON.
    parsed = json.loads(text)
    assert isinstance(parsed, dict)


def test_successive_updates_overwrite():
    """U-INVIG-SF-05b: Each update fully replaces the previous snapshot."""
    collector = StatusFeedCollector()

    snap1 = HubSnapshot(
        exam_id="exam-A", state="pen_sync", timer_remaining_sec=0,
        wifi_connected=True, wifi_band="5GHz", wifi_signal_dbm=-40,
        storage_sd_ok=True, storage_usb_ok=True, storage_degraded=False,
        sync_complete=10, sync_in_progress=5, sync_failed=0, sync_pending=25,
    )
    collector.update(snap1)
    assert collector.to_dict()["exam_id"] == "exam-A"

    snap2 = HubSnapshot(
        exam_id="exam-B", state="uploading", timer_remaining_sec=0,
        wifi_connected=False, wifi_band="", wifi_signal_dbm=0,
        storage_sd_ok=True, storage_usb_ok=False, storage_degraded=True,
        sync_complete=40, sync_in_progress=0, sync_failed=0, sync_pending=0,
    )
    collector.update(snap2)
    d = collector.to_dict()
    assert d["exam_id"] == "exam-B"
    assert d["state"] == "uploading"
    assert d["storage"]["degraded"] is True
