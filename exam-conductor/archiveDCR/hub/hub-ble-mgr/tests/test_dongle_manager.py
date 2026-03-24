"""Unit tests for DongleManager.

Test IDs: U-BLE-DM-01 through U-BLE-DM-08.
Validation level: L3 (unit, no I/O — all BLE operations mocked).
"""

from __future__ import annotations

import pytest

from src.config import MAX_DONGLES, MAX_PENS_PER_DONGLE
from src.dongle_manager import (
    BleAdapter,
    DongleManager,
    DongleState,
    DongleStatus,
)


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

class MockAdapter:
    """Mock BLE adapter that returns a fixed list of dongles."""

    def __init__(self, dongles: list[DongleState] | None = None) -> None:
        self.dongles = dongles or []

    async def enumerate(self) -> list[DongleState]:
        return list(self.dongles)


def _make_dongle(mac: str, hci: str = "hci0") -> DongleState:
    return DongleState(mac=mac, hci_path=hci, usb_port=f"/usb/{mac}")


# ---------------------------------------------------------------------------
# U-BLE-DM-01: Enumerate dongles
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enumerate_dongles():
    """U-BLE-DM-01: refresh() populates dongle registry from adapter."""
    mgr = DongleManager()
    adapter = MockAdapter([
        _make_dongle("AA:BB:CC:DD:EE:01", "hci0"),
        _make_dongle("AA:BB:CC:DD:EE:02", "hci1"),
    ])

    result = await mgr.refresh(adapter)
    assert len(result) == 2
    assert mgr.get_dongle("AA:BB:CC:DD:EE:01") is not None
    assert mgr.get_dongle("AA:BB:CC:DD:EE:02") is not None
    # Both should be HEALTHY after initial enumeration.
    assert mgr.get_dongle("AA:BB:CC:DD:EE:01").status == DongleStatus.HEALTHY
    assert mgr.get_dongle("AA:BB:CC:DD:EE:02").status == DongleStatus.HEALTHY


# ---------------------------------------------------------------------------
# U-BLE-DM-02: Hot-plug adds new dongle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hot_plug_adds_dongle():
    """U-BLE-DM-02: A new dongle appearing in a subsequent refresh is added."""
    mgr = DongleManager()
    adapter = MockAdapter([_make_dongle("AA:BB:CC:DD:EE:01")])
    await mgr.refresh(adapter)
    assert len(mgr.dongles) == 1

    adapter.dongles.append(_make_dongle("AA:BB:CC:DD:EE:02", "hci1"))
    await mgr.refresh(adapter)
    assert len(mgr.dongles) == 2
    assert mgr.get_dongle("AA:BB:CC:DD:EE:02").status == DongleStatus.HEALTHY


# ---------------------------------------------------------------------------
# U-BLE-DM-03: Unplug marks dongle FAILED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unplug_marks_failed():
    """U-BLE-DM-03: A dongle disappearing from refresh is marked FAILED."""
    mgr = DongleManager()
    d1 = _make_dongle("AA:BB:CC:DD:EE:01")
    d2 = _make_dongle("AA:BB:CC:DD:EE:02", "hci1")
    adapter = MockAdapter([d1, d2])
    await mgr.refresh(adapter)

    # Remove d2 from adapter.
    adapter.dongles = [d1]
    await mgr.refresh(adapter)

    assert mgr.get_dongle("AA:BB:CC:DD:EE:02").status == DongleStatus.FAILED


# ---------------------------------------------------------------------------
# U-BLE-DM-04: Re-plug recovers dongle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_replug_recovers_dongle():
    """U-BLE-DM-04: A failed dongle re-appearing is marked HEALTHY."""
    mgr = DongleManager()
    d1 = _make_dongle("AA:BB:CC:DD:EE:01")
    adapter = MockAdapter([d1])
    await mgr.refresh(adapter)

    # Unplug.
    adapter.dongles = []
    await mgr.refresh(adapter)
    assert mgr.get_dongle("AA:BB:CC:DD:EE:01").status == DongleStatus.FAILED

    # Re-plug.
    adapter.dongles = [d1]
    await mgr.refresh(adapter)
    assert mgr.get_dongle("AA:BB:CC:DD:EE:01").status == DongleStatus.HEALTHY


# ---------------------------------------------------------------------------
# U-BLE-DM-05: Pen limit per dongle
# ---------------------------------------------------------------------------

def test_pen_limit_per_dongle():
    """U-BLE-DM-05: assign_pen fails when dongle is at MAX_PENS_PER_DONGLE."""
    mgr = DongleManager()
    dongle = _make_dongle("AA:BB:CC:DD:EE:01")
    dongle.status = DongleStatus.HEALTHY
    mgr._dongles["AA:BB:CC:DD:EE:01"] = dongle

    # Fill to capacity.
    for i in range(MAX_PENS_PER_DONGLE):
        assert mgr.assign_pen("AA:BB:CC:DD:EE:01", f"PEN:{i:02d}")

    # One more should fail.
    assert not mgr.assign_pen("AA:BB:CC:DD:EE:01", "PEN:OVERFLOW")
    assert dongle.connected_pens == MAX_PENS_PER_DONGLE


# ---------------------------------------------------------------------------
# U-BLE-DM-06: Overflow redirection to next dongle
# ---------------------------------------------------------------------------

def test_overflow_redirection():
    """U-BLE-DM-06: find_available_dongle returns dongle with capacity."""
    mgr = DongleManager()

    d1 = _make_dongle("D1")
    d1.status = DongleStatus.HEALTHY
    d2 = _make_dongle("D2", "hci1")
    d2.status = DongleStatus.HEALTHY
    mgr._dongles["D1"] = d1
    mgr._dongles["D2"] = d2

    # Fill d1.
    for i in range(MAX_PENS_PER_DONGLE):
        mgr.assign_pen("D1", f"PEN:{i:02d}")

    # d1 is full; find_available_dongle should return d2.
    available = mgr.find_available_dongle()
    assert available is not None
    assert available.mac == "D2"


# ---------------------------------------------------------------------------
# U-BLE-DM-07: collect_orphaned_pens clears tracking
# ---------------------------------------------------------------------------

def test_collect_orphaned_pens():
    """U-BLE-DM-07: Orphan collection returns pen MACs and clears dongle."""
    mgr = DongleManager()
    d1 = _make_dongle("D1")
    d1.status = DongleStatus.HEALTHY
    mgr._dongles["D1"] = d1

    mgr.assign_pen("D1", "PEN:01")
    mgr.assign_pen("D1", "PEN:02")
    assert d1.connected_pens == 2

    orphans = mgr.collect_orphaned_pens("D1")
    assert set(orphans) == {"PEN:01", "PEN:02"}
    assert d1.connected_pens == 0
    assert d1.pen_macs == set()


# ---------------------------------------------------------------------------
# U-BLE-DM-08: MAX_DONGLES cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_dongles_cap():
    """U-BLE-DM-08: Dongles beyond MAX_DONGLES are ignored."""
    mgr = DongleManager()
    dongles = [
        _make_dongle(f"D{i}", f"hci{i}")
        for i in range(MAX_DONGLES + 2)
    ]
    adapter = MockAdapter(dongles)
    await mgr.refresh(adapter)

    assert len(mgr.dongles) == MAX_DONGLES


# ---------------------------------------------------------------------------
# U-BLE-DM-09: Health FSM transitions
# ---------------------------------------------------------------------------

def test_health_fsm_valid_transitions():
    """U-BLE-DM-09: Valid health transitions are accepted."""
    mgr = DongleManager()
    d = _make_dongle("D1")
    d.status = DongleStatus.HEALTHY
    mgr._dongles["D1"] = d

    result = mgr.transition_health("D1", DongleStatus.DEGRADED, "slow")
    assert result is not None
    assert result.status == DongleStatus.DEGRADED

    result = mgr.transition_health("D1", DongleStatus.FAILED, "timeout")
    assert result is not None
    assert result.status == DongleStatus.FAILED


def test_health_fsm_invalid_transition():
    """U-BLE-DM-09b: Invalid health transition returns None."""
    mgr = DongleManager()
    d = _make_dongle("D1")
    d.status = DongleStatus.UNKNOWN
    mgr._dongles["D1"] = d

    # unknown -> degraded is not a valid transition.
    result = mgr.transition_health("D1", DongleStatus.DEGRADED)
    assert result is None
    assert d.status == DongleStatus.UNKNOWN


# ---------------------------------------------------------------------------
# U-BLE-DM-10: summary builds correct dict
# ---------------------------------------------------------------------------

def test_summary():
    """U-BLE-DM-10: summary() returns dongle list and totals."""
    mgr = DongleManager()
    d1 = _make_dongle("D1")
    d1.status = DongleStatus.HEALTHY
    mgr._dongles["D1"] = d1
    mgr.assign_pen("D1", "PEN:01")

    s = mgr.summary()
    assert s["total_connected"] == 1
    assert s["total_capacity"] == MAX_PENS_PER_DONGLE - 1
    assert len(s["dongles"]) == 1
    assert s["dongles"][0]["mac"] == "D1"
