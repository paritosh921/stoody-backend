"""Unit tests for PenDiscovery.

Test IDs: U-BLE-PD-01 through U-BLE-PD-06.
Validation level: L3 (unit, no I/O — all BLE scan operations mocked).
"""

from __future__ import annotations

import asyncio

import pytest

from src.config import PEN_GATT_SERVICE_UUID
from src.dongle_manager import DongleManager, DongleState, DongleStatus
from src.pen_discovery import PenDiscovery, PenInfo


# ---------------------------------------------------------------------------
# Mock scanner
# ---------------------------------------------------------------------------

class MockScanner:
    """Mock BLE scanner that yields predefined pens via callback."""

    def __init__(self, pens_per_dongle: dict[str, list[PenInfo]] | None = None) -> None:
        self.pens_per_dongle = pens_per_dongle or {}
        self.started: list[str] = []
        self.stopped: list[str] = []

    async def start_scan(self, dongle, callback, timeout_sec):
        self.started.append(dongle.mac)
        pens = self.pens_per_dongle.get(dongle.mac, [])
        for pen in pens:
            pen.dongle_mac = dongle.mac
            await callback(pen)

    async def stop_scan(self, dongle):
        self.stopped.append(dongle.mac)


def _setup_mgr(*macs: str) -> DongleManager:
    """Create a DongleManager with pre-registered healthy dongles."""
    mgr = DongleManager()
    for mac in macs:
        d = DongleState(mac=mac, hci_path=f"hci-{mac}")
        d.status = DongleStatus.HEALTHY
        mgr._dongles[mac] = d
    return mgr


def _pen(mac: str, rssi: int = -50, battery: int = 80) -> PenInfo:
    """Create a PenInfo with the ExamPen service UUID."""
    return PenInfo(
        mac=mac,
        rssi=rssi,
        battery_pct=battery,
        service_uuids=[PEN_GATT_SERVICE_UUID],
    )


def _non_pen(mac: str) -> PenInfo:
    """Create a PenInfo without the ExamPen service UUID."""
    return PenInfo(mac=mac, rssi=-60, service_uuids=["0000180f-0000-1000-8000-00805f9b34fb"])


# ---------------------------------------------------------------------------
# U-BLE-PD-01: Basic scan discovers pens
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scan_discovers_pens():
    """U-BLE-PD-01: start_scan discovers pens from all dongles."""
    mgr = _setup_mgr("D1", "D2")
    discovered: list[PenInfo] = []

    async def on_pen(p: PenInfo) -> None:
        discovered.append(p)

    scanner = MockScanner({
        "D1": [_pen("PEN:01"), _pen("PEN:02")],
        "D2": [_pen("PEN:03")],
    })
    disc = PenDiscovery(mgr, scanner, on_pen_discovered=on_pen)

    await disc.start_scan(timeout_sec=5)
    # Allow tasks to complete.
    await asyncio.sleep(0.1)

    assert len(disc.discovered_pens) == 3
    assert len(discovered) == 3
    assert {"PEN:01", "PEN:02", "PEN:03"} == set(disc.discovered_pens.keys())


# ---------------------------------------------------------------------------
# U-BLE-PD-02: Deduplication keeps strongest RSSI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dedup_keeps_strongest_rssi():
    """U-BLE-PD-02: Same pen seen on two dongles — keep stronger signal."""
    mgr = _setup_mgr("D1", "D2")
    scanner = MockScanner({
        "D1": [_pen("PEN:01", rssi=-70)],
        "D2": [_pen("PEN:01", rssi=-40)],
    })
    disc = PenDiscovery(mgr, scanner)

    await disc.start_scan(timeout_sec=5)
    await asyncio.sleep(0.1)

    assert len(disc.discovered_pens) == 1
    pen = disc.get_pen("PEN:01")
    assert pen is not None
    assert pen.rssi == -40
    assert pen.dongle_mac == "D2"


# ---------------------------------------------------------------------------
# U-BLE-PD-03: Non-pen devices filtered by UUID
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_pen_filtered():
    """U-BLE-PD-03: Devices without pen GATT UUID are ignored."""
    mgr = _setup_mgr("D1")
    scanner = MockScanner({
        "D1": [_non_pen("OTHER:01"), _pen("PEN:01")],
    })
    disc = PenDiscovery(mgr, scanner)

    await disc.start_scan(timeout_sec=5)
    await asyncio.sleep(0.1)

    assert len(disc.discovered_pens) == 1
    assert "PEN:01" in disc.discovered_pens
    assert "OTHER:01" not in disc.discovered_pens


# ---------------------------------------------------------------------------
# U-BLE-PD-04: Failed dongle skipped during scan
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_dongle_skipped():
    """U-BLE-PD-04: Failed dongles are not scanned."""
    mgr = _setup_mgr("D1", "D2")
    mgr.get_dongle("D2").status = DongleStatus.FAILED

    scanner = MockScanner({
        "D1": [_pen("PEN:01")],
        "D2": [_pen("PEN:02")],  # Should never be reached.
    })
    disc = PenDiscovery(mgr, scanner)

    await disc.start_scan(timeout_sec=5)
    await asyncio.sleep(0.1)

    assert "D1" in scanner.started
    assert "D2" not in scanner.started
    assert len(disc.discovered_pens) == 1


# ---------------------------------------------------------------------------
# U-BLE-PD-05: Stop scan cancels tasks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_scan():
    """U-BLE-PD-05: stop_scan() sets scanning=False and calls stop on scanner."""
    mgr = _setup_mgr("D1")
    scanner = MockScanner({"D1": [_pen("PEN:01")]})
    disc = PenDiscovery(mgr, scanner)

    await disc.start_scan(timeout_sec=5)
    await asyncio.sleep(0.1)
    assert disc.scanning

    await disc.stop_scan()
    assert not disc.scanning
    assert "D1" in scanner.stopped


# ---------------------------------------------------------------------------
# U-BLE-PD-06: clear_discovered resets table
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_discovered():
    """U-BLE-PD-06: clear_discovered() empties the pen table."""
    mgr = _setup_mgr("D1")
    scanner = MockScanner({"D1": [_pen("PEN:01"), _pen("PEN:02")]})
    disc = PenDiscovery(mgr, scanner)

    await disc.start_scan(timeout_sec=5)
    await asyncio.sleep(0.1)
    assert len(disc.discovered_pens) == 2

    disc.clear_discovered()
    assert len(disc.discovered_pens) == 0


# ---------------------------------------------------------------------------
# U-BLE-PD-07: No dongles available logs warning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_dongles_available():
    """U-BLE-PD-07: start_scan with no available dongles sets scanning=False."""
    mgr = DongleManager()
    scanner = MockScanner()
    disc = PenDiscovery(mgr, scanner)

    await disc.start_scan(timeout_sec=5)
    assert not disc.scanning
