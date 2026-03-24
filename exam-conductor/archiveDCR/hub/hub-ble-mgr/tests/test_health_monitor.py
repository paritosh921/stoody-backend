"""Unit tests for HealthMonitor.

Test IDs: U-BLE-HM-01 through U-BLE-HM-06.
Validation level: L3 (unit, no I/O — health probes mocked).
"""

from __future__ import annotations

import asyncio

import pytest

from src.config import HEALTH_SLOW_RESPONSE_SEC
from src.dongle_manager import DongleManager, DongleState, DongleStatus
from src.health_monitor import HealthMonitor, ProbeResult


# ---------------------------------------------------------------------------
# Mock probe
# ---------------------------------------------------------------------------

class MockProbe:
    """Mock health probe returning preset results per dongle."""

    def __init__(self, results: dict[str, ProbeResult] | None = None) -> None:
        self.results = results or {}
        self.probe_count = 0

    async def probe(self, dongle: DongleState) -> ProbeResult:
        self.probe_count += 1
        if dongle.mac in self.results:
            return self.results[dongle.mac]
        return ProbeResult(
            dongle_mac=dongle.mac, responsive=True, response_time_sec=0.1,
        )


def _setup_mgr(*macs: str) -> DongleManager:
    mgr = DongleManager()
    for mac in macs:
        d = DongleState(mac=mac, hci_path=f"hci-{mac}")
        d.status = DongleStatus.HEALTHY
        mgr._dongles[mac] = d
    return mgr


# ---------------------------------------------------------------------------
# U-BLE-HM-01: Healthy probe keeps dongle HEALTHY
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_healthy_stays_healthy():
    """U-BLE-HM-01: Responsive fast probe keeps HEALTHY status."""
    mgr = _setup_mgr("D1")
    probe = MockProbe({
        "D1": ProbeResult(dongle_mac="D1", responsive=True, response_time_sec=0.1),
    })
    events: list[tuple[str, DongleStatus, str]] = []

    async def on_change(mac, status, detail):
        events.append((mac, status, detail))

    mon = HealthMonitor(mgr, probe, on_health_change=on_change)
    await mon.check_all()

    assert mgr.get_dongle("D1").status == DongleStatus.HEALTHY
    assert len(events) == 0  # No state change.


# ---------------------------------------------------------------------------
# U-BLE-HM-02: Slow response transitions to DEGRADED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slow_response_degrades():
    """U-BLE-HM-02: Slow response time causes HEALTHY -> DEGRADED transition."""
    mgr = _setup_mgr("D1")
    probe = MockProbe({
        "D1": ProbeResult(
            dongle_mac="D1",
            responsive=True,
            response_time_sec=HEALTH_SLOW_RESPONSE_SEC + 1.0,
            detail="slow hciconfig",
        ),
    })
    events: list[tuple[str, DongleStatus, str]] = []

    async def on_change(mac, status, detail):
        events.append((mac, status, detail))

    mon = HealthMonitor(mgr, probe, on_health_change=on_change)
    await mon.check_all()

    assert mgr.get_dongle("D1").status == DongleStatus.DEGRADED
    assert len(events) == 1
    assert events[0][1] == DongleStatus.DEGRADED


# ---------------------------------------------------------------------------
# U-BLE-HM-03: Unresponsive dongle transitions to FAILED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unresponsive_fails():
    """U-BLE-HM-03: Non-responsive dongle causes transition to FAILED."""
    mgr = _setup_mgr("D1")
    probe = MockProbe({
        "D1": ProbeResult(
            dongle_mac="D1",
            responsive=False,
            detail="no response to hciconfig",
        ),
    })
    events: list[tuple[str, DongleStatus, str]] = []

    async def on_change(mac, status, detail):
        events.append((mac, status, detail))

    mon = HealthMonitor(mgr, probe, on_health_change=on_change)
    await mon.check_all()

    assert mgr.get_dongle("D1").status == DongleStatus.FAILED
    assert len(events) == 1
    assert events[0][1] == DongleStatus.FAILED


# ---------------------------------------------------------------------------
# U-BLE-HM-04: Recovery from DEGRADED back to HEALTHY
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recovery_from_degraded():
    """U-BLE-HM-04: A degraded dongle recovers to HEALTHY on good probe."""
    mgr = _setup_mgr("D1")
    mgr.transition_health("D1", DongleStatus.DEGRADED, "was slow")

    probe = MockProbe({
        "D1": ProbeResult(dongle_mac="D1", responsive=True, response_time_sec=0.1),
    })
    events: list[tuple[str, DongleStatus, str]] = []

    async def on_change(mac, status, detail):
        events.append((mac, status, detail))

    mon = HealthMonitor(mgr, probe, on_health_change=on_change)
    await mon.check_all()

    assert mgr.get_dongle("D1").status == DongleStatus.HEALTHY
    assert len(events) == 1
    assert events[0][1] == DongleStatus.HEALTHY


# ---------------------------------------------------------------------------
# U-BLE-HM-05: Hot-unplug notification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hot_unplug_notification():
    """U-BLE-HM-05: notify_dongle_removed marks dongle FAILED immediately."""
    mgr = _setup_mgr("D1")
    probe = MockProbe()
    events: list[tuple[str, DongleStatus, str]] = []

    async def on_change(mac, status, detail):
        events.append((mac, status, detail))

    mon = HealthMonitor(mgr, probe, on_health_change=on_change)
    await mon.notify_dongle_removed("D1")

    assert mgr.get_dongle("D1").status == DongleStatus.FAILED
    assert len(events) == 1
    assert events[0] == ("D1", DongleStatus.FAILED, "hot-unplug detected")


# ---------------------------------------------------------------------------
# U-BLE-HM-06: Background task lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_background_task_lifecycle():
    """U-BLE-HM-06: start()/stop() manage the background task correctly."""
    mgr = _setup_mgr("D1")
    probe = MockProbe()
    mon = HealthMonitor(mgr, probe, interval_sec=0.05)

    assert not mon.running
    mon.start()
    assert mon.running

    # Let it run a couple of cycles.
    await asyncio.sleep(0.15)
    assert probe.probe_count >= 2

    mon.stop()
    assert not mon.running


# ---------------------------------------------------------------------------
# U-BLE-HM-07: Can-scan=False degrades
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cannot_scan_degrades():
    """U-BLE-HM-07: Responsive but unable to scan -> DEGRADED."""
    mgr = _setup_mgr("D1")
    probe = MockProbe({
        "D1": ProbeResult(
            dongle_mac="D1",
            responsive=True,
            response_time_sec=0.1,
            can_scan=False,
            detail="scan test failed",
        ),
    })
    events: list[tuple[str, DongleStatus, str]] = []

    async def on_change(mac, status, detail):
        events.append((mac, status, detail))

    mon = HealthMonitor(mgr, probe, on_health_change=on_change)
    await mon.check_all()

    assert mgr.get_dongle("D1").status == DongleStatus.DEGRADED


# ---------------------------------------------------------------------------
# U-BLE-HM-08: Probe exception treated as failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_exception_is_failure():
    """U-BLE-HM-08: An exception during probe marks dongle as FAILED."""
    mgr = _setup_mgr("D1")

    class ExplodingProbe:
        async def probe(self, dongle):
            raise OSError("USB error")

    events: list[tuple[str, DongleStatus, str]] = []

    async def on_change(mac, status, detail):
        events.append((mac, status, detail))

    mon = HealthMonitor(mgr, ExplodingProbe(), on_health_change=on_change)
    await mon.check_all()

    assert mgr.get_dongle("D1").status == DongleStatus.FAILED
    assert len(events) == 1
