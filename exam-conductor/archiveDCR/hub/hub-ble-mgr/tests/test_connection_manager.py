"""Unit tests for ConnectionManager.

Test IDs: U-BLE-CM-01 through U-BLE-CM-07.
Validation level: L3 (unit, no I/O — BLE connector mocked).
"""

from __future__ import annotations

import pytest

from src.config import CONNECTION_MAX_RETRIES, MAX_PENS_PER_DONGLE
from src.connection_manager import ConnectionManager, ConnectionRecord
from src.dongle_manager import DongleManager, DongleState, DongleStatus


# ---------------------------------------------------------------------------
# Mock connector
# ---------------------------------------------------------------------------

class MockConnector:
    """Mock BLE connector for testing."""

    def __init__(self, *, fail_count: int = 0, fail_always: bool = False) -> None:
        self.connections: list[tuple[str, str]] = []
        self.disconnections: list[str] = []
        self._fail_count = fail_count
        self._fail_always = fail_always
        self._attempt: int = 0

    async def connect(self, pen_mac: str, dongle_mac: str, timeout: float) -> str:
        self._attempt += 1
        if self._fail_always or self._attempt <= self._fail_count:
            raise ConnectionError(f"Mock connect failure #{self._attempt}")
        self.connections.append((pen_mac, dongle_mac))
        return f"conn-{pen_mac}"

    async def disconnect(self, pen_mac: str) -> None:
        self.disconnections.append(pen_mac)


def _setup_mgr(*macs: str) -> DongleManager:
    mgr = DongleManager()
    for mac in macs:
        d = DongleState(mac=mac, hci_path=f"hci-{mac}")
        d.status = DongleStatus.HEALTHY
        mgr._dongles[mac] = d
    return mgr


# ---------------------------------------------------------------------------
# U-BLE-CM-01: Successful connection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_pen_success():
    """U-BLE-CM-01: connect_pen succeeds and tracks the connection."""
    mgr = _setup_mgr("D1")
    connector = MockConnector()
    cm = ConnectionManager(mgr, connector)

    record = await cm.connect_pen("PEN:01", "D1")
    assert record is not None
    assert record.pen_mac == "PEN:01"
    assert record.dongle_mac == "D1"
    assert record.connection_id == "conn-PEN:01"
    assert record.retries == 0

    # Dongle tracking updated.
    d1 = mgr.get_dongle("D1")
    assert d1.connected_pens == 1
    assert "PEN:01" in d1.pen_macs


# ---------------------------------------------------------------------------
# U-BLE-CM-02: Connection with retries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_pen_with_retries():
    """U-BLE-CM-02: Connection succeeds after transient failures."""
    mgr = _setup_mgr("D1")
    connector = MockConnector(fail_count=2)  # First 2 attempts fail.
    cm = ConnectionManager(mgr, connector)

    record = await cm.connect_pen("PEN:01", "D1")
    assert record is not None
    assert record.retries == 2
    assert record.connection_id == "conn-PEN:01"


# ---------------------------------------------------------------------------
# U-BLE-CM-03: All retries exhausted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_all_retries_exhausted():
    """U-BLE-CM-03: connect_pen returns None after all retries fail."""
    mgr = _setup_mgr("D1")
    connector = MockConnector(fail_always=True)
    cm = ConnectionManager(mgr, connector)

    record = await cm.connect_pen("PEN:01", "D1")
    assert record is None
    assert len(cm.connections) == 0


# ---------------------------------------------------------------------------
# U-BLE-CM-04: Disconnect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_pen():
    """U-BLE-CM-04: disconnect_pen removes tracking and calls BLE disconnect."""
    mgr = _setup_mgr("D1")
    connector = MockConnector()
    cm = ConnectionManager(mgr, connector)

    await cm.connect_pen("PEN:01", "D1")
    assert "PEN:01" in cm.connections

    result = await cm.disconnect_pen("PEN:01")
    assert result is True
    assert "PEN:01" not in cm.connections
    assert "PEN:01" in connector.disconnections
    assert mgr.get_dongle("D1").connected_pens == 0


@pytest.mark.asyncio
async def test_disconnect_unknown_pen():
    """U-BLE-CM-04b: Disconnecting an unknown pen returns False."""
    mgr = _setup_mgr("D1")
    connector = MockConnector()
    cm = ConnectionManager(mgr, connector)

    result = await cm.disconnect_pen("PEN:UNKNOWN")
    assert result is False


# ---------------------------------------------------------------------------
# U-BLE-CM-05: Auto-select dongle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_select_dongle():
    """U-BLE-CM-05: connect_pen auto-selects a dongle if none specified."""
    mgr = _setup_mgr("D1", "D2")
    connector = MockConnector()
    cm = ConnectionManager(mgr, connector)

    record = await cm.connect_pen("PEN:01")
    assert record is not None
    assert record.dongle_mac in ("D1", "D2")


# ---------------------------------------------------------------------------
# U-BLE-CM-06: Overflow redirect when specified dongle is full
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_overflow_redirect():
    """U-BLE-CM-06: If specified dongle is full, redirect to another."""
    mgr = _setup_mgr("D1", "D2")
    connector = MockConnector()
    cm = ConnectionManager(mgr, connector)

    # Fill D1.
    for i in range(MAX_PENS_PER_DONGLE):
        await cm.connect_pen(f"PEN:{i:02d}", "D1")

    # Next pen requests D1 but should overflow to D2.
    record = await cm.connect_pen("PEN:OVERFLOW", "D1")
    assert record is not None
    assert record.dongle_mac == "D2"


# ---------------------------------------------------------------------------
# U-BLE-CM-07: Dongle failure re-queues pens
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dongle_failure_requeue():
    """U-BLE-CM-07: Pens from a failed dongle are reconnected on another."""
    mgr = _setup_mgr("D1", "D2")
    connector = MockConnector()
    disconnected: list[tuple[str, str]] = []

    async def on_disconnected(pen_mac: str, dongle_mac: str) -> None:
        disconnected.append((pen_mac, dongle_mac))

    cm = ConnectionManager(
        mgr, connector, on_disconnected=on_disconnected,
    )

    await cm.connect_pen("PEN:01", "D1")
    await cm.connect_pen("PEN:02", "D1")

    unplaced = await cm.handle_dongle_failure("D1")

    # Both pens should be reconnected on D2.
    assert len(unplaced) == 0
    assert mgr.get_dongle("D1").status == DongleStatus.FAILED
    assert mgr.get_dongle("D2").connected_pens == 2

    # Disconnect events fired for both pens leaving D1.
    assert len(disconnected) == 2


@pytest.mark.asyncio
async def test_dongle_failure_no_capacity():
    """U-BLE-CM-07b: Pens unplaced when no capacity on remaining dongles."""
    mgr = _setup_mgr("D1")
    connector = MockConnector()
    cm = ConnectionManager(mgr, connector)

    await cm.connect_pen("PEN:01", "D1")

    # D1 is the only dongle and it fails — no alternative.
    unplaced = await cm.handle_dongle_failure("D1")
    assert "PEN:01" in unplaced


# ---------------------------------------------------------------------------
# U-BLE-CM-08: disconnect_all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_all():
    """U-BLE-CM-08: disconnect_all removes all connections."""
    mgr = _setup_mgr("D1")
    connector = MockConnector()
    cm = ConnectionManager(mgr, connector)

    await cm.connect_pen("PEN:01", "D1")
    await cm.connect_pen("PEN:02", "D1")
    assert len(cm.connections) == 2

    await cm.disconnect_all()
    assert len(cm.connections) == 0
    assert mgr.get_dongle("D1").connected_pens == 0
