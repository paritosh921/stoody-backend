"""Pen discovery and BLE scan orchestration.

Manages scanning across multiple dongles with staggered activation (H5)
and deduplicates discovered pens across all dongles.

Domain logic is separated from the BLE adapter layer -- the ``BleScanner``
protocol is injected so discovery logic is testable without hardware.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Protocol

from src.config import (
    DEFAULT_SCAN_TIMEOUT_SEC,
    PEN_GATT_SERVICE_UUID,
    SCAN_STAGGER_DELAY_SEC,
)
from src.dongle_manager import DongleManager, DongleState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PenInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class PenInfo:
    """Discovered pen metadata."""

    mac: str
    rssi: int = -100
    battery_pct: int = 0
    dongle_mac: str = ""
    connected: bool = False
    service_uuids: list[str] = field(default_factory=list)

    def has_pen_service(self) -> bool:
        """True if this device advertises the ExamPen GATT service UUID."""
        return PEN_GATT_SERVICE_UUID in self.service_uuids

    def to_dict(self) -> dict:
        return {
            "pen_mac": self.mac,
            "rssi": self.rssi,
            "battery_pct": self.battery_pct,
            "dongle_mac": self.dongle_mac,
            "connected": self.connected,
        }


# ---------------------------------------------------------------------------
# BLE scanner protocol (injected dependency)
# ---------------------------------------------------------------------------

# Callback signature: called when a pen is discovered during scan.
DiscoveryCallback = Callable[[PenInfo], Coroutine[Any, Any, None]]


class BleScanner(Protocol):
    """Abstract BLE scan interface.

    Production implementation wraps bleak.BleakScanner.
    Tests inject a mock that yields synthetic PenInfo objects.
    """

    async def start_scan(
        self,
        dongle: DongleState,
        callback: DiscoveryCallback,
        timeout_sec: int,
    ) -> None:
        """Start scanning on a specific dongle, calling *callback* for each
        device that advertises the pen GATT service UUID."""
        ...  # pragma: no cover

    async def stop_scan(self, dongle: DongleState) -> None:
        """Stop an active scan on *dongle*."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# PenDiscovery
# ---------------------------------------------------------------------------

class PenDiscovery:
    """Orchestrates pen discovery across multiple BLE dongles.

    Key behaviors:
      - Stagger scan activation across dongles (500 ms delay per H5).
      - Deduplicate pens seen on multiple dongles (keep strongest RSSI).
      - Filter by PEN_GATT_SERVICE_UUID.
      - Emit discovery events via a callback.
    """

    def __init__(
        self,
        dongle_mgr: DongleManager,
        scanner: BleScanner,
        on_pen_discovered: DiscoveryCallback | None = None,
    ) -> None:
        self._dongle_mgr = dongle_mgr
        self._scanner = scanner
        self._on_pen_discovered = on_pen_discovered

        # pen_mac -> PenInfo (deduplicated across dongles).
        self._discovered: dict[str, PenInfo] = {}
        self._scanning: bool = False
        self._scan_tasks: list[asyncio.Task] = []

    # -- Public interface ---------------------------------------------------

    @property
    def scanning(self) -> bool:
        return self._scanning

    @property
    def discovered_pens(self) -> dict[str, PenInfo]:
        """All pens discovered so far (deduplicated)."""
        return dict(self._discovered)

    def get_pen(self, pen_mac: str) -> PenInfo | None:
        return self._discovered.get(pen_mac)

    async def start_scan(
        self,
        *,
        timeout_sec: int = DEFAULT_SCAN_TIMEOUT_SEC,
    ) -> None:
        """Start staggered BLE scans on all healthy dongles."""
        if self._scanning:
            logger.warning("Scan already in progress")
            return

        self._scanning = True
        dongles = [
            d for d in self._dongle_mgr.dongles.values()
            if d.is_available
        ]

        if not dongles:
            logger.warning("No available dongles for scanning")
            self._scanning = False
            return

        logger.info(
            "Starting scan on %d dongle(s), stagger=%.1fs, timeout=%ds",
            len(dongles), SCAN_STAGGER_DELAY_SEC, timeout_sec,
        )

        for i, dongle in enumerate(dongles):
            if i > 0:
                await asyncio.sleep(SCAN_STAGGER_DELAY_SEC)
            task = asyncio.create_task(
                self._run_dongle_scan(dongle, timeout_sec),
                name=f"scan-{dongle.mac}",
            )
            self._scan_tasks.append(task)

    async def stop_scan(self) -> None:
        """Stop all active scans."""
        if not self._scanning:
            return

        self._scanning = False

        for task in self._scan_tasks:
            task.cancel()

        for dongle in self._dongle_mgr.dongles.values():
            if dongle.is_available:
                try:
                    await self._scanner.stop_scan(dongle)
                except Exception:
                    logger.debug(
                        "Error stopping scan on %s (may already be stopped)",
                        dongle.mac,
                    )

        self._scan_tasks.clear()
        logger.info("Scanning stopped")

    def clear_discovered(self) -> None:
        """Reset the discovered pen table (e.g., between exam sessions)."""
        self._discovered.clear()

    # -- Internal -----------------------------------------------------------

    async def _run_dongle_scan(
        self, dongle: DongleState, timeout_sec: int,
    ) -> None:
        """Scan a single dongle and funnel results through dedup."""
        try:
            await self._scanner.start_scan(
                dongle, self._on_raw_discovery, timeout_sec,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Scan error on dongle %s", dongle.mac)

    async def _on_raw_discovery(self, pen_info: PenInfo) -> None:
        """Dedup callback: keep pen with strongest RSSI."""
        if not pen_info.has_pen_service():
            return  # Not an ExamPen — ignore.

        existing = self._discovered.get(pen_info.mac)
        if existing is not None:
            # Keep the record from the dongle with better signal.
            if pen_info.rssi > existing.rssi:
                self._discovered[pen_info.mac] = pen_info
            return

        self._discovered[pen_info.mac] = pen_info
        logger.info(
            "Pen discovered: %s (RSSI=%d, dongle=%s)",
            pen_info.mac, pen_info.rssi, pen_info.dongle_mac,
        )

        if self._on_pen_discovered is not None:
            await self._on_pen_discovered(pen_info)
