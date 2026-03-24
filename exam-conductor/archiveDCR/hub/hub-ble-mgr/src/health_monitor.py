"""Dongle health monitoring.

Periodically checks each dongle's status and detects failures.
Emits health-change events via callback when a dongle transitions state.

Domain logic only -- the ``HealthProbe`` protocol abstracts the actual
hciconfig / D-Bus calls so the health FSM is testable without hardware.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Protocol

from src.config import HEALTH_CHECK_INTERVAL_SEC, HEALTH_SLOW_RESPONSE_SEC
from src.dongle_manager import DongleManager, DongleState, DongleStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health probe protocol (injected)
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result from probing a single dongle's health."""

    dongle_mac: str
    responsive: bool
    response_time_sec: float = 0.0
    can_scan: bool = True
    detail: str = ""


class HealthProbe(Protocol):
    """Abstract interface for dongle health probes.

    Production implementation runs ``hciconfig hciX`` and checks the
    output.  Tests inject a mock.
    """

    async def probe(self, dongle: DongleState) -> ProbeResult:
        """Probe a single dongle and return its health status."""
        ...  # pragma: no cover


# Callback for health-change events.
HealthChangeCallback = Callable[
    [str, DongleStatus, str], Coroutine[Any, Any, None]
]
# Signature: (dongle_mac, new_status, detail) -> awaitable


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    """Periodic health checker for all managed BLE dongles.

    Runs a background loop that probes each dongle at
    ``HEALTH_CHECK_INTERVAL_SEC`` intervals (default 10 s).  On
    status change it invokes the ``on_health_change`` callback so
    that the IPC layer can emit ``ble.dongle.health.event``.
    """

    def __init__(
        self,
        dongle_mgr: DongleManager,
        probe: HealthProbe,
        on_health_change: HealthChangeCallback | None = None,
        interval_sec: float = HEALTH_CHECK_INTERVAL_SEC,
    ) -> None:
        self._dongle_mgr = dongle_mgr
        self._probe = probe
        self._on_health_change = on_health_change
        self._interval = interval_sec
        self._task: asyncio.Task | None = None

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the periodic health-check background task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._loop(), name="health-monitor",
        )
        logger.info("Health monitor started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        """Cancel the background task."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
            logger.info("Health monitor stopped")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- Single check round -------------------------------------------------

    async def check_all(self) -> list[ProbeResult]:
        """Run one health-check round across all dongles.

        Returns the list of probe results.  Called by the background
        loop and also available for on-demand diagnostics.
        """
        results: list[ProbeResult] = []

        for dongle in list(self._dongle_mgr.dongles.values()):
            result = await self._probe_dongle(dongle)
            results.append(result)

        return results

    # -- Dongle hot-unplug notification ------------------------------------

    async def notify_dongle_removed(self, dongle_mac: str) -> None:
        """Called when a D-Bus ``Adapter1.Removed`` signal fires.

        Immediately marks the dongle as FAILED without waiting for the
        next periodic check.
        """
        updated = self._dongle_mgr.transition_health(
            dongle_mac, DongleStatus.FAILED, "hot-unplug detected",
        )
        if updated is not None and self._on_health_change is not None:
            await self._on_health_change(
                dongle_mac, DongleStatus.FAILED, "hot-unplug detected",
            )

    # -- Internal -----------------------------------------------------------

    async def _loop(self) -> None:
        """Background loop: check all dongles at fixed intervals."""
        try:
            while True:
                await asyncio.sleep(self._interval)
                await self.check_all()
        except asyncio.CancelledError:
            pass

    async def _probe_dongle(self, dongle: DongleState) -> ProbeResult:
        """Probe a single dongle and update its health state."""
        try:
            result = await self._probe.probe(dongle)
        except Exception as exc:
            logger.warning("Probe failed for %s: %s", dongle.mac, exc)
            result = ProbeResult(
                dongle_mac=dongle.mac,
                responsive=False,
                detail=f"probe exception: {exc}",
            )

        new_status = self._classify(result)
        old_status = dongle.status

        if new_status != old_status:
            self._dongle_mgr.transition_health(
                dongle.mac, new_status, result.detail,
            )
            if self._on_health_change is not None:
                await self._on_health_change(
                    dongle.mac, new_status, result.detail,
                )

        return result

    @staticmethod
    def _classify(result: ProbeResult) -> DongleStatus:
        """Derive a DongleStatus from a probe result."""
        if not result.responsive:
            return DongleStatus.FAILED

        if (
            result.response_time_sec > HEALTH_SLOW_RESPONSE_SEC
            or not result.can_scan
        ):
            return DongleStatus.DEGRADED

        return DongleStatus.HEALTHY
