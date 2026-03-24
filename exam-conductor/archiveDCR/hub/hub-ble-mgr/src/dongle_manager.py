"""Core dongle state management.

Tracks per-dongle state (healthy/degraded/failed), enforces the 8-pen cap,
and provides overflow redirection when a dongle is full or failed.

Domain logic only -- no I/O imports. BLE adapter enumeration and D-Bus
signal handling are injected via the ``BleAdapter`` protocol so that all
state-machine logic is testable without hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from src.config import MAX_DONGLES, MAX_PENS_PER_DONGLE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health FSM
# ---------------------------------------------------------------------------

class DongleStatus(str, Enum):
    """Dongle health states (matches dongle_registry.status CHECK constraint)."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


# Valid transitions: unknown -> healthy, healthy <-> degraded, * -> failed
_VALID_TRANSITIONS: set[tuple[DongleStatus, DongleStatus]] = {
    (DongleStatus.UNKNOWN, DongleStatus.HEALTHY),
    (DongleStatus.UNKNOWN, DongleStatus.FAILED),
    (DongleStatus.HEALTHY, DongleStatus.DEGRADED),
    (DongleStatus.HEALTHY, DongleStatus.FAILED),
    (DongleStatus.DEGRADED, DongleStatus.HEALTHY),
    (DongleStatus.DEGRADED, DongleStatus.FAILED),
    # Recovery after re-plug:
    (DongleStatus.FAILED, DongleStatus.HEALTHY),
}


# ---------------------------------------------------------------------------
# DongleState dataclass
# ---------------------------------------------------------------------------

@dataclass
class DongleState:
    """Runtime state for a single BLE dongle."""

    mac: str
    hci_path: str
    usb_port: str = ""
    status: DongleStatus = DongleStatus.UNKNOWN
    connected_pens: int = 0
    firmware: str = ""
    # Set of pen MACs currently connected through this dongle.
    pen_macs: set[str] = field(default_factory=set)

    @property
    def has_capacity(self) -> bool:
        """True if this dongle can accept another pen connection."""
        return (
            self.connected_pens < MAX_PENS_PER_DONGLE
            and self.status in (DongleStatus.HEALTHY, DongleStatus.DEGRADED)
        )

    @property
    def is_available(self) -> bool:
        """True if dongle is not failed."""
        return self.status != DongleStatus.FAILED

    def to_dict(self) -> dict:
        """Serialize for IPC payloads."""
        return {
            "mac": self.mac,
            "hci_path": self.hci_path,
            "usb_port": self.usb_port,
            "status": self.status.value,
            "connected_pens": self.connected_pens,
            "firmware": self.firmware,
        }


# ---------------------------------------------------------------------------
# BLE adapter protocol (injected dependency)
# ---------------------------------------------------------------------------

class BleAdapter(Protocol):
    """Abstract interface for BLE adapter enumeration.

    Production implementation uses hciconfig / BlueZ D-Bus.
    Tests inject a mock.
    """

    async def enumerate(self) -> list[DongleState]:
        """Return a list of currently visible HCI adapters."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# DongleManager
# ---------------------------------------------------------------------------

class DongleManager:
    """Manages the set of BLE dongles and their per-pen allocation.

    Responsibilities:
      - Track which dongles are present and their health status.
      - Enforce MAX_PENS_PER_DONGLE per dongle (A1.1).
      - Redirect overflow to the next available dongle.
      - Handle dongle hot-plug / unplug.
      - Provide a list of pens to re-queue when a dongle fails (H3).
    """

    def __init__(self) -> None:
        # mac -> DongleState
        self._dongles: dict[str, DongleState] = {}

    # -- Accessors ----------------------------------------------------------

    @property
    def dongles(self) -> dict[str, DongleState]:
        return dict(self._dongles)

    def get_dongle(self, mac: str) -> DongleState | None:
        return self._dongles.get(mac)

    @property
    def total_capacity(self) -> int:
        """Total available pen slots across all healthy/degraded dongles."""
        return sum(
            MAX_PENS_PER_DONGLE - d.connected_pens
            for d in self._dongles.values()
            if d.is_available
        )

    @property
    def total_connected(self) -> int:
        return sum(d.connected_pens for d in self._dongles.values())

    # -- Enumeration --------------------------------------------------------

    async def refresh(self, adapter: BleAdapter) -> list[DongleState]:
        """Re-enumerate dongles from the adapter layer.

        New dongles are added. Existing ones retain their pen connections.
        Missing dongles are marked FAILED.
        """
        found = await adapter.enumerate()
        found_macs = {d.mac for d in found}

        # Mark disappeared dongles as failed.
        for mac, state in self._dongles.items():
            if mac not in found_macs and state.status != DongleStatus.FAILED:
                state.status = DongleStatus.FAILED
                logger.warning("Dongle %s disappeared — marked FAILED", mac)

        # Add/update found dongles.
        for d in found:
            if d.mac in self._dongles:
                existing = self._dongles[d.mac]
                existing.hci_path = d.hci_path
                existing.usb_port = d.usb_port
                existing.firmware = d.firmware
                if existing.status == DongleStatus.FAILED:
                    existing.status = DongleStatus.HEALTHY
                    logger.info("Dongle %s re-appeared — marked HEALTHY", d.mac)
            else:
                if len(self._dongles) >= MAX_DONGLES:
                    logger.warning(
                        "Dongle %s ignored — at MAX_DONGLES (%d)",
                        d.mac, MAX_DONGLES,
                    )
                    continue
                d.status = DongleStatus.HEALTHY
                self._dongles[d.mac] = d
                logger.info("Dongle %s added (hci=%s)", d.mac, d.hci_path)

        return list(self._dongles.values())

    # -- Health transitions -------------------------------------------------

    def transition_health(
        self, mac: str, new_status: DongleStatus, detail: str = "",
    ) -> DongleState | None:
        """Attempt a health-state transition for a dongle.

        Returns the updated DongleState, or None if the transition is
        invalid or the dongle is unknown.
        """
        state = self._dongles.get(mac)
        if state is None:
            return None

        pair = (state.status, new_status)
        if pair not in _VALID_TRANSITIONS and state.status != new_status:
            logger.warning(
                "Invalid dongle transition %s: %s -> %s",
                mac, state.status.value, new_status.value,
            )
            return None

        old = state.status
        state.status = new_status
        if old != new_status:
            logger.info(
                "Dongle %s: %s -> %s (%s)",
                mac, old.value, new_status.value, detail,
            )
        return state

    # -- Pen allocation -----------------------------------------------------

    def find_available_dongle(self, *, exclude_mac: str = "") -> DongleState | None:
        """Find the first dongle with capacity, optionally excluding one.

        Used for initial allocation and overflow redirection.
        """
        for d in self._dongles.values():
            if d.mac == exclude_mac:
                continue
            if d.has_capacity:
                return d
        return None

    def assign_pen(self, dongle_mac: str, pen_mac: str) -> bool:
        """Record that *pen_mac* is connected through *dongle_mac*.

        Returns False if dongle is full, failed, or unknown.
        """
        state = self._dongles.get(dongle_mac)
        if state is None or not state.has_capacity:
            return False
        if pen_mac in state.pen_macs:
            return True  # already tracked
        state.pen_macs.add(pen_mac)
        state.connected_pens = len(state.pen_macs)
        return True

    def release_pen(self, dongle_mac: str, pen_mac: str) -> bool:
        """Remove *pen_mac* from *dongle_mac* tracking."""
        state = self._dongles.get(dongle_mac)
        if state is None:
            return False
        state.pen_macs.discard(pen_mac)
        state.connected_pens = len(state.pen_macs)
        return True

    def collect_orphaned_pens(self, dongle_mac: str) -> list[str]:
        """Return pen MACs from a failed dongle and clear its tracking.

        Called during dongle failure to re-queue pens (H3).
        """
        state = self._dongles.get(dongle_mac)
        if state is None:
            return []
        orphans = list(state.pen_macs)
        state.pen_macs.clear()
        state.connected_pens = 0
        return orphans

    # -- Summary for health report ------------------------------------------

    def summary(self) -> dict:
        """Build a summary dict suitable for IPC health responses."""
        return {
            "dongles": [d.to_dict() for d in self._dongles.values()],
            "total_connected": self.total_connected,
            "total_capacity": self.total_capacity,
        }
