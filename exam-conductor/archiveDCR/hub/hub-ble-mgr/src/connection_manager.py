"""Pen connection lifecycle management.

Tracks which pen is connected through which dongle, handles graceful
disconnect, and re-queues pens when a dongle fails (H3).

Domain logic only -- the ``BleConnector`` protocol is injected for
testability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Protocol

from src.config import CONNECTION_MAX_RETRIES, CONNECTION_TIMEOUT_SEC
from src.dongle_manager import DongleManager, DongleStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection record
# ---------------------------------------------------------------------------

@dataclass
class ConnectionRecord:
    """Tracks one active pen connection."""

    pen_mac: str
    dongle_mac: str
    connection_id: str = ""
    retries: int = 0


# ---------------------------------------------------------------------------
# BLE connector protocol (injected)
# ---------------------------------------------------------------------------

class BleConnector(Protocol):
    """Abstract interface for BLE connect/disconnect.

    Production implementation wraps bleak.BleakClient.
    Tests inject a mock.
    """

    async def connect(
        self, pen_mac: str, dongle_mac: str, timeout: float,
    ) -> str:
        """Connect to *pen_mac* via *dongle_mac*. Returns connection_id."""
        ...  # pragma: no cover

    async def disconnect(self, pen_mac: str) -> None:
        """Disconnect *pen_mac*."""
        ...  # pragma: no cover


# Callback types for IPC event emission.
ConnectCallback = Callable[[ConnectionRecord], Coroutine[Any, Any, None]]
DisconnectCallback = Callable[[str, str], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages pen connection lifecycle across all dongles.

    Responsibilities:
      - Connect a pen to a specific dongle (or auto-select).
      - Disconnect a pen gracefully.
      - Re-queue pens from a failed dongle to remaining dongles (H3).
      - Track connection table: pen_mac -> ConnectionRecord.
    """

    def __init__(
        self,
        dongle_mgr: DongleManager,
        connector: BleConnector,
        on_connected: ConnectCallback | None = None,
        on_disconnected: DisconnectCallback | None = None,
    ) -> None:
        self._dongle_mgr = dongle_mgr
        self._connector = connector
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        # pen_mac -> ConnectionRecord
        self._connections: dict[str, ConnectionRecord] = {}

    # -- Accessors ----------------------------------------------------------

    @property
    def connections(self) -> dict[str, ConnectionRecord]:
        return dict(self._connections)

    def get_connection(self, pen_mac: str) -> ConnectionRecord | None:
        return self._connections.get(pen_mac)

    def pen_dongle(self, pen_mac: str) -> str | None:
        """Return the dongle MAC a pen is connected through, or None."""
        rec = self._connections.get(pen_mac)
        return rec.dongle_mac if rec else None

    # -- Connect ------------------------------------------------------------

    async def connect_pen(
        self,
        pen_mac: str,
        dongle_mac: str | None = None,
    ) -> ConnectionRecord | None:
        """Connect to a pen, optionally on a specific dongle.

        If *dongle_mac* is None, auto-selects the first dongle with
        capacity.  Returns the ConnectionRecord on success, None on
        failure (all retries exhausted or no capacity).
        """
        if pen_mac in self._connections:
            logger.info("Pen %s already connected", pen_mac)
            return self._connections[pen_mac]

        # Resolve dongle.
        if dongle_mac is None:
            dongle = self._dongle_mgr.find_available_dongle()
            if dongle is None:
                logger.error("No dongle with capacity for pen %s", pen_mac)
                return None
            dongle_mac = dongle.mac
        else:
            dongle = self._dongle_mgr.get_dongle(dongle_mac)
            if dongle is None or not dongle.has_capacity:
                # Try to redirect to another dongle.
                dongle = self._dongle_mgr.find_available_dongle(
                    exclude_mac=dongle_mac,
                )
                if dongle is None:
                    logger.error(
                        "Dongle %s full/unavailable and no alternative for %s",
                        dongle_mac, pen_mac,
                    )
                    return None
                dongle_mac = dongle.mac
                logger.info(
                    "Redirecting pen %s to dongle %s (overflow)",
                    pen_mac, dongle_mac,
                )

        # Attempt connection with retries.
        record = ConnectionRecord(pen_mac=pen_mac, dongle_mac=dongle_mac)

        for attempt in range(1, CONNECTION_MAX_RETRIES + 1):
            try:
                conn_id = await self._connector.connect(
                    pen_mac, dongle_mac, CONNECTION_TIMEOUT_SEC,
                )
                record.connection_id = conn_id
                record.retries = attempt - 1
                break
            except Exception as exc:
                logger.warning(
                    "Connect attempt %d/%d for %s failed: %s",
                    attempt, CONNECTION_MAX_RETRIES, pen_mac, exc,
                )
                record.retries = attempt
                if attempt == CONNECTION_MAX_RETRIES:
                    logger.error(
                        "All %d connect attempts exhausted for pen %s",
                        CONNECTION_MAX_RETRIES, pen_mac,
                    )
                    return None

        # Track connection.
        self._connections[pen_mac] = record
        self._dongle_mgr.assign_pen(dongle_mac, pen_mac)

        logger.info(
            "Pen %s connected via dongle %s (id=%s, retries=%d)",
            pen_mac, dongle_mac, record.connection_id, record.retries,
        )

        if self._on_connected is not None:
            await self._on_connected(record)

        return record

    # -- Disconnect ---------------------------------------------------------

    async def disconnect_pen(self, pen_mac: str) -> bool:
        """Gracefully disconnect a pen. Returns True if it was connected."""
        record = self._connections.pop(pen_mac, None)
        if record is None:
            return False

        try:
            await self._connector.disconnect(pen_mac)
        except Exception:
            logger.warning("Error during disconnect of %s (ignoring)", pen_mac)

        self._dongle_mgr.release_pen(record.dongle_mac, pen_mac)
        logger.info("Pen %s disconnected from dongle %s", pen_mac, record.dongle_mac)

        if self._on_disconnected is not None:
            await self._on_disconnected(pen_mac, record.dongle_mac)

        return True

    # -- Dongle failure re-queue (H3) ---------------------------------------

    async def handle_dongle_failure(
        self, dongle_mac: str,
    ) -> list[str]:
        """Re-queue pens from a failed dongle to remaining dongles.

        Returns list of pen MACs that could NOT be reassigned (no
        capacity on remaining dongles).
        """
        self._dongle_mgr.transition_health(
            dongle_mac, DongleStatus.FAILED, "dongle failure detected",
        )
        orphans = self._dongle_mgr.collect_orphaned_pens(dongle_mac)

        if not orphans:
            return []

        logger.warning(
            "Dongle %s failed — re-queuing %d pens", dongle_mac, len(orphans),
        )

        unplaced: list[str] = []
        for pen_mac in orphans:
            # Remove old connection record.
            self._connections.pop(pen_mac, None)

            if self._on_disconnected is not None:
                await self._on_disconnected(pen_mac, dongle_mac)

            # Try to reconnect on another dongle.
            new_dongle = self._dongle_mgr.find_available_dongle(
                exclude_mac=dongle_mac,
            )
            if new_dongle is None:
                logger.warning("No capacity to reassign pen %s", pen_mac)
                unplaced.append(pen_mac)
                continue

            result = await self.connect_pen(pen_mac, new_dongle.mac)
            if result is None:
                unplaced.append(pen_mac)

        if unplaced:
            logger.warning(
                "%d pen(s) could not be reassigned: %s",
                len(unplaced), unplaced,
            )

        return unplaced

    # -- Bulk disconnect ----------------------------------------------------

    async def disconnect_all(self) -> None:
        """Disconnect all pens (e.g., session teardown)."""
        pen_macs = list(self._connections.keys())
        for pen_mac in pen_macs:
            await self.disconnect_pen(pen_mac)
