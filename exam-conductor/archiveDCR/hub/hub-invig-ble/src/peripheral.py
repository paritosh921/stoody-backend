"""BLE peripheral (GATT server) for the invigilator channel.

Advertises the invigilator GATT service (``ble-gatt-spec.md`` Section 2)
with the following characteristics:

- **Auth** (write + indicate): invigilator writes 12-byte code, receives
  1-byte result (0x01 accept, 0x00 reject).
- **Command** (write): invigilator sends command header + JSON payload.
  Requires prior auth.
- **Status feed** (notify): hub pushes 1 Hz JSON status to the app.
- **MAC list** (read + notify): JSON array of pen discovery rows.

This module abstracts the BLE library behind a ``BlePeripheralBackend``
protocol so that all domain logic is testable without real BLE hardware.
On production hardware the backend would be implemented via ``bless`` or
``bluez-peripheral``; in tests a mock backend is injected.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from src.auth_handler import AuthHandler, AuthResult
from src.command_handler import (
    CMD_MANUAL_REGISTER,
    CommandHandler,
    CommandResult,
    create_provisional_binding,
)
from src.config import (
    BLE_PERIPHERAL_NAME,
    CHAR_AUTH_UUID,
    CHAR_COMMAND_UUID,
    CHAR_MAC_LIST_UUID,
    CHAR_STATUS_FEED_UUID,
    GATT_SERVICE_UUID,
    MODULE_ID,
)
from src.status_feed import StatusFeedCollector

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol (implemented by real BLE library or test mock)
# ---------------------------------------------------------------------------

class BlePeripheralBackend(Protocol):
    """Abstract BLE peripheral operations."""

    async def start_advertising(self, name: str, service_uuid: str) -> None: ...

    async def stop_advertising(self) -> None: ...

    async def send_indication(
        self, char_uuid: str, data: bytes, address: str,
    ) -> None: ...

    async def send_notification(
        self, char_uuid: str, data: bytes,
    ) -> None: ...

    async def update_characteristic(
        self, char_uuid: str, data: bytes,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Callback types the peripheral expects from main.py
# ---------------------------------------------------------------------------

class PeripheralCallbacks(Protocol):
    """Callbacks the peripheral fires on domain events."""

    async def on_auth_result(self, result: AuthResult) -> None: ...

    async def on_command(self, result: CommandResult) -> None: ...


# ---------------------------------------------------------------------------
# InvigilatorPeripheral
# ---------------------------------------------------------------------------

class InvigilatorPeripheral:
    """Manages the invigilator GATT service lifecycle.

    Wires together auth, command handling, and status feed with the BLE
    backend and fires callbacks to the main loop for IPC dispatch.
    """

    def __init__(
        self,
        backend: BlePeripheralBackend,
        auth_handler: AuthHandler,
        command_handler: CommandHandler,
        status_feed: StatusFeedCollector,
        callbacks: PeripheralCallbacks,
    ) -> None:
        self._backend = backend
        self._auth = auth_handler
        self._cmd = command_handler
        self._status_feed = status_feed
        self._callbacks = callbacks
        self._pen_macs: list[dict[str, Any]] = []

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Begin BLE advertising."""
        await self._backend.start_advertising(
            BLE_PERIPHERAL_NAME, GATT_SERVICE_UUID,
        )
        log.info(
            "Invigilator peripheral advertising as '%s'",
            BLE_PERIPHERAL_NAME,
        )

    async def stop(self) -> None:
        """Stop BLE advertising and clean up."""
        await self._backend.stop_advertising()
        log.info("Invigilator peripheral stopped")

    # -- Characteristic write handlers (called by BLE backend callbacks) ----

    async def on_auth_write(self, data: bytes, address: str) -> None:
        """Handle a write to the Auth characteristic.

        *data* is the 12-byte ASCII auth code.  *address* is the remote
        BLE device address.
        """
        code = data[:12].decode("ascii", errors="replace").rstrip("\x00")
        result = self._auth.authenticate(code, address)

        # Send indication: 0x01 accept, 0x00 reject.
        indication = b"\x01" if result.success else b"\x00"
        await self._backend.send_indication(
            CHAR_AUTH_UUID, indication, address,
        )

        await self._callbacks.on_auth_result(result)
        log.info(
            "Auth attempt from %s: %s (%s)",
            address, "accepted" if result.success else "rejected", result.reason,
        )

    async def on_command_write(self, data: bytes, address: str) -> None:
        """Handle a write to the Command characteristic.

        *data* follows the wire format in ``ble-gatt-spec.md`` Section 4.
        """
        result = self._cmd.handle(data, address)
        await self._callbacks.on_command(result)

        if result.accepted:
            log.info(
                "Command accepted: %s (request_id=%s) from %s",
                result.cmd_name, result.request_id, address,
            )
        else:
            log.warning(
                "Command rejected: cmd_id=%d error=%s from %s",
                result.cmd_id, result.error_code, address,
            )

    # -- Status feed --------------------------------------------------------

    async def push_status(self) -> None:
        """Push current status JSON to the Status Feed characteristic."""
        data = self._status_feed.to_json_bytes()
        await self._backend.send_notification(CHAR_STATUS_FEED_UUID, data)

    # -- MAC list -----------------------------------------------------------

    def update_pen_macs(self, macs: list[dict[str, Any]]) -> None:
        """Update the cached pen MAC list (from BLE scan results)."""
        self._pen_macs = macs

    async def push_mac_list(self) -> None:
        """Push current MAC list to the MAC List characteristic."""
        data = json.dumps(self._pen_macs, separators=(",", ":")).encode("utf-8")
        await self._backend.update_characteristic(CHAR_MAC_LIST_UUID, data)
        await self._backend.send_notification(CHAR_MAC_LIST_UUID, data)
