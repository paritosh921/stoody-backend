"""Bleak-based BLE client factory — real hardware adapter.

Wraps the ``bleak`` library to satisfy the ``BleClient`` and
``BleClientFactory`` protocols defined in ``gatt_reader.py``.

This file is imported only from ``main.py`` and is NOT imported
during tests — tests inject mock factories instead.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from bleak import BleakClient as _BleakClient
except ImportError:
    _BleakClient = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class BleakClientAdapter:
    """Adapter that wraps a ``bleak.BleakClient`` to match ``BleClient``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def read_gatt_char(self, uuid: str) -> bytes:
        return bytes(await self._client.read_gatt_char(uuid))

    async def write_gatt_char(
        self, uuid: str, data: bytes, response: bool = True
    ) -> None:
        await self._client.write_gatt_char(uuid, data, response=response)

    async def start_notify(self, uuid: str, callback: Any) -> None:
        await self._client.start_notify(uuid, callback)

    async def stop_notify(self, uuid: str) -> None:
        await self._client.stop_notify(uuid)

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected


class BleakClientFactory:
    """Creates BLE connections using bleak (real hardware)."""

    async def connect(
        self, pen_mac: str, timeout: float
    ) -> BleakClientAdapter:
        if _BleakClient is None:
            raise RuntimeError("bleak is not installed")
        client = _BleakClient(pen_mac, timeout=timeout)
        await client.connect()
        logger.info("BLE connected to %s", pen_mac)
        return BleakClientAdapter(client)

    async def disconnect(self, client: BleakClientAdapter) -> None:
        await client._client.disconnect()
        logger.info("BLE disconnected")
