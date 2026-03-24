"""Async JSON-lines IPC client for outbound requests to hub-supervisor.

Connects to the supervisor's IPC socket and supports request/reply with
correlation IDs.  Graceful degradation: if the supervisor is unreachable,
``request()`` returns ``None`` instead of raising.

Self-contained so ``hub-invig-ble`` can be built before ``hub-common``.
"""

from __future__ import annotations

import asyncio
import json
import logging

from src.ipc_handlers import Envelope

log = logging.getLogger(__name__)

# Whether the platform supports Unix domain sockets.
_HAS_UNIX = hasattr(asyncio, "open_unix_connection")


class IpcClient:
    """Persistent IPC client that connects to a peer module's socket.

    On Linux it connects via Unix domain socket; on Windows it falls
    back to TCP loopback (the peer address must be ``localhost:<port>``).
    """

    def __init__(
        self,
        address: str,
        *,
        use_tcp: bool | None = None,
    ) -> None:
        self._address = address
        self._use_tcp = use_tcp if use_tcp is not None else (not _HAS_UNIX)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Open connection.  Returns ``True`` on success, ``False`` on failure."""
        try:
            if self._use_tcp:
                host, port_str = self._address.rsplit(":", 1)
                self._reader, self._writer = await asyncio.open_connection(
                    host, int(port_str),
                )
            else:
                self._reader, self._writer = (
                    await asyncio.open_unix_connection(self._address)
                )
            self._connected = True
            log.info("IpcClient connected to %s", self._address)
            return True
        except (OSError, ConnectionRefusedError):
            log.warning("IpcClient: could not connect to %s", self._address)
            self._connected = False
            return False

    async def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (OSError, ConnectionError):
                pass
        self._connected = False

    async def request(
        self,
        envelope: Envelope,
        timeout_sec: float = 2.0,
    ) -> Envelope | None:
        """Send *envelope* and wait for the correlated reply.

        Returns ``None`` if the supervisor is unreachable, the connection
        is lost, or the reply does not arrive within *timeout_sec*.
        """
        if not self._connected:
            if not await self.connect():
                return None

        assert self._writer is not None
        assert self._reader is not None

        try:
            self._writer.write(envelope.to_line())
            await self._writer.drain()
            raw = await asyncio.wait_for(
                self._reader.readline(), timeout=timeout_sec,
            )
            if not raw:
                self._connected = False
                return None
            return Envelope.from_line(raw)
        except (
            asyncio.TimeoutError,
            OSError,
            ConnectionError,
            json.JSONDecodeError,
            KeyError,
        ):
            self._connected = False
            return None

    async def send(self, envelope: Envelope) -> None:
        """Fire-and-forget send (no reply expected)."""
        if not self._connected:
            if not await self.connect():
                return
        assert self._writer is not None
        try:
            self._writer.write(envelope.to_line())
            await self._writer.drain()
        except (OSError, ConnectionError):
            self._connected = False
