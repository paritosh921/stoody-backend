"""Async JSON-lines IPC server with client broadcast support.

Follows the same wire protocol as ``hub_common.ipc_protocol.IpcServer``
(one JSON object per newline-terminated line, UTF-8), and the same
pattern established by ``hub-timer/src/ipc_server.py``.

Transport abstraction: Unix domain socket on Linux, TCP loopback on
Windows (so the test-suite runs everywhere).

This module is intentionally self-contained so that ``hub-ble-mgr`` can
be built before ``hub-common`` is packaged as a wheel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Envelope  (mirrors hub_common.ipc_protocol.IpcEnvelope)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Envelope:
    """Minimal IPC envelope compatible with hub-common."""

    msg_type: str
    source: str
    target: str
    payload: dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    expects_reply: bool = False
    sent_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )

    def to_line(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode() + b"\n"

    @classmethod
    def from_line(cls, raw: bytes | str) -> "Envelope":
        text = raw.decode() if isinstance(raw, bytes) else raw
        d: dict[str, Any] = json.loads(text.strip())
        return cls(
            msg_id=d["msg_id"],
            msg_type=d["msg_type"],
            source=d["source"],
            target=d["target"],
            sent_at=d["sent_at"],
            correlation_id=d.get("correlation_id"),
            expects_reply=d.get("expects_reply", False),
            payload=d.get("payload", {}),
        )

    def make_error(self, code: str, message: str, *, source: str) -> "Envelope":
        return Envelope(
            msg_type=self.msg_type.rsplit(".", 1)[0] + ".error",
            source=source,
            target=self.source,
            correlation_id=self.msg_id,
            payload={"code": code, "message": message},
        )


# Type alias for registered handler callbacks.
HandlerFn = Callable[[Envelope], Awaitable[Envelope | None]]

# Whether the platform supports Unix domain sockets.
_HAS_UNIX = hasattr(asyncio, "start_unix_server")


# ---------------------------------------------------------------------------
# BleIpcServer
# ---------------------------------------------------------------------------

class BleIpcServer:
    """Async server that accepts IPC connections and supports broadcast.

    On Linux it binds a Unix domain socket; on Windows (for tests) it
    falls back to a TCP loopback socket.  The ``address`` property
    returns the connectable address after ``start()`` completes.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        module_id: str,
        use_tcp: bool | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._module_id = module_id
        self._use_tcp = use_tcp if use_tcp is not None else (not _HAS_UNIX)
        self._handlers: dict[str, HandlerFn] = {}
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._tcp_port: int | None = None

    # -- public API ---------------------------------------------------------

    @property
    def address(self) -> str:
        """Connectable address (socket path or ``localhost:<port>``)."""
        if self._use_tcp and self._tcp_port is not None:
            return f"localhost:{self._tcp_port}"
        return self._socket_path

    @property
    def tcp_port(self) -> int | None:
        return self._tcp_port

    def register(self, msg_type: str, handler: HandlerFn) -> None:
        self._handlers[msg_type] = handler

    async def start(self) -> None:
        if self._use_tcp:
            self._server = await asyncio.start_server(
                self._on_connection, "127.0.0.1", 0,
            )
            sock = self._server.sockets[0]
            self._tcp_port = sock.getsockname()[1]
            log.info(
                "BleIpcServer [%s] listening on TCP 127.0.0.1:%d",
                self._module_id, self._tcp_port,
            )
        else:
            parent = os.path.dirname(self._socket_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
            self._server = await asyncio.start_unix_server(
                self._on_connection, path=self._socket_path,
            )
            log.info(
                "BleIpcServer [%s] listening on %s",
                self._module_id, self._socket_path,
            )

    async def stop(self) -> None:
        for w in list(self._clients):
            w.close()
        self._clients.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if not self._use_tcp and os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass

    async def broadcast(self, envelope: Envelope | Any) -> None:
        """Send *envelope* to every connected client.

        Accepts both our local Envelope and hub-common's IpcEnvelope
        (or any object with a ``to_line`` / ``to_dict`` method).
        """
        if hasattr(envelope, "to_line"):
            data = envelope.to_line()
        else:
            data = json.dumps(
                envelope if isinstance(envelope, dict) else vars(envelope),
                separators=(",", ":"),
            ).encode() + b"\n"

        dead: list[asyncio.StreamWriter] = []
        for writer in list(self._clients):
            try:
                writer.write(data)
                await writer.drain()
            except (ConnectionError, OSError):
                dead.append(writer)
        for w in dead:
            self._clients.discard(w)

    # -- connection handler -------------------------------------------------

    async def _on_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._clients.add(writer)
        log.debug("BleIpcServer: client connected (%d total)", len(self._clients))
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    env = Envelope.from_line(line)
                except (json.JSONDecodeError, KeyError):
                    log.warning("BleIpcServer: malformed line ignored")
                    continue
                await self._dispatch(env, writer)
        except asyncio.CancelledError:
            pass
        except (ConnectionError, OSError):
            pass
        finally:
            self._clients.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    async def _dispatch(
        self, envelope: Envelope, writer: asyncio.StreamWriter,
    ) -> None:
        handler = self._handlers.get(envelope.msg_type)
        if handler is None:
            log.warning("BleIpcServer: no handler for %s", envelope.msg_type)
            if envelope.expects_reply:
                err = envelope.make_error(
                    "unknown_message_type",
                    f"No handler for {envelope.msg_type}",
                    source=self._module_id,
                )
                writer.write(err.to_line())
                await writer.drain()
            return
        try:
            reply = await handler(envelope)
        except Exception:
            log.exception("BleIpcServer: handler error for %s", envelope.msg_type)
            if envelope.expects_reply:
                err = envelope.make_error(
                    "handler_error", "Internal error", source=self._module_id,
                )
                writer.write(err.to_line())
                await writer.drain()
            return
        if reply is not None:
            writer.write(reply.to_line())
            await writer.drain()
