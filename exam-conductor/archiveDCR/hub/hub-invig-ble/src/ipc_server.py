"""Async JSON-lines IPC server for hub-invig-ble.

Same wire protocol as ``hub-timer``'s ``ipc_server.py``: one JSON object
per newline-terminated line, UTF-8.  Supports client tracking for broadcast
and transport abstraction (Unix domain socket on Linux, TCP loopback on
Windows for tests).

Self-contained so ``hub-invig-ble`` can be built before ``hub-common`` is
packaged as a wheel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Awaitable, Callable

from src.ipc_handlers import Envelope

log = logging.getLogger(__name__)

# Type alias for registered handler callbacks.
HandlerFn = Callable[[Envelope], Awaitable[Envelope | None]]

# Whether the platform supports Unix domain sockets.
_HAS_UNIX = hasattr(asyncio, "start_unix_server")


class InvigIpcServer:
    """Async IPC server with broadcast and handler dispatch.

    On Linux it binds a Unix domain socket; on Windows (for tests) it
    falls back to a TCP loopback socket.
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
                "InvigIpcServer [%s] listening on TCP 127.0.0.1:%d",
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
                "InvigIpcServer [%s] listening on %s",
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

    async def broadcast(self, envelope: Envelope) -> None:
        """Send *envelope* to every connected client."""
        data = envelope.to_line()
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
        log.debug("InvigIpcServer: client connected (%d total)", len(self._clients))
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    env = Envelope.from_line(line)
                except (json.JSONDecodeError, KeyError):
                    log.warning("InvigIpcServer: malformed line ignored")
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
            log.warning(
                "InvigIpcServer: no handler for %s", envelope.msg_type,
            )
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
            log.exception(
                "InvigIpcServer: handler error for %s", envelope.msg_type,
            )
            if envelope.expects_reply:
                err = envelope.make_error(
                    "handler_error", "Internal error",
                    source=self._module_id,
                )
                writer.write(err.to_line())
                await writer.drain()
            return
        if reply is not None:
            writer.write(reply.to_line())
            await writer.drain()
