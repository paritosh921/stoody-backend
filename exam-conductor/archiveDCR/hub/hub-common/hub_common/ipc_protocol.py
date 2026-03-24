"""IPC envelope, Unix-domain-socket client/server, and JSON-lines transport.

Transport spec:
- Unix domain sockets, JSON-lines encoding (one JSON object per \\n-terminated line).
- UTF-8 encoding throughout.
- Timestamps are ISO 8601 UTC with ``Z`` suffix.
- Request/reply uses ``correlation_id``; fire-and-forget events do not require ACK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SEC = 10.0


@dataclass(slots=True)
class IpcEnvelope:
    """Canonical IPC message envelope (matches ``hub/ipc-protocol.md`` Section 2)."""

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

    # -- serialization ------------------------------------------------------

    def to_json_line(self) -> bytes:
        """Serialize to a newline-terminated UTF-8 JSON bytes line."""
        return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8") + b"\n"

    @classmethod
    def from_json_line(cls, line: bytes | str) -> "IpcEnvelope":
        """Deserialize from a single JSON-lines entry."""
        raw = line.decode("utf-8") if isinstance(line, bytes) else line
        data: dict[str, Any] = json.loads(raw.strip())
        return cls(
            msg_id=data["msg_id"],
            msg_type=data["msg_type"],
            source=data["source"],
            target=data["target"],
            sent_at=data["sent_at"],
            correlation_id=data.get("correlation_id"),
            expects_reply=data.get("expects_reply", False),
            payload=data.get("payload", {}),
        )

    # -- convenience factories -----------------------------------------------

    def make_reply(
        self,
        reply_type: str,
        payload: dict[str, Any],
        *,
        source: str | None = None,
    ) -> "IpcEnvelope":
        """Create a reply envelope correlated to this request."""
        return IpcEnvelope(
            msg_type=reply_type,
            source=source or self.target,
            target=self.source,
            correlation_id=self.msg_id,
            expects_reply=False,
            payload=payload,
        )

    def make_error(
        self,
        code: str,
        message: str,
        *,
        source: str | None = None,
    ) -> "IpcEnvelope":
        """Create an error reply envelope."""
        error_type = self.msg_type.rsplit(".", 1)[0] + ".error"
        return self.make_reply(
            error_type,
            {"code": code, "message": message},
            source=source,
        )


# Type alias for handler callbacks.
HandlerFn = Callable[[IpcEnvelope], Awaitable[IpcEnvelope | None]]


# ---------------------------------------------------------------------------
# IpcClient — connects to a module's Unix socket, sends requests
# ---------------------------------------------------------------------------

class IpcClient:
    """Async client that connects to a single Unix domain socket endpoint."""

    def __init__(self, socket_path: str, *, source_id: str) -> None:
        self._socket_path = socket_path
        self._source_id = source_id
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str, asyncio.Future[IpcEnvelope]] = {}
        self._recv_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        if hasattr(asyncio, "open_unix_connection"):
            self._reader, self._writer = await asyncio.open_unix_connection(
                self._socket_path,
            )
        else:
            # Windows fallback: socket_path is "host:port"
            host, port = self._socket_path.rsplit(":", 1)
            self._reader, self._writer = await asyncio.open_connection(
                host, int(port),
            )
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            self._recv_task = None
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None

    # -- sending ------------------------------------------------------------

    async def send(self, envelope: IpcEnvelope) -> None:
        """Fire-and-forget send."""
        if self._writer is None:
            raise RuntimeError("IpcClient is not connected")
        self._writer.write(envelope.to_json_line())
        await self._writer.drain()

    async def request(
        self,
        envelope: IpcEnvelope,
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,
    ) -> IpcEnvelope:
        """Send *envelope* and wait for a correlated reply (or timeout)."""
        envelope.expects_reply = True
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[IpcEnvelope] = loop.create_future()
        self._pending[envelope.msg_id] = fut
        try:
            await self.send(envelope)
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"IPC request {envelope.msg_type} timed out after {timeout}s"
            ) from None
        finally:
            self._pending.pop(envelope.msg_id, None)

    # -- receive loop -------------------------------------------------------

    async def _recv_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    env = IpcEnvelope.from_json_line(line)
                except (json.JSONDecodeError, KeyError):
                    logger.warning("IpcClient: malformed line ignored")
                    continue
                cid = env.correlation_id
                if cid and cid in self._pending:
                    self._pending[cid].set_result(env)
                else:
                    logger.debug(
                        "IpcClient: unsolicited message %s (no pending future)",
                        env.msg_type,
                    )
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# IpcServer — listens on a Unix socket, dispatches by msg_type
# ---------------------------------------------------------------------------

class IpcServer:
    """Async server that listens on a Unix domain socket and dispatches
    incoming messages to registered handlers by ``msg_type``."""

    def __init__(self, socket_path: str, *, module_id: str) -> None:
        self._socket_path = socket_path
        self._module_id = module_id
        self._handlers: dict[str, HandlerFn] = {}
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.StreamWriter] = set()

    def register(self, msg_type: str, handler: HandlerFn) -> None:
        """Register *handler* for a given ``msg_type``."""
        self._handlers[msg_type] = handler

    async def start(self) -> None:
        if hasattr(asyncio, "start_unix_server"):
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                path=self._socket_path,
            )
        else:
            # Windows fallback: socket_path is "host:port"
            host, port = self._socket_path.rsplit(":", 1)
            self._server = await asyncio.start_server(
                self._handle_connection,
                host, int(port),
            )
        logger.info("IpcServer [%s] listening on %s", self._module_id, self._socket_path)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        await self._server.serve_forever()

    # -- broadcast ----------------------------------------------------------

    async def broadcast(self, envelope: IpcEnvelope) -> None:
        """Send *envelope* to every connected client."""
        data = envelope.to_json_line()
        dead: list[asyncio.StreamWriter] = []
        for writer in list(self._clients):
            try:
                writer.write(data)
                await writer.drain()
            except (ConnectionError, OSError):
                dead.append(writer)
        for writer in dead:
            self._clients.discard(writer)
            try:
                writer.close()
            except OSError:
                pass

    # -- connection handler -------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername") or "unknown"
        logger.debug("IpcServer: new connection from %s", peer)
        self._clients.add(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    env = IpcEnvelope.from_json_line(line)
                except (json.JSONDecodeError, KeyError):
                    logger.warning("IpcServer: malformed line from %s", peer)
                    continue
                await self._dispatch(env, writer)
        except asyncio.CancelledError:
            pass
        finally:
            self._clients.discard(writer)
            writer.close()
            await writer.wait_closed()

    async def _dispatch(
        self,
        envelope: IpcEnvelope,
        writer: asyncio.StreamWriter,
    ) -> None:
        handler = self._handlers.get(envelope.msg_type)
        if handler is None:
            logger.warning(
                "IpcServer [%s]: no handler for %s",
                self._module_id,
                envelope.msg_type,
            )
            if envelope.expects_reply:
                err = envelope.make_error(
                    "unknown_message_type",
                    f"No handler registered for {envelope.msg_type}",
                    source=self._module_id,
                )
                writer.write(err.to_json_line())
                await writer.drain()
            return

        try:
            reply = await handler(envelope)
        except Exception:
            logger.exception(
                "IpcServer [%s]: handler error for %s",
                self._module_id,
                envelope.msg_type,
            )
            if envelope.expects_reply:
                err = envelope.make_error(
                    "handler_error",
                    "Internal handler error",
                    source=self._module_id,
                )
                writer.write(err.to_json_line())
                await writer.drain()
            return

        if reply is not None:
            writer.write(reply.to_json_line())
            await writer.drain()
