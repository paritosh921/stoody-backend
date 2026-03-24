"""NATS JetStream connection factory and helpers.

Provides:
- Async NATS connection with automatic reconnect
- JetStream context creation
- JSON publish helper
- Subscribe helper with consumer-group (queue) support
- Environment-based configuration
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Awaitable

import nats
from nats.aio.client import Client as NatsConn
from nats.js.api import ConsumerConfig, DeliverPolicy
from nats.js.client import JetStreamContext

from exampen_common.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_NATS_URL: str = os.getenv("NATS_URL", "nats://localhost:4222")
_NATS_CREDS: str | None = os.getenv("NATS_CREDS")
_RECONNECT_DELAY: float = float(os.getenv("NATS_RECONNECT_DELAY", "2"))
_MAX_RECONNECT: int = int(os.getenv("NATS_MAX_RECONNECT", "-1"))  # infinite

# Type alias for message handler callbacks
MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class NatsClient:
    """Thin async wrapper around ``nats-py`` with JetStream support."""

    def __init__(self, conn: NatsConn, js: JetStreamContext) -> None:
        self._conn = conn
        self._js = js

    # -- properties --------------------------------------------------------

    @property
    def connection(self) -> NatsConn:
        return self._conn

    @property
    def jetstream(self) -> JetStreamContext:
        return self._js

    # -- publish -----------------------------------------------------------

    async def publish(
        self,
        subject: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Publish a JSON-serialized message to a JetStream subject."""
        data = json.dumps(payload, default=str).encode("utf-8")
        await self._js.publish(subject, data, headers=headers)
        _log.debug("published to %s (%d bytes)", subject, len(data))

    # -- subscribe ---------------------------------------------------------

    async def subscribe(
        self,
        subject: str,
        handler: MessageHandler,
        *,
        durable: str | None = None,
        queue: str | None = None,
        deliver_policy: DeliverPolicy = DeliverPolicy.ALL,
    ) -> nats.js.client.PushSubscription:
        """Subscribe to a JetStream subject with JSON deserialization.

        Parameters
        ----------
        subject:
            NATS subject pattern (e.g. ``"stroke.raw"``).
        handler:
            Async callback receiving a deserialized ``dict``.
        durable:
            Durable consumer name for replay guarantees.
        queue:
            Queue group name for load-balanced consumption.
        deliver_policy:
            JetStream deliver policy (default: ALL).
        """
        config = ConsumerConfig(
            deliver_policy=deliver_policy,
            durable_name=durable,
        )

        async def _on_msg(msg: nats.aio.msg.Msg) -> None:
            try:
                payload = json.loads(msg.data.decode("utf-8"))
                await handler(payload)
                await msg.ack()
            except Exception:
                _log.exception(
                    "handler error for %s (will be redelivered)", subject
                )
                await msg.nak()

        sub = await self._js.subscribe(
            subject,
            cb=_on_msg,
            queue=queue or "",
            config=config,
        )
        _log.info(
            "subscribed to %s (durable=%s, queue=%s)",
            subject,
            durable,
            queue,
        )
        return sub

    # -- lifecycle ---------------------------------------------------------

    async def close(self) -> None:
        """Drain and close the underlying NATS connection."""
        if not self._conn.is_closed:
            await self._conn.drain()
            _log.info("NATS connection drained and closed")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def create_nats_client(
    url: str = _NATS_URL,
    creds: str | None = _NATS_CREDS,
) -> NatsClient:
    """Create and return a connected :class:`NatsClient`.

    The connection is configured with automatic reconnect.
    """
    connect_opts: dict[str, Any] = {
        "servers": [url],
        "reconnect_time_wait": _RECONNECT_DELAY,
        "max_reconnect_attempts": _MAX_RECONNECT,
        "error_cb": _on_error,
        "disconnected_cb": _on_disconnect,
        "reconnected_cb": _on_reconnect,
    }
    if creds:
        connect_opts["user_credentials"] = creds

    conn = await nats.connect(**connect_opts)
    js = conn.jetstream()
    _log.info("NATS connected to %s", url)
    return NatsClient(conn, js)


# ---------------------------------------------------------------------------
# Internal callbacks
# ---------------------------------------------------------------------------


async def _on_error(exc: Exception) -> None:
    _log.error("NATS error: %s", exc)


async def _on_disconnect() -> None:
    _log.warning("NATS disconnected — will attempt reconnect")


async def _on_reconnect() -> None:
    _log.info("NATS reconnected")
