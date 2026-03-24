"""NATS JetStream client for ExamPen DCR.

Adapted from exam-conductor/DCR/libs/exampen-common-py for use within the
Stoody backend process.  Key differences from the standalone library version:

- No module-level env reads; URL is a constructor parameter.
- Follows the same async patterns used by the rest of the Stoody backend
  (Motor-style async/await, structured logging).
- Factory function ``create_nats_client()`` returns a ready-to-use instance.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

import nats
from nats.aio.client import Client as NatsConn
from nats.aio.msg import Msg
from nats.js.api import ConsumerConfig, DeliverPolicy
from nats.js.client import JetStreamContext, PushSubscription

logger = logging.getLogger(__name__)

# Type alias for message handler callbacks
MessageHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class NatsClient:
    """Thin async wrapper around ``nats-py`` with JetStream support.

    Parameters
    ----------
    url:
        NATS server URL (e.g. ``"nats://localhost:4222"``).
    creds:
        Optional path to a NATS credentials file.
    reconnect_delay:
        Seconds to wait between reconnect attempts (default 2).
    max_reconnect:
        Maximum reconnect attempts (-1 = infinite, default).
    """

    def __init__(
        self,
        url: str = "nats://localhost:4222",
        *,
        creds: Optional[str] = None,
        reconnect_delay: float = 2.0,
        max_reconnect: int = -1,
    ) -> None:
        self._url = url
        self._creds = creds
        self._reconnect_delay = reconnect_delay
        self._max_reconnect = max_reconnect
        self._conn: Optional[NatsConn] = None
        self._js: Optional[JetStreamContext] = None

    # -- properties --------------------------------------------------------

    @property
    def connection(self) -> NatsConn:
        """Return the underlying NATS connection (raises if not connected)."""
        if self._conn is None or self._conn.is_closed:
            raise RuntimeError("NatsClient is not connected. Call connect() first.")
        return self._conn

    @property
    def jetstream(self) -> JetStreamContext:
        """Return the JetStream context (raises if not connected)."""
        if self._js is None:
            raise RuntimeError("NatsClient is not connected. Call connect() first.")
        return self._js

    @property
    def is_connected(self) -> bool:
        """Return True if the underlying connection is open."""
        return self._conn is not None and not self._conn.is_closed

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Establish the NATS connection and create the JetStream context."""
        if self.is_connected:
            logger.debug("NatsClient already connected to %s", self._url)
            return

        connect_opts: Dict[str, Any] = {
            "servers": [self._url],
            "reconnect_time_wait": self._reconnect_delay,
            "max_reconnect_attempts": self._max_reconnect,
            "error_cb": self._on_error,
            "disconnected_cb": self._on_disconnect,
            "reconnected_cb": self._on_reconnect,
        }
        if self._creds:
            connect_opts["user_credentials"] = self._creds

        self._conn = await nats.connect(**connect_opts)
        self._js = self._conn.jetstream()
        logger.info("NATS connected to %s", self._url)

    async def close(self) -> None:
        """Drain and close the underlying NATS connection."""
        if self._conn is not None and not self._conn.is_closed:
            await self._conn.drain()
            logger.info("NATS connection drained and closed")
        self._conn = None
        self._js = None

    # -- publish -----------------------------------------------------------

    async def publish(
        self,
        subject: str,
        data: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish a JSON-serialized message to a JetStream subject.

        Parameters
        ----------
        subject:
            NATS subject (e.g. ``"exampen.stroke.raw"``).
        data:
            Dictionary payload; will be serialized with ``json.dumps``.
        headers:
            Optional NATS headers.
        """
        js = self.jetstream
        payload = json.dumps(data, default=str).encode("utf-8")
        await js.publish(subject, payload, headers=headers)
        logger.debug("Published to %s (%d bytes)", subject, len(payload))

    # -- subscribe ---------------------------------------------------------

    async def subscribe(
        self,
        subject: str,
        handler: MessageHandler,
        *,
        queue_group: Optional[str] = None,
        durable: Optional[str] = None,
        deliver_policy: DeliverPolicy = DeliverPolicy.ALL,
    ) -> PushSubscription:
        """Subscribe to a JetStream subject with JSON deserialization.

        Parameters
        ----------
        subject:
            NATS subject pattern (e.g. ``"exampen.stroke.raw"``).
        handler:
            Async callback receiving a deserialized ``dict``.
        queue_group:
            Queue group name for load-balanced consumption.
        durable:
            Durable consumer name for replay guarantees.
        deliver_policy:
            JetStream deliver policy (default: ALL).
        """
        js = self.jetstream
        config = ConsumerConfig(
            deliver_policy=deliver_policy,
            durable_name=durable,
        )

        async def _on_msg(msg: Msg) -> None:
            try:
                payload = json.loads(msg.data.decode("utf-8"))
                await handler(payload)
                await msg.ack()
            except Exception:
                logger.exception(
                    "Handler error for %s (message will be redelivered)", subject
                )
                await msg.nak()

        sub = await js.subscribe(
            subject,
            cb=_on_msg,
            queue=queue_group or "",
            config=config,
        )
        logger.info(
            "Subscribed to %s (durable=%s, queue=%s)",
            subject,
            durable,
            queue_group,
        )
        return sub

    # -- internal callbacks ------------------------------------------------

    @staticmethod
    async def _on_error(exc: Exception) -> None:
        logger.error("NATS error: %s", exc)

    @staticmethod
    async def _on_disconnect() -> None:
        logger.warning("NATS disconnected - will attempt reconnect")

    @staticmethod
    async def _on_reconnect() -> None:
        logger.info("NATS reconnected")
