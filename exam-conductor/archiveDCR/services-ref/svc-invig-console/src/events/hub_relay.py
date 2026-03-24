"""Subscribe to NATS hub status events and relay to WebSocket clients.

On each hub status update:
1. Store the latest hub data keyed by exam_id.
2. Notify all connected WebSocket clients for that exam.

This module owns ZERO authoritative state -- it is a read-only relay.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Awaitable

from exampen_common.logging import get_logger
from exampen_common.nats_client import NatsClient

from src.config import NATS_HUB_STATUS_SUBJECT

_log = get_logger(__name__)

# Type alias for WebSocket notification callbacks
WsNotifyCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

# NATS subject for hub status updates (per-exam)
# Pattern: EXAMPEN.hub.status.<exam_id>
_SUBJECT = NATS_HUB_STATUS_SUBJECT


class HubRelay:
    """Manages NATS subscription for hub status and fan-out to WS clients.

    Stores the latest hub status per exam_id so that newly connecting
    WebSocket clients can receive an immediate snapshot.
    """

    def __init__(self, nats: NatsClient) -> None:
        self._nats = nats
        self._sub: Any = None
        # Latest hub status per exam_id
        self._latest: dict[str, dict[str, Any]] = {}
        # Lock for concurrent access to _latest
        self._lock = asyncio.Lock()
        # Registered per-exam notification callbacks (exam_id -> set of callbacks)
        self._listeners: dict[str, set[WsNotifyCallback]] = {}

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to hub status NATS subject."""
        self._sub = await self._nats.subscribe(
            _SUBJECT,
            handler=self._on_hub_status,
            durable="invig-console-hub-status",
            queue="invig-console",
        )
        _log.info("Hub relay started, subscribed to %s", _SUBJECT)

    async def stop(self) -> None:
        """Unsubscribe and clean up."""
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None
        self._listeners.clear()
        _log.info("Hub relay stopped")

    # -- Public API ---------------------------------------------------------

    def get_latest(self, exam_id: str) -> dict[str, Any]:
        """Return the most recent hub status for *exam_id*, or empty dict."""
        return self._latest.get(exam_id, {})

    async def register_listener(
        self,
        exam_id: str,
        callback: WsNotifyCallback,
    ) -> None:
        """Register a WebSocket notification callback for *exam_id*."""
        async with self._lock:
            if exam_id not in self._listeners:
                self._listeners[exam_id] = set()
            self._listeners[exam_id].add(callback)

    async def unregister_listener(
        self,
        exam_id: str,
        callback: WsNotifyCallback,
    ) -> None:
        """Remove a previously registered callback."""
        async with self._lock:
            listeners = self._listeners.get(exam_id)
            if listeners:
                listeners.discard(callback)
                if not listeners:
                    del self._listeners[exam_id]

    # -- Internal -----------------------------------------------------------

    async def _on_hub_status(self, payload: dict[str, Any]) -> None:
        """Handle an incoming hub status message from NATS."""
        exam_id = payload.get("exam_id", "")
        if not exam_id:
            _log.warning("Hub status message missing exam_id, ignoring")
            return

        async with self._lock:
            self._latest[exam_id] = payload
            listeners = list(self._listeners.get(exam_id, set()))

        # Fan-out to all registered WebSocket callbacks for this exam
        for callback in listeners:
            try:
                await callback(exam_id, payload)
            except Exception:
                _log.exception(
                    "Error notifying WS listener for exam %s", exam_id,
                )
