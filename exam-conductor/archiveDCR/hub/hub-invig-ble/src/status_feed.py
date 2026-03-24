"""Real-time status feed for the invigilator mobile app.

Collects hub status from the supervisor via IPC snapshot requests and
formats it as the JSON schema defined in ``ble-gatt-spec.md`` Section 5.

The ``StatusFeedCollector`` is a pure formatter that transforms an IPC
snapshot result into the BLE-ready JSON bytes.  The async scheduling
(1 Hz cadence) is handled by the caller (``main.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Status snapshot (data from supervisor)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HubSnapshot:
    """Supervisor FSM snapshot enriched with subsystem data.

    Maps to the ``fsm.snapshot.result`` payload plus supplementary
    fields gathered from timer, store, and BLE manager.
    """

    exam_id: str | None
    state: str
    timer_remaining_sec: int
    wifi_connected: bool
    wifi_band: str
    wifi_signal_dbm: int
    storage_sd_ok: bool
    storage_usb_ok: bool
    storage_degraded: bool
    sync_complete: int
    sync_in_progress: int
    sync_failed: int
    sync_pending: int


# ---------------------------------------------------------------------------
# Default (empty) snapshot
# ---------------------------------------------------------------------------

_EMPTY_SNAPSHOT = HubSnapshot(
    exam_id=None,
    state="idle",
    timer_remaining_sec=0,
    wifi_connected=False,
    wifi_band="",
    wifi_signal_dbm=0,
    storage_sd_ok=True,
    storage_usb_ok=True,
    storage_degraded=False,
    sync_complete=0,
    sync_in_progress=0,
    sync_failed=0,
    sync_pending=0,
)


# ---------------------------------------------------------------------------
# Collector / formatter
# ---------------------------------------------------------------------------

class StatusFeedCollector:
    """Formats hub status into the BLE status-feed JSON schema.

    Call :meth:`update` to ingest a new supervisor snapshot, then
    :meth:`to_json_bytes` to get the UTF-8 encoded notification payload.
    """

    def __init__(self) -> None:
        self._snapshot: HubSnapshot = _EMPTY_SNAPSHOT

    @property
    def snapshot(self) -> HubSnapshot:
        return self._snapshot

    def update(self, snapshot: HubSnapshot) -> None:
        """Replace the current snapshot with a fresh one."""
        self._snapshot = snapshot

    def update_from_ipc(self, payload: dict[str, Any]) -> None:
        """Update from a raw ``fsm.snapshot.result`` IPC payload dict."""
        timer = payload.get("timer", {})
        storage = payload.get("storage", {})
        wifi = payload.get("wifi", {})
        sync = payload.get("sync", {})

        self._snapshot = HubSnapshot(
            exam_id=payload.get("exam_id"),
            state=payload.get("state", "idle"),
            timer_remaining_sec=timer.get("remaining_sec", 0),
            wifi_connected=wifi.get("connected", False),
            wifi_band=wifi.get("band", ""),
            wifi_signal_dbm=wifi.get("signal_dbm", 0),
            storage_sd_ok=storage.get("sd_ok", True),
            storage_usb_ok=storage.get("usb_ok", True),
            storage_degraded=storage.get("degraded", False),
            sync_complete=sync.get("complete", 0),
            sync_in_progress=sync.get("in_progress", 0),
            sync_failed=sync.get("failed", 0),
            sync_pending=sync.get("pending", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the status-feed JSON object (ble-gatt-spec.md Section 5)."""
        s = self._snapshot
        return {
            "exam_id": s.exam_id,
            "state": s.state,
            "timer_remaining_sec": s.timer_remaining_sec,
            "wifi": {
                "connected": s.wifi_connected,
                "band": s.wifi_band,
                "signal_dbm": s.wifi_signal_dbm,
            },
            "storage": {
                "sd_ok": s.storage_sd_ok,
                "usb_ok": s.storage_usb_ok,
                "degraded": s.storage_degraded,
            },
            "sync": {
                "complete": s.sync_complete,
                "in_progress": s.sync_in_progress,
                "failed": s.sync_failed,
                "pending": s.sync_pending,
            },
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the status-feed object to UTF-8 JSON bytes.

        Suitable for writing directly to the BLE Status Feed characteristic.
        """
        return json.dumps(
            self.to_dict(), separators=(",", ":"),
        ).encode("utf-8")
