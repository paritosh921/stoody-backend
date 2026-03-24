"""Per-pen sync state machine — ZERO I/O, pure domain logic.

State transitions follow HUB_DEPLOYMENT_SPEC.md Section 4.2 and
FAILURE_MITIGATION_REGISTER.md A1.7 (3 retries, 30s timeout each).

The domain layer must NEVER import asyncio, bleak, or any I/O library
(CLAUDE.md Per-Service Layer Rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Self


class SyncStatus(Enum):
    """Mirrors ``pen_sync_status.status`` CHECK constraint in SQLite schema."""

    PENDING = auto()
    CONNECTING = auto()
    SYNCING = auto()
    COMPLETE = auto()
    FAILED = auto()
    TIMEOUT = auto()


class SyncEvent(Enum):
    """Events that drive state transitions."""

    START = auto()
    CONNECTED = auto()
    CHUNK_RECEIVED = auto()
    ALL_CHUNKS_RECEIVED = auto()
    CHECKSUM_MATCH = auto()
    CHECKSUM_MISMATCH = auto()
    STORE_CONFIRMED = auto()
    DISCONNECT = auto()
    TIMEOUT = auto()
    ABORT = auto()
    RETRY = auto()


# Valid transitions: (current_status, event) -> new_status
_TRANSITIONS: dict[tuple[SyncStatus, SyncEvent], SyncStatus] = {
    (SyncStatus.PENDING, SyncEvent.START): SyncStatus.CONNECTING,
    (SyncStatus.CONNECTING, SyncEvent.CONNECTED): SyncStatus.SYNCING,
    (SyncStatus.CONNECTING, SyncEvent.TIMEOUT): SyncStatus.TIMEOUT,
    (SyncStatus.CONNECTING, SyncEvent.DISCONNECT): SyncStatus.FAILED,
    (SyncStatus.CONNECTING, SyncEvent.ABORT): SyncStatus.FAILED,
    (SyncStatus.SYNCING, SyncEvent.CHUNK_RECEIVED): SyncStatus.SYNCING,
    (SyncStatus.SYNCING, SyncEvent.CHECKSUM_MATCH): SyncStatus.SYNCING,
    (SyncStatus.SYNCING, SyncEvent.STORE_CONFIRMED): SyncStatus.COMPLETE,
    (SyncStatus.SYNCING, SyncEvent.CHECKSUM_MISMATCH): SyncStatus.FAILED,
    (SyncStatus.SYNCING, SyncEvent.DISCONNECT): SyncStatus.FAILED,
    (SyncStatus.SYNCING, SyncEvent.TIMEOUT): SyncStatus.TIMEOUT,
    (SyncStatus.SYNCING, SyncEvent.ABORT): SyncStatus.FAILED,
    # Retry from failed/timeout brings back to connecting
    (SyncStatus.FAILED, SyncEvent.RETRY): SyncStatus.CONNECTING,
    (SyncStatus.TIMEOUT, SyncEvent.RETRY): SyncStatus.CONNECTING,
}


@dataclass(slots=True)
class SyncState:
    """Tracks per-pen sync progress and integrity verification.

    All fields are plain data — no I/O, no async, no side effects.
    """

    pen_mac: str
    exam_id: str
    status: SyncStatus = SyncStatus.PENDING
    bytes_expected: int = 0
    bytes_received: int = 0
    total_chunks: int = 0
    chunks_received: int = 0
    chunks_store_confirmed: int = 0
    checksum_expected: str = ""
    checksum_actual: str = ""
    retries_remaining: int = 3
    last_confirmed_chunk: int = -1
    error_detail: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            SyncStatus.COMPLETE,
            SyncStatus.FAILED,
            SyncStatus.TIMEOUT,
        )

    @property
    def can_retry(self) -> bool:
        return (
            self.status in (SyncStatus.FAILED, SyncStatus.TIMEOUT)
            and self.retries_remaining > 0
        )

    @property
    def progress_pct(self) -> float:
        if self.bytes_expected <= 0:
            return 0.0
        return min(100.0, (self.bytes_received / self.bytes_expected) * 100.0)

    @property
    def all_chunks_stored(self) -> bool:
        return (
            self.total_chunks > 0
            and self.chunks_store_confirmed >= self.total_chunks
        )

    @property
    def checksum_verified(self) -> bool:
        return (
            self.checksum_expected != ""
            and self.checksum_actual != ""
            and self.checksum_expected == self.checksum_actual
        )


def transition(state: SyncState, event: SyncEvent) -> SyncState:
    """Apply *event* to *state* and return the updated state.

    Raises ``ValueError`` if the transition is not allowed.
    """
    key = (state.status, event)
    new_status = _TRANSITIONS.get(key)
    if new_status is None:
        raise ValueError(
            f"Invalid transition: {state.status.name} + {event.name}"
        )

    state.status = new_status

    if event is SyncEvent.RETRY:
        state.retries_remaining -= 1
        state.error_detail = ""

    if event is SyncEvent.DISCONNECT:
        state.error_detail = "BLE disconnect"

    if event is SyncEvent.TIMEOUT:
        state.error_detail = "Timeout"

    if event is SyncEvent.ABORT:
        state.error_detail = "Aborted by supervisor"

    if event is SyncEvent.CHECKSUM_MISMATCH:
        state.error_detail = (
            f"Checksum mismatch: expected {state.checksum_expected}, "
            f"got {state.checksum_actual}"
        )

    return state


def record_chunk(state: SyncState, chunk_bytes: int) -> SyncState:
    """Record that a chunk has been received from the pen."""
    state.chunks_received += 1
    state.bytes_received += chunk_bytes
    return state


def record_store_confirm(
    state: SyncState, chunk_index: int
) -> SyncState:
    """Record that hub-store confirmed durable write for a chunk."""
    state.chunks_store_confirmed += 1
    state.last_confirmed_chunk = max(
        state.last_confirmed_chunk, chunk_index
    )
    return state


def set_buffer_info(
    state: SyncState,
    total_bytes: int,
    total_chunks: int,
    checksum_crc32: str,
) -> SyncState:
    """Populate buffer metadata after reading pen's Buffer Status char."""
    state.bytes_expected = total_bytes
    state.total_chunks = total_chunks
    state.checksum_expected = checksum_crc32
    return state
