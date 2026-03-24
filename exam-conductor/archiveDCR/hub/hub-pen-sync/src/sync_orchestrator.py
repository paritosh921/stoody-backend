"""Orchestrates the full pen sync flow.

For each pen: connect -> query size -> start transfer -> receive chunks
-> pass each chunk to hub-store via IPC -> verify checksum -> ACK pen.

CRITICAL DATA SAFETY (STATE_OWNERSHIP_MAP.md):
  Pen buffer clear (0x03) ONLY after hub-store confirms dual-write
  for ALL chunks AND checksum matches. Data is irreplaceable once cleared.

Retry semantics: 3 retries on disconnect, 30s timeout each
(FAILURE_MITIGATION_REGISTER.md A1.7).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from src.chunk_manager import ChunkBuffer
from src.config import PenSyncConfig
from src.gatt_reader import (
    BleClient,
    clear_pen_buffer,
    read_buffer_status,
    receive_chunks,
    start_sync,
)
from src.store_client import (
    publish_complete,
    publish_progress,
    send_chunk_to_store,
)
from src.sync_state import (
    SyncEvent,
    SyncState,
    SyncStatus,
    record_chunk,
    record_store_confirm,
    set_buffer_info,
    transition,
)

logger = logging.getLogger(__name__)

MODULE_ID = "hub-pen-sync"


class BleClientFactory(Protocol):
    """Creates BLE client connections (abstracted for testing)."""

    async def connect(self, pen_mac: str, timeout: float) -> BleClient: ...
    async def disconnect(self, client: BleClient) -> None: ...


class SyncOrchestrator:
    """Coordinates pen sync across GATT reader, chunk buffer, and hub-store."""

    def __init__(
        self,
        config: PenSyncConfig,
        store_client: Any,
        ble_factory: BleClientFactory,
        event_publisher: Any,
    ) -> None:
        self._cfg = config
        self._store = store_client
        self._ble = ble_factory
        self._publish = event_publisher
        self._active_syncs: dict[str, SyncState] = {}
        self._abort_flags: dict[str, asyncio.Event] = {}

    def get_state(self, pen_mac: str) -> SyncState | None:
        return self._active_syncs.get(pen_mac)

    async def sync_pen(
        self, exam_id: str, pen_mac: str, dongle_mac: str
    ) -> SyncState:
        """Run the full sync flow for a single pen with retry logic."""
        state = SyncState(
            pen_mac=pen_mac,
            exam_id=exam_id,
            retries_remaining=self._cfg.max_retries,
        )
        self._active_syncs[pen_mac] = state
        self._abort_flags[pen_mac] = asyncio.Event()

        try:
            state = await self._sync_with_retries(state, dongle_mac)
        finally:
            self._abort_flags.pop(pen_mac, None)

        self._active_syncs[pen_mac] = state
        return state

    async def abort_pen(self, pen_mac: str, reason: str) -> None:
        """Signal abort for an in-progress sync."""
        flag = self._abort_flags.get(pen_mac)
        if flag:
            flag.set()
        state = self._active_syncs.get(pen_mac)
        if state and not state.is_terminal:
            transition(state, SyncEvent.ABORT)
            await publish_complete(self._publish, state)

    # --------------------------------------------------------------- retry

    async def _sync_with_retries(
        self, state: SyncState, dongle_mac: str
    ) -> SyncState:
        """Attempt sync up to max_retries times."""
        first_attempt = True
        while True:
            if first_attempt:
                transition(state, SyncEvent.START)
                first_attempt = False
            # After RETRY, state is already CONNECTING — no START needed
            await publish_progress(self._publish, state)

            try:
                state = await self._run_single_attempt(state, dongle_mac)
            except asyncio.TimeoutError:
                if not state.is_terminal:
                    transition(state, SyncEvent.TIMEOUT)
                logger.warning(
                    "Sync timeout for %s (retries: %d)",
                    state.pen_mac, state.retries_remaining,
                )
            except ConnectionError:
                if not state.is_terminal:
                    transition(state, SyncEvent.DISCONNECT)
                logger.warning(
                    "BLE disconnect for %s (retries: %d)",
                    state.pen_mac, state.retries_remaining,
                )

            if state.status == SyncStatus.COMPLETE:
                await publish_complete(self._publish, state)
                return state

            if state.can_retry:
                transition(state, SyncEvent.RETRY)
                logger.info(
                    "Retrying %s (attempt %d/%d)",
                    state.pen_mac,
                    self._cfg.max_retries - state.retries_remaining + 1,
                    self._cfg.max_retries,
                )
                continue

            await publish_complete(self._publish, state)
            return state

    # ---------------------------------------------------------- single attempt

    async def _run_single_attempt(
        self, state: SyncState, dongle_mac: str
    ) -> SyncState:
        """Execute one sync attempt: connect, read, store, verify."""
        client = await asyncio.wait_for(
            self._ble.connect(
                state.pen_mac, self._cfg.ble_connect_timeout_sec
            ),
            timeout=self._cfg.retry_timeout_sec,
        )
        transition(state, SyncEvent.CONNECTED)
        await publish_progress(self._publish, state)

        try:
            return await self._transfer_and_verify(client, state)
        finally:
            try:
                await self._ble.disconnect(client)
            except Exception:
                logger.debug("disconnect failed for %s", state.pen_mac)

    async def _transfer_and_verify(
        self, client: BleClient, state: SyncState
    ) -> SyncState:
        """Read buffer status, transfer chunks, verify, clear."""
        buf_status = await read_buffer_status(client)

        if buf_status.total_bytes == 0:
            state.checksum_expected = buf_status.checksum_crc32
            state.checksum_actual = buf_status.checksum_crc32
            transition(state, SyncEvent.STORE_CONFIRMED)
            return state

        set_buffer_info(
            state,
            total_bytes=buf_status.total_bytes,
            total_chunks=0,
            checksum_crc32=buf_status.checksum_crc32,
        )
        await start_sync(client)

        chunk_buf = ChunkBuffer(
            expected_total_bytes=buf_status.total_bytes,
            expected_buffer_crc32=buf_status.checksum_crc32,
        )
        state = await self._receive_and_store(client, state, chunk_buf)

        if state.status != SyncStatus.SYNCING:
            return state

        ok, actual_crc, _ = chunk_buf.verify_whole_buffer()
        state.checksum_actual = actual_crc

        if not ok:
            transition(state, SyncEvent.CHECKSUM_MISMATCH)
            return state

        transition(state, SyncEvent.CHECKSUM_MATCH)

        if state.all_chunks_stored:
            transition(state, SyncEvent.STORE_CONFIRMED)
            # CRITICAL: clear ONLY after confirmed dual-write + checksum
            await clear_pen_buffer(client)
            logger.info("Pen %s buffer cleared after dual-write", state.pen_mac)

        return state

    # ----------------------------------------------------------- chunk rx

    async def _receive_and_store(
        self,
        client: BleClient,
        state: SyncState,
        chunk_buf: ChunkBuffer,
    ) -> SyncState:
        """Receive chunks via GATT and store each via IPC."""
        abort = self._abort_flags.get(state.pen_mac)

        async def on_chunk(raw: bytes) -> None:
            nonlocal state
            if abort and abort.is_set():
                raise asyncio.CancelledError("Sync aborted")

            entry, err = chunk_buf.add_raw_chunk(raw)
            if err or entry is None:
                logger.warning("Chunk err for %s: %s", state.pen_mac, err)
                return

            if state.total_chunks == 0 and chunk_buf.total_chunks > 0:
                state.total_chunks = chunk_buf.total_chunks

            record_chunk(state, len(entry.payload))

            ok = await send_chunk_to_store(
                self._store, state.exam_id, state.pen_mac,
                entry.index, entry.payload, self._cfg.store_write_timeout_sec,
            )
            if ok:
                record_store_confirm(state, entry.index)

            await publish_progress(self._publish, state)

        try:
            await receive_chunks(client, self._cfg, on_chunk)
        except asyncio.TimeoutError:
            raise
        except asyncio.CancelledError:
            transition(state, SyncEvent.ABORT)
        except Exception as exc:
            if not state.is_terminal:
                transition(state, SyncEvent.DISCONNECT)
            logger.warning("receive_chunks failed: %s", exc)

        return state
