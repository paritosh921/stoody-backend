"""Hub-store IPC client and event publishing helpers.

Encapsulates store.write.request IPC calls and pen.sync.* event
publishing, keeping the orchestrator focused on flow control.
"""

from __future__ import annotations

import base64
import logging
import zlib
from typing import Any

from hub_common.ipc_protocol import IpcEnvelope
from hub_common.message_types import STORE_WRITE_REQUEST

from src.sync_state import SyncState

logger = logging.getLogger(__name__)

MODULE_ID = "hub-pen-sync"


async def send_chunk_to_store(
    store: Any,
    exam_id: str,
    pen_mac: str,
    chunk_index: int,
    payload: bytes,
    timeout: float,
) -> bool:
    """Send a chunk to hub-store via IPC. Returns True if SD confirmed."""
    chunk_b64 = base64.b64encode(payload).decode("ascii")
    crc = format(zlib.crc32(payload) & 0xFFFFFFFF, "08x")

    env = IpcEnvelope(
        msg_type=STORE_WRITE_REQUEST,
        source=MODULE_ID,
        target="hub-store",
        expects_reply=True,
        payload={
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "chunk_index": chunk_index,
            "chunk_b64": chunk_b64,
            "checksum_crc32": crc,
        },
    )

    try:
        reply = await store.request(env, timeout=timeout)
    except TimeoutError:
        logger.error(
            "Store write timed out: %s/%s chunk %d", exam_id, pen_mac, chunk_index
        )
        return False

    if ".error" in reply.msg_type:
        logger.error(
            "Store write failed: %s/%s chunk %d: %s",
            exam_id, pen_mac, chunk_index,
            reply.payload.get("message", "unknown"),
        )
        return False

    if not reply.payload.get("sd_persisted", False):
        logger.error("SD not confirmed: %s/%s chunk %d", exam_id, pen_mac, chunk_index)
        return False

    return True


async def publish_progress(
    publish_fn: Any, state: SyncState
) -> None:
    """Emit pen.sync.progress.event to supervisor, TUI, invig."""
    payload = {
        "exam_id": state.exam_id,
        "pen_mac": state.pen_mac,
        "chunk_index": state.chunks_received,
        "total_chunks": state.total_chunks,
        "bytes_received": state.bytes_received,
        "status": state.status.name.lower(),
    }
    for target in ("hub-supervisor", "hub-tui", "hub-invig-ble"):
        env = IpcEnvelope(
            msg_type="pen.sync.progress.event",
            source=MODULE_ID,
            target=target,
            expects_reply=False,
            payload=payload,
        )
        try:
            await publish_fn(env)
        except Exception:
            logger.debug("Progress publish failed to %s", target)


async def publish_complete(
    publish_fn: Any, state: SyncState
) -> None:
    """Emit pen.sync.complete.event to supervisor and uplink."""
    payload = {
        "exam_id": state.exam_id,
        "pen_mac": state.pen_mac,
        "total_chunks": state.total_chunks,
        "checksum_crc32": state.checksum_actual or state.checksum_expected,
        "status": state.status.name.lower(),
    }
    for target in ("hub-supervisor", "hub-uplink"):
        env = IpcEnvelope(
            msg_type="pen.sync.complete.event",
            source=MODULE_ID,
            target=target,
            expects_reply=False,
            payload=payload,
        )
        try:
            await publish_fn(env)
        except Exception:
            logger.debug("Complete publish failed to %s", target)
