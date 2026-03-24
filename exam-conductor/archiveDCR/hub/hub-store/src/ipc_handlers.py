"""IPC message handlers for store.* messages.

Maps incoming ``store.write.request``, ``store.read.request``, and
``store.health.event`` triggers to the dual-writer and reader modules,
then returns well-formed IPC reply envelopes.
"""

from __future__ import annotations

import base64
import logging
import shutil
import zlib

from hub_common.ipc_protocol import IpcEnvelope
from hub_common.message_types import (
    STORE_HEALTH_EVENT,
    STORE_READ_REQUEST,
    STORE_READ_RESULT,
    STORE_WRITE_REQUEST,
    STORE_WRITE_RESULT,
)

from src.config import StoreConfig
from src.dual_writer import DualWriter
from src.reader import get_chunk, get_pen_data

logger = logging.getLogger(__name__)

MODULE_ID = "hub-store"


class StoreHandlers:
    """Stateful handler registry for hub-store IPC messages."""

    def __init__(self, writer: DualWriter, config: StoreConfig) -> None:
        self._writer = writer
        self._cfg = config

    # ---------------------------------------------------------------- write

    async def handle_write_request(self, env: IpcEnvelope) -> IpcEnvelope:
        """Handle ``store.write.request``: dual-write a pen chunk."""
        p = env.payload
        result = await self._writer.write_pen_chunk(
            exam_id=p["exam_id"],
            pen_mac=p["pen_mac"],
            chunk_index=p["chunk_index"],
            chunk_b64=p["chunk_b64"],
            checksum_crc32=p["checksum_crc32"],
        )

        if result.error:
            return env.make_error("storage_write_failed", result.error, source=MODULE_ID)

        if not result.sd_persisted:
            return env.make_error(
                "storage_write_failed",
                "SD write failed",
                source=MODULE_ID,
            )

        return env.make_reply(
            STORE_WRITE_RESULT,
            {
                "exam_id": result.exam_id,
                "pen_mac": result.pen_mac,
                "chunk_index": result.chunk_index,
                "sd_persisted": result.sd_persisted,
                "usb_persisted": result.usb_persisted,
            },
            source=MODULE_ID,
        )

    # ---------------------------------------------------------------- read

    async def handle_read_request(self, env: IpcEnvelope) -> IpcEnvelope:
        """Handle ``store.read.request``: return a stored chunk."""
        p = env.payload
        exam_id: str = p["exam_id"]
        pen_mac: str = p["pen_mac"]
        chunk_index: int = p["chunk_index"]

        data = get_chunk(self._cfg, exam_id, pen_mac, chunk_index)
        if not data:
            return env.make_error(
                "storage_read_failed",
                f"Chunk {chunk_index} not found for {exam_id}/{pen_mac}",
                source=MODULE_ID,
            )

        crc = format(zlib.crc32(data) & 0xFFFFFFFF, "08x")
        return env.make_reply(
            STORE_READ_RESULT,
            {
                "exam_id": exam_id,
                "pen_mac": pen_mac,
                "chunk_index": chunk_index,
                "chunk_b64": base64.b64encode(data).decode("ascii"),
                "checksum_crc32": crc,
            },
            source=MODULE_ID,
        )

    # ---------------------------------------------------------------- health

    def build_health_event(self, target: str) -> IpcEnvelope:
        """Build a ``store.health.event`` envelope with current status."""
        sd_ok = self._cfg.sd_base.is_dir()
        usb_ok = self._cfg.usb_available()

        free_bytes = 0
        try:
            usage = shutil.disk_usage(str(self._cfg.sd_base))
            free_bytes = usage.free
        except OSError:
            sd_ok = False

        return IpcEnvelope(
            msg_type=STORE_HEALTH_EVENT,
            source=MODULE_ID,
            target=target,
            expects_reply=False,
            payload={
                "sd_ok": sd_ok,
                "usb_ok": usb_ok,
                "degraded": self._writer.degraded,
                "free_bytes": free_bytes,
            },
        )
