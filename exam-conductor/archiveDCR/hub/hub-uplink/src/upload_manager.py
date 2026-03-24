"""Core upload logic for hub-uplink.

Reads chunks from hub-store via IPC (``store.read.request``), uploads
each to ``POST /api/v1/strokes/ingest`` on svc-stroke-ingest, and
records backend ACKs in the SQLite ``upload_ledger``.

KEY INVARIANTS (STATE_OWNERSHIP_MAP.md Section 3):
  - Ledger updated ONLY after backend ACK (not before).
  - Pen marked "uploaded" ONLY when ALL chunks ACKd.
  - Backend deduplicates via idempotency key ``{exam_id}:{pen_mac}:{chunk_index}``.
  - Retry indefinitely until all chunks ACKd.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from hub_common.ipc_protocol import IpcClient, IpcEnvelope
from hub_common.message_types import STORE_READ_REQUEST

from src.config import UplinkConfig
from src.ledger import UploadLedger

logger = logging.getLogger(__name__)

MODULE_ID = "hub-uplink"


@dataclass(slots=True)
class PenUploadSpec:
    """Specifies a pen whose chunks should be uploaded."""

    exam_id: str
    pen_mac: str
    total_chunks: int
    upload_path: str  # "wifi" | "mobile"


class UploadManager:
    """Orchestrates per-pen chunk upload with resume and retry."""

    def __init__(
        self,
        config: UplinkConfig,
        ledger: UploadLedger,
        store_client: IpcClient,
        *,
        progress_callback: Any | None = None,
    ) -> None:
        self._cfg = config
        self._ledger = ledger
        self._store = store_client
        self._on_progress = progress_callback

    # -- public API ---------------------------------------------------------

    async def upload_pen(self, spec: PenUploadSpec) -> bool:
        """Upload all pending chunks for one pen.

        Returns True when every chunk has been ACKd.
        Retries indefinitely on transient failures.
        """
        self._ledger.init_pen(
            spec.exam_id, spec.pen_mac, spec.total_chunks, spec.upload_path,
        )
        pending = self._ledger.get_pending_chunks(spec.exam_id, spec.pen_mac)
        if not pending:
            self._ledger.mark_upload_complete(spec.exam_id, spec.pen_mac)
            return True

        for chunk_index in pending:
            await self._upload_chunk_with_retry(spec, chunk_index)

        self._ledger.mark_upload_complete(spec.exam_id, spec.pen_mac)
        logger.info(
            "Pen %s/%s upload complete (%d chunks)",
            spec.exam_id, spec.pen_mac, spec.total_chunks,
        )
        return True

    # -- chunk-level upload -------------------------------------------------

    async def _upload_chunk_with_retry(
        self, spec: PenUploadSpec, chunk_index: int,
    ) -> None:
        """Upload a single chunk, retrying indefinitely on failure."""
        while True:
            try:
                chunk_data = await self._read_chunk_from_store(
                    spec.exam_id, spec.pen_mac, chunk_index,
                )
                if chunk_data is None:
                    raise RuntimeError(
                        f"store.read.request failed for chunk {chunk_index}"
                    )

                accepted = await self._post_chunk(spec, chunk_index, chunk_data)
                if accepted:
                    # ACK received — update ledger AFTER backend confirmation
                    self._ledger.mark_chunk_acked(
                        spec.exam_id, spec.pen_mac, chunk_index,
                    )
                    if self._on_progress:
                        acked = spec.total_chunks - len(
                            self._ledger.get_pending_chunks(
                                spec.exam_id, spec.pen_mac,
                            ),
                        )
                        await self._on_progress(
                            spec.exam_id, spec.pen_mac, chunk_index,
                            acked, spec.total_chunks, spec.upload_path,
                        )
                    return

            except Exception:
                logger.exception(
                    "Chunk %d upload failed for %s/%s — retrying in %ds",
                    chunk_index, spec.exam_id, spec.pen_mac,
                    self._cfg.retry_interval_sec,
                )

            await asyncio.sleep(self._cfg.retry_interval_sec)

    # -- store IPC ----------------------------------------------------------

    async def _read_chunk_from_store(
        self, exam_id: str, pen_mac: str, chunk_index: int,
    ) -> dict | None:
        """Read a chunk from hub-store via IPC ``store.read.request``."""
        request = IpcEnvelope(
            msg_type=STORE_READ_REQUEST,
            source=MODULE_ID,
            target="hub-store",
            expects_reply=True,
            payload={
                "exam_id": exam_id,
                "pen_mac": pen_mac,
                "chunk_index": chunk_index,
            },
        )
        reply = await self._store.request(request, timeout=10.0)
        if "error" in reply.msg_type or "code" in reply.payload:
            logger.error("Store read error: %s", reply.payload)
            return None
        return reply.payload

    # -- HTTP upload --------------------------------------------------------

    async def _post_chunk(
        self,
        spec: PenUploadSpec,
        chunk_index: int,
        chunk_data: dict,
    ) -> bool:
        """POST one chunk to svc-stroke-ingest.  Returns True on 202."""
        url = self._cfg.backend_url.rstrip("/") + self._cfg.ingest_endpoint
        idempotency_key = f"{spec.exam_id}:{spec.pen_mac}:{chunk_index}"

        body = {
            "exam_id": spec.exam_id,
            "pen_mac": spec.pen_mac,
            "chunk_index": chunk_index,
            "total_chunks": spec.total_chunks,
            "payload_base64": chunk_data["chunk_b64"],
            "checksum_crc32": chunk_data["checksum_crc32"],
            "upload_path": spec.upload_path,
            "idempotency_key": idempotency_key,
        }

        timeout = aiohttp.ClientTimeout(total=self._cfg.upload_timeout_sec)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=timeout) as resp:
                if resp.status == 202:
                    return True
                logger.warning(
                    "Ingest returned %d for chunk %d of %s/%s",
                    resp.status, chunk_index, spec.exam_id, spec.pen_mac,
                )
                return False
