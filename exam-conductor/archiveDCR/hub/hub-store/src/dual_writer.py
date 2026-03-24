"""Core dual-write engine.

Implements the critical data-safety protocol from HUB_DEPLOYMENT_SPEC.md
Section 3.3 and FAILURE_MITIGATION_REGISTER.md S4:

  SD write + fsync  -->  USB write + fsync  -->  return success
  If USB write fails --> log warning, set degraded flag, continue SD-only

Every ``os.fsync()`` call is the hard guarantee that data has reached
durable media before the caller (hub-pen-sync) ACKs the pen.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import StoreConfig

if TYPE_CHECKING:
    from src.ledger import ChunkLedger

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # 64 KiB per pre-chunked upload file


def _safe_dir_name(name: str) -> str:
    """Sanitize a name for use as a directory component.

    On Linux (the RPi target) colons in MAC addresses are fine, but on
    Windows they are invalid path characters.  Replace them with dashes.
    """
    return name.replace(":", "-")


@dataclass(slots=True)
class WriteResult:
    """Outcome of a single dual-write operation."""

    exam_id: str
    pen_mac: str
    chunk_index: int
    sd_persisted: bool
    usb_persisted: bool
    degraded: bool = False
    error: str | None = None


@dataclass(slots=True)
class MetaInfo:
    """Contents of ``strokes.meta.json``."""

    bytes: int
    checksum_crc32: str
    pages: list[int] = field(default_factory=list)
    sync_ts: str = ""


class DualWriter:
    """Append-mode dual-write to SD + USB with ``os.fsync()`` gates."""

    def __init__(
        self,
        config: StoreConfig,
        ledger: ChunkLedger | None = None,
    ) -> None:
        self._cfg = config
        self._degraded = False
        self._ledger = ledger

    @property
    def degraded(self) -> bool:
        return self._degraded

    # ------------------------------------------------------------------ write

    async def write_pen_chunk(
        self,
        exam_id: str,
        pen_mac: str,
        chunk_index: int,
        chunk_b64: str,
        checksum_crc32: str,
    ) -> WriteResult:
        """Dual-write a single chunk.

        1. Decode base-64 payload.
        2. Verify CRC-32 of decoded bytes.
        3. SD write + fsync.
        4. USB write + fsync (best-effort if USB unavailable).
        5. Write pre-chunked upload file.
        6. Update metadata (SD + USB mirror).
        7. Record chunk in SQLite ledger.
        """
        raw = base64.b64decode(chunk_b64)

        actual_crc = format(zlib.crc32(raw) & 0xFFFFFFFF, "08x")
        if actual_crc != checksum_crc32:
            self._record_failure(
                exam_id, pen_mac,
                f"CRC mismatch: expected {checksum_crc32}, got {actual_crc}",
            )
            return WriteResult(
                exam_id=exam_id,
                pen_mac=pen_mac,
                chunk_index=chunk_index,
                sd_persisted=False,
                usb_persisted=False,
                error=f"CRC mismatch: expected {checksum_crc32}, got {actual_crc}",
            )

        sd_ok = self._write_and_fsync(
            self._cfg.sd_data, exam_id, pen_mac, chunk_index, raw
        )

        usb_ok = False
        if self._cfg.usb_available():
            usb_ok = self._write_and_fsync(
                self._cfg.usb_data, exam_id, pen_mac, chunk_index, raw
            )
            if not usb_ok:
                self._degraded = True
                logger.warning(
                    "USB write failed for %s/%s chunk %d — degraded mode",
                    exam_id, pen_mac, chunk_index,
                )
        else:
            self._degraded = True
            logger.warning("USB mount not available — degraded mode")

        if sd_ok:
            self._update_metadata(
                self._cfg.sd_data, exam_id, pen_mac, len(raw), actual_crc
            )
            # Mirror metadata to USB
            if usb_ok:
                self._update_metadata(
                    self._cfg.usb_data, exam_id, pen_mac, len(raw), actual_crc
                )
            # Record in ledger
            self._record_chunk(exam_id, pen_mac, len(raw))
        else:
            self._record_failure(exam_id, pen_mac, "SD write failed")

        return WriteResult(
            exam_id=exam_id,
            pen_mac=pen_mac,
            chunk_index=chunk_index,
            sd_persisted=sd_ok,
            usb_persisted=usb_ok,
            degraded=self._degraded,
        )

    # -------------------------------------------------------- sync lifecycle

    def mark_sync_complete(
        self, exam_id: str, pen_mac: str, checksum: str
    ) -> None:
        """Mark a pen's sync as complete after checksum verification."""
        if self._ledger:
            self._ledger.mark_sync_complete(exam_id, pen_mac, checksum)

    def init_upload_ledger(
        self, exam_id: str, pen_mac: str, total_chunks: int
    ) -> None:
        """Prepare the upload ledger row after chunking is done."""
        if self._ledger:
            self._ledger.init_upload_ledger(exam_id, pen_mac, total_chunks)

    # ---------------------------------------------------------------- internals

    def _record_chunk(
        self, exam_id: str, pen_mac: str, num_bytes: int
    ) -> None:
        if self._ledger:
            try:
                self._ledger.record_chunk_received(exam_id, pen_mac, num_bytes)
            except Exception:
                logger.exception("Ledger record_chunk_received failed")

    def _record_failure(
        self, exam_id: str, pen_mac: str, detail: str
    ) -> None:
        if self._ledger:
            try:
                self._ledger.mark_sync_failed(exam_id, pen_mac, detail)
            except Exception:
                logger.exception("Ledger mark_sync_failed failed")

    @staticmethod
    def _write_and_fsync(
        base: Path,
        exam_id: str,
        pen_mac: str,
        chunk_index: int,
        data: bytes,
    ) -> bool:
        """Append *data* to ``strokes_raw.bin`` AND write a pre-chunked file.

        Returns ``True`` on success.  Performs ``os.fsync()`` after each
        write — this is the critical data-safety guarantee.
        """
        pen_dir = base / exam_id / _safe_dir_name(pen_mac)
        pen_dir.mkdir(parents=True, exist_ok=True)

        # -- append to strokes_raw.bin ----------------------------------
        raw_path = pen_dir / "strokes_raw.bin"
        try:
            fd = os.open(str(raw_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            logger.exception("Failed to write %s", raw_path)
            return False

        # -- write pre-chunked upload file ------------------------------
        chunks_dir = pen_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunks_dir / f"chunk_{chunk_index:03d}.bin"
        try:
            fd = os.open(str(chunk_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            logger.exception("Failed to write chunk %s", chunk_path)
            return False

        return True

    @staticmethod
    def _update_metadata(
        base: Path,
        exam_id: str,
        pen_mac: str,
        new_bytes: int,
        crc: str,
    ) -> None:
        """Update ``strokes.meta.json`` with cumulative stats and fsync."""
        pen_dir = base / exam_id / _safe_dir_name(pen_mac)
        pen_dir.mkdir(parents=True, exist_ok=True)
        meta_path = pen_dir / "strokes.meta.json"

        existing = MetaInfo(bytes=0, checksum_crc32="")
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    d = json.load(f)
                existing = MetaInfo(**d)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        existing.bytes += new_bytes
        existing.checksum_crc32 = crc
        existing.sync_ts = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        blob = json.dumps({
            "bytes": existing.bytes,
            "checksum_crc32": existing.checksum_crc32,
            "pages": existing.pages,
            "sync_ts": existing.sync_ts,
        })

        # Write + fsync for data-safety
        fd = os.open(str(meta_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            os.write(fd, blob.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
