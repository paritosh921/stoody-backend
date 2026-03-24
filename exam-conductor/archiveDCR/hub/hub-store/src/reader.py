"""Read-only functions for hub-store data.

STATE_OWNERSHIP_MAP.md Section 2.1 mandates that these functions MUST
NOT mutate durable state.  Tests in ``test_reader.py`` assert this
invariant by snapshotting the filesystem before and after each call.

Functions:
  - ``get_pen_data``  — full ``strokes_raw.bin`` for one pen
  - ``get_chunk``     — single pre-chunked upload file
  - ``list_pen_data`` — enumerate pens that have data for an exam
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import StoreConfig

logger = logging.getLogger(__name__)


def _safe_dir_name(name: str) -> str:
    """Sanitize a name for filesystem use (colons invalid on Windows)."""
    return name.replace(":", "-")


def get_pen_data(
    config: StoreConfig,
    exam_id: str,
    pen_mac: str,
) -> bytes:
    """Return the raw stroke bytes for *pen_mac* in *exam_id*.

    Reads from SD primary path.  Returns empty bytes if the file does
    not exist (caller decides whether that is an error).
    """
    raw_path = config.sd_data / exam_id / _safe_dir_name(pen_mac) / "strokes_raw.bin"
    if not raw_path.is_file():
        logger.debug("No data at %s", raw_path)
        return b""
    return raw_path.read_bytes()


def get_chunk(
    config: StoreConfig,
    exam_id: str,
    pen_mac: str,
    chunk_index: int,
) -> bytes:
    """Return the bytes of a single pre-chunked upload file.

    File layout: ``{sd_data}/{exam_id}/{pen_mac}/chunks/chunk_{idx:03d}.bin``
    """
    chunk_path = (
        config.sd_data
        / exam_id
        / _safe_dir_name(pen_mac)
        / "chunks"
        / f"chunk_{chunk_index:03d}.bin"
    )
    if not chunk_path.is_file():
        logger.debug("Chunk not found: %s", chunk_path)
        return b""
    return chunk_path.read_bytes()


def list_pen_data(
    config: StoreConfig,
    exam_id: str,
) -> list[str]:
    """Return pen MAC addresses that have stored data for *exam_id*.

    Lists subdirectories of ``{sd_data}/{exam_id}/`` that contain a
    ``strokes_raw.bin`` file.
    """
    exam_dir = config.sd_data / exam_id
    if not exam_dir.is_dir():
        return []

    result: list[str] = []
    for entry in sorted(exam_dir.iterdir()):
        if entry.is_dir() and (entry / "strokes_raw.bin").is_file():
            result.append(entry.name)
    return result
