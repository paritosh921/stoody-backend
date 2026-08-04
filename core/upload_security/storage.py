"""Private quarantine and released upload storage."""

from __future__ import annotations

import os
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import aiofiles

from config_async import settings


_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")
PRIVATE_DIRECTORY_MODE = 0o2750
PRIVATE_FILE_MODE = 0o640


def safe_filename(filename: str | None, *, fallback: str = "upload.bin") -> str:
    cleaned = _SAFE_SEGMENT.sub("_", filename or fallback).strip("._")
    return cleaned or fallback


def safe_storage_segment(value: str | None, *, fallback: str = "unknown") -> str:
    cleaned = _SAFE_SEGMENT.sub("_", value or fallback).strip("._")
    return cleaned or fallback


class PrivateUploadStorage:
    def __init__(
        self,
        *,
        local_root: str | Path | None = None,
        quarantine_prefix: str | None = None,
        released_prefix: str | None = None,
        rejected_prefix: str | None = None,
    ) -> None:
        self.local_root = Path(local_root or settings.UPLOAD_PRIVATE_LOCAL_DIR)
        self.quarantine_prefix = quarantine_prefix or settings.UPLOAD_QUARANTINE_PREFIX
        self.released_prefix = released_prefix or settings.UPLOAD_RELEASED_PREFIX
        self.rejected_prefix = rejected_prefix or settings.UPLOAD_REJECTED_PREFIX

    async def write_quarantine(
        self,
        *,
        data: bytes,
        tenant: str | None,
        upload_id: str,
        original_filename: str | None,
    ) -> str:
        tenant_segment = safe_storage_segment(tenant)
        name = f"{uuid.uuid4().hex}_{safe_filename(original_filename)}"
        path = self.local_root / self.quarantine_prefix / tenant_segment / safe_storage_segment(upload_id) / name
        _ensure_private_directory(path.parent, root=self.local_root)
        async with aiofiles.open(path, "wb") as handle:
            await handle.write(data)
        _chmod_best_effort(path, PRIVATE_FILE_MODE)
        return str(path)

    async def release_clean(
        self,
        *,
        quarantine_path: str,
        tenant: str | None,
        policy_id: str,
        upload_id: str,
        safe_filename: str,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        tenant_segment = safe_storage_segment(tenant)
        path = (
            self.local_root
            / self.released_prefix
            / tenant_segment
            / safe_storage_segment(policy_id)
            / safe_storage_segment(upload_id)
            / safe_filename
        )
        _ensure_private_directory(path.parent, root=self.local_root)
        source = Path(quarantine_path)
        await _copy_file(source, path)
        _chmod_best_effort(path, PRIVATE_FILE_MODE)
        await _write_metadata_sidecar(path, content_type=content_type, metadata=metadata)
        await _delete_file_and_empty_parents(source, stop_at=self.local_root / self.quarantine_prefix)
        return str(path)

    async def write_released_bytes(
        self,
        *,
        data: bytes,
        tenant: str | None,
        policy_id: str,
        upload_id: str,
        safe_filename: str,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        tenant_segment = safe_storage_segment(tenant)
        path = (
            self.local_root
            / self.released_prefix
            / tenant_segment
            / safe_storage_segment(policy_id)
            / safe_storage_segment(upload_id)
            / safe_filename
        )
        _ensure_private_directory(path.parent, root=self.local_root)
        async with aiofiles.open(path, "wb") as handle:
            await handle.write(data)
        _chmod_best_effort(path, PRIVATE_FILE_MODE)
        await _write_metadata_sidecar(path, content_type=content_type, metadata=metadata)
        return str(path)

    async def read_released_path(self, released_path: str) -> bytes:
        """Read a scanner-released artefact without permitting arbitrary paths."""
        if not released_path:
            raise FileNotFoundError("Released upload path is missing")
        candidate = Path(released_path).resolve(strict=False)
        released_root = (self.local_root / self.released_prefix).resolve(strict=False)
        if candidate == released_root or released_root not in candidate.parents:
            raise ValueError("Released upload path is outside private upload storage")
        if not candidate.is_file():
            raise FileNotFoundError(f"Released upload is unavailable: {candidate}")
        return await _read_file(candidate)

    async def delete_released_path(self, released_path: str) -> bool:
        """Delete a released local artefact after durable object-store transfer.

        The path is constrained to this storage instance's released prefix so
        an upload record can never turn this cleanup into arbitrary deletion.
        """
        if not released_path:
            return False
        try:
            candidate = Path(released_path).resolve(strict=False)
            released_root = (self.local_root / self.released_prefix).resolve(
                strict=False
            )
        except OSError:
            return False
        if candidate == released_root or released_root not in candidate.parents:
            return False

        await _delete_file_and_empty_parents(candidate, stop_at=released_root)
        await _delete_file_and_empty_parents(
            Path(f"{candidate}.metadata.json"),
            stop_at=released_root,
        )
        return True

    async def mark_rejected(
        self,
        *,
        quarantine_path: str,
        tenant: str | None,
        upload_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        source = Path(quarantine_path)
        if not source.exists():
            return quarantine_path
        tenant_segment = safe_storage_segment(tenant)
        path = self.local_root / self.rejected_prefix / tenant_segment / safe_storage_segment(upload_id) / source.name
        _ensure_private_directory(path.parent, root=self.local_root)
        await _copy_file(source, path)
        _chmod_best_effort(path, PRIVATE_FILE_MODE)
        await _write_metadata_sidecar(path, content_type="application/octet-stream", metadata=metadata)
        return str(path)


async def _copy_file(source: Path, destination: Path) -> None:
    import asyncio

    await asyncio.to_thread(shutil.copyfile, source, destination)


async def _read_file(path: Path) -> bytes:
    import asyncio

    return await asyncio.to_thread(path.read_bytes)


async def _delete_file_and_empty_parents(path: Path, *, stop_at: Path) -> None:
    import asyncio

    def _delete() -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return
        stop = stop_at.resolve(strict=False)
        current = path.parent
        while True:
            try:
                current_resolved = current.resolve(strict=False)
            except OSError:
                break
            if current_resolved == stop or stop not in current_resolved.parents:
                break
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    await asyncio.to_thread(_delete)


async def _write_metadata_sidecar(
    path: Path,
    *,
    content_type: str,
    metadata: dict[str, Any] | None,
) -> None:
    if not metadata:
        return
    sidecar = Path(f"{path}.metadata.json")
    payload = {
        "content_type": content_type,
        "metadata": metadata,
    }
    async with aiofiles.open(sidecar, "w", encoding="utf-8") as handle:
        await handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))
    _chmod_best_effort(sidecar, PRIVATE_FILE_MODE)


def _ensure_private_directory(path: Path, *, root: Path | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for directory in _private_directory_chain(path, root=root):
        _chmod_best_effort(directory, PRIVATE_DIRECTORY_MODE)


def _private_directory_chain(path: Path, *, root: Path | None = None) -> list[Path]:
    if root is None:
        return [path]
    try:
        relative = path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return [path]

    directories = [root]
    current = root
    for part in relative.parts:
        current = current / part
        directories.append(current)
    return directories


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass
