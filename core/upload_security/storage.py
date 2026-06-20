"""Private quarantine and released upload storage."""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import aiofiles

from config_async import settings


_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


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
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as handle:
            await handle.write(data)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        await _copy_file(Path(quarantine_path), path)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as handle:
            await handle.write(data)
        return str(path)

    async def mark_rejected(self, *, quarantine_path: str, tenant: str | None, upload_id: str) -> str:
        source = Path(quarantine_path)
        if not source.exists():
            return quarantine_path
        tenant_segment = safe_storage_segment(tenant)
        path = self.local_root / self.rejected_prefix / tenant_segment / safe_storage_segment(upload_id) / source.name
        path.parent.mkdir(parents=True, exist_ok=True)
        await _copy_file(source, path)
        return str(path)


async def _copy_file(source: Path, destination: Path) -> None:
    import asyncio

    await asyncio.to_thread(shutil.copyfile, source, destination)
