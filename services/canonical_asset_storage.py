"""Durable, tenant-scoped storage for immutable PCR paper assets.

PCR grading workers must be able to read the exact question paper and teacher
solution that were frozen when an exam was finalized.  A host-local absolute
path is not a durable reference when API and grading processes run on
different machines (or when a production database is inspected locally).

New PCR assets are therefore promoted to the private object store whenever it
is configured.  Local storage remains available for development, and legacy
absolute paths can be safely rebased to the configured private-upload root
without allowing arbitrary filesystem reads.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config_async import settings
from core.upload_security.storage import (
    PrivateUploadStorage,
    safe_filename,
    safe_storage_segment,
)
from utils.s3_storage import (
    PrivateObjectStorageError,
    delete_private_object,
    download_private_object,
    is_s3_enabled,
    upload_private_object,
)


logger = logging.getLogger(__name__)

PRIVATE_CANONICAL_ASSET_S3_PREFIX = "private/exampen/canonical-assets"
DEFAULT_MAX_CANONICAL_ASSET_BYTES = 64 * 1024 * 1024


class CanonicalAssetStorageError(RuntimeError):
    """Raised when an immutable PCR asset cannot be stored safely."""


@dataclass(frozen=True)
class CanonicalAssetTransfer:
    """One scan-stage local upload and its canonical storage reference."""

    upload_id: str
    local_path: str
    storage_path: str

    @property
    def promoted_to_s3(self) -> bool:
        return self.storage_path.startswith("s3://")


def canonical_object_storage_required() -> bool:
    """Return whether this runtime may retain canonical assets only locally."""

    configured = os.getenv("CANONICAL_ASSET_REQUIRE_OBJECT_STORAGE")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    return not bool(settings.DEBUG_MODE)


def canonical_asset_object_key(
    *,
    tenant_db: str,
    document_id: str,
    artifact_kind: str,
    upload_id: str,
    sha256: str,
    filename: str,
) -> str:
    """Build an immutable private key without user-controlled path segments."""

    digest = str(sha256 or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CanonicalAssetStorageError("Canonical asset SHA-256 is invalid")
    return "/".join(
        (
            PRIVATE_CANONICAL_ASSET_S3_PREFIX,
            safe_storage_segment(tenant_db),
            safe_storage_segment(document_id),
            safe_storage_segment(artifact_kind),
            safe_storage_segment(upload_id),
            f"{digest}-{safe_filename(filename, fallback='asset.pdf')}",
        )
    )


async def store_canonical_asset(
    *,
    data: bytes,
    local_path: str,
    upload_id: str,
    tenant_db: str,
    document_id: str,
    artifact_kind: str,
    filename: str,
    content_type: str,
    sha256: str,
) -> CanonicalAssetTransfer:
    """Promote one clean PCR asset to durable private storage when available."""

    if not data:
        raise CanonicalAssetStorageError("Canonical asset payload is empty")
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != str(sha256 or "").strip().lower():
        raise CanonicalAssetStorageError("Canonical asset integrity check failed")

    if not is_s3_enabled():
        if canonical_object_storage_required():
            raise CanonicalAssetStorageError(
                "Private object storage is required for canonical PCR assets"
            )
        return CanonicalAssetTransfer(
            upload_id=upload_id,
            local_path=local_path,
            storage_path=local_path,
        )

    try:
        storage_path = await upload_private_object(
            data,
            object_key=canonical_asset_object_key(
                tenant_db=tenant_db,
                document_id=document_id,
                artifact_kind=artifact_kind,
                upload_id=upload_id,
                sha256=actual_sha256,
                filename=filename,
            ),
            content_type=content_type or "application/pdf",
            metadata={
                "purpose": "immutable_pcr_asset",
                "artifact": safe_storage_segment(artifact_kind),
                "tenant": safe_storage_segment(tenant_db),
                "document": safe_storage_segment(document_id),
                "sha256": actual_sha256,
            },
        )
    except PrivateObjectStorageError as exc:
        raise CanonicalAssetStorageError(
            "Canonical PCR asset could not be stored in private object storage"
        ) from exc

    return CanonicalAssetTransfer(
        upload_id=upload_id,
        local_path=local_path,
        storage_path=storage_path,
    )


async def finalize_canonical_transfer(
    transfer: CanonicalAssetTransfer,
    *,
    tenant_db: Any | None = None,
) -> None:
    """Update upload audit metadata and remove the redundant local staging file."""

    if not transfer.promoted_to_s3:
        return
    now = datetime.now(timezone.utc)
    if tenant_db is not None:
        try:
            await tenant_db["upload_security_verdicts"].update_one(
                {"upload_id": transfer.upload_id},
                {
                    "$set": {
                        "released_storage_path": transfer.storage_path,
                        "storage_backend": "s3",
                        "storage_transfer_status": "complete",
                        "storage_transfer_updated_at": now,
                    }
                },
            )
        except Exception:
            logger.exception(
                "Could not update canonical asset upload audit for %s",
                transfer.upload_id,
            )

    try:
        deleted = await PrivateUploadStorage().delete_released_path(
            transfer.local_path
        )
        if not deleted:
            logger.warning(
                "Canonical asset was promoted but its local staging file was not removed: %s",
                transfer.local_path,
            )
    except Exception:
        logger.exception(
            "Could not remove canonical asset local staging file: %s",
            transfer.local_path,
        )


async def rollback_canonical_transfers(
    transfers: Iterable[CanonicalAssetTransfer],
) -> None:
    """Remove S3 objects created by a document upload that did not commit."""

    for transfer in transfers:
        if not transfer.promoted_to_s3:
            continue
        try:
            await delete_private_object(
                transfer.storage_path,
                allowed_key_prefix=PRIVATE_CANONICAL_ASSET_S3_PREFIX,
            )
        except PrivateObjectStorageError:
            logger.exception(
                "Could not roll back uncommitted canonical asset %s",
                transfer.storage_path,
            )


def canonical_local_asset_candidates(
    storage_path: str,
    *,
    backend_root: str | Path | None = None,
    private_root: str | Path | None = None,
) -> list[Path]:
    """Resolve approved local candidates for current and legacy storage paths.

    Persisted Linux paths are rebased only from known upload-root anchors.
    Arbitrary absolute paths and traversal outside the two approved roots are
    excluded.
    """

    raw = str(storage_path or "").strip()
    if not raw or raw.startswith("s3://"):
        return []

    backend = Path(backend_root or Path(__file__).resolve().parents[1]).resolve(
        strict=False
    )
    private = Path(private_root or settings.UPLOAD_PRIVATE_LOCAL_DIR).resolve(
        strict=False
    )
    public_uploads = (backend / "uploads").resolve(strict=False)
    allowed_roots = (private, public_uploads)
    normalized = raw.replace("\\", "/")
    normalized_lower = normalized.lower()
    proposed: list[Path] = []

    direct = Path(raw)
    if direct.is_absolute():
        proposed.append(direct)
    elif normalized.startswith(("clean/", "derived/")):
        proposed.append(private / Path(normalized))
    else:
        proposed.append(backend / Path(normalized))

    for marker in (
        "/var/lib/stoody/uploads/",
        "/data/private_uploads/",
    ):
        index = normalized_lower.find(marker)
        if index >= 0:
            tail = normalized[index + len(marker) :]
            if tail:
                proposed.append(private / Path(tail))

    uploads_marker = "/uploads/"
    uploads_index = normalized_lower.find(uploads_marker)
    if uploads_index >= 0:
        tail = normalized[uploads_index + len(uploads_marker) :]
        if tail:
            proposed.append(public_uploads / Path(tail))

    candidates: list[Path] = []
    seen: set[str] = set()
    for proposed_path in proposed:
        try:
            candidate = proposed_path.resolve(strict=False)
        except OSError:
            continue
        if not any(
            candidate == root or root in candidate.parents for root in allowed_roots
        ):
            continue
        identity = os.path.normcase(str(candidate))
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(candidate)
    return candidates


async def read_canonical_asset(
    storage_path: str,
    *,
    max_bytes: int = DEFAULT_MAX_CANONICAL_ASSET_BYTES,
) -> bytes | None:
    """Read a bounded canonical asset from private S3 or an approved local root."""

    if not storage_path:
        return None
    if storage_path.startswith("s3://"):
        try:
            return await download_private_object(storage_path, max_bytes=max_bytes)
        except PrivateObjectStorageError as exc:
            logger.error("Could not read canonical private object: %s", exc)
            return None

    candidates = canonical_local_asset_candidates(storage_path)
    if not candidates:
        logger.error(
            "Refusing canonical asset outside approved upload roots: %s",
            storage_path,
        )
        return None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > max_bytes:
            logger.error(
                "Canonical asset has an invalid size (%s bytes): %s",
                size,
                candidate,
            )
            return None
        return await asyncio.to_thread(candidate.read_bytes)
    logger.error(
        "Canonical asset is absent from every approved local candidate: %s",
        storage_path,
    )
    return None
