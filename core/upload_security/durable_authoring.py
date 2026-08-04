"""Durable private-object handoff for scanned authoring PDFs.

Upload security intentionally scans files on local private storage. This module
owns the boundary after that scan: immutable bytes are verified, promoted to
private S3, reflected in the upload audit, and only then removed from staging.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from inspect import iscoroutinefunction
from typing import Any

from core.upload_security.storage import PrivateUploadStorage, safe_storage_segment
from utils.s3_storage import upload_private_object


logger = logging.getLogger(__name__)
AUTHORING_OBJECT_PREFIX = "private/content/authoring"


@dataclass(frozen=True)
class AuthoringPromotion:
    storage_uri: str
    data: bytes
    sha256: str
    upload_id: str
    released_path: str


async def stage_released_authoring_pdf(
    *,
    released_path: str,
    expected_sha256: str,
    filename: str,
    content_type: str,
    tenant_db: str,
    document_id: str,
    artifact_role: str,
    upload_id: str,
    data: bytes | None = None,
    storage: PrivateUploadStorage | None = None,
) -> AuthoringPromotion:
    """Verify a clean local PDF and copy it to content-addressed private S3."""
    private_storage = storage or PrivateUploadStorage()
    payload = bytes(data) if data else await private_storage.read_released_path(released_path)
    if not payload:
        raise ValueError("Released authoring PDF is empty")

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    expected = str(expected_sha256 or "").strip().lower()
    if not expected or actual_sha256 != expected:
        raise ValueError("Released authoring PDF failed SHA-256 verification")

    tenant = safe_storage_segment(tenant_db, fallback="tenant")
    document = safe_storage_segment(document_id, fallback="document")
    role = safe_storage_segment(artifact_role, fallback="document")
    object_key = f"{AUTHORING_OBJECT_PREFIX}/{tenant}/{document}/{role}/{actual_sha256}.pdf"
    storage_uri = await upload_private_object(
        payload,
        object_key=object_key,
        content_type=content_type or "application/pdf",
        metadata={
            "document_id": document,
            "artifact_role": role,
            "sha256": actual_sha256,
            "source": "secure_authoring_upload",
            "filename": safe_storage_segment(filename, fallback=f"{role}.pdf"),
        },
    )
    return AuthoringPromotion(
        storage_uri=storage_uri,
        data=payload,
        sha256=actual_sha256,
        upload_id=str(upload_id or "").strip(),
        released_path=released_path,
    )


async def complete_authoring_promotion(
    db: Any,
    promotion: AuthoringPromotion,
    *,
    storage: PrivateUploadStorage | None = None,
) -> None:
    """Commit the S3 handoff to the audit record, then remove local staging."""
    if promotion.upload_id:
        update = {
            "$set": {
                "storage_backend": "s3",
                "storage_transfer_status": "complete",
                "storage_transfer_updated_at": datetime.now(timezone.utc),
                "released_storage_path": promotion.storage_uri,
            }
        }
        update_with_manager = getattr(db, "mongo_update_one", None)
        if iscoroutinefunction(update_with_manager):
            await update_with_manager(
                "upload_security_verdicts",
                {"upload_id": promotion.upload_id},
                update,
            )
        else:
            await db["upload_security_verdicts"].update_one(
                {"upload_id": promotion.upload_id},
                update,
            )

    private_storage = storage or PrivateUploadStorage()
    deleted = await private_storage.delete_released_path(promotion.released_path)
    if not deleted:
        logger.warning(
            "Private S3 handoff completed but local authoring staging cleanup failed: %s",
            promotion.released_path,
        )


async def promote_clean_authoring_pdf(
    db: Any,
    clean_upload: Any,
    *,
    tenant_db: str,
    document_id: str,
    artifact_role: str,
) -> AuthoringPromotion:
    """Promote one CleanUpload and complete its durable storage handoff."""
    promotion = await stage_released_authoring_pdf(
        released_path=clean_upload.released_storage_path,
        expected_sha256=clean_upload.sha256,
        filename=clean_upload.original_filename,
        content_type=clean_upload.content_type or "application/pdf",
        tenant_db=tenant_db,
        document_id=document_id,
        artifact_role=artifact_role,
        upload_id=clean_upload.upload_id,
        data=clean_upload.bytes,
    )
    await complete_authoring_promotion(db, promotion)
    return promotion
