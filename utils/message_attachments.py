"""
Shared helper for validating and storing message attachments.

Used by both superadmin_async.py and admin_async.py for bidirectional messaging.
"""

import logging
from typing import Any, Dict, List

from fastapi import HTTPException, UploadFile
from core.upload_security.service import secure_upload_many

logger = logging.getLogger(__name__)

async def upload_message_attachments(
    files: List[UploadFile],
    *,
    actor: Dict[str, Any] | None,
    db: Any,
    purpose_metadata_base: Dict[str, Any],
    authorization_subject_base: str,
) -> List[dict]:
    """
    Validate and upload message attachments to S3.

    Returns a list of metadata dicts:
        [{"filename", "content_type", "size_bytes", "storage_path"}, ...]

    Raises HTTPException(400) on validation failure.
    """
    if not files:
        return []

    clean_uploads = await secure_upload_many(
        files=files,
        policy_id="support_message_attachment",
        actor=actor,
        db=db,
        purpose_metadata_factory=lambda upload, index: {
            **purpose_metadata_base,
            "purpose": "support_message_attachment",
            "index": index,
        },
        authorization_subject_factory=lambda upload, index: f"{authorization_subject_base}:attachment:{index}",
        include_bytes=False,
    )

    return [
        {
            "filename": clean_upload.original_filename or "attachment",
            "content_type": clean_upload.content_type,
            "size_bytes": clean_upload.size_bytes,
            "storage_path": clean_upload.released_storage_path,
            "upload_id": clean_upload.upload_id,
            "sha256": clean_upload.sha256,
        }
        for clean_upload in clean_uploads
    ]
