"""
Shared helper for uploading message attachments to S3.

Used by both superadmin_async.py and admin_async.py for bidirectional messaging.
"""

import uuid
import logging
from typing import List

from fastapi import HTTPException, UploadFile
from utils.s3_storage import upload_file as s3_upload_file

logger = logging.getLogger(__name__)

ALLOWED_MESSAGE_FILE_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/jpg"}
MAX_MESSAGE_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB per file
MAX_MESSAGE_TOTAL_SIZE_BYTES = 100 * 1024 * 1024  # 100MB total per message
MAX_MESSAGE_FILES = 10


async def upload_message_attachments(files: List[UploadFile]) -> List[dict]:
    """
    Validate and upload message attachments to S3.

    Returns a list of metadata dicts:
        [{"filename", "content_type", "size_bytes", "storage_path"}, ...]

    Raises HTTPException(400) on validation failure.
    """
    if not files:
        return []

    if len(files) > MAX_MESSAGE_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {MAX_MESSAGE_FILES} files per message.",
        )

    attachment_metadata = []
    total_size = 0

    for file in files:
        # Validate content type
        content_type = (file.content_type or "").lower()
        if content_type not in ALLOWED_MESSAGE_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' has unsupported type '{content_type}'. "
                       f"Allowed: PDF, PNG, JPG.",
            )

        # Read file data
        file_data = await file.read()
        file_size = len(file_data)

        # Validate individual file size
        if file_size > MAX_MESSAGE_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is {file_size / (1024*1024):.1f}MB. "
                       f"Maximum is 20MB per file.",
            )

        total_size += file_size
        if total_size > MAX_MESSAGE_TOTAL_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Total attachment size exceeds 100MB limit.",
            )

        # Upload to S3
        safe_filename = (file.filename or "attachment").replace("/", "_").replace("\\", "_")
        s3_path = f"uploads/message-attachments/{uuid.uuid4()}_{safe_filename}"

        success, storage_path = await s3_upload_file(
            file_data, s3_path, content_type=content_type
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file '{file.filename}'.",
            )

        attachment_metadata.append({
            "filename": file.filename or "attachment",
            "content_type": content_type,
            "size_bytes": file_size,
            "storage_path": storage_path,
        })

    return attachment_metadata
