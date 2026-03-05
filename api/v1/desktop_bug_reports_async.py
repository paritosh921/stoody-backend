"""
Desktop bug report ingestion API.
Accepts support messages submitted from Stoody desktop agent Help tab.
"""

import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from utils.s3_storage import upload_file

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
}
MAX_TOTAL_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 20MB total
MAX_ATTACHMENTS = 8


def _safe_filename(value: str) -> str:
    name = (value or "").strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:150] or f"attachment_{uuid.uuid4().hex[:8]}"


@router.post("/desktop-bug-reports/submit")
@limiter.limit("20/hour")
async def submit_desktop_bug_report(
    request: Request,
    title: str = Form(..., min_length=1, max_length=200),
    description: str = Form(..., min_length=1, max_length=5000),
    timestamp: Optional[str] = Form(default=None, max_length=80),
    app_version: str = Form(default="unknown", max_length=50),
    pen_mac: Optional[str] = Form(default=None, max_length=32),
    pen_connected: bool = Form(default=False),
    os_info: str = Form(default="", max_length=300),
    machine: str = Form(default="", max_length=200),
    attachments: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Store a desktop bug report for support triage in tenant DB."""
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context required",
        )

    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )

    user_id = str(current_user.get("user_id") or "unknown")
    now = datetime.utcnow()
    ticket_id = f"bug_{uuid.uuid4().hex[:12]}"
    uploads = [f for f in attachments if f and f.filename]
    if len(uploads) > MAX_ATTACHMENTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many images. Maximum {MAX_ATTACHMENTS} files allowed.",
        )

    attachment_docs: List[Dict[str, Any]] = []
    total_bytes = 0
    for upload in uploads:
        content_type = (upload.content_type or "").strip().lower()
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported attachment type: {upload.content_type}. Images only.",
            )
        file_data = await upload.read()
        file_size = len(file_data)
        total_bytes += file_size
        if total_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Total image size exceeds 20MB limit ({MAX_TOTAL_ATTACHMENT_BYTES} bytes).",
            )

        safe_name = _safe_filename(upload.filename or "image")
        local_path = os.path.join(
            "uploads",
            "desktop-bug-reports",
            db_name,
            user_id,
            ticket_id,
            safe_name,
        )
        success, storage_path = await upload_file(
            file_data=file_data,
            local_path=local_path,
            content_type=content_type,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store attachment '{safe_name}'",
            )
        attachment_docs.append(
            {
                "filename": safe_name,
                "content_type": content_type,
                "size_bytes": file_size,
                "storage_path": storage_path,
            }
        )

    await tenant_db["desktop_bug_reports"].insert_one(
        {
            "ticket_id": ticket_id,
            "title": title.strip(),
            "description": description.strip(),
            "reported_at_client": timestamp,
            "created_at": now,
            "app_version": app_version,
            "pen_mac": (pen_mac or "").upper() if pen_mac else None,
            "pen_connected": bool(pen_connected),
            "os_info": os_info.strip(),
            "machine": machine.strip(),
            "attachments": attachment_docs,
            "attachment_count": len(attachment_docs),
            "attachment_total_bytes": sum(a.get("size_bytes", 0) for a in attachment_docs),
            "user_id": user_id,
            "username": current_user.get("username"),
            "user_type": current_user.get("user_type"),
            "db_name": db_name,
            "source": "desktop_agent_help_form",
        }
    )

    logger.info(
        "Desktop bug report submitted: ticket=%s user=%s db=%s",
        ticket_id,
        user_id,
        db_name,
    )
    return {
        "success": True,
        "ticket_id": ticket_id,
        "created_at": now.isoformat(),
        "attachment_count": len(attachment_docs),
    }
