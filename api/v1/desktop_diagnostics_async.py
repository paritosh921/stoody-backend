"""
Desktop diagnostics upload API.
Accepts diagnostics zip bundles from Stoody desktop agent for support triage.
"""

import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from utils.s3_storage import upload_file

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

MAX_ZIP_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/zip", "application/octet-stream"}


def _safe_filename(value: str) -> str:
    name = (value or "").strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120] or "diagnostics.zip"


@router.post("/desktop-diagnostics/upload")
@limiter.limit("5/hour")
async def upload_desktop_diagnostics(
    request: Request,
    diagnostics_zip: UploadFile = File(...),
    package_id: str = Form(..., min_length=8, max_length=120),
    app_version: str = Form("unknown", max_length=50),
    note: str = Form("", max_length=1000),
    pen_mac: Optional[str] = Form(None, max_length=32),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Store an uploaded diagnostics bundle for the authenticated user."""
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

    filename = _safe_filename(diagnostics_zip.filename or f"{package_id}.zip")
    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip diagnostics files are allowed",
        )

    content_type = (diagnostics_zip.content_type or "").strip().lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {diagnostics_zip.content_type}",
        )

    data = await diagnostics_zip.read()
    size_bytes = len(data)
    if size_bytes <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded diagnostics file is empty",
        )
    if size_bytes > MAX_ZIP_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Diagnostics file too large (max {MAX_ZIP_SIZE_BYTES} bytes)",
        )

    ticket_id = f"diag_{uuid.uuid4().hex[:12]}"
    user_id = str(current_user.get("user_id") or "unknown")
    local_path = os.path.join(
        "uploads",
        "desktop-diagnostics",
        db_name,
        user_id,
        f"{ticket_id}_{filename}",
    )

    success, storage_path = await upload_file(
        file_data=data,
        local_path=local_path,
        content_type="application/zip",
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store diagnostics bundle",
        )

    now = datetime.utcnow()
    await tenant_db["desktop_diagnostics"].insert_one(
        {
            "ticket_id": ticket_id,
            "package_id": package_id,
            "filename": filename,
            "storage_path": storage_path,
            "size_bytes": size_bytes,
            "content_type": "application/zip",
            "uploaded_at": now,
            "app_version": app_version,
            "note": note.strip() if note else "",
            "pen_mac": (pen_mac or "").upper() if pen_mac else None,
            "user_id": user_id,
            "username": current_user.get("username"),
            "user_type": current_user.get("user_type"),
            "db_name": db_name,
            "source": "desktop_agent",
        }
    )

    logger.info(
        "Desktop diagnostics uploaded: ticket=%s user=%s size=%d",
        ticket_id,
        user_id,
        size_bytes,
    )
    return {
        "success": True,
        "ticket_id": ticket_id,
        "package_id": package_id,
        "uploaded_at": now.isoformat(),
        "size_bytes": size_bytes,
    }
