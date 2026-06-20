"""
Desktop diagnostics upload API.
Accepts diagnostics zip bundles from Stoody desktop agent for support triage.
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from core.upload_security.service import secure_upload

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

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

    ticket_id = f"diag_{uuid.uuid4().hex[:12]}"
    user_id = str(current_user.get("user_id") or "unknown")
    filename = _safe_filename(diagnostics_zip.filename or f"{package_id}.zip")
    clean_upload = await secure_upload(
        file=diagnostics_zip,
        policy_id="desktop_diagnostics_zip",
        actor=current_user,
        db=tenant_db,
        purpose_metadata={
            "purpose": "desktop_diagnostics_zip",
            "collection": "desktop_diagnostics",
            "ticket_id": ticket_id,
            "package_id": package_id,
            "created_by": user_id,
        },
        authorization_subject=f"desktop_diagnostics:{db_name}:{ticket_id}",
        include_bytes=False,
    )

    now = datetime.utcnow()
    await tenant_db["desktop_diagnostics"].insert_one(
        {
            "ticket_id": ticket_id,
            "package_id": package_id,
            "filename": filename,
            "storage_path": clean_upload.released_storage_path,
            "size_bytes": clean_upload.size_bytes,
            "content_type": clean_upload.content_type or "application/zip",
            "upload_id": clean_upload.upload_id,
            "sha256": clean_upload.sha256,
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
        clean_upload.size_bytes,
    )
    return {
        "success": True,
        "ticket_id": ticket_id,
        "package_id": package_id,
        "uploaded_at": now.isoformat(),
        "size_bytes": clean_upload.size_bytes,
    }
