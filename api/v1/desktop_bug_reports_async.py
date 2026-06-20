"""
Desktop bug report ingestion API.
Accepts support messages submitted from Stoody desktop agent Help tab.
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from core.upload_security.service import secure_upload_many

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

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

    clean_uploads = await secure_upload_many(
        files=uploads,
        policy_id="desktop_bug_image",
        actor=current_user,
        db=tenant_db,
        purpose_metadata_factory=lambda upload, index: {
            "purpose": "desktop_bug_image",
            "collection": "desktop_bug_reports",
            "ticket_id": ticket_id,
            "index": index,
            "created_by": user_id,
        },
        authorization_subject_factory=lambda upload, index: f"desktop_bug:{db_name}:{ticket_id}:{index}",
        include_bytes=False,
    )

    attachment_docs: List[Dict[str, Any]] = []
    for clean_upload in clean_uploads:
        safe_name = _safe_filename(clean_upload.original_filename or "image")
        attachment_docs.append(
            {
                "filename": safe_name,
                "content_type": clean_upload.content_type,
                "size_bytes": clean_upload.size_bytes,
                "storage_path": clean_upload.released_storage_path,
                "upload_id": clean_upload.upload_id,
                "sha256": clean_upload.sha256,
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
