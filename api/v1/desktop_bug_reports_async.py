"""
Desktop bug report ingestion API.
Accepts support messages submitted from Stoody desktop agent Help tab.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class DesktopBugReportSubmitRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=5000)
    timestamp: Optional[str] = Field(default=None, max_length=80)
    app_version: str = Field(default="unknown", max_length=50)
    pen_mac: Optional[str] = Field(default=None, max_length=32)
    pen_connected: bool = Field(default=False)
    os_info: str = Field(default="", max_length=300)
    machine: str = Field(default="", max_length=200)


@router.post("/desktop-bug-reports/submit")
@limiter.limit("20/hour")
async def submit_desktop_bug_report(
    request: Request,
    payload: DesktopBugReportSubmitRequest,
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

    await tenant_db["desktop_bug_reports"].insert_one(
        {
            "ticket_id": ticket_id,
            "title": payload.title.strip(),
            "description": payload.description.strip(),
            "reported_at_client": payload.timestamp,
            "created_at": now,
            "app_version": payload.app_version,
            "pen_mac": (payload.pen_mac or "").upper() if payload.pen_mac else None,
            "pen_connected": bool(payload.pen_connected),
            "os_info": payload.os_info.strip(),
            "machine": payload.machine.strip(),
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
    }

