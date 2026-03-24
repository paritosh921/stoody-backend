"""Photo-based copy upload fallback endpoints.

Routes are mounted at ``/api/v1/exampen/copies``.

When pen strokes are unavailable (pen malfunction, network failure),
invigilators or students can upload photographs of the answer sheets.
These are stored as references to S3/local objects and linked to the
exam + student.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
    require_exampen_role,
)

logger = logging.getLogger(__name__)
router = APIRouter()

COLLECTION = "exampen_copy_uploads"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db


async def _publish_event(request: Request, subject: str, data: dict) -> None:
    nats = getattr(request.app.state, "exampen_nats", None)
    if nats is None or not nats.is_connected:
        return
    try:
        await nats.publish(subject, data)
    except Exception:
        logger.warning("NATS publish to %s failed (non-fatal)", subject)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class CopyUploadBody(BaseModel):
    exam_id: str
    student_id: str
    page_number: int = 1
    image_url: str  # S3 key or presigned URL
    mime_type: str = "image/jpeg"
    file_size_bytes: int | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_copy(
    body: CopyUploadBody,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("principal", "hod", "invigilator", "student")
    ),
) -> dict[str, Any]:
    """Record a photo-based copy upload.

    The actual image binary is assumed to have been uploaded directly to
    S3 (presigned URL).  This endpoint records the metadata and triggers
    downstream processing via NATS.
    """
    db = await _get_tenant_db(request, user)
    coll = db[COLLECTION]

    now = datetime.now(timezone.utc)
    doc = {
        "_id": uuid4().hex,
        "tenant_id": user.tenant_id,
        "exam_id": body.exam_id,
        "student_id": body.student_id,
        "page_number": body.page_number,
        "image_url": body.image_url,
        "mime_type": body.mime_type,
        "file_size_bytes": body.file_size_bytes,
        "notes": body.notes,
        "uploaded_by": user.user_id,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
    }
    await coll.insert_one(doc)

    await _publish_event(request, "exampen.copy.ready", {
        "copy_id": doc["_id"],
        "exam_id": body.exam_id,
        "student_id": body.student_id,
        "tenant_id": user.tenant_id,
    })

    return doc


@router.get("/{exam_id}")
async def list_copies(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
    student_id: str | None = Query(None),
) -> dict[str, Any]:
    """List uploaded copies for an exam, optionally filtered by student.

    Students may only see their own copies.
    """
    db = await _get_tenant_db(request, user)
    coll = db[COLLECTION]

    query: dict[str, Any] = {
        "exam_id": exam_id,
        "tenant_id": user.tenant_id,
    }

    is_student = "student" in user.exampen_roles
    if is_student:
        # Students can only see their own
        query["student_id"] = user.user_id
    elif student_id:
        query["student_id"] = student_id

    cursor = coll.find(query).sort("page_number", 1)
    items = await cursor.to_list(length=500)
    return {"exam_id": exam_id, "copies": items}
