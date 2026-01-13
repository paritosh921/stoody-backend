"""
Async strokes API for SkillBot
Reads stroke data produced by the Stoody BLE agent (tenant Mongo `strokes` collection).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from core.permissions import has_permission
from core.pen_tokens import decode_pen_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strokes", tags=["strokes"])


class StrokeItem(BaseModel):
    id: str = Field(..., description="Mongo document id")
    session_id: str
    pen_mac: str
    page_number: Optional[int] = None
    timestamp: datetime
    strokes: List[Dict[str, Any]]
    book_type: Optional[str] = None
    canvas_background: Optional[str] = None
    page_style: Optional[str] = None


class StrokeListResponse(BaseModel):
    count: int
    strokes: List[StrokeItem]

class StrokeIngestRequest(BaseModel):
    session_id: str
    pen_mac: str
    page_number: Optional[int] = None
    timestamp: Optional[datetime] = None
    strokes: List[Dict[str, Any]]
    book_type: Optional[str] = None
    canvas_background: Optional[str] = None
    page_style: Optional[str] = None


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


@router.get("", response_model=StrokeListResponse)
async def list_strokes(
    pen_mac: Optional[str] = Query(None, description="Filter by pen MAC (optional)"),
    page_number: Optional[int] = Query(None, description="Filter by page number (optional)"),
    limit: int = Query(50, ge=1, le=500, description="Max documents to return"),
    before: Optional[datetime] = Query(None, description="Only strokes before this timestamp"),
    after: Optional[datetime] = Query(None, description="Only strokes after this timestamp"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
) -> StrokeListResponse:
    """
    Return stroke batches for the authenticated user, sourced from the Stoody agent's
    `strokes` collection in the tenant DB. This lets the SkillBot frontend load the
    same stroke memory that the Stoody BLE agent writes.
    """
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

    # Tenant isolation via per-tenant DB
    user_type = current_user.get("user_type")
    if user_type in ["admin", "b2c_admin"] and not has_permission(current_user, "view_strokes"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

    if user_type == "student":
        # Students can only see their own strokes
        valid_user_ids = [current_user["user_id"]]
    else:
        # Admin/tutor: Get all student user_ids for this tenant
        cursor = tenant_db["students"].find({}, {"_id": 1})
        tenant_students = await cursor.to_list(length=10000)
        valid_user_ids = [str(s["_id"]) for s in tenant_students]

        if not valid_user_ids:
            # No students in this tenant - return empty
            return StrokeListResponse(count=0, strokes=[])

    query: Dict[str, Any] = {"user_id": {"$in": valid_user_ids}}
    if pen_mac:
        query["pen_mac"] = pen_mac.upper()
    if page_number is not None:
        query["page_number"] = page_number
    if before or after:
        ts: Dict[str, Any] = {}
        if before:
            ts["$lt"] = before
        if after:
            ts["$gt"] = after
        query["timestamp"] = ts

    try:
        cursor = (
            tenant_db["strokes"]
            .find(query)
            .sort("timestamp", -1)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        items = [_serialize(d) for d in docs]
        return StrokeListResponse(count=len(items), strokes=items)
    except Exception as exc:
        logger.error("Failed to fetch strokes: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch strokes",
        ) from exc


@router.post("/ingest")
async def ingest_strokes(
    request: Request,
    payload: StrokeIngestRequest,
    db: DatabaseManager = Depends(get_database),
):
    token = request.headers.get("X-Pen-Token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing pen token",
        )

    token_data = decode_pen_token(token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired pen token",
        )

    db_name = token_data.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid pen token context",
        )

    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )

    pen_doc = await tenant_db["pen_tokens"].find_one({
        "token": token,
        "active": True
    })
    if not pen_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pen token not registered",
        )

    expires_at = pen_doc.get("expires_at")
    if expires_at and expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pen token expired",
        )

    if pen_doc.get("pen_mac") and pen_doc["pen_mac"] != payload.pen_mac:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Pen not authorized for this token",
        )

    if not pen_doc.get("pen_mac"):
        await tenant_db["pen_tokens"].update_one(
            {"_id": pen_doc["_id"]},
            {"$set": {"pen_mac": payload.pen_mac}}
        )

    stroke_doc = {
        "session_id": payload.session_id,
        "pen_mac": payload.pen_mac,
        "page_number": payload.page_number,
        "timestamp": payload.timestamp or datetime.utcnow(),
        "strokes": payload.strokes,
        "book_type": payload.book_type,
        "canvas_background": payload.canvas_background,
        "page_style": payload.page_style,
        "user_id": token_data.get("sub"),
    }

    await tenant_db["strokes"].insert_one(stroke_doc)

    return {"success": True}

