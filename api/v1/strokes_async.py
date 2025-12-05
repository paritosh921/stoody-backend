"""
Async strokes API for SkillBot
Reads stroke data produced by the Stoody BLE agent (shared Mongo `strokes` collection).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager

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
    `strokes` collection. This lets the SkillBot frontend load the same stroke memory
    that the Stoody BLE agent writes.
    """
    if not db.mongo_db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB not configured",
        )

    query: Dict[str, Any] = {"user_id": current_user["user_id"]}
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
            db.mongo_db["strokes"]
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

