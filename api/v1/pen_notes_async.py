"""
Read-only endpoints for the frontend to fetch pen notes from the shared MongoDB.

Pen notes are written by the pen server (stoody-ble-agent/server/) into the
per-tenant database's `pen_notes` collection.  Both servers share the same
MongoDB Atlas cluster, so the main backend can read directly — no proxy needed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pen-notes", tags=["pen-notes"])


@router.get("")
async def list_pen_notes(
    date_from: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: int = Query(50, ge=1, le=200),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """List pen note summaries by date (page counts, no stroke data)."""
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

    user_id = current_user["user_id"]
    user_type = current_user.get("user_type")

    # Students see only their own notes; admins/tutors see all in tenant
    query: Dict[str, Any] = {}
    if user_type == "student":
        query["user_id"] = user_id

    if date_from or date_to:
        date_filter: Dict[str, str] = {}
        if date_from:
            date_filter["$gte"] = date_from
        if date_to:
            date_filter["$lte"] = date_to
        query["date"] = date_filter

    pen_notes = tenant_db["pen_notes"]
    cursor = pen_notes.find(
        query,
        {"pages.strokes": 0},  # Exclude stroke data for summaries
    ).sort("date", -1).limit(limit)

    docs = await cursor.to_list(length=limit)
    for doc in docs:
        doc["id"] = str(doc.pop("_id"))
        for page in doc.get("pages", []):
            page.pop("strokes", None)

    total = await pen_notes.count_documents(query)
    return {"notes": docs, "total": total}


@router.get("/{note_id}")
async def get_pen_note(
    note_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get a full pen note with all pages and strokes."""
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

    user_id = current_user["user_id"]
    user_type = current_user.get("user_type")
    pen_notes = tenant_db["pen_notes"]

    try:
        oid = ObjectId(note_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid note ID")

    query: Dict[str, Any] = {"_id": oid}
    if user_type == "student":
        query["user_id"] = user_id

    doc = await pen_notes.find_one(query)
    if not doc:
        raise HTTPException(status_code=404, detail="Note not found")

    doc["id"] = str(doc.pop("_id"))
    return doc


@router.get("/{note_id}/pages/{page_number}")
async def get_pen_note_page(
    note_id: str,
    page_number: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get strokes for a specific page within a pen note."""
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

    user_id = current_user["user_id"]
    user_type = current_user.get("user_type")
    pen_notes = tenant_db["pen_notes"]

    try:
        oid = ObjectId(note_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid note ID")

    query: Dict[str, Any] = {"_id": oid}
    if user_type == "student":
        query["user_id"] = user_id

    doc = await pen_notes.find_one(
        query,
        {"pages": {"$elemMatch": {"page_number": page_number}}},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Note not found")

    pages = doc.get("pages", [])
    if not pages:
        raise HTTPException(status_code=404, detail=f"Page {page_number} not found")

    return pages[0]
