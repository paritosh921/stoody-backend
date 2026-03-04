"""
Async strokes API for SkillBot
Reads stroke data produced by the Stoody BLE agent (tenant Mongo `strokes` collection).
Also provides canvas page persistence endpoints for server-side stroke storage.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from pymongo import ReplaceOne

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


# ─── Canvas Pages — server-side stroke persistence ─────────────────────────────
# One document per (user_id, book_type, page_number). The pen is just an input
# device; pen_mac is optional metadata, NOT part of the unique key.

class CanvasPageStroke(BaseModel):
    id: str
    points: List[List[float]]
    strokeWidth: float = 1.3
    color: str = "#000000"
    tool: str = "pen"
    timestamp: Optional[float] = None
    svgPath: Optional[str] = None


class CanvasPageUpsert(BaseModel):
    book_type: str = Field(..., min_length=1, max_length=10)
    page_number: int = Field(..., ge=0)
    strokes: List[CanvasPageStroke]
    page_style: Optional[str] = None
    canvas_background: Optional[str] = None
    stroke_count: Optional[int] = None
    pen_mac: Optional[str] = None
    client_last_modified: Optional[float] = None
    version: Optional[int] = None

    @field_validator("stroke_count", mode="before")
    @classmethod
    def _default_stroke_count(cls, v, info):
        if v is not None:
            return v
        strokes = info.data.get("strokes") if info.data else []
        return len(strokes or [])


class CanvasPageBatchRequest(BaseModel):
    pages: List[CanvasPageUpsert] = Field(..., max_length=20)


class BulkLoadRequest(BaseModel):
    pages: List[Dict[str, Any]] = Field(
        ...,
        max_length=50,
        description="List of {book_type, page_number} dicts",
    )


class CanvasPageMeta(BaseModel):
    book_type: str
    page_number: int
    stroke_count: int = 0
    last_modified: Optional[str] = None
    client_last_modified: Optional[float] = None
    version: int = 1


async def _get_canvas_collection(
    current_user: Dict[str, Any], db: DatabaseManager
):
    """Return the correct canvas_pages collection (tenant or B2C)."""
    is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
    if is_b2c:
        b2c_db = await db.get_b2c_db()
        if b2c_db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="B2C database not available",
            )
        return b2c_db["canvas_pages"]

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
    return tenant_db["canvas_pages"]


def _page_doc(user_id: str, admin_id: Optional[str], page: CanvasPageUpsert, now: datetime) -> Dict[str, Any]:
    """Build the MongoDB document for a canvas page upsert."""
    return {
        "user_id": user_id,
        "admin_id": admin_id,
        "book_type": page.book_type.upper(),
        "page_number": page.page_number,
        "strokes": [s.model_dump() for s in page.strokes],
        "page_style": page.page_style,
        "canvas_background": page.canvas_background,
        "stroke_count": page.stroke_count,
        "pen_mac": page.pen_mac,
        "last_modified": now,
        "client_last_modified": page.client_last_modified,
        "version": (page.version or 0) + 1,
    }


@router.put("/pages")
async def upsert_canvas_page(
    page: CanvasPageUpsert,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Upsert a single canvas page. Identity key: (user_id, book_type, page_number)."""
    user_id = current_user["user_id"]
    admin_id = current_user.get("admin_id")
    collection = await _get_canvas_collection(current_user, db)

    now = datetime.now(timezone.utc)
    filt = {
        "user_id": user_id,
        "book_type": page.book_type.upper(),
        "page_number": page.page_number,
    }
    doc = _page_doc(user_id, admin_id, page, now)

    # Optimistic concurrency: if client sends a version, require it to match
    if page.version is not None:
        filt["version"] = page.version

    result = await collection.replace_one(filt, doc, upsert=True)

    if result.matched_count == 0 and result.upserted_id is None and page.version is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Version conflict — page was modified by another session",
        )

    return {
        "success": True,
        "version": doc["version"],
        "last_modified": now.isoformat(),
    }


@router.post("/pages/batch")
async def batch_upsert_canvas_pages(
    body: CanvasPageBatchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Upsert up to 20 canvas pages in one request."""
    user_id = current_user["user_id"]
    admin_id = current_user.get("admin_id")
    collection = await _get_canvas_collection(current_user, db)

    now = datetime.now(timezone.utc)
    ops = []
    for page in body.pages:
        filt = {
            "user_id": user_id,
            "book_type": page.book_type.upper(),
            "page_number": page.page_number,
        }
        doc = _page_doc(user_id, admin_id, page, now)
        ops.append(ReplaceOne(filt, doc, upsert=True))

    if ops:
        result = await collection.bulk_write(ops, ordered=False)
        return {
            "success": True,
            "upserted": result.upserted_count,
            "modified": result.modified_count,
        }
    return {"success": True, "upserted": 0, "modified": 0}


@router.get("/pages")
async def list_canvas_pages(
    book_type: Optional[str] = Query(None, description="Filter by book type"),
    since: Optional[str] = Query(None, description="ISO datetime — only pages modified after this"),
    limit: int = Query(200, ge=1, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """List page metadata (no strokes) for the current user. Supports incremental sync via `since`."""
    user_id = current_user["user_id"]
    collection = await _get_canvas_collection(current_user, db)

    query: Dict[str, Any] = {"user_id": user_id}
    if book_type:
        query["book_type"] = book_type.upper()
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            query["last_modified"] = {"$gt": since_dt}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid `since` datetime format")

    projection = {
        "strokes": 0,  # Exclude the heavy strokes array
    }
    cursor = (
        collection.find(query, projection)
        .sort("last_modified", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)

    pages = []
    for d in docs:
        pages.append({
            "book_type": d.get("book_type"),
            "page_number": d.get("page_number"),
            "stroke_count": d.get("stroke_count", 0),
            "last_modified": d.get("last_modified", "").isoformat() if d.get("last_modified") else None,
            "client_last_modified": d.get("client_last_modified"),
            "version": d.get("version", 1),
        })

    return {"count": len(pages), "pages": pages}


@router.get("/pages/{book_type}/{page_number}")
async def get_canvas_page(
    book_type: str,
    page_number: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Load a single canvas page with full strokes."""
    user_id = current_user["user_id"]
    collection = await _get_canvas_collection(current_user, db)

    doc = await collection.find_one({
        "user_id": user_id,
        "book_type": book_type.upper(),
        "page_number": page_number,
    })
    if not doc:
        raise HTTPException(status_code=404, detail="Page not found")

    doc.pop("_id", None)
    if isinstance(doc.get("last_modified"), datetime):
        doc["last_modified"] = doc["last_modified"].isoformat()
    return doc


@router.post("/pages/bulk-load")
async def bulk_load_canvas_pages(
    body: BulkLoadRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Load up to 50 pages in one request (full strokes included)."""
    user_id = current_user["user_id"]
    collection = await _get_canvas_collection(current_user, db)

    or_clauses = []
    for p in body.pages:
        bt = p.get("book_type")
        pn = p.get("page_number")
        if bt is not None and pn is not None:
            or_clauses.append({"book_type": str(bt).upper(), "page_number": int(pn)})

    if not or_clauses:
        return {"pages": []}

    query = {"user_id": user_id, "$or": or_clauses}
    cursor = collection.find(query)
    docs = await cursor.to_list(length=len(or_clauses))

    pages = []
    for d in docs:
        d.pop("_id", None)
        if isinstance(d.get("last_modified"), datetime):
            d["last_modified"] = d["last_modified"].isoformat()
        pages.append(d)

    return {"pages": pages}

