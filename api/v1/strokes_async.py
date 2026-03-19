"""
Async strokes API for SkillBot
Reads stroke data produced by the Stoody BLE agent (tenant Mongo `strokes` collection).
Also provides canvas page persistence endpoints for server-side stroke storage.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from pymongo import ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError, DuplicateKeyError

from api.v1.auth_async import get_current_user, get_database
from api.v1.copy_sets_async import resolve_copy_id as _resolve_copy_id
from core.database import DatabaseManager
from core.permissions import has_permission
from core.pen_tokens import decode_pen_token
from core.user_identity import canonical_canvas_user_id

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
        valid_user_ids = _resolve_canvas_user_ids(current_user)
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

    # Fire-and-forget: queue page for AI classification (60s debounce)
    # Legacy ingest path uses pen token auth — copy_id not available here.
    # The canvas_pages upsert path (which is the primary flow) passes copy_id.
    try:
        from services.note_classification_service import queue_classification
        asyncio.create_task(
            queue_classification(
                db, db_name, token_data.get("sub"),
                payload.pen_mac, payload.book_type, payload.page_number,
            )
        )
    except Exception:
        pass  # classification is best-effort, never block stroke ingestion

    return {"success": True}


# ─── Canvas Pages — server-side stroke persistence ─────────────────────────────
# One document per (user_id, book_type, page_number). The pen is just an input
# device; pen_mac is optional metadata, NOT part of the unique key.

class CanvasPageStroke(BaseModel):
    id: str
    points: List[Any]  # Accept both [[x,y,p,...]] arrays and [{x,y,pressure}] dicts
    strokeWidth: float = 1.3
    color: str = "#000000"
    tool: str = "pen"
    timestamp: Optional[float] = None
    svgPath: Optional[str] = None
    baseWidthMm: Optional[float] = None
    sourceMode: Optional[str] = None
    startedAt: Optional[float] = None
    endedAt: Optional[float] = None
    pageNumber: Optional[int] = None
    bookType: Optional[str] = None

    @field_validator("points", mode="before")
    @classmethod
    def _normalise_points(cls, v: Any) -> list:
        """Accept both [{x,y,pressure}] dicts and [[x,y,p]] arrays."""
        if not isinstance(v, list):
            return v
        out = []
        for pt in v:
            if isinstance(pt, dict):
                # Convert {x, y, pressure, ...} → [x, y, pressure, ...]
                x = pt.get("x", 0)
                y = pt.get("y", 0)
                p = pt.get("pressure", 0.5)
                arr = [x, y, p]
                # Preserve extra fields if present (tiltX, tiltY, timestamp)
                for extra in ("tiltX", "tiltY", "timestamp"):
                    if extra in pt:
                        arr.append(pt[extra])
                out.append(arr)
            else:
                out.append(pt)
        return out


class CanvasPageUpsert(BaseModel):
    book_type: str = Field(..., min_length=1, max_length=10)
    page_number: int = Field(..., ge=0)
    strokes: List[CanvasPageStroke]
    copy_id: Optional[str] = None
    page_style: Optional[str] = None
    canvas_background: Optional[str] = None
    stroke_count: Optional[int] = None
    pen_mac: Optional[str] = None
    source: Optional[str] = None
    client_last_modified: Optional[float] = None
    version: Optional[int] = None
    session_id: Optional[str] = None
    first_activity: Optional[float] = None
    last_activity: Optional[float] = None
    device_id: Optional[str] = None

    @field_validator("stroke_count", mode="before")
    @classmethod
    def _default_stroke_count(cls, v, info):
        strokes = info.data.get("strokes") if info.data else []
        computed = len(strokes or [])
        # Trust the computed length over client-provided value.
        # If client provides a valid positive count and no strokes are available
        # in info.data yet, fall back to the provided value.
        if computed > 0:
            return computed
        if v is not None and v > 0:
            return v
        return computed


class CanvasPageBatchRequest(BaseModel):
    pages: List[CanvasPageUpsert] = Field(..., max_length=20)


class BulkLoadRequest(BaseModel):
    pages: List[Dict[str, Any]] = Field(
        ...,
        max_length=50,
        description="List of {book_type, page_number, copy_id?} dicts",
    )
    copy_id: Optional[str] = None


class CanvasPageMeta(BaseModel):
    book_type: str
    page_number: int
    copy_id: Optional[str] = None
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


def _resolve_canvas_user_ids(current_user: Dict[str, Any]) -> List[Any]:
    """Return all known user-id variants used by legacy and migrated pen data."""
    user_id = current_user["user_id"]
    from bson import ObjectId as _ObjectId

    username = current_user.get("username")
    user_ids: list[Any] = []
    if username:
        user_ids.append(username)
    user_ids.extend([user_id, str(user_id)])
    try:
        if _ObjectId.is_valid(user_id):
            user_ids.append(_ObjectId(user_id))
    except Exception:
        pass

    deduped: list[Any] = []
    seen: set[str] = set()
    for item in user_ids:
        marker = f"{type(item).__name__}:{item}"
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


_canonical_canvas_user_id = canonical_canvas_user_id  # back-compat alias


def _extract_duplicate_index_details(exc: Exception) -> tuple[Optional[str], Optional[dict]]:
    """Return duplicate-index name and key values without leaking the full failed op."""
    if isinstance(exc, DuplicateKeyError):
        details = getattr(exc, "details", {}) or {}
        errmsg = str(details.get("errmsg") or exc)
        index_name = None
        if " index: " in errmsg:
            index_name = errmsg.split(" index: ", 1)[1].split(" dup key:", 1)[0].strip()
        return index_name, details.get("keyValue")

    if isinstance(exc, BulkWriteError):
        details = getattr(exc, "details", {}) or {}
        write_errors = details.get("writeErrors") or []
        if write_errors:
            first = write_errors[0] or {}
            errmsg = str(first.get("errmsg") or exc)
            index_name = None
            if " index: " in errmsg:
                index_name = errmsg.split(" index: ", 1)[1].split(" dup key:", 1)[0].strip()
            return index_name, first.get("keyValue")

    return None, None


def _raise_sanitized_canvas_write_error(exc: Exception) -> None:
    """Map duplicate-key write failures to a concise HTTP error without logging stroke payloads."""
    index_name, key_value = _extract_duplicate_index_details(exc)
    if index_name:
        logger.warning(
            "Canvas page write failed due to duplicate index conflict on %s with key=%s",
            index_name,
            key_value,
        )
        if index_name in {"uniq_canvas_page", "uniq_canvas_pages_user_book_page"}:
            detail = (
                "Canvas page write blocked by legacy Mongo index without copy_id. "
                "Run the copy-set migration and drop the legacy canvas_pages unique index."
            )
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail) from exc

    if isinstance(exc, (DuplicateKeyError, BulkWriteError)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Canvas page write failed due to a duplicate key conflict.",
        ) from exc

    raise exc


def _page_doc(
    user_id: str,
    admin_id: Optional[str],
    page: CanvasPageUpsert,
    now: datetime,
    *,
    copy_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the MongoDB document for a canvas page upsert."""
    doc: Dict[str, Any] = {
        "user_id": user_id,
        "admin_id": admin_id,
        "copy_id": copy_id or page.copy_id,
        "book_type": page.book_type.upper(),
        "page_number": page.page_number,
        "strokes": [s.model_dump() for s in page.strokes],
        "page_style": page.page_style,
        "canvas_background": page.canvas_background,
        "stroke_count": page.stroke_count,
        "pen_mac": page.pen_mac,
        "source": page.source,
        "last_modified": now,
        "client_last_modified": page.client_last_modified,
        "version": (page.version or 0) + 1,
    }
    if page.session_id:
        doc["session_id"] = page.session_id
    if page.first_activity is not None:
        doc["first_activity"] = page.first_activity
    if page.last_activity is not None:
        doc["last_activity"] = page.last_activity
    if page.device_id:
        doc["device_id"] = page.device_id
    return doc


def _incoming_stroke_docs(page: CanvasPageUpsert) -> List[Dict[str, Any]]:
    return [s.model_dump() for s in page.strokes]


def _merge_stroke_docs(
    existing_strokes: List[Dict[str, Any]],
    incoming_strokes: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    seen_ids = {
        str(stroke.get("id") or "")
        for stroke in existing_strokes
        if stroke.get("id")
    }
    merged = list(existing_strokes)
    added = 0
    for stroke in incoming_strokes:
        stroke_id = str(stroke.get("id") or "")
        if stroke_id and stroke_id in seen_ids:
            continue
        if stroke_id:
            seen_ids.add(stroke_id)
        merged.append(stroke)
        added += 1
    return merged, added


def _is_stale_canvas_page_update(existing_doc: Dict[str, Any], page: CanvasPageUpsert) -> bool:
    existing_version = int(existing_doc.get("version", 0) or 0)
    if page.version is not None:
        # Canvas page writes are additive. Equal-version writes are still
        # conflict-prone because the client may have loaded only a subset of
        # strokes for that page. Merge on >= so same-version saves do not
        # replace older strokes that were already present on the server.
        return existing_version >= page.version

    existing_last_modified = existing_doc.get("client_last_modified")
    incoming_last_modified = page.client_last_modified
    if existing_last_modified is not None and incoming_last_modified is not None:
        return float(existing_last_modified) > float(incoming_last_modified)

    # Versionless writes are treated as stale/merge-only by default when a page already exists.
    return True


def _build_merged_page_doc(
    *,
    existing_doc: Dict[str, Any],
    user_id: str,
    admin_id: Optional[str],
    page: CanvasPageUpsert,
    now: datetime,
    copy_id: Optional[str] = None,
) -> tuple[Dict[str, Any], int]:
    incoming_strokes = _incoming_stroke_docs(page)
    merged_strokes, added_count = _merge_stroke_docs(
        list(existing_doc.get("strokes") or []),
        incoming_strokes,
    )

    first_activity_candidates = [
        value for value in (existing_doc.get("first_activity"), page.first_activity)
        if value is not None
    ]
    last_activity_candidates = [
        value for value in (existing_doc.get("last_activity"), page.last_activity)
        if value is not None
    ]
    client_last_modified_candidates = [
        value for value in (existing_doc.get("client_last_modified"), page.client_last_modified)
        if value is not None
    ]

    resolved_copy_id = copy_id or page.copy_id or existing_doc.get("copy_id")
    merged_doc: Dict[str, Any] = {
        "user_id": user_id,
        "admin_id": admin_id,
        "copy_id": resolved_copy_id,
        "book_type": page.book_type.upper(),
        "page_number": page.page_number,
        "strokes": merged_strokes,
        "page_style": page.page_style if page.page_style is not None else existing_doc.get("page_style"),
        "canvas_background": (
            page.canvas_background
            if page.canvas_background is not None
            else existing_doc.get("canvas_background")
        ),
        "stroke_count": len(merged_strokes),
        "pen_mac": page.pen_mac or existing_doc.get("pen_mac"),
        "source": page.source or existing_doc.get("source"),
        "last_modified": now,
        "client_last_modified": (
            max(client_last_modified_candidates)
            if client_last_modified_candidates else None
        ),
        "version": int(existing_doc.get("version", 0) or 0) + 1,
    }
    session_id = page.session_id or existing_doc.get("session_id")
    if session_id:
        merged_doc["session_id"] = session_id
    if first_activity_candidates:
        merged_doc["first_activity"] = min(first_activity_candidates)
    if last_activity_candidates:
        merged_doc["last_activity"] = max(last_activity_candidates)
    device_id = page.device_id or existing_doc.get("device_id")
    if device_id:
        merged_doc["device_id"] = device_id
    return merged_doc, added_count


def _build_metadata_refresh(
    existing_doc: Dict[str, Any],
    page: CanvasPageUpsert,
    now: datetime,
) -> Dict[str, Any]:
    """Build a $set dict for audit/activity fields when no new strokes were added.

    Returns an empty dict if nothing needs updating.
    """
    updates: Dict[str, Any] = {"last_modified": now}

    if page.device_id and page.device_id != existing_doc.get("device_id"):
        updates["device_id"] = page.device_id
    if page.session_id and page.session_id != existing_doc.get("session_id"):
        updates["session_id"] = page.session_id
    if page.first_activity is not None:
        existing_first = existing_doc.get("first_activity")
        if existing_first is None or page.first_activity < existing_first:
            updates["first_activity"] = page.first_activity
    if page.last_activity is not None:
        existing_last = existing_doc.get("last_activity")
        if existing_last is None or page.last_activity > existing_last:
            updates["last_activity"] = page.last_activity
    if page.client_last_modified is not None:
        existing_clm = existing_doc.get("client_last_modified")
        if existing_clm is None or page.client_last_modified > existing_clm:
            updates["client_last_modified"] = page.client_last_modified

    return updates


async def _upsert_notes_canvas_classification(
    classification_collection,
    *,
    user_id: str,
    user_ids: List[Any],
    page: CanvasPageUpsert,
    now: datetime,
    copy_id: Optional[str] = None,
) -> None:
    pen_mac = (page.pen_mac or "").upper() or None
    resolved_copy_id = copy_id or page.copy_id

    stroke_count = page.stroke_count if page.stroke_count is not None else len(page.strokes or [])
    identity_fields: Dict[str, Any] = {
        "user_id": user_id,
        "pen_mac": pen_mac,
        "book_type": page.book_type.upper(),
        "page_number": page.page_number,
    }
    if resolved_copy_id:
        identity_fields["copy_id"] = resolved_copy_id
    mutable_fields: Dict[str, Any] = {
        "stroke_count_at_classification": stroke_count,
        "updated_at": now,
        "last_activity": page.last_activity if page.last_activity is not None else now,
    }
    if page.session_id:
        mutable_fields["session_id"] = page.session_id
    if page.first_activity is not None:
        mutable_fields["first_activity"] = page.first_activity

    set_on_insert: Dict[str, Any] = {
        **identity_fields,
        "subject": "Unorganised",
        "topic": "General",
        "classification_source": "system",
        "confidence": 0.0,
        "ocr_text": "",
        "thumbnail_url": None,
        "is_favorite": False,
        "is_archived": False,
        "created_at": now,
        "original_subject": None,
        "original_topic": None,
    }

    # Find existing with $in (legacy user_id compat) + pen_mac to match unique index
    try:
        find_query: Dict[str, Any] = {
            "user_id": {"$in": user_ids},
            "book_type": page.book_type.upper(),
            "page_number": page.page_number,
        }
        if resolved_copy_id:
            find_query["copy_id"] = resolved_copy_id
        if pen_mac:
            find_query["pen_mac"] = pen_mac

        existing = await classification_collection.find_one(find_query, {"_id": 1, "user_id": 1})
        if existing:
            if existing.get("user_id") != user_id:
                # Old user_id variant — a canonical doc may already exist.
                # Delete the old-variant doc and upsert on the canonical one.
                await classification_collection.delete_one({"_id": existing["_id"]})
                await classification_collection.update_one(
                    identity_fields,
                    {
                        "$setOnInsert": set_on_insert,
                        "$set": mutable_fields,
                    },
                    upsert=True,
                )
            else:
                await classification_collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": mutable_fields},
                )
        else:
            await classification_collection.update_one(
                identity_fields,
                {
                    "$setOnInsert": set_on_insert,
                    "$set": mutable_fields,
                },
                upsert=True,
            )
    except DuplicateKeyError:
        # Race condition or leftover duplicates — just update the canonical doc
        try:
            await classification_collection.update_one(
                identity_fields,
                {"$set": mutable_fields},
            )
        except Exception:
            pass  # Classification is non-critical, don't crash the caller
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("note_classifications upsert failed: %s", exc)


@router.put("/pages")
async def upsert_canvas_page(
    page: CanvasPageUpsert,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Upsert a single canvas page. Identity key: (user_id, copy_id, book_type, page_number)."""
    user_id = _canonical_canvas_user_id(current_user)
    admin_id = current_user.get("admin_id")
    collection = await _get_canvas_collection(current_user, db)
    classification_collection = collection.database["note_classifications"]
    user_ids = _resolve_canvas_user_ids(current_user)

    # Resolve copy_id: explicit in payload → user's active copy → auto-create default
    copy_id = await _resolve_copy_id(page.copy_id, current_user, db)

    now = datetime.now(timezone.utc)
    page_filter: Dict[str, Any] = {
        "user_id": {"$in": user_ids},
        "copy_id": copy_id,
        "book_type": page.book_type.upper(),
        "page_number": page.page_number,
    }
    existing = await collection.find_one(page_filter)

    if existing is None:
        doc = _page_doc(user_id, admin_id, page, now, copy_id=copy_id)
        upsert_filt: Dict[str, Any] = {
            "user_id": user_id,
            "copy_id": copy_id,
            "book_type": page.book_type.upper(),
            "page_number": page.page_number,
        }
        try:
            result = await collection.replace_one(upsert_filt, doc, upsert=True)
        except DuplicateKeyError as exc:
            _raise_sanitized_canvas_write_error(exc)
    else:
        doc, added_count = _build_merged_page_doc(
            existing_doc=existing,
            user_id=user_id,
            admin_id=admin_id,
            page=page,
            now=now,
            copy_id=copy_id,
        )
        if added_count == 0:
            # No new strokes, but still refresh audit/activity metadata
            metadata_update = _build_metadata_refresh(existing, page, now)
            if metadata_update:
                await collection.update_one({"_id": existing["_id"]}, {"$set": metadata_update})
            await _upsert_notes_canvas_classification(
                classification_collection,
                user_id=user_id,
                user_ids=user_ids,
                page=page,
                now=now,
                copy_id=copy_id,
            )
            return {
                "success": True,
                "version": int(existing.get("version", 1) or 1),
                "last_modified": now.isoformat(),
            }
        try:
            result = await collection.replace_one({"_id": existing["_id"]}, doc)
        except DuplicateKeyError as exc:
            _raise_sanitized_canvas_write_error(exc)

    await _upsert_notes_canvas_classification(
        classification_collection,
        user_id=user_id,
        user_ids=user_ids,
        page=page,
        now=now,
        copy_id=copy_id,
    )

    return {
        "success": True,
        "version": doc["version"],
        "last_modified": now.isoformat(),
        "copy_id": copy_id,
    }


@router.post("/pages/batch")
async def batch_upsert_canvas_pages(
    body: CanvasPageBatchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Upsert up to 20 canvas pages in one request."""
    user_id = _canonical_canvas_user_id(current_user)
    admin_id = current_user.get("admin_id")
    collection = await _get_canvas_collection(current_user, db)
    classification_collection = collection.database["note_classifications"]
    user_ids = _resolve_canvas_user_ids(current_user)

    # Resolve copy_id once for the entire batch.  Individual pages may
    # carry their own copy_id but when absent the batch defaults to the
    # user's active copy.
    batch_copy_id = await _resolve_copy_id(
        body.pages[0].copy_id if body.pages else None,
        current_user,
        db,
    )

    now = datetime.now(timezone.utc)

    # Pre-fetch existing docs to handle legacy user_id variants.
    page_keys = [
        (p.copy_id or batch_copy_id, p.book_type.upper(), p.page_number) for p in body.pages
    ]
    existing_map: Dict[tuple, Any] = {}
    if page_keys:
        or_clauses = [
            {"copy_id": cid, "book_type": bt, "page_number": pn} for cid, bt, pn in page_keys
        ]
        existing_cursor = collection.find(
            {"user_id": {"$in": user_ids}, "$or": or_clauses},
            {
                "_id": 1,
                "copy_id": 1,
                "book_type": 1,
                "page_number": 1,
                "strokes": 1,
                "version": 1,
                "client_last_modified": 1,
                "page_style": 1,
                "canvas_background": 1,
                "pen_mac": 1,
                "source": 1,
                "session_id": 1,
                "first_activity": 1,
                "last_activity": 1,
                "device_id": 1,
            },
        )
        async for edoc in existing_cursor:
            existing_map[(edoc.get("copy_id"), edoc["book_type"], edoc["page_number"])] = edoc

    ops = []
    changed_pages: List[tuple[CanvasPageUpsert, str]] = []  # (page, resolved_copy_id)
    for page in body.pages:
        copy_id = page.copy_id or batch_copy_id
        key = (copy_id, page.book_type.upper(), page.page_number)
        existing = existing_map.get(key)
        if existing is None:
            doc = _page_doc(user_id, admin_id, page, now, copy_id=copy_id)
            filt = {
                "user_id": user_id,
                "copy_id": copy_id,
                "book_type": page.book_type.upper(),
                "page_number": page.page_number,
            }
            ops.append(ReplaceOne(filt, doc, upsert=True))
            changed_pages.append((page, copy_id))
        else:
            doc, added_count = _build_merged_page_doc(
                existing_doc=existing,
                user_id=user_id,
                admin_id=admin_id,
                page=page,
                now=now,
                copy_id=copy_id,
            )
            if added_count == 0:
                # No new strokes — still refresh audit/activity metadata
                metadata_update = _build_metadata_refresh(existing, page, now)
                if metadata_update:
                    ops.append(UpdateOne({"_id": existing["_id"]}, {"$set": metadata_update}))
                changed_pages.append((page, copy_id))
                continue
            ops.append(ReplaceOne({"_id": existing["_id"]}, doc))
            changed_pages.append((page, copy_id))

    if ops:
        try:
            result = await collection.bulk_write(ops, ordered=False)
        except BulkWriteError as exc:
            _raise_sanitized_canvas_write_error(exc)
        for page, page_copy_id in changed_pages:
            await _upsert_notes_canvas_classification(
                classification_collection,
                user_id=user_id,
                user_ids=user_ids,
                page=page,
                now=now,
                copy_id=page_copy_id,
            )
        return {
            "success": True,
            "upserted": result.upserted_count,
            "modified": result.modified_count,
        }
    return {"success": True, "upserted": 0, "modified": 0}


@router.get("/pages")
async def list_canvas_pages(
    copy_id: Optional[str] = Query(None, description="Filter by copy set ID"),
    book_type: Optional[str] = Query(None, description="Filter by book type"),
    since: Optional[str] = Query(None, description="ISO datetime — only pages modified after this"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """List page metadata for the current user from canvas_pages."""
    user_ids = _resolve_canvas_user_ids(current_user)
    collection = await _get_canvas_collection(current_user, db)

    query: Dict[str, Any] = {"user_id": {"$in": user_ids}}
    if copy_id:
        query["copy_id"] = copy_id
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
    fetch_window = min(max(limit + offset, limit), 5000)
    cursor = (
        collection.find(query, projection)
        .sort("last_modified", -1)
        .limit(fetch_window)
    )
    docs = await cursor.to_list(length=fetch_window)

    pages = []
    for d in docs:
        page_meta: Dict[str, Any] = {
            "copy_id": d.get("copy_id"),
            "book_type": d.get("book_type"),
            "page_number": d.get("page_number"),
            "stroke_count": d.get("stroke_count", 0),
            "last_modified": d.get("last_modified", "").isoformat() if d.get("last_modified") else None,
            "client_last_modified": d.get("client_last_modified"),
            "version": d.get("version", 1),
        }
        if d.get("session_id"):
            page_meta["session_id"] = d["session_id"]
        if d.get("first_activity") is not None:
            page_meta["first_activity"] = d["first_activity"]
        if d.get("last_activity") is not None:
            page_meta["last_activity"] = d["last_activity"]
        pages.append(page_meta)

    # Sort by last_modified descending and paginate
    pages.sort(
        key=lambda p: p.get("last_modified") or "",
        reverse=True,
    )
    sliced = pages[offset: offset + limit]

    return {"count": len(sliced), "pages": sliced}


@router.get("/pages/{book_type}/{page_number}")
async def get_canvas_page(
    book_type: str,
    page_number: int,
    copy_id: Optional[str] = Query(None, description="Copy set ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Load a single canvas page with full strokes."""
    user_ids = _resolve_canvas_user_ids(current_user)
    collection = await _get_canvas_collection(current_user, db)

    page_filter: Dict[str, Any] = {
        "user_id": {"$in": user_ids},
        "book_type": book_type.upper(),
        "page_number": page_number,
    }
    if copy_id:
        page_filter["copy_id"] = copy_id
    doc = await collection.find_one(page_filter)

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
    user_ids = _resolve_canvas_user_ids(current_user)
    collection = await _get_canvas_collection(current_user, db)

    or_clauses = []
    for p in body.pages:
        bt = p.get("book_type")
        pn = p.get("page_number")
        if bt is not None and pn is not None:
            clause: Dict[str, Any] = {"book_type": str(bt).upper(), "page_number": int(pn)}
            cid = p.get("copy_id") or body.copy_id
            if cid:
                clause["copy_id"] = cid
            or_clauses.append(clause)

    if not or_clauses:
        return {"pages": []}

    query: Dict[str, Any] = {"user_id": {"$in": user_ids}, "$or": or_clauses}
    cursor = collection.find(query)
    docs = await cursor.to_list(length=len(or_clauses))

    pages = []
    for d in docs:
        d.pop("_id", None)
        if isinstance(d.get("last_modified"), datetime):
            d["last_modified"] = d["last_modified"].isoformat()
        pages.append(d)

    return {"pages": pages}




