"""
Student Copies & Pins API - Async Routes

Provides endpoints for:
1. Listing student copy pages (written with smart pen)
2. Retrieving strokes for a specific copy page
3. Creating/managing pinned copies (PDF snapshots)

This integrates with the BLE agent's stroke storage to provide a "My Copy" 
feature in the Learning Mode of the student frontend.
"""

import asyncio
import logging
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
import httpx

from fastapi import APIRouter, HTTPException, Request, Depends, Query, status, Header
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database, get_auth_manager
from core.auth import AuthManager
from utils.s3_storage import upload_file, download_file as s3_download_file, get_public_url
from utils.stroke_pdf_generator import generate_copy_pdf, generate_copy_thumbnail, build_svg_from_strokes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/copies", tags=["Student Copies"])

DEFAULT_PEN_BACKEND_URL = "https://pkg5ywq5pcb5mby95dxirdffxi7oc2m7.stoody.in"


def _resolve_pen_backend_history_url() -> Optional[str]:
    """
    Resolve the pen backend strokes-history endpoint.
    Accepts either:
    - PEN_BACKEND_URL=https://host
    - PEN_BACKEND_URL=https://host/api/v1
    """
    base = (
        os.getenv("PEN_BACKEND_URL")
        or os.getenv("STOODY_PEN_BACKEND_URL")
        or os.getenv("VITE_PEN_BACKEND_URL")
        or DEFAULT_PEN_BACKEND_URL
    )
    if not base:
        return None

    normalized = base.rstrip("/")
    if normalized.endswith("/api/v1"):
        return f"{normalized}/agent/strokes/history"
    return f"{normalized}/api/v1/agent/strokes/history"


def _resolve_forward_auth_header(request: Optional[Request]) -> Optional[str]:
    """
    Resolve a bearer token to forward to pen backend fallback APIs.
    Prefers Authorization header; falls back to cookie-auth token.
    """
    if request is None:
        return None

    auth_header = request.headers.get("Authorization")
    if auth_header:
        return auth_header

    cookie_token = request.cookies.get("stoody_auth_token")
    if cookie_token:
        return f"Bearer {cookie_token}"

    return None


async def _fetch_remote_strokes(
    authorization_header: Optional[str],
    *,
    pen_mac: Optional[str] = None,
    page_number: Optional[int] = None,
    book_type: Optional[str] = None,
    limit: int = 2000,
    order: str = "desc",
) -> List[Dict[str, Any]]:
    """
    Fetch stroke batches from pen backend for the authenticated user.
    Used as a fallback path when local tenant DB strokes are unavailable/empty.
    """
    endpoint = _resolve_pen_backend_history_url()
    if not endpoint or not authorization_header:
        return []

    params: Dict[str, Any] = {
        "limit": max(1, min(int(limit), 5000)),
        "order": "asc" if order == "asc" else "desc",
    }
    if pen_mac:
        params["pen_mac"] = pen_mac.upper()
    if page_number is not None:
        params["page_number"] = page_number
    if book_type:
        params["book_type"] = book_type

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                endpoint,
                params=params,
                headers={"Authorization": authorization_header},
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Pen backend stroke fallback failed: %s %s",
                    resp.status_code,
                    resp.text[:200] if resp.text else "",
                )
                return []
            data = resp.json()
            strokes = data.get("strokes", [])
            return strokes if isinstance(strokes, list) else []
    except Exception as exc:
        logger.warning("Pen backend stroke fallback error: %s", exc)
        return []


# Custom dependency that accepts token from query param (for image sources)
async def get_current_user_from_token_or_query(
    request: Request,
    token: Optional[str] = Query(None, description="JWT token for image source auth"),
    authorization: Optional[str] = Header(None),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Dict[str, Any]:
    """
    Get current user from either Authorization header or query param token.
    This is needed for endpoints used in <img src=...> tags where headers can't be set.
    """
    # Try Authorization header first
    jwt_token = None
    if authorization and authorization.startswith("Bearer "):
        jwt_token = authorization[7:]
    elif token:
        jwt_token = token
    
    if not jwt_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Check if token is revoked
        from core.token_blacklist import token_blacklist
        if token_blacklist.is_revoked(jwt_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_data = await auth_manager.verify_token_and_get_user(jwt_token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CopyPageSummary(BaseModel):
    """Summary of a copy page (aggregated from stroke batches)"""
    pen_mac: str
    book_type: Optional[str] = None
    page_number: int
    total_strokes: int
    first_activity: datetime
    last_activity: datetime
    canvas_background: Optional[str] = None
    

class CopyPageListResponse(BaseModel):
    """Response for listing copy pages"""
    success: bool = True
    total: int
    pages: List[CopyPageSummary]


class CopyPageDetailResponse(BaseModel):
    """Response for a single copy page with strokes"""
    success: bool = True
    pen_mac: str
    book_type: Optional[str] = None
    page_number: int
    stroke_batches: List[Dict[str, Any]]
    total_strokes: int


class CreatePinRequest(BaseModel):
    """Request to pin a copy page"""
    pen_mac: str
    book_type: Optional[str] = "A5"
    page_number: int
    title: Optional[str] = None
    linked_document_id: Optional[str] = None
    linked_page: Optional[int] = None


class PinnedCopy(BaseModel):
    """A pinned copy (PDF snapshot)"""
    id: str
    user_id: str
    pen_mac: str
    book_type: Optional[str] = None
    page_number: int
    title: str
    pdf_path: str
    thumbnail_path: Optional[str] = None
    linked_document_id: Optional[str] = None
    linked_page: Optional[int] = None
    created_at: datetime


class PinnedCopyListResponse(BaseModel):
    """Response for listing pinned copies"""
    success: bool = True
    total: int
    pins: List[Dict[str, Any]]


# ============================================================================
# Helpers — canvas_pages fallback (server-side canvas sync collection)
# ============================================================================

async def _get_canvas_pages_collection(current_user: Dict[str, Any], db: DatabaseManager):
    """Return the canvas_pages collection for the user's tenant/B2C database."""
    is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
    if is_b2c:
        return db.b2c_db["canvas_pages"]
    db_name = current_user.get("db_name")
    if not db_name:
        return None
    tenant_db = await db.get_tenant_db(db_name)
    return tenant_db["canvas_pages"] if tenant_db is not None else None


async def _list_canvas_pages_for_user(
    current_user: Dict[str, Any],
    db: DatabaseManager,
    *,
    pen_mac: Optional[str] = None,
    book_type: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Query canvas_pages collection and return summaries compatible with CopyPageSummary."""
    col = await _get_canvas_pages_collection(current_user, db)
    if col is None:
        return []

    # Build all user_id variants for legacy compat
    user_id = current_user.get("user_id")
    user_identifiers: list = [user_id, str(user_id)]
    username = current_user.get("username")
    if username:
        user_identifiers.append(username)
    try:
        if ObjectId.is_valid(user_id):
            user_identifiers.append(ObjectId(user_id))
    except Exception:
        pass

    query: Dict[str, Any] = {"user_id": {"$in": user_identifiers}, "stroke_count": {"$gt": 0}}
    if pen_mac:
        query["pen_mac"] = pen_mac.upper()
    if book_type:
        query["book_type"] = book_type.upper()

    try:
        cursor = col.find(query, {"strokes": 0}).sort("last_modified", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
    except Exception as exc:
        logger.warning("canvas_pages query failed: %s", exc)
        return []

    results: List[Dict[str, Any]] = []
    for d in docs:
        pn = d.get("page_number")
        if pn is None:
            continue

        last_mod = d.get("last_modified")
        if isinstance(last_mod, datetime):
            last_activity = last_mod
        elif isinstance(last_mod, str):
            try:
                last_activity = datetime.fromisoformat(last_mod.replace("Z", "+00:00"))
            except Exception:
                last_activity = datetime.utcnow()
        else:
            last_activity = datetime.utcnow()

        first_act = d.get("first_activity")
        if isinstance(first_act, (int, float)):
            first_activity = datetime.utcfromtimestamp(first_act / 1000 if first_act > 1e12 else first_act)
        else:
            first_activity = last_activity

        last_act = d.get("last_activity")
        if isinstance(last_act, (int, float)):
            last_activity_ts = datetime.utcfromtimestamp(last_act / 1000 if last_act > 1e12 else last_act)
            if last_activity_ts > last_activity:
                last_activity = last_activity_ts

        results.append({
            "pen_mac": d.get("pen_mac") or "canvas",
            "book_type": d.get("book_type"),
            "page_number": pn,
            "total_strokes": d.get("stroke_count", 0),
            "first_activity": first_activity,
            "last_activity": last_activity,
            "canvas_background": d.get("canvas_background"),
        })
    return results


async def _get_canvas_page_as_batches(
    current_user: Dict[str, Any],
    db: DatabaseManager,
    *,
    page_number: int,
    book_type: Optional[str] = None,
    pen_mac: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load a single canvas page from canvas_pages and wrap it as a list of
    stroke batches (same shape the copies endpoints expect).
    """
    col = await _get_canvas_pages_collection(current_user, db)
    if col is None:
        return []

    # Build all user_id variants for legacy compat
    user_id = current_user.get("user_id")
    user_identifiers: list = [user_id, str(user_id)]
    username = current_user.get("username")
    if username:
        user_identifiers.append(username)
    try:
        if ObjectId.is_valid(user_id):
            user_identifiers.append(ObjectId(user_id))
    except Exception:
        pass

    query: Dict[str, Any] = {"user_id": {"$in": user_identifiers}, "page_number": page_number}
    if book_type:
        query["book_type"] = book_type.upper()

    try:
        doc = await col.find_one(query)
    except Exception as exc:
        logger.warning("canvas_pages single-page query failed: %s", exc)
        return []

    if not doc or not doc.get("strokes"):
        return []

    last_mod = doc.get("last_modified")
    ts: Any = last_mod if isinstance(last_mod, datetime) else datetime.utcnow()

    return [{
        "_id": doc.get("_id"),
        "session_id": doc.get("session_id"),
        "timestamp": ts,
        "strokes": doc["strokes"],
        "canvas_background": doc.get("canvas_background"),
        "page_style": doc.get("page_style"),
        "book_type": doc.get("book_type"),
        "pen_mac": doc.get("pen_mac") or "canvas",
        "source": "canvas_pages",
    }]


# ============================================================================
# COPY PAGE ENDPOINTS - List and retrieve student's copy pages
# ============================================================================

@router.get("", response_model=CopyPageListResponse)
async def list_copy_pages(
    request: Request,
    pen_mac: Optional[str] = Query(None, description="Filter by pen MAC address"),
    book_type: Optional[str] = Query(None, description="Filter by book type (A4, A5)"),
    limit: int = Query(50, ge=1, le=200, description="Max pages to return"),
    debug_all: bool = Query(False, description="Debug: Show all strokes without user filter"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    List copy pages for the authenticated student.

    Returns a summary of all pages written with the smart pen or live canvas,
    grouped by (pen_mac, book_type, page_number) with stroke counts.
    Merges data from both `strokes` (pen batches) and `canvas_pages` (canvas sync).
    """
    try:
        user_id = current_user.get("user_id")
        db_name = current_user.get("db_name")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"

        logger.info(f"Listing copy pages for user_id={user_id}, db_name={db_name}, is_b2c={is_b2c}, debug_all={debug_all}")

        # Build match stage for aggregation
        if debug_all:
            # DEBUG MODE: Show all strokes to diagnose user_id matching issues
            match_stage = {}
            logger.warning(f"DEBUG MODE: Showing all strokes without user filter")
        else:
            # Normal mode: filter by user_id or username
            # Strokes might be keyed by user_id (str/ObjectId) OR username
            user_identifiers = [user_id, str(user_id)]

            # Add username if available
            username = current_user.get("username")
            if username:
                user_identifiers.append(username)

            # Also try ObjectId format
            try:
                if ObjectId.is_valid(user_id):
                    user_identifiers.append(ObjectId(user_id))
            except:
                pass

            match_stage = {"user_id": {"$in": user_identifiers}}
        if pen_mac:
            match_stage["pen_mac"] = pen_mac.upper()
        if book_type:
            match_stage["book_type"] = book_type

        # Aggregation pipeline to group strokes by page
        pipeline = [
            {"$match": match_stage},
            {"$group": {
                "_id": {
                    "pen_mac": "$pen_mac",
                    "book_type": "$book_type",
                    "page_number": "$page_number"
                },
                "total_strokes": {"$sum": {"$size": {"$ifNull": ["$strokes", []]}}},
                "first_activity": {"$min": "$timestamp"},
                "last_activity": {"$max": "$timestamp"},
                "canvas_background": {"$first": "$canvas_background"}
            }},
            {"$sort": {"last_activity": -1}},
            {"$limit": limit}
        ]

        # Execute aggregation on strokes collection
        if is_b2c:
            cursor = db.b2c_db["strokes"].aggregate(pipeline)
        else:
            tenant_db = await db.get_tenant_db(db_name)
            if tenant_db is None:
                logger.error(f"Tenant database not available: {db_name}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Tenant database not available"
                )
            cursor = tenant_db["strokes"].aggregate(pipeline)

        # Process results from strokes collection
        pages = []
        async for doc in cursor:
            page_id = doc["_id"]
            # Skip pages without a page number
            if page_id.get("page_number") is None:
                continue

            pages.append({
                "pen_mac": page_id.get("pen_mac", ""),
                "book_type": page_id.get("book_type"),
                "page_number": page_id.get("page_number", 0),
                "total_strokes": doc.get("total_strokes", 0),
                "first_activity": doc.get("first_activity"),
                "last_activity": doc.get("last_activity"),
                "canvas_background": doc.get("canvas_background")
            })

        # ── Merge canvas_pages (server-side canvas sync) ──
        if not debug_all:
            canvas_pages = await _list_canvas_pages_for_user(
                current_user, db, pen_mac=pen_mac, book_type=book_type, limit=limit,
            )
            if canvas_pages:
                # Build a lookup of existing pages from strokes collection
                existing: Dict[tuple, int] = {}
                for idx, p in enumerate(pages):
                    key = (str(p.get("book_type") or "").upper(), p.get("page_number"))
                    existing[key] = idx

                for cp in canvas_pages:
                    key = (str(cp.get("book_type") or "").upper(), cp.get("page_number"))
                    if key in existing:
                        # Merge: keep the entry with more strokes, update activity times
                        idx = existing[key]
                        old = pages[idx]
                        if cp["total_strokes"] > old.get("total_strokes", 0):
                            pages[idx] = cp
                        else:
                            # Update activity bounds from canvas_pages
                            cp_last = cp.get("last_activity")
                            old_last = old.get("last_activity")
                            if cp_last and old_last and cp_last > old_last:
                                old["last_activity"] = cp_last
                    else:
                        pages.append(cp)

                # Re-sort by last_activity descending and trim to limit
                pages.sort(
                    key=lambda p: p.get("last_activity") or datetime.min,
                    reverse=True,
                )
                pages = pages[:limit]

        # Fallback path: fetch from pen backend if tenant strokes are empty/missing.
        if not pages and not debug_all:
            remote_docs = await _fetch_remote_strokes(
                _resolve_forward_auth_header(request),
                pen_mac=pen_mac,
                book_type=book_type,
                limit=max(limit * 50, 2000),
                order="desc",
            )
            if remote_docs:
                grouped: Dict[tuple[str, Optional[str], int], Dict[str, Any]] = {}
                for doc in remote_docs:
                    page_no = doc.get("page_number")
                    if page_no is None:
                        continue
                    try:
                        page_no_int = int(page_no)
                    except Exception:
                        continue
                    key_pen_mac = str(doc.get("pen_mac", "")).upper()
                    key_book_type = doc.get("book_type")
                    key = (key_pen_mac, key_book_type, page_no_int)

                    ts = doc.get("timestamp")
                    try:
                        if isinstance(ts, str):
                            ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        elif isinstance(ts, datetime):
                            ts_dt = ts
                        else:
                            ts_dt = datetime.utcnow()
                    except Exception:
                        ts_dt = datetime.utcnow()

                    strokes_count = len(doc.get("strokes", []) or [])
                    if key not in grouped:
                        grouped[key] = {
                            "pen_mac": key_pen_mac,
                            "book_type": key_book_type,
                            "page_number": page_no_int,
                            "total_strokes": strokes_count,
                            "first_activity": ts_dt,
                            "last_activity": ts_dt,
                            "canvas_background": doc.get("canvas_background"),
                        }
                    else:
                        entry = grouped[key]
                        entry["total_strokes"] += strokes_count
                        if ts_dt < entry["first_activity"]:
                            entry["first_activity"] = ts_dt
                        if ts_dt > entry["last_activity"]:
                            entry["last_activity"] = ts_dt
                        if not entry.get("canvas_background"):
                            entry["canvas_background"] = doc.get("canvas_background")

                pages = sorted(
                    grouped.values(),
                    key=lambda p: p.get("last_activity") or datetime.min,
                    reverse=True,
                )[:limit]
                logger.info(
                    "Copy pages fallback via pen backend: %d pages for user %s",
                    len(pages),
                    user_id,
                )

        logger.info(f"Found {len(pages)} copy pages for user {user_id}")

        return {
            "success": True,
            "total": len(pages),
            "pages": pages
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list copy pages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list copy pages: {str(e)}"
        )


@router.get("/{pen_mac}/{page_number}")
async def get_copy_page(
    request: Request,
    pen_mac: str,
    page_number: int,
    book_type: Optional[str] = Query(None, description="Filter by book type"),
    debug_all: bool = Query(False, description="Debug: Skip user filter"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all strokes for a specific copy page.
    
    Returns all stroke batches for the given pen_mac + page_number combination.
    """
    try:
        user_id = current_user.get("user_id")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        
        logger.info(f"Getting copy page: pen_mac={pen_mac}, page={page_number} for user {user_id}, debug_all={debug_all}")
        
        query = {
            "pen_mac": pen_mac.upper(),
            "page_number": page_number
        }
        if not debug_all:
            # Check for user_id (str/ObjectId) OR username
            user_identifiers = [user_id, str(user_id)]
            username = current_user.get("username")
            if username:
                user_identifiers.append(username)
                
            query["user_id"] = {"$in": user_identifiers}
        if book_type:
            query["book_type"] = book_type
        
        # Fetch stroke batches from strokes collection
        if is_b2c:
            cursor = db.b2c_db["strokes"].find(query).sort("timestamp", 1)
            stroke_batches = await cursor.to_list(length=1000)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            if tenant_db is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Tenant database not available"
                )
            cursor = tenant_db["strokes"].find(query).sort("timestamp", 1)
            stroke_batches = await cursor.to_list(length=1000)

        # Always merge canvas_pages data (has newer strokes from live canvas)
        if not debug_all:
            canvas_batches = await _get_canvas_page_as_batches(
                current_user, db,
                page_number=page_number,
                book_type=book_type,
                pen_mac=pen_mac,
            )
            if canvas_batches:
                if stroke_batches:
                    # Merge: append canvas_pages batch, dedup happens at stroke level
                    stroke_batches = stroke_batches + canvas_batches
                else:
                    stroke_batches = canvas_batches

        # Fallback: pen backend remote API
        if not stroke_batches and not debug_all:
            stroke_batches = await _fetch_remote_strokes(
                _resolve_forward_auth_header(request),
                pen_mac=pen_mac,
                page_number=page_number,
                book_type=book_type,
                limit=5000,
                order="asc",
            )
            if stroke_batches:
                logger.info(
                    "Copy page fallback via pen backend: pen_mac=%s page=%s user=%s",
                    pen_mac, page_number, user_id,
                )

        if not stroke_batches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Copy page not found: pen_mac={pen_mac}, page={page_number}"
            )

        # Serialize stroke batches, dedup strokes by id across all batches
        serialized_batches = []
        total_strokes = 0
        detected_book_type = None
        seen_stroke_ids: set = set()

        for batch in stroke_batches:
            raw_strokes = batch.get("strokes", [])

            if not detected_book_type:
                detected_book_type = batch.get("book_type")

            # Dedup strokes by id across all batches
            unique_strokes = []
            for s in raw_strokes:
                sid = s.get("id") or s.get("strokeId")
                if sid and sid in seen_stroke_ids:
                    continue
                if sid:
                    seen_stroke_ids.add(sid)
                unique_strokes.append(s)

            if not unique_strokes:
                continue

            total_strokes += len(unique_strokes)

            ts_value = batch.get("timestamp")
            if isinstance(ts_value, datetime):
                ts_iso = ts_value.isoformat()
            else:
                ts_iso = str(ts_value) if ts_value else None

            serialized_batches.append({
                "id": str(batch.get("_id") or batch.get("id") or ""),
                "session_id": batch.get("session_id"),
                "timestamp": ts_iso,
                "strokes": unique_strokes,
                "canvas_background": batch.get("canvas_background"),
                "page_style": batch.get("page_style")
            })

        # Temporary _debug — check browser Network tab
        _debug = {
            "user_id_from_jwt": str(user_id),
            "username_from_jwt": current_user.get("username"),
            "user_identifiers_queried": [str(u) for u in (user_identifiers if not debug_all else [])],
            "query_used": {k: str(v) for k, v in query.items()},
            "strokes_batches_found": len(stroke_batches) if stroke_batches else 0,
            "canvas_batches_found": len(canvas_batches) if not debug_all and canvas_batches else 0,
        }
        # Sample distinct user_ids in strokes for this pen_mac
        try:
            if not is_b2c and tenant_db is not None:
                _debug["distinct_user_ids_for_pen"] = await tenant_db["strokes"].distinct(
                    "user_id", {"pen_mac": pen_mac.upper()}
                )
                _debug["total_strokes_for_pen"] = await tenant_db["strokes"].count_documents(
                    {"pen_mac": pen_mac.upper()}
                )
                _debug["total_strokes_for_pen_and_page"] = await tenant_db["strokes"].count_documents(
                    {"pen_mac": pen_mac.upper(), "page_number": page_number}
                )
        except Exception:
            pass

        return {
            "success": True,
            "pen_mac": pen_mac.upper(),
            "book_type": detected_book_type or book_type,
            "page_number": page_number,
            "stroke_batches": serialized_batches,
            "total_strokes": total_strokes,
            "_debug": _debug,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get copy page: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get copy page: {str(e)}"
        )


@router.get("/{pen_mac}/{page_number}/svg")
async def get_copy_page_svg(
    request: Request,
    pen_mac: str,
    page_number: int,
    book_type: Optional[str] = Query(None, description="Book type"),
    background: str = Query("#FFFBF0", description="Background color"),
    debug_all: bool = Query(False, description="Debug: Skip user filter"),
    current_user: Dict[str, Any] = Depends(get_current_user_from_token_or_query),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get the copy page rendered as an SVG.
    
    Useful for previews and client-side rendering.
    """
    try:
        user_id = current_user.get("user_id")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        
        # Build query (skip user_id filter in debug mode)
        query = {
            "pen_mac": pen_mac.upper(),
            "page_number": page_number
        }
        if not debug_all:
            # Check for user_id (str/ObjectId) OR username
            user_identifiers = [user_id, str(user_id)]
            username = current_user.get("username")
            if username:
                user_identifiers.append(username)
                
            query["user_id"] = {"$in": user_identifiers}
        if book_type:
            query["book_type"] = book_type
        
        # Fetch stroke batches from strokes collection
        if is_b2c:
            cursor = db.b2c_db["strokes"].find(query).sort("timestamp", 1)
            stroke_batches = await cursor.to_list(length=1000)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            if tenant_db is not None:
                cursor = tenant_db["strokes"].find(query).sort("timestamp", 1)
                stroke_batches = await cursor.to_list(length=1000)
            else:
                stroke_batches = []

        # Always merge canvas_pages data (has newer strokes from live canvas)
        if not debug_all:
            canvas_batches = await _get_canvas_page_as_batches(
                current_user, db,
                page_number=page_number,
                book_type=book_type,
                pen_mac=pen_mac,
            )
            if canvas_batches:
                if stroke_batches:
                    stroke_batches = stroke_batches + canvas_batches
                else:
                    stroke_batches = canvas_batches

        # Fallback: pen backend remote API
        if not stroke_batches and not debug_all:
            stroke_batches = await _fetch_remote_strokes(
                _resolve_forward_auth_header(request),
                pen_mac=pen_mac,
                page_number=page_number,
                book_type=book_type,
                limit=5000,
                order="asc",
            )
            if stroke_batches:
                logger.info(
                    "SVG fallback via pen backend: pen_mac=%s page=%s",
                    pen_mac,
                    page_number,
                )

        if not stroke_batches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Copy page not found"
            )

        # Detect book type from strokes if not provided
        detected_book_type = book_type
        if not detected_book_type:
            for batch in stroke_batches:
                if batch.get("book_type"):
                    detected_book_type = batch.get("book_type")
                    break

        # Dedup strokes across batches (when merging strokes + canvas_pages)
        seen_stroke_ids: set = set()
        deduped_batches = []
        for batch in stroke_batches:
            raw_strokes = batch.get("strokes", [])
            unique = []
            for s in raw_strokes:
                sid = s.get("id") or s.get("strokeId")
                if sid and sid in seen_stroke_ids:
                    continue
                if sid:
                    seen_stroke_ids.add(sid)
                unique.append(s)
            if unique:
                deduped_batch = {**batch, "strokes": unique}
                deduped_batches.append(deduped_batch)

        # Build SVG
        svg_content = build_svg_from_strokes(
            deduped_batches,
            book_type=detected_book_type,
            background_color=background
        )

        return Response(
            content=svg_content,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": f'inline; filename="copy-page-{page_number}.svg"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate SVG: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate SVG: {str(e)}"
        )


# ============================================================================
# PIN ENDPOINTS - Create and manage pinned copies (PDF snapshots)
# ============================================================================

# Add explicit OPTIONS handler for CORS preflight
@router.options("/pins")
async def pins_options():
    """Handle CORS preflight for pins endpoint"""
    return Response(status_code=200)


@router.post("/pins", status_code=status.HTTP_201_CREATED)
async def create_pin(
    payload: CreatePinRequest,
    raw_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Pin a copy page as a PDF.
    
    This creates a PDF snapshot of the copy page and stores it for later access.
    """
    try:
        user_id = current_user.get("user_id")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        
        logger.info(f"Creating pin for user {user_id}: pen_mac={payload.pen_mac}, page={payload.page_number}")
        
        # Fetch stroke batches for this page
        # Check for user_id (str/ObjectId) OR username
        user_identifiers = [user_id, str(user_id)]
        username = current_user.get("username")
        if username:
            user_identifiers.append(username)

        query = {
            "user_id": {"$in": user_identifiers},
            "pen_mac": payload.pen_mac.upper(),
            "page_number": payload.page_number
        }
        if payload.book_type:
            query["book_type"] = payload.book_type
        
        if is_b2c:
            cursor = db.b2c_db["strokes"].find(query).sort("timestamp", 1)
            stroke_batches = await cursor.to_list(length=1000)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            if tenant_db is not None:
                cursor = tenant_db["strokes"].find(query).sort("timestamp", 1)
                stroke_batches = await cursor.to_list(length=1000)
            else:
                stroke_batches = []

        # Always merge canvas_pages data (has newer strokes from live canvas)
        canvas_batches = await _get_canvas_page_as_batches(
            current_user, db,
            page_number=payload.page_number,
            book_type=payload.book_type,
            pen_mac=payload.pen_mac,
        )
        if canvas_batches:
            if stroke_batches:
                stroke_batches = stroke_batches + canvas_batches
            else:
                stroke_batches = canvas_batches

        # Fallback: pen backend remote API
        if not stroke_batches:
            stroke_batches = await _fetch_remote_strokes(
                _resolve_forward_auth_header(raw_request),
                pen_mac=payload.pen_mac,
                page_number=payload.page_number,
                book_type=payload.book_type,
                limit=5000,
                order="asc",
            )
            if stroke_batches:
                logger.info(
                    "Pin creation fallback via pen backend: pen_mac=%s page=%s user=%s",
                    payload.pen_mac,
                    payload.page_number,
                    user_id,
                )

        if not stroke_batches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Copy page not found - no strokes available to pin"
            )
        
        # Generate PDF
        pdf_bytes = await generate_copy_pdf(
            stroke_batches,
            book_type=payload.book_type
        )
        
        if not pdf_bytes:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate PDF - please ensure svglib and reportlab are installed"
            )
        
        # Generate thumbnail (optional, may fail if cairosvg not available)
        thumbnail_bytes = await generate_copy_thumbnail(
            stroke_batches,
            book_type=payload.book_type,
            scale=0.15
        )
        
        # Generate unique ID and paths
        pin_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pdf_local_path = f"uploads/pinned_copies/{user_id}/{pin_id}_{timestamp}.pdf"
        thumbnail_local_path = f"uploads/pinned_copies/{user_id}/{pin_id}_{timestamp}_thumb.png" if thumbnail_bytes else None
        
        # Upload PDF to storage
        success, pdf_storage_path = await upload_file(
            pdf_bytes,
            pdf_local_path,
            content_type="application/pdf"
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store PDF"
            )
        
        # Upload thumbnail if available
        thumbnail_storage_path = None
        if thumbnail_bytes and thumbnail_local_path:
            success, thumbnail_storage_path = await upload_file(
                thumbnail_bytes,
                thumbnail_local_path,
                content_type="image/png"
            )
        
        # Create pin record
        auto_title = payload.title or f"Copy Page {payload.page_number}"
        if payload.linked_document_id:
            auto_title = f"{auto_title} - Learning Notes"
        
        pin_doc = {
            "_id": pin_id,
            "user_id": user_id,
            "pen_mac": payload.pen_mac.upper(),
            "book_type": payload.book_type,
            "page_number": payload.page_number,
            "title": auto_title,
            "pdf_path": pdf_storage_path,
            "thumbnail_path": thumbnail_storage_path,
            "linked_document_id": payload.linked_document_id,
            "linked_page": payload.linked_page,
            "stroke_count": sum(len(b.get("strokes", [])) for b in stroke_batches),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert into database
        if is_b2c:
            await db.b2c_insert_one("pinned_copies", pin_doc)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            await tenant_db["pinned_copies"].insert_one(pin_doc)
        
        logger.info(f"Created pin {pin_id} for user {user_id}")

        # Queue note classification for this page (best-effort)
        try:
            from services.note_classification_service import queue_classification
            cls_db_name = current_user.get("db_name")
            if not cls_db_name and is_b2c:
                from config_async import MONGODB_DB_STOODY
                cls_db_name = MONGODB_DB_STOODY
            asyncio.create_task(
                queue_classification(
                    db, cls_db_name, user_id,
                    payload.pen_mac, payload.book_type, payload.page_number,
                )
            )
        except Exception:
            pass

        return {
            "success": True,
            "pin_id": pin_id,
            "title": auto_title,
            "pdf_path": pdf_storage_path,
            "thumbnail_path": thumbnail_storage_path,
            "message": "Copy page pinned successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create pin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pin: {str(e)}"
        )


@router.get("/pins", response_model=PinnedCopyListResponse)
async def list_pins(
    limit: int = Query(50, ge=1, le=200, description="Max pins to return"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    List all pinned copies for the authenticated student.
    """
    try:
        user_id = current_user.get("user_id")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        
        logger.info(f"Listing pins for user {user_id}")
        
        query = {"user_id": user_id}
        
        if is_b2c:
            cursor = db.b2c_db["pinned_copies"].find(query).sort("created_at", -1).limit(limit)
            pins = await cursor.to_list(length=limit)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            cursor = tenant_db["pinned_copies"].find(query).sort("created_at", -1).limit(limit)
            pins = await cursor.to_list(length=limit)
        
        # Serialize pins
        serialized_pins = []
        for pin in pins:
            serialized_pins.append({
                "id": str(pin.get("_id")),
                "user_id": pin.get("user_id"),
                "pen_mac": pin.get("pen_mac"),
                "book_type": pin.get("book_type"),
                "page_number": pin.get("page_number"),
                "title": pin.get("title"),
                "pdf_url": get_public_url(pin.get("pdf_path")) if pin.get("pdf_path") else None,
                "thumbnail_url": get_public_url(pin.get("thumbnail_path")) if pin.get("thumbnail_path") else None,
                "linked_document_id": pin.get("linked_document_id"),
                "linked_page": pin.get("linked_page"),
                "stroke_count": pin.get("stroke_count", 0),
                "created_at": pin.get("created_at").isoformat() if pin.get("created_at") else None
            })
        
        return {
            "success": True,
            "total": len(serialized_pins),
            "pins": serialized_pins
        }
        
    except Exception as e:
        logger.error(f"Failed to list pins: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list pins: {str(e)}"
        )


@router.get("/pins/{pin_id}")
async def get_pin(
    pin_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get details of a specific pinned copy.
    """
    try:
        user_id = current_user.get("user_id")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        
        query = {"_id": pin_id, "user_id": user_id}
        
        if is_b2c:
            pin = await db.b2c_find_one("pinned_copies", query)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            pin = await tenant_db["pinned_copies"].find_one(query)
        
        if not pin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pinned copy not found"
            )
        
        return {
            "success": True,
            "pin": {
                "id": str(pin.get("_id")),
                "user_id": pin.get("user_id"),
                "pen_mac": pin.get("pen_mac"),
                "book_type": pin.get("book_type"),
                "page_number": pin.get("page_number"),
                "title": pin.get("title"),
                "pdf_url": get_public_url(pin.get("pdf_path")) if pin.get("pdf_path") else None,
                "thumbnail_url": get_public_url(pin.get("thumbnail_path")) if pin.get("thumbnail_path") else None,
                "linked_document_id": pin.get("linked_document_id"),
                "linked_page": pin.get("linked_page"),
                "stroke_count": pin.get("stroke_count", 0),
                "created_at": pin.get("created_at").isoformat() if pin.get("created_at") else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pin: {str(e)}"
        )


@router.get("/pins/{pin_id}/file")
async def get_pin_file(
    pin_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Stream the PDF file for a pinned copy.
    """
    try:
        user_id = current_user.get("user_id")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        
        query = {"_id": pin_id, "user_id": user_id}
        
        if is_b2c:
            pin = await db.b2c_find_one("pinned_copies", query)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            pin = await tenant_db["pinned_copies"].find_one(query)
        
        if not pin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pinned copy not found"
            )
        
        pdf_path = pin.get("pdf_path")
        if not pdf_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF file not found"
            )
        
        # Download from storage
        pdf_bytes = await s3_download_file(pdf_path)
        
        if not pdf_bytes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF file not found in storage"
            )
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{pin.get("title", "copy")}.pdf"',
                "Content-Length": str(len(pdf_bytes))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pin file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pin file: {str(e)}"
        )


@router.delete("/pins/{pin_id}")
async def delete_pin(
    pin_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Delete a pinned copy.
    """
    try:
        user_id = current_user.get("user_id")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        
        query = {"_id": pin_id, "user_id": user_id}
        
        if is_b2c:
            result = await db.b2c_delete_one("pinned_copies", query)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            result = await tenant_db["pinned_copies"].delete_one(query)
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pinned copy not found"
            )
        
        # Note: We don't delete the PDF from storage to allow for potential recovery
        # A cleanup job could be added later to remove orphaned files
        
        return {
            "success": True,
            "message": "Pinned copy deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete pin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete pin: {str(e)}"
        )


@router.patch("/pins/{pin_id}")
async def update_pin(
    pin_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Update a pinned copy's title.
    """
    try:
        user_id = current_user.get("user_id")
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        
        body = await request.json()
        new_title = body.get("title")
        
        if not new_title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title is required"
            )
        
        query = {"_id": pin_id, "user_id": user_id}
        update = {"$set": {"title": new_title, "updated_at": datetime.utcnow()}}
        
        if is_b2c:
            result = await db.b2c_update_one("pinned_copies", query, update)
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            result = await tenant_db["pinned_copies"].update_one(query, update)
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Pinned copy not found"
            )
        
        return {
            "success": True,
            "message": "Pinned copy updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update pin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update pin: {str(e)}"
        )
