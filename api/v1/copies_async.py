"""
Student Copies & Pins API - Async Routes

Provides endpoints for:
1. Listing student copy pages (written with smart pen)
2. Retrieving strokes for a specific copy page
3. Creating/managing pinned copies (PDF snapshots)

Data source: ``canvas_pages`` collection (single source of truth).
"""

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, HTTPException, Request, Depends, Query, status, Header
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from core.user_identity import canonical_canvas_user_id, canvas_user_id_variants
from api.v1.auth_async import get_current_user, get_database, get_auth_manager
from core.auth import AuthManager
from utils.s3_storage import upload_file, download_file as s3_download_file, get_public_url
from utils.stroke_pdf_generator import generate_copy_pdf, generate_copy_thumbnail, build_svg_from_strokes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/copies", tags=["Student Copies"])


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
# Helpers — canvas_pages (single source of truth)
# ============================================================================

def _build_user_id_variants(current_user: Dict[str, Any]) -> list:
    """Build all user_id variants including ObjectId for canvas_pages queries."""
    variants: list = list(canvas_user_id_variants(current_user))
    uid = current_user.get("user_id")
    try:
        if uid and ObjectId.is_valid(uid):
            variants.append(ObjectId(uid))
    except Exception:
        pass
    return variants


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

    user_identifiers = _build_user_id_variants(current_user)

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

    user_identifiers = _build_user_id_variants(current_user)

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

@router.get("")
async def list_copy_pages(
    request: Request,
    pen_mac: Optional[str] = Query(None, description="Filter by pen MAC address"),
    book_type: Optional[str] = Query(None, description="Filter by book type (A4, A5)"),
    limit: int = Query(50, ge=1, le=200, description="Max pages to return"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    List copy pages for the authenticated student.

    Returns a summary of all pages from the ``canvas_pages`` collection,
    grouped by (pen_mac, book_type, page_number) with stroke counts.
    """
    try:
        pages = await _list_canvas_pages_for_user(
            current_user, db, pen_mac=pen_mac, book_type=book_type, limit=limit,
        )

        return {
            "success": True,
            "total": len(pages),
            "pages": pages,
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
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all strokes for a specific copy page.

    Returns all stroke batches for the given pen_mac + page_number combination.
    """
    try:
        stroke_batches = await _get_canvas_page_as_batches(
            current_user, db,
            page_number=page_number,
            book_type=book_type,
            pen_mac=pen_mac,
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

            # Filter strokes to this page + matching book_type + dedup by id
            batch_bt = (batch.get("book_type") or "").upper()
            req_bt = (book_type or detected_book_type or "").upper()
            unique_strokes = []
            for s in raw_strokes:
                # Skip strokes that belong to a different page
                s_pn = s.get("pageNumber") if s.get("pageNumber") is not None else s.get("pageNo")
                if s_pn is not None and int(s_pn) != page_number:
                    continue

                # Skip strokes whose book type doesn't match the request
                s_bt = (s.get("bookType") or "").upper()
                if s_bt and req_bt and s_bt != req_bt:
                    continue

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

        return {
            "success": True,
            "pen_mac": pen_mac.upper(),
            "book_type": detected_book_type or book_type,
            "page_number": page_number,
            "stroke_batches": serialized_batches,
            "total_strokes": total_strokes,
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
    current_user: Dict[str, Any] = Depends(get_current_user_from_token_or_query),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get the copy page rendered as an SVG.

    Useful for previews and client-side rendering.
    """
    try:
        stroke_batches = await _get_canvas_page_as_batches(
            current_user, db,
            page_number=page_number,
            book_type=book_type,
            pen_mac=pen_mac,
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

        # Filter strokes to this page + matching book_type + dedup across batches
        req_bt = (detected_book_type or "").upper()
        seen_stroke_ids: set = set()
        deduped_batches = []
        for batch in stroke_batches:
            raw_strokes = batch.get("strokes", [])
            unique = []
            for s in raw_strokes:
                # Skip strokes belonging to a different page
                s_pn = s.get("pageNumber") if s.get("pageNumber") is not None else s.get("pageNo")
                if s_pn is not None and int(s_pn) != page_number:
                    continue

                # Skip strokes whose book type doesn't match
                s_bt = (s.get("bookType") or "").upper()
                if s_bt and req_bt and s_bt != req_bt:
                    continue

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
        user_id = canonical_canvas_user_id(current_user)
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"

        logger.info(f"Creating pin for user {user_id}: pen_mac={payload.pen_mac}, page={payload.page_number}")

        # Fetch stroke batches from canvas_pages
        stroke_batches = await _get_canvas_page_as_batches(
            current_user, db,
            page_number=payload.page_number,
            book_type=payload.book_type,
            pen_mac=payload.pen_mac,
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

        # Create pin record — use canonical (username-style) user_id
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
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        uid_match = {"$in": canvas_user_id_variants(current_user)}

        query = {"user_id": uid_match}

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
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        uid_match = {"$in": canvas_user_id_variants(current_user)}

        query = {"_id": pin_id, "user_id": uid_match}

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
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        uid_match = {"$in": canvas_user_id_variants(current_user)}

        query = {"_id": pin_id, "user_id": uid_match}

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
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        uid_match = {"$in": canvas_user_id_variants(current_user)}

        query = {"_id": pin_id, "user_id": uid_match}

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
        is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
        uid_match = {"$in": canvas_user_id_variants(current_user)}

        body = await request.json()
        new_title = body.get("title")

        if not new_title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title is required"
            )

        query = {"_id": pin_id, "user_id": uid_match}
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
            "message": "Pin updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update pin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update pin: {str(e)}"
        )
