"""
Student Copies & Pins API - Async Routes

Provides endpoints for:
1. Listing student copy pages (written with smart pen)
2. Retrieving strokes for a specific copy page
3. Creating/managing pinned copies (PDF snapshots)

Data source: ``canvas_pages`` collection (single source of truth).

Conducted-exam bridge (SWM-012):
    When copy pages contain conducted-exam metadata (``exam_id``), the pin
    creation flow bridges the underlying stroke artifacts into the shared
    ingest substrate via ``IngestService.ingest_submission()``.  Non-exam
    copy pages are unaffected.
"""

import asyncio
import logging
import re
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
from api.v1.copy_sets_async import resolve_copy_id as _resolve_copy_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/copies", tags=["Student Copies"])


# ---------------------------------------------------------------------------
# Conducted-exam ingest bridge (SWM-012)
# ---------------------------------------------------------------------------

async def _maybe_bridge_to_ingest(
    *,
    current_user: Dict[str, Any],
    db: DatabaseManager,
    stroke_batches: List[Dict[str, Any]],
    pen_mac: str,
    page_number: int,
) -> Optional[Dict[str, Any]]:
    """Conditionally bridge copy page strokes into the shared ingest substrate.

    The bridge fires only when ALL of the following are true:
    - At least one stroke batch carries ``exam_id`` metadata
    - ``student_id`` and ``admin_id`` can be resolved
    - The exam-conductor ingest module is importable

    Returns the IngestResult dict on success, None otherwise.
    Failures are logged but never propagated — copy functionality is
    unaffected.
    """
    # Detect conducted-exam metadata on any batch
    exam_id: Optional[str] = None
    admin_id: Optional[str] = None
    collected_strokes: List[Dict[str, Any]] = []

    for batch in stroke_batches:
        batch_exam_id = batch.get("exam_id")
        if batch_exam_id:
            exam_id = batch_exam_id
            admin_id = admin_id or batch.get("admin_id")
            collected_strokes.extend(batch.get("strokes", []))

    if not exam_id:
        # Not a conducted-exam page — skip ingest entirely
        return None

    student_id = current_user.get("user_id") or current_user.get("student_id")
    admin_id = admin_id or current_user.get("admin_id")

    if not student_id or not admin_id:
        logger.debug(
            "Skipping ingest bridge for exam=%s: missing student_id or admin_id",
            exam_id,
        )
        return None

    # Attempt import — graceful degradation
    try:
        from api.v1._exampen_imports import load_exampen
        ingest_mod = load_exampen("ingest.service")
        IngestService = ingest_mod.IngestService
    except (ImportError, AttributeError):
        logger.debug(
            "exam-conductor ingest not available — skipping copy ingest bridge"
        )
        return None

    try:
        db_name = current_user.get("db_name")
        if not db_name:
            return None

        tenant_db = await db.get_tenant_db(db_name)
        if tenant_db is None:
            return None

        service = IngestService(tenant_db)
        await service.initialize()

        # Determine source from pen_mac
        source = "ble_pen" if pen_mac and pen_mac.lower() != "canvas" else "camera"

        pages = [
            {
                "page_number": page_number,
                "raw_strokes": collected_strokes,
            }
        ]

        result = await service.ingest_submission(
            exam_id=exam_id,
            student_id=student_id,
            admin_id=admin_id,
            source=source,
            pen_mac=pen_mac.upper() if source == "ble_pen" else None,
            pages=pages,
        )

        logger.info(
            "Copy ingest bridge: submission_id=%s exam=%s student=%s "
            "page=%d already_existed=%s",
            result.submission_id,
            exam_id,
            student_id,
            page_number,
            result.already_existed,
        )
        return result.model_dump()

    except Exception:
        logger.exception(
            "Copy ingest bridge failed for exam=%s student=%s page=%d "
            "(non-fatal, copy operation continues)",
            exam_id,
            student_id,
            page_number,
        )
        return None


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
    id: Optional[str] = None
    copy_id: Optional[str] = None
    pen_mac: str
    book_type: Optional[str] = None
    page_number: int
    total_strokes: int
    first_activity: datetime
    last_activity: datetime
    canvas_background: Optional[str] = None
    session_id: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    confidence: Optional[float] = None
    is_favorite: bool = False
    is_archived: bool = False
    classification_source: Optional[str] = None
    classification_status: str = "unclassified"
    classification_error: Optional[str] = None
    queue_status: Optional[str] = None
    queue_attempts: Optional[int] = None
    queue_process_after: Optional[datetime] = None
    queue_updated_at: Optional[datetime] = None


class CopyPageListResponse(BaseModel):
    """Response for listing copy pages"""
    success: bool = True
    total: int
    pages: List[CopyPageSummary]


class CopyPageDetailResponse(BaseModel):
    """Response for a single copy page with strokes"""
    success: bool = True
    copy_id: Optional[str] = None
    pen_mac: str
    book_type: Optional[str] = None
    page_number: int
    stroke_batches: List[Dict[str, Any]]
    total_strokes: int


class CreatePinRequest(BaseModel):
    """Request to pin a copy page"""
    pen_mac: str
    copy_id: Optional[str] = None
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


async def _get_note_classifications_collection(current_user: Dict[str, Any], db: DatabaseManager):
    """Return the note_classifications collection for the user's tenant/B2C database."""
    is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
    if is_b2c:
        return db.b2c_db["note_classifications"]
    db_name = current_user.get("db_name")
    if not db_name:
        return None
    tenant_db = await db.get_tenant_db(db_name)
    return tenant_db["note_classifications"] if tenant_db is not None else None


async def _get_classification_queue_collection(current_user: Dict[str, Any], db: DatabaseManager):
    """Return the classification_queue collection for the user's tenant/B2C database."""
    is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
    if is_b2c:
        return db.b2c_db["classification_queue"]
    db_name = current_user.get("db_name")
    if not db_name:
        return None
    tenant_db = await db.get_tenant_db(db_name)
    return tenant_db["classification_queue"] if tenant_db is not None else None


async def _list_canvas_pages_for_user(
    current_user: Dict[str, Any],
    db: DatabaseManager,
    *,
    pen_mac: Optional[str] = None,
    book_type: Optional[str] = None,
    copy_id: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Query canvas_pages collection and return summaries compatible with CopyPageSummary."""
    col = await _get_canvas_pages_collection(current_user, db)
    if col is None:
        return []
    classify_col = await _get_note_classifications_collection(current_user, db)
    queue_col = await _get_classification_queue_collection(current_user, db)

    user_identifiers = _build_user_id_variants(current_user)

    query: Dict[str, Any] = {"user_id": {"$in": user_identifiers}, "stroke_count": {"$gt": 0}}
    if copy_id:
        query["copy_id"] = copy_id
    if pen_mac:
        query["pen_mac"] = pen_mac.upper()
    if book_type:
        bt_val = book_type.upper()
        query["book_type"] = {"$in": [bt_val, None]} if bt_val == "STANDARD" else bt_val

    try:
        cursor = col.find(query, {"strokes": 0}).sort("last_modified", -1).limit(limit * 5)
        docs = await cursor.to_list(length=limit * 5)
    except Exception as exc:
        logger.warning("canvas_pages query failed: %s", exc)
        return []

    # Classification lookup keyed by (copy_id, book_type, page_number).
    # pen_mac is metadata, not page identity.
    classifications: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    if classify_col is not None and docs:
        page_numbers = sorted({int(d.get("page_number")) for d in docs if d.get("page_number") is not None})
        book_types: list = sorted({str(d.get("book_type") or "STANDARD").upper() for d in docs if d.get("page_number") is not None})
        if "STANDARD" in book_types:
            book_types.append(None)  # also match null/missing book_type
        class_query: Dict[str, Any] = {
            "user_id": {"$in": user_identifiers},
            "page_number": {"$in": page_numbers},
            "book_type": {"$in": book_types},
        }
        if copy_id:
            class_query["copy_id"] = copy_id
        try:
            class_docs = await classify_col.find(class_query).to_list(length=None)
            for doc in class_docs:
                key = (
                    str(doc.get("copy_id") or "default"),
                    (doc.get("book_type") or "STANDARD").upper(),
                    int(doc.get("page_number")),
                )
                existing = classifications.get(key)
                if existing is None or (doc.get("updated_at") or doc.get("created_at") or datetime.min) >= (existing.get("updated_at") or existing.get("created_at") or datetime.min):
                    classifications[key] = doc
        except Exception as exc:
            logger.warning("note_classifications query failed for copies: %s", exc)

    queue_state: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    if queue_col is not None and docs:
        page_numbers = sorted({int(d.get("page_number")) for d in docs if d.get("page_number") is not None})
        book_types = sorted({str(d.get("book_type") or "STANDARD").upper() for d in docs if d.get("page_number") is not None})
        queue_query: Dict[str, Any] = {
            "user_id": current_user.get("user_id"),
            "page_number": {"$in": page_numbers},
            "book_type": {"$in": book_types},
            "status": {"$in": ["pending", "processing", "failed", "completed", "cancelled", "cancelling"]},
        }
        if copy_id:
            queue_query["copy_id"] = copy_id
        try:
            queue_docs = await queue_col.find(queue_query).to_list(length=None)
            for doc in queue_docs:
                key = (
                    str(doc.get("copy_id") or "default"),
                    (doc.get("book_type") or "STANDARD").upper(),
                    int(doc.get("page_number")),
                )
                existing = queue_state.get(key)
                if existing is None or (doc.get("updated_at") or doc.get("queued_at") or datetime.min) >= (existing.get("updated_at") or existing.get("queued_at") or datetime.min):
                    queue_state[key] = doc
        except Exception as exc:
            logger.warning("classification_queue query failed for copies: %s", exc)

    # Grouping keyed by (copy_id, book_type, page_number)
    grouped: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for d in docs:
        pn = d.get("page_number")
        if pn is None:
            continue
        bt = (d.get("book_type") or "STANDARD").upper()

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

        cid = str(d.get("copy_id") or "default")
        key = (cid, bt, int(pn))
        existing = grouped.get(key)
        if existing is None:
            classification = classifications.get(key)
            queue_doc = queue_state.get(key)
            if classification and classification.get("is_archived"):
                continue
            classification_status = "unclassified"
            classification_error = None
            classification_source = classification.get("classification_source") if classification else None
            queue_status = queue_doc.get("status") if queue_doc else None
            queue_attempts = int(queue_doc.get("attempts", 0)) if queue_doc else None
            queue_process_after = queue_doc.get("process_after") if queue_doc else None
            queue_updated_at = queue_doc.get("updated_at") if queue_doc else None
            if queue_status in {"pending", "processing", "cancelling"}:
                classification_status = "pending"
            elif queue_status == "failed":
                classification_status = "failed"
                classification_error = queue_doc.get("error")
            elif classification_source == "pending_ai":
                classification_status = "failed"
                if queue_status == "completed":
                    classification_error = "AI job completed but the page classification was not saved."
                elif queue_status == "cancelled":
                    classification_error = "AI classification was cancelled."
                else:
                    classification_error = "AI classification is stuck. Retry or cancel to continue."
            elif classification:
                classification_status = "classified"
            grouped[key] = {
                "id": str(classification.get("_id")) if classification and classification.get("_id") else None,
                "copy_id": d.get("copy_id"),
                "pen_mac": d.get("pen_mac") or "canvas",
                "book_type": bt,
                "page_number": int(pn),
                "total_strokes": int(d.get("stroke_count", 0) or 0),
                "first_activity": first_activity,
                "last_activity": last_activity,
                "canvas_background": d.get("canvas_background"),
                "session_id": d.get("session_id"),
                "subject": classification.get("subject") if classification else None,
                "topic": classification.get("topic") if classification else None,
                "confidence": classification.get("confidence") if classification else None,
                "is_favorite": bool(classification.get("is_favorite")) if classification else False,
                "is_archived": bool(classification.get("is_archived")) if classification else False,
                "classification_source": classification_source,
                "classification_status": classification_status,
                "classification_error": classification_error,
                "queue_status": queue_status,
                "queue_attempts": queue_attempts,
                "queue_process_after": queue_process_after,
                "queue_updated_at": queue_updated_at,
            }
            continue

        if first_activity < existing["first_activity"]:
            existing["first_activity"] = first_activity

        if last_activity >= existing["last_activity"]:
            existing["last_activity"] = last_activity
            existing["pen_mac"] = d.get("pen_mac") or existing["pen_mac"]
            existing["canvas_background"] = d.get("canvas_background") or existing.get("canvas_background")
            existing["session_id"] = d.get("session_id") or existing.get("session_id")

        classification = classifications.get(key)
        queue_doc = queue_state.get(key)
        if classification:
            if classification.get("is_archived"):
                continue
            existing["id"] = existing.get("id") or (str(classification.get("_id")) if classification.get("_id") else None)
            existing["subject"] = classification.get("subject") or existing.get("subject")
            existing["topic"] = classification.get("topic") or existing.get("topic")
            existing["confidence"] = classification.get("confidence") if classification.get("confidence") is not None else existing.get("confidence")
            existing["is_favorite"] = bool(classification.get("is_favorite")) if classification.get("is_favorite") is not None else existing.get("is_favorite", False)
            existing["is_archived"] = bool(classification.get("is_archived"))
            existing["classification_source"] = classification.get("classification_source") or existing.get("classification_source")

        queue_status = queue_doc.get("status") if queue_doc else None
        existing["queue_status"] = queue_status
        existing["queue_attempts"] = int(queue_doc.get("attempts", 0)) if queue_doc else None
        existing["queue_process_after"] = queue_doc.get("process_after") if queue_doc else None
        existing["queue_updated_at"] = queue_doc.get("updated_at") if queue_doc else None

        if queue_status in {"pending", "processing", "cancelling"}:
            existing["classification_status"] = "pending"
            existing["classification_error"] = None
        elif queue_status == "failed":
            existing["classification_status"] = "failed"
            existing["classification_error"] = queue_doc.get("error")
        elif existing.get("classification_source") == "pending_ai":
            existing["classification_status"] = "failed"
            if queue_status == "completed":
                existing["classification_error"] = "AI job completed but the page classification was not saved."
            elif queue_status == "cancelled":
                existing["classification_error"] = "AI classification was cancelled."
            else:
                existing["classification_error"] = "AI classification is stuck. Retry or cancel to continue."
        elif classification:
            existing["classification_status"] = "classified"
            existing["classification_error"] = None

        existing["total_strokes"] = max(existing["total_strokes"], int(d.get("stroke_count", 0) or 0))

    results = sorted(
        [page for page in grouped.values() if not page.get("is_archived")],
        key=lambda p: p["last_activity"],
        reverse=True,
    )
    return results[:limit]


async def _get_canvas_page_as_batches(
    current_user: Dict[str, Any],
    db: DatabaseManager,
    *,
    page_number: int,
    book_type: Optional[str] = None,
    pen_mac: Optional[str] = None,
    copy_id: Optional[str] = None,
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
    if copy_id:
        query["copy_id"] = copy_id
    if book_type:
        bt_val = book_type.upper()
        query["book_type"] = {"$in": [bt_val, None]} if bt_val == "STANDARD" else bt_val
    if pen_mac:
        query["pen_mac"] = {"$regex": f"^{re.escape(pen_mac)}$", "$options": "i"}

    try:
        docs = await col.find(query).sort("last_modified", 1).to_list(length=None)
    except Exception as exc:
        logger.warning("canvas_pages single-page query failed: %s", exc)
        return []

    if not docs:
        return []

    batches: List[Dict[str, Any]] = []
    for doc in docs:
        if not doc.get("strokes"):
            continue

        last_mod = doc.get("last_modified")
        ts: Any = last_mod if isinstance(last_mod, datetime) else datetime.utcnow()

        batches.append({
            "_id": doc.get("_id"),
            "session_id": doc.get("session_id"),
            "timestamp": ts,
            "strokes": doc["strokes"],
            "canvas_background": doc.get("canvas_background"),
            "page_style": doc.get("page_style"),
            "book_type": doc.get("book_type"),
            "pen_mac": doc.get("pen_mac") or "canvas",
            "source": "canvas_pages",
        })

    return batches


# ============================================================================
# COPY PAGE ENDPOINTS - List and retrieve student's copy pages
# ============================================================================

@router.get("")
async def list_copy_pages(
    request: Request,
    pen_mac: Optional[str] = Query(None, description="Filter by pen MAC address"),
    book_type: Optional[str] = Query(None, description="Filter by book type (A4, A5)"),
    copy_id: Optional[str] = Query(None, description="Filter by copy set ID"),
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
            current_user, db, pen_mac=pen_mac, book_type=book_type, copy_id=copy_id, limit=limit,
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
    copy_id: Optional[str] = Query(None, description="Filter by copy set ID"),
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
            copy_id=copy_id,
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
            "copy_id": copy_id,
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
    copy_id: Optional[str] = Query(None, description="Filter by copy set ID"),
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
            copy_id=copy_id,
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
                "Content-Disposition": f'inline; filename="copy-page-{page_number}.svg"',
                "Cache-Control": "no-store, must-revalidate",
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
# BULK DELETE ENDPOINT - Delete copy pages by identity
# ============================================================================

class CopyPageIdentity(BaseModel):
    pen_mac: str
    copy_id: Optional[str] = None
    book_type: str = "A5"
    page_number: int


class BulkDeleteCopyPagesRequest(BaseModel):
    pages: List[CopyPageIdentity] = Field(..., max_length=100)


@router.post("/bulk-delete")
async def bulk_delete_copy_pages(
    body: BulkDeleteCopyPagesRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """
    Delete copy pages by identity (pen_mac, book_type, page_number).

    Works for both classified and unclassified pages — does not require a
    classification_id.
    """
    from core.user_identity import canonical_canvas_user_id
    from utils.page_delete import delete_page_by_identity

    is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
    if is_b2c:
        tenant_db = db.b2c_db
    else:
        tenant_db = await db.get_tenant_db(current_user.get("db_name"))

    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = canonical_canvas_user_id(current_user)
    variants = _build_user_id_variants(current_user)

    # Deduplicate by (pen_mac, copy_id, book_type, page_number)
    seen: set = set()
    unique_pages: list = []
    for p in body.pages:
        key = (p.pen_mac.upper(), p.copy_id, p.book_type.upper(), p.page_number)
        if key not in seen:
            seen.add(key)
            unique_pages.append(p)

    deleted_count = 0
    failed: list = []
    for p in unique_pages:
        try:
            result = await delete_page_by_identity(
                tenant_db,
                pen_mac=p.pen_mac,
                book_type=p.book_type,
                page_number=p.page_number,
                user_id=user_id,
                user_id_variants=variants,
                copy_id=p.copy_id,
            )
            if result["had_data"]:
                deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete copy page {p.pen_mac}/{p.book_type}/{p.page_number}: {e}")
            failed.append({
                "pen_mac": p.pen_mac,
                "book_type": p.book_type,
                "page_number": p.page_number,
                "error": str(e),
            })

    return {
        "success": len(failed) == 0,
        "deleted_count": deleted_count,
        "failed": failed,
    }


# ============================================================================
# CONDUCTED-EXAM INGEST BRIDGE (SWM-012)
# ============================================================================

class CopyPageIngestRequest(BaseModel):
    """Request to explicitly bridge a copy page into the ingest substrate."""
    pen_mac: str
    copy_id: Optional[str] = None
    book_type: Optional[str] = "A5"
    page_number: int
    exam_id: str = Field(..., description="Conducted exam identifier")
    admin_id: Optional[str] = Field(
        default=None,
        description="Admin identity (resolved from user context if omitted)",
    )


@router.post("/ingest-bridge", status_code=status.HTTP_200_OK)
async def ingest_bridge_copy_page(
    payload: CopyPageIngestRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Explicitly bridge a copy page's strokes into the conducted-exam
    ingest substrate.

    This endpoint is called when the caller knows a copy page belongs to
    a conducted exam and wants to ensure its artifacts are persisted in
    ``evalpen_submissions`` / ``evalpen_answer_pages``.

    Idempotent: re-calling with the same (exam_id, student_id) returns
    the existing submission without error (ING-03).

    Non-exam copy pages should NOT call this endpoint.
    """
    student_id = current_user.get("user_id") or current_user.get("student_id")
    admin_id = payload.admin_id or current_user.get("admin_id")

    if not student_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot resolve student_id from authentication context",
        )
    if not admin_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="admin_id is required (pass explicitly or ensure it is in the auth token)",
        )

    # Fetch stroke data
    stroke_batches = await _get_canvas_page_as_batches(
        current_user, db,
        page_number=payload.page_number,
        book_type=payload.book_type,
        pen_mac=payload.pen_mac,
        copy_id=payload.copy_id,
    )

    if not stroke_batches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Copy page not found — no strokes available to ingest",
        )

    # Inject exam_id into batches so the bridge helper detects it
    for batch in stroke_batches:
        batch["exam_id"] = payload.exam_id
        batch["admin_id"] = admin_id

    result = await _maybe_bridge_to_ingest(
        current_user=current_user,
        db=db,
        stroke_batches=stroke_batches,
        pen_mac=payload.pen_mac,
        page_number=payload.page_number,
    )

    if result is None:
        return {
            "success": True,
            "ingest_available": False,
            "message": "Ingest substrate is not available. "
                       "Artifacts will be ingested via backfill.",
        }

    return {
        "success": True,
        "ingest_available": True,
        "submission_id": result.get("submission_id"),
        "page_count": result.get("page_count", 0),
        "content_hash": result.get("content_hash"),
        "already_existed": result.get("already_existed", False),
        "message": "Copy page artifacts ingested into conducted-exam substrate.",
    }


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
            copy_id=payload.copy_id,
        )

        if not stroke_batches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Copy page not found - no strokes available to pin"
            )

        # SWM-012: Bridge conducted-exam artifacts to ingest substrate
        # (best-effort, non-blocking for the pin flow)
        await _maybe_bridge_to_ingest(
            current_user=current_user,
            db=db,
            stroke_batches=stroke_batches,
            pen_mac=payload.pen_mac,
            page_number=payload.page_number,
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
            "copy_id": payload.copy_id,
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
