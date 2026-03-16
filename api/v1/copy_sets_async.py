"""
Copy Sets API - Async Routes

Provides endpoints for managing student copy sets (physical notebook copies).
Each student can create, select, rename, archive, and delete copy sets.
All canvas page writes, hydration, and list views are scoped to copy_id.

The pen has no copy concept — copy_id is resolved from the user's
active_copy_id pointer, stored on the student profile document.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from core.user_identity import canonical_canvas_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/copy-sets", tags=["Copy Sets"])
COPY_SET_TITLE_MAX_LENGTH = 11


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class CopySetCreate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=COPY_SET_TITLE_MAX_LENGTH)


class CopySetUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=COPY_SET_TITLE_MAX_LENGTH)


class CopySetResponse(BaseModel):
    copy_id: str
    user_id: str
    title: str
    is_archived: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# COLLECTION / DB HELPERS  (importable by strokes_async, copies_async)
# ============================================================================

async def get_copy_sets_collection(
    current_user: Dict[str, Any], db: DatabaseManager
):
    """Return the ``copy_sets`` collection for the user's tenant / B2C database."""
    is_b2c = (
        current_user.get("is_b2c", False)
        or current_user.get("user_type") == "b2c_user"
    )
    if is_b2c:
        b2c_db = await db.get_b2c_db()
        if b2c_db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="B2C database not available",
            )
        return b2c_db["copy_sets"]

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
    return tenant_db["copy_sets"]


async def _get_user_collection(
    current_user: Dict[str, Any], db: DatabaseManager
):
    """Return the collection containing the current user's profile document."""
    is_b2c = (
        current_user.get("is_b2c", False)
        or current_user.get("user_type") == "b2c_user"
    )
    if is_b2c:
        b2c_db = await db.get_b2c_db()
        if b2c_db is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="B2C database not available",
            )
        return b2c_db["users"]

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
    user_type = current_user.get("user_type", "student")
    if user_type == "admin":
        return tenant_db["admins"]
    if user_type == "tutor":
        return tenant_db["tutors"]
    return tenant_db["students"]


def _build_user_filter(uid: str) -> Dict[str, Any]:
    """Build a filter that matches the user document by ``_id`` (ObjectId or string)."""
    parts: List[Any] = [uid]
    try:
        if ObjectId.is_valid(uid):
            parts.append(ObjectId(uid))
    except Exception:
        pass
    if len(parts) == 1:
        return {"_id": parts[0]}
    return {"_id": {"$in": parts}}


def _copy_set_to_response(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Serialise a copy_sets document to a JSON-safe dict."""
    return {
        "copy_id": str(doc["_id"]),
        "user_id": doc.get("user_id", ""),
        "title": doc.get("title", "Copy 1"),
        "is_archived": doc.get("is_archived", False),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


async def _next_copy_number(col, user_id: str) -> int:
    """Return the next auto-increment number for copy naming."""
    count = await col.count_documents({"user_id": user_id})
    return count + 1


async def _set_active_copy_id(
    current_user: Dict[str, Any],
    db: DatabaseManager,
    copy_id: str,
) -> None:
    """Update ``active_copy_id`` on the user's profile document."""
    user_col = await _get_user_collection(current_user, db)
    uid = current_user.get("user_id")
    await user_col.update_one(
        _build_user_filter(uid),
        {"$set": {"active_copy_id": copy_id}},
    )


async def _get_active_copy_id_from_db(
    current_user: Dict[str, Any],
    db: DatabaseManager,
) -> Optional[str]:
    """Read the user's ``active_copy_id`` directly from MongoDB."""
    user_col = await _get_user_collection(current_user, db)
    uid = current_user.get("user_id")
    user_doc = await user_col.find_one(
        _build_user_filter(uid), {"active_copy_id": 1}
    )
    if user_doc:
        return user_doc.get("active_copy_id")
    return None


# ---- public helpers used by strokes_async / copies_async -----------------

async def ensure_active_copy(
    current_user: Dict[str, Any],
    db: DatabaseManager,
) -> Dict[str, Any]:
    """Guarantee the user has an active, non-archived copy set.

    Creates a default ``Copy 1`` if none exists.  Returns the copy-set
    document (with ``_id``).
    """
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    # 1. Try the pointer stored on the user profile.
    active_copy_id = await _get_active_copy_id_from_db(current_user, db)
    if active_copy_id:
        try:
            copy_set = await col.find_one(
                {"_id": ObjectId(active_copy_id), "user_id": user_id, "is_archived": False}
            )
            if copy_set:
                return copy_set
        except Exception:
            pass

    # 2. Pointer was stale or missing — pick the oldest non-archived copy.
    copy_set = await col.find_one(
        {"user_id": user_id, "is_archived": False},
        sort=[("created_at", 1)],
    )
    if copy_set:
        await _set_active_copy_id(current_user, db, str(copy_set["_id"]))
        return copy_set

    # 3. No copy sets at all — create a default.
    now = datetime.now(timezone.utc)
    doc: Dict[str, Any] = {
        "user_id": user_id,
        "title": "Copy 1",
        "is_archived": False,
        "created_at": now,
        "updated_at": now,
    }
    result = await col.insert_one(doc)
    doc["_id"] = result.inserted_id
    await _set_active_copy_id(current_user, db, str(result.inserted_id))
    return doc


async def resolve_copy_id(
    copy_id: Optional[str],
    current_user: Dict[str, Any],
    db: DatabaseManager,
    *,
    allow_archived: bool = False,
) -> str:
    """Resolve an explicit *copy_id* or fall back to the user's active copy.

    When *copy_id* is provided it is validated against the user's copy_sets
    to ensure ownership.  Archived copies are rejected for writes unless
    *allow_archived* is ``True`` (reads may pass it).

    Importable by ``strokes_async`` and ``copies_async`` so every
    write/read path can transparently handle the ``copy_id`` field.
    """
    if copy_id:
        col = await get_copy_sets_collection(current_user, db)
        user_id = canonical_canvas_user_id(current_user)
        try:
            doc = await col.find_one({"_id": ObjectId(copy_id), "user_id": user_id})
        except Exception:
            doc = None
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid copy_id — copy set not found or not owned by current user",
            )
        if doc.get("is_archived") and not allow_archived:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot write to an archived copy set",
            )
        return copy_id
    copy_set = await ensure_active_copy(current_user, db)
    return str(copy_set["_id"])


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("")
async def list_copy_sets(
    archived: bool = Query(False, description="If true, list archived copy sets"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """List copy sets for the current user."""
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    query: Dict[str, Any] = {"user_id": user_id, "is_archived": archived}
    cursor = col.find(query).sort("created_at", 1)
    docs = await cursor.to_list(length=100)

    active_copy_id = await _get_active_copy_id_from_db(current_user, db)

    return {
        "success": True,
        "copy_sets": [_copy_set_to_response(d) for d in docs],
        "active_copy_id": active_copy_id,
    }


@router.get("/active")
async def get_active_copy_set(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Return the user's active copy set (creates a default if none exists)."""
    copy_set = await ensure_active_copy(current_user, db)
    return {
        "success": True,
        "copy_set": _copy_set_to_response(copy_set),
    }


@router.post("")
async def create_copy_set(
    body: CopySetCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Create a new copy set and auto-activate it."""
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    now = datetime.now(timezone.utc)
    title = body.title
    if not title:
        number = await _next_copy_number(col, user_id)
        title = f"Copy {number}"[:COPY_SET_TITLE_MAX_LENGTH]

    doc: Dict[str, Any] = {
        "user_id": user_id,
        "title": title,
        "is_archived": False,
        "created_at": now,
        "updated_at": now,
    }
    result = await col.insert_one(doc)
    doc["_id"] = result.inserted_id
    copy_id_str = str(result.inserted_id)

    await _set_active_copy_id(current_user, db, copy_id_str)

    return {
        "success": True,
        "copy_set": _copy_set_to_response(doc),
    }


@router.patch("/{copy_id}")
async def rename_copy_set(
    copy_id: str,
    body: CopySetUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Rename a copy set."""
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    now = datetime.now(timezone.utc)
    result = await col.update_one(
        {"_id": ObjectId(copy_id), "user_id": user_id},
        {"$set": {"title": body.title, "updated_at": now}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Copy set not found")

    doc = await col.find_one({"_id": ObjectId(copy_id)})
    return {"success": True, "copy_set": _copy_set_to_response(doc)}


@router.post("/{copy_id}/activate")
async def activate_copy_set(
    copy_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Activate a copy set (set it as the user's active copy)."""
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    copy_set = await col.find_one({"_id": ObjectId(copy_id), "user_id": user_id})
    if not copy_set:
        raise HTTPException(status_code=404, detail="Copy set not found")
    if copy_set.get("is_archived"):
        raise HTTPException(
            status_code=400, detail="Cannot activate an archived copy set — unarchive it first"
        )

    await _set_active_copy_id(current_user, db, copy_id)

    return {"success": True, "copy_set": _copy_set_to_response(copy_set)}


@router.post("/{copy_id}/archive")
async def archive_copy_set(
    copy_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Archive a copy set.

    If this is the active copy, promotes another non-archived copy or
    creates a new blank copy.
    """
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    copy_set = await col.find_one({"_id": ObjectId(copy_id), "user_id": user_id})
    if not copy_set:
        raise HTTPException(status_code=404, detail="Copy set not found")
    if copy_set.get("is_archived"):
        return {"success": True, "copy_set": _copy_set_to_response(copy_set)}

    now = datetime.now(timezone.utc)
    await col.update_one(
        {"_id": ObjectId(copy_id)},
        {"$set": {"is_archived": True, "updated_at": now}},
    )
    copy_set["is_archived"] = True
    copy_set["updated_at"] = now

    # If this was the active copy, promote another.
    active_copy_id = await _get_active_copy_id_from_db(current_user, db)
    if active_copy_id == copy_id:
        next_copy = await col.find_one(
            {"user_id": user_id, "is_archived": False, "_id": {"$ne": ObjectId(copy_id)}},
            sort=[("created_at", 1)],
        )
        if next_copy:
            await _set_active_copy_id(current_user, db, str(next_copy["_id"]))
        else:
            # All copies archived — create a fresh blank one.
            new_number = await _next_copy_number(col, user_id)
            new_doc: Dict[str, Any] = {
                "user_id": user_id,
                "title": f"Copy {new_number}",
                "is_archived": False,
                "created_at": now,
                "updated_at": now,
            }
            result = await col.insert_one(new_doc)
            await _set_active_copy_id(current_user, db, str(result.inserted_id))

    return {"success": True, "copy_set": _copy_set_to_response(copy_set)}


@router.post("/{copy_id}/unarchive")
async def unarchive_copy_set(
    copy_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Unarchive a copy set."""
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    now = datetime.now(timezone.utc)
    result = await col.update_one(
        {"_id": ObjectId(copy_id), "user_id": user_id},
        {"$set": {"is_archived": False, "updated_at": now}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Copy set not found")

    doc = await col.find_one({"_id": ObjectId(copy_id)})
    return {"success": True, "copy_set": _copy_set_to_response(doc)}


@router.delete("/{copy_id}")
async def delete_copy_set(
    copy_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Delete a copy set and **all** its pages / classifications."""
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    copy_set = await col.find_one({"_id": ObjectId(copy_id), "user_id": user_id})
    if not copy_set:
        raise HTTPException(status_code=404, detail="Copy set not found")

    # Resolve the target database for data collections.
    is_b2c = (
        current_user.get("is_b2c", False)
        or current_user.get("user_type") == "b2c_user"
    )
    if is_b2c:
        target_db = await db.get_b2c_db()
    else:
        target_db = await db.get_tenant_db(current_user.get("db_name"))

    if target_db:
        page_filter: Dict[str, Any] = {"user_id": user_id, "copy_id": copy_id}
        await target_db["canvas_pages"].delete_many(page_filter)
        await target_db["note_classifications"].delete_many(page_filter)
        await target_db["classification_queue"].delete_many(page_filter)

    await col.delete_one({"_id": ObjectId(copy_id)})

    # If this was the active copy, promote another (ensure_active_copy
    # will auto-create a default if the user has no copies left).
    active_copy_id = await _get_active_copy_id_from_db(current_user, db)
    if active_copy_id == copy_id:
        await ensure_active_copy(current_user, db)

    return {"success": True, "deleted_copy_id": copy_id}
