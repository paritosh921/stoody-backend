"""
Copy Sets API - Async Routes

Provides endpoints for managing student copy sets (physical notebook copies).
Each student can create, select, rename, archive, and delete copy sets.
All canvas page writes, hydration, and list views are scoped to copy_id.

The pen has no copy concept — copy_id is resolved from the user's
active_copy_id pointer, stored on the student profile document.

Conducted-exam bridge (SWM-012):
    Copy sets that contain conducted-exam pages can be detected via
    ``copy_set_has_exam_pages()``.  A dedicated ``/copy-sets/{copy_id}/exam-status``
    endpoint allows callers to check whether a copy set has been ingested
    into the shared ingest substrate.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from core.user_identity import canonical_canvas_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/copy-sets", tags=["Copy Sets"])
COPY_SET_TITLE_MAX_LENGTH = 11
DEFAULT_NEW_COPY_TITLE = "New Copy"
DEFAULT_PRACTICE_COPY_TITLE = "Practice"
PRACTICE_COPY_PURPOSE = "practice"
DEFAULT_COPY_SET_TITLES = (
    DEFAULT_NEW_COPY_TITLE,
    DEFAULT_PRACTICE_COPY_TITLE,
)
CANVAS_ONLY_COPY_ID_RE = re.compile(r"^(?:tally|online)-[A-Za-z0-9_.:-]{1,240}$")


def _is_canvas_only_copy_id(copy_id: str) -> bool:
    """Return true for virtual canvas scopes that are not physical copy sets."""
    return bool(CANVAS_ONLY_COPY_ID_RE.fullmatch(copy_id))


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
    purpose: Optional[str] = None
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
        "purpose": doc.get("purpose"),
        "is_archived": doc.get("is_archived", False),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


async def _next_copy_number(col, user_id: str) -> int:
    """Return the next auto-increment number for copy naming."""
    count = await col.count_documents({"user_id": user_id})
    return count + 1


async def ensure_practice_copy_set_for_user(
    current_user: Dict[str, Any],
    db: DatabaseManager,
) -> Dict[str, Any]:
    """Resolve the one semantic Practice copy owned by the current user.

    ``title`` is display data and is not a safe identity key. New deployments
    enforce a partial unique ``(user_id, purpose)`` index; this helper also
    claims one legacy title-only Practice copy so existing users keep their
    pages. The final upsert is idempotent and duplicate-key retry makes the
    endpoint safe when web and mobile enter Practice concurrently.
    """
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)
    now = datetime.now(timezone.utc)

    existing = await col.find_one(
        {"user_id": user_id, "purpose": PRACTICE_COPY_PURPOSE}
    )
    if existing:
        if (
            existing.get("is_archived")
            or existing.get("title") != DEFAULT_PRACTICE_COPY_TITLE
        ):
            await col.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "title": DEFAULT_PRACTICE_COPY_TITLE,
                        "is_archived": False,
                        "updated_at": now,
                    }
                },
            )
            existing.update(
                {
                    "title": DEFAULT_PRACTICE_COPY_TITLE,
                    "is_archived": False,
                    "updated_at": now,
                }
            )
        return existing

    legacy_filter = {
        "user_id": user_id,
        "title": DEFAULT_PRACTICE_COPY_TITLE,
        "purpose": {"$exists": False},
    }
    try:
        claimed = await col.find_one_and_update(
            legacy_filter,
            {
                "$set": {
                    "purpose": PRACTICE_COPY_PURPOSE,
                    "is_archived": False,
                    "updated_at": now,
                }
            },
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        claimed = None
    if claimed:
        return claimed

    semantic_filter = {"user_id": user_id, "purpose": PRACTICE_COPY_PURPOSE}
    try:
        resolved = await col.find_one_and_update(
            semantic_filter,
            {
                "$setOnInsert": {
                    "user_id": user_id,
                    "purpose": PRACTICE_COPY_PURPOSE,
                    "title": DEFAULT_PRACTICE_COPY_TITLE,
                    "is_archived": False,
                    "created_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        resolved = await col.find_one(semantic_filter)

    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The Practice notebook could not be resolved.",
        )
    return resolved


async def ensure_default_copy_sets_for_user(
    current_user: Dict[str, Any],
    db: DatabaseManager,
) -> List[Dict[str, Any]]:
    """Ensure the canonical default copy pair exists for a user.

    New users should start with:
    - ``New Copy``
    - ``Practice``

    If the user has no active copy pointer yet, ``New Copy`` becomes active.
    Returns the created/found copy-set documents sorted by creation time.
    """
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)
    now = datetime.now(timezone.utc)

    existing_cursor = col.find(
        {"user_id": user_id, "title": {"$in": list(DEFAULT_COPY_SET_TITLES)}}
    )
    existing_docs = await existing_cursor.to_list(length=10)
    existing_by_title = {doc.get("title"): doc for doc in existing_docs}

    created_docs: List[Dict[str, Any]] = []
    for title in DEFAULT_COPY_SET_TITLES:
        if title == DEFAULT_PRACTICE_COPY_TITLE:
            doc = await ensure_practice_copy_set_for_user(current_user, db)
            existing_by_title[title] = doc
            continue
        if title in existing_by_title:
            continue
        doc: Dict[str, Any] = {
            "user_id": user_id,
            "title": title,
            "is_archived": False,
            "created_at": now,
            "updated_at": now,
        }
        result = await col.insert_one(doc)
        doc["_id"] = result.inserted_id
        created_docs.append(doc)
        existing_by_title[title] = doc

    active_copy_id = await _get_active_copy_id_from_db(current_user, db)
    if not active_copy_id:
        preferred = existing_by_title.get(DEFAULT_NEW_COPY_TITLE)
        if preferred:
            await _set_active_copy_id(current_user, db, str(preferred["_id"]))

    docs = [existing_by_title[title] for title in DEFAULT_COPY_SET_TITLES if title in existing_by_title]
    docs.extend(created_docs)
    deduped: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        if "_id" in doc:
            deduped[str(doc["_id"])] = doc
    return sorted(deduped.values(), key=lambda doc: doc.get("created_at") or now)


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

    Creates the default ``New Copy`` + ``Practice`` pair if none exists.
    Returns the copy-set
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

    # 3. No copy sets at all — create the default pair and activate New Copy.
    docs = await ensure_default_copy_sets_for_user(current_user, db)
    for doc in docs:
        if doc.get("title") == DEFAULT_NEW_COPY_TITLE:
            return doc
    return docs[0]


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
        if _is_canvas_only_copy_id(str(copy_id)):
            return str(copy_id)

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


@router.post("/practice/resolve")
async def resolve_practice_copy_set(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Return the caller's canonical Practice copy, creating it if needed."""
    copy_set = await ensure_practice_copy_set_for_user(current_user, db)
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
    if title == DEFAULT_PRACTICE_COPY_TITLE:
        doc = await ensure_practice_copy_set_for_user(current_user, db)
        await _set_active_copy_id(current_user, db, str(doc["_id"]))
        return {
            "success": True,
            "copy_set": _copy_set_to_response(doc),
        }
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
    copy_set = await col.find_one({"_id": ObjectId(copy_id), "user_id": user_id})
    if not copy_set:
        raise HTTPException(status_code=404, detail="Copy set not found")
    if copy_set.get("purpose") == PRACTICE_COPY_PURPOSE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The system Practice notebook cannot be renamed.",
        )

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
    if copy_set.get("purpose") == PRACTICE_COPY_PURPOSE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The system Practice notebook cannot be archived.",
        )
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
    if copy_set.get("purpose") == PRACTICE_COPY_PURPOSE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The system Practice notebook cannot be deleted.",
        )

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


# ============================================================================
# CONDUCTED-EXAM BRIDGE HELPERS (SWM-012)
# ============================================================================

async def copy_set_has_exam_pages(
    copy_id: str,
    current_user: Dict[str, Any],
    db: DatabaseManager,
) -> bool:
    """Check whether any canvas page in this copy set carries conducted-exam
    metadata (``exam_id``).

    Importable by ``copies_async`` and other modules that need to know
    whether a copy set is associated with a conducted exam.
    """
    is_b2c = (
        current_user.get("is_b2c", False)
        or current_user.get("user_type") == "b2c_user"
    )
    if is_b2c:
        target_db = await db.get_b2c_db()
    else:
        db_name = current_user.get("db_name")
        target_db = await db.get_tenant_db(db_name) if db_name else None

    if target_db is None:
        return False

    user_id = canonical_canvas_user_id(current_user)
    page = await target_db["canvas_pages"].find_one(
        {
            "user_id": user_id,
            "copy_id": copy_id,
            "exam_id": {"$exists": True, "$ne": None},
        },
        projection={"_id": 1},
    )
    return page is not None


async def _get_exam_ingest_status_for_copy(
    copy_id: str,
    current_user: Dict[str, Any],
    db: DatabaseManager,
) -> Dict[str, Any]:
    """Return ingest status for conducted-exam pages within a copy set.

    Checks the ``evalpen_submissions`` collection for submissions matching
    the exam IDs found in this copy set's pages.  Returns a summary dict.
    """
    is_b2c = (
        current_user.get("is_b2c", False)
        or current_user.get("user_type") == "b2c_user"
    )
    if is_b2c:
        target_db = await db.get_b2c_db()
    else:
        db_name = current_user.get("db_name")
        target_db = await db.get_tenant_db(db_name) if db_name else None

    if target_db is None:
        return {"has_exam_pages": False, "exams": []}

    user_id = canonical_canvas_user_id(current_user)

    # Find distinct exam_ids in this copy set's pages
    canvas_col = target_db["canvas_pages"]
    pipeline = [
        {
            "$match": {
                "user_id": user_id,
                "copy_id": copy_id,
                "exam_id": {"$exists": True, "$ne": None},
            }
        },
        {"$group": {"_id": "$exam_id"}},
    ]
    cursor = canvas_col.aggregate(pipeline)
    exam_ids = [doc["_id"] async for doc in cursor]

    if not exam_ids:
        return {"has_exam_pages": False, "exams": []}

    # Check ingest status for each exam_id (graceful if collection missing)
    student_id = current_user.get("user_id") or current_user.get("student_id")
    exams_status: List[Dict[str, Any]] = []

    try:
        sub_col = target_db["evalpen_submissions"]
        for eid in exam_ids:
            query: Dict[str, Any] = {"exam_id": eid}
            if student_id:
                query["student_id"] = student_id
            sub = await sub_col.find_one(query, projection={
                "submission_id": 1,
                "content_hash": 1,
                "page_count": 1,
                "segmentation_status": 1,
            })
            exams_status.append({
                "exam_id": eid,
                "ingested": sub is not None,
                "submission_id": sub.get("submission_id") if sub else None,
                "page_count": sub.get("page_count", 0) if sub else 0,
                "segmentation_status": sub.get("segmentation_status") if sub else None,
            })
    except Exception:
        # Collection may not exist yet — that's fine
        for eid in exam_ids:
            exams_status.append({
                "exam_id": eid,
                "ingested": False,
                "submission_id": None,
                "page_count": 0,
                "segmentation_status": None,
            })

    return {
        "has_exam_pages": True,
        "exams": exams_status,
    }


@router.get("/{copy_id}/exam-status")
async def get_copy_set_exam_status(
    copy_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Check whether a copy set contains conducted-exam pages and their
    ingest status in the shared ingest substrate.

    Returns ``has_exam_pages: false`` for regular (non-exam) copy sets.
    For exam copy sets, returns per-exam ingest status including
    ``submission_id`` and ``segmentation_status``.
    """
    col = await get_copy_sets_collection(current_user, db)
    user_id = canonical_canvas_user_id(current_user)

    copy_set = await col.find_one({"_id": ObjectId(copy_id), "user_id": user_id})
    if not copy_set:
        raise HTTPException(status_code=404, detail="Copy set not found")

    status_info = await _get_exam_ingest_status_for_copy(
        copy_id, current_user, db
    )

    return {
        "success": True,
        "copy_id": copy_id,
        **status_info,
    }
