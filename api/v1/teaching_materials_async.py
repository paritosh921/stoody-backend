"""
Teaching Materials API

Tutor-uploaded media (images, videos, PDFs, PPTs) that can be:
  - Uploaded from the web frontend (stoody-frontend) by a logged-in tutor
  - Listed and pulled by the smartboard app (sb-android) after passcode pairing

Auth: standard tutor JWT — works for both web login AND smartboard pairing
(smartboard JWT is a tutor JWT with the `smartboard_cloud_access` feature flag).

Storage: S3 via utils.s3_storage with prefix
    teaching-materials/{tenant_id}/{tutor_id}/{uuid}.{ext}

Per-tutor quota: STOODY_MATERIALS_QUOTA_BYTES (default 5 GB).

NOTE: deliberately NOT using `from __future__ import annotations`.
FastAPI introspects annotations to build the OpenAPI schema and stringified
ForwardRefs (e.g. for UploadFile) break that.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from utils.s3_storage import (
    upload_file as s3_upload_file,
    delete_file as s3_delete_file,
    get_public_url,
    is_s3_enabled,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/teaching-materials", tags=["Teaching Materials"])
limiter = Limiter(key_func=get_remote_address)

COLLECTION = "teaching_materials"

ALLOWED_TYPES = {"image", "video", "pdf", "ppt"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024              # 50 MB per file
TUTOR_QUOTA_BYTES = int(os.getenv("STOODY_MATERIALS_QUOTA_BYTES", str(5 * 1024 * 1024 * 1024)))  # 5 GB
SIGNED_URL_TTL = 3600                                 # 1 hour

# MIME type allowlist by category. Browsers and curl set these reliably.
MIME_BY_TYPE: Dict[str, set] = {
    "image": {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"},
    "video": {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"},
    "pdf":   {"application/pdf"},
    "ppt":   {
        "application/vnd.ms-powerpoint",                                              # .ppt
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    },
}


# ── Pydantic models ─────────────────────────────────────────────────────────

class MaterialOut(BaseModel):
    id: str
    type: str
    filename: str
    mime_type: str
    size_bytes: int
    url: str               # short-lived signed URL
    thumbnail_url: Optional[str] = None
    page_count: Optional[int] = None
    uploaded_at: datetime


class MaterialListResponse(BaseModel):
    items: List[MaterialOut]
    total: int
    page: int
    page_size: int
    quota_used_bytes: int
    quota_total_bytes: int


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_tutor_identity(user: Dict[str, Any]) -> Dict[str, str]:
    tutor_id = user.get("tutor_id") or user.get("user_id") or user.get("_id")
    tenant_id = user.get("tenant_id") or user.get("institution_id") or user.get("db_name")
    if not tutor_id or not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid token — missing tutor or tenant context")
    return {"tutor_id": str(tutor_id), "tenant_id": str(tenant_id)}


def _validate_type_mime(material_type: str, mime: str, filename: str) -> None:
    if material_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid type. Allowed: {', '.join(sorted(ALLOWED_TYPES))}")
    allowed_mimes = MIME_BY_TYPE[material_type]
    if mime not in allowed_mimes:
        raise HTTPException(
            status_code=400,
            detail=f"MIME {mime!r} not allowed for type {material_type!r}. Allowed: {sorted(allowed_mimes)}"
        )
    if any(c in filename for c in ("..", "/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid filename")


def _s3_key(tenant_id: str, tutor_id: str, mat_id: str, filename: str) -> str:
    ext = (os.path.splitext(filename)[1] or "").lower()[:8]
    return f"teaching-materials/{tenant_id}/{tutor_id}/{mat_id}{ext}"


async def _tutor_quota_used(db: DatabaseManager, tutor_id: str, tenant_id: str) -> int:
    """Sum of size_bytes for this tutor. Returns 0 if no docs."""
    try:
        pipeline = [
            {"$match": {"tutor_id": tutor_id, "tenant_id": tenant_id}},
            {"$group": {"_id": None, "total": {"$sum": "$size_bytes"}}},
        ]
        results = await db.mongo_aggregate(COLLECTION, pipeline)
        if results and len(results) > 0:
            return int(results[0].get("total") or 0)
    except Exception as e:
        logger.warning(f"quota aggregate failed: {e}")
    return 0


def _to_out(doc: Dict[str, Any]) -> MaterialOut:
    return MaterialOut(
        id=str(doc.get("_id")),
        type=doc["type"],
        filename=doc["filename"],
        mime_type=doc["mime_type"],
        size_bytes=int(doc["size_bytes"]),
        url=get_public_url(doc["s3_key"], expires_in=SIGNED_URL_TTL),
        thumbnail_url=(
            get_public_url(doc["thumbnail_s3_key"], expires_in=SIGNED_URL_TTL)
            if doc.get("thumbnail_s3_key") else None
        ),
        page_count=doc.get("page_count"),
        uploaded_at=doc.get("uploaded_at") or datetime.now(timezone.utc),
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/upload", response_model=MaterialOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("60/hour")
async def upload_material(
    request: Request,
    file: UploadFile = File(...),
    material_type: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
) -> MaterialOut:
    identity = _extract_tutor_identity(current_user)
    tutor_id = identity["tutor_id"]
    tenant_id = identity["tenant_id"]

    if not is_s3_enabled():
        raise HTTPException(status_code=503, detail="S3 storage not configured")

    _validate_type_mime(material_type, file.content_type or "", file.filename or "untitled")

    file_bytes = await file.read()
    size = len(file_bytes)
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size} bytes). Max {MAX_FILE_SIZE_BYTES} bytes."
        )
    if size <= 0:
        raise HTTPException(status_code=400, detail="Empty file")

    used = await _tutor_quota_used(db, tutor_id, tenant_id)
    if used + size > TUTOR_QUOTA_BYTES:
        raise HTTPException(
            status_code=507,  # Insufficient Storage
            detail=f"Quota exceeded ({used + size}/{TUTOR_QUOTA_BYTES} bytes). Delete some files first."
        )

    mat_id = uuid.uuid4().hex
    s3_key = _s3_key(tenant_id, tutor_id, mat_id, file.filename or "file")

    success, storage_path = await s3_upload_file(
        file_data=file_bytes,
        local_path=s3_key,
        content_type=file.content_type or "application/octet-stream",
    )
    if not success:
        raise HTTPException(status_code=502, detail="Upload to storage failed")

    page_count: Optional[int] = None
    if material_type == "pdf":
        try:
            import io
            from pypdf import PdfReader
            page_count = len(PdfReader(io.BytesIO(file_bytes)).pages)
        except Exception as e:
            logger.warning(f"PDF page count failed for {mat_id}: {e}")

    now = datetime.now(timezone.utc)
    doc = {
        "tutor_id": tutor_id,
        "tenant_id": tenant_id,
        "type": material_type,
        "filename": file.filename or f"{material_type}-{mat_id}",
        "mime_type": file.content_type or "application/octet-stream",
        "size_bytes": size,
        "s3_key": storage_path,
        "thumbnail_s3_key": None,   # TODO: generate thumbnails for video/pdf in a follow-up
        "page_count": page_count,
        "uploaded_at": now,
        "updated_at": now,
    }
    inserted_id = await db.mongo_insert_one(COLLECTION, doc)
    if not inserted_id:
        # Best-effort: try to clean the orphan S3 object
        try:
            await s3_delete_file(storage_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="DB insert failed")

    doc["_id"] = inserted_id
    logger.info(f"[MATERIALS] tutor={tutor_id} uploaded {material_type} id={inserted_id} size={size}")
    return _to_out(doc)


@router.get("", response_model=MaterialListResponse)
@limiter.limit("300/minute")
async def list_materials(
    request: Request,
    material_type: Optional[str] = Query(None, description="Filter by image/video/pdf/ppt"),
    page: int = Query(1, ge=1),
    page_size: int = Query(40, ge=1, le=200),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
) -> MaterialListResponse:
    identity = _extract_tutor_identity(current_user)
    tutor_id = identity["tutor_id"]
    tenant_id = identity["tenant_id"]

    filt: Dict[str, Any] = {"tutor_id": tutor_id, "tenant_id": tenant_id}
    if material_type:
        if material_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail="Invalid material_type")
        filt["type"] = material_type

    total = await db.mongo_count(COLLECTION, filt)
    skip = (page - 1) * page_size
    docs = await db.mongo_find(
        COLLECTION, filt,
        sort=[("uploaded_at", -1)],
        skip=skip, limit=page_size,
    )

    items = [_to_out(d) for d in docs]
    used = await _tutor_quota_used(db, tutor_id, tenant_id)

    return MaterialListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        quota_used_bytes=used,
        quota_total_bytes=TUTOR_QUOTA_BYTES,
    )


@router.get("/{material_id}", response_model=MaterialOut)
@limiter.limit("300/minute")
async def get_material(
    request: Request,
    material_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
) -> MaterialOut:
    identity = _extract_tutor_identity(current_user)
    try:
        oid_filter = {"_id": ObjectId(material_id)}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid material id")

    doc = await db.mongo_find_one(COLLECTION, oid_filter)
    if not doc:
        raise HTTPException(status_code=404, detail="Material not found")
    if doc.get("tutor_id") != identity["tutor_id"] or doc.get("tenant_id") != identity["tenant_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this material")

    return _to_out(doc)


@router.delete("/{material_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
async def delete_material(
    request: Request,
    material_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
) -> None:
    identity = _extract_tutor_identity(current_user)
    try:
        oid_filter = {"_id": ObjectId(material_id)}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid material id")

    doc = await db.mongo_find_one(COLLECTION, oid_filter)
    if not doc:
        raise HTTPException(status_code=404, detail="Material not found")
    if doc.get("tutor_id") != identity["tutor_id"] or doc.get("tenant_id") != identity["tenant_id"]:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Delete S3 object first (best-effort), then DB row.
    try:
        await s3_delete_file(doc["s3_key"])
        if doc.get("thumbnail_s3_key"):
            await s3_delete_file(doc["thumbnail_s3_key"])
    except Exception as e:
        logger.warning(f"S3 delete failed for material {material_id}: {e}")

    await db.mongo_delete_one(COLLECTION, oid_filter)
    logger.info(f"[MATERIALS] tutor={identity['tutor_id']} deleted material {material_id}")
