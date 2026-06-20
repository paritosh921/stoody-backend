"""
ExamPen Camera Fallback Upload API — photographed answer page ingest.

Handles:
  - Upload photographed answer pages from mobile app
  - PCR-only route enforcement (reject DCR exams)
  - Canonical image artifact persistence with exam/student/page provenance

Architecture:
    IMPLEMENTATION_PLAN.md §UP-005
    new-docs/api/copy-upload.openapi.yaml

Ownership Declaration:
    - Writes:  evalpen_answer_pages (via IngestService), exampen_camera_uploads (tracking)
    - Reads from: exampen_exams (lifecycle + exam_type validation)
    - Never writes to: exampen_exams

Hard constraints:
    - C1: MongoDB only
    - Camera fallback is PCR-only. DCR exams MUST be rejected with 400.
    - Exam must be in 'uploading' or 'collection_closed' lifecycle state
    - Provenance must include source="camera", captured_by="mobile"
    - Must never overwrite or pretend to be pen-originated data
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from core.upload_security.service import secure_upload
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for camera upload",
        )
    return current_user


# ---------------------------------------------------------------------------
# Tenant DB helper
# ---------------------------------------------------------------------------

async def _get_tenant_db(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Any:
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context missing from token",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CameraUploadAck(BaseModel):
    artifact_id: str
    exam_id: str
    student_id: str
    page_number: int
    exam_type: str
    routed_engine: str
    deduplicated: bool
    accepted_at: str


class CameraUploadStatus(BaseModel):
    exam_id: str
    total_uploads: int
    by_student: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

_indexes_ensured = False


async def _ensure_indexes(collection) -> None:
    global _indexes_ensured
    if _indexes_ensured:
        return
    await collection.create_index(
        [("exam_id", 1), ("student_id", 1), ("page_number", 1)],
        unique=True,
    )
    await collection.create_index("content_hash")
    _indexes_ensured = True


def _fmt(v) -> Optional[str]:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if v is not None:
        return str(v)
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{exam_id}/{student_id}/{page_num}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a photographed answer page",
    responses={
        400: {"description": "Camera fallback not allowed for DCR exams, or exam not in uploadable state"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found or student not in roster"},
        409: {"description": "Page already uploaded for this student"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def upload_camera_page(
    exam_id: str,
    student_id: str,
    page_num: int,
    image: UploadFile = File(..., description="Photographed answer page image"),
    source: str = Form("camera", description="camera or photographed_copy"),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> CameraUploadAck:
    """Upload a photographed answer page for a conducted exam.

    Camera fallback is PCR-only. Attempts to upload to a DCR exam
    are rejected with 400. The image is persisted as a canonical
    answer page artifact with camera provenance.
    """
    tenant_db = await _get_tenant_db(db, current_user)

    # Validate exam
    exam_doc = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    # DCR rejection guard
    exam_type = exam_doc.get("exam_type", "")
    if exam_type == "dcr":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera fallback not allowed for DCR exams",
        )

    # Lifecycle validation
    lifecycle = exam_doc.get("lifecycle_state", "draft")
    if lifecycle not in ("collection_closed", "uploading"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exam {exam_id} is in state '{lifecycle}' — must be 'collection_closed' or 'uploading' for camera upload",
        )

    # Roster validation
    roster = exam_doc.get("roster", [])
    if roster and student_id not in roster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student {student_id} not found in exam roster",
        )

    clean_upload = await secure_upload(
        file=image,
        policy_id="camera_answer_image",
        actor=current_user,
        db=db,
        purpose_metadata={
            "purpose": "camera_answer_image",
            "collection": "exampen_camera_uploads",
            "exam_id": exam_id,
            "student_id": student_id,
            "page_number": page_num,
            "created_by": current_user.get("user_id", "unknown"),
        },
        authorization_subject=f"camera:{exam_id}:{student_id}:{page_num}",
    )

    content_hash = clean_upload.sha256

    camera_col = tenant_db["exampen_camera_uploads"]
    await _ensure_indexes(camera_col)

    # Deduplication check
    existing = await camera_col.find_one({
        "exam_id": exam_id,
        "student_id": student_id,
        "page_number": page_num,
    })

    if existing:
        # Same page already uploaded — return dedup ack
        return CameraUploadAck(
            artifact_id=existing.get("artifact_id", str(existing["_id"])),
            exam_id=exam_id,
            student_id=student_id,
            page_number=page_num,
            exam_type="pcr",
            routed_engine="pcr",
            deduplicated=True,
            accepted_at=_fmt(existing.get("uploaded_at")) or datetime.now(timezone.utc).isoformat(),
        )

    now = datetime.now(timezone.utc)
    artifact_id = f"cam-{exam_id}-{student_id}-p{page_num}"

    # Store image artifact reference
    # Actual image bytes would go to S3/GridFS in production;
    # here we store the hash and metadata for canonical tracking.
    doc = {
        "artifact_id": artifact_id,
        "exam_id": exam_id,
        "student_id": student_id,
        "page_number": page_num,
        "source": source,
        "captured_by": "mobile",
        "content_hash": content_hash,
        "content_type": clean_upload.content_type or "image/jpeg",
        "file_size_bytes": clean_upload.size_bytes,
        "original_filename": clean_upload.original_filename,
        "upload_id": clean_upload.upload_id,
        "storage_path": clean_upload.released_storage_path,
        "uploaded_by": current_user.get("user_id", "unknown"),
        "uploaded_at": now,
        "routed_engine": "pcr",
    }

    try:
        await camera_col.insert_one(doc)
    except Exception as exc:
        if hasattr(exc, "code") and exc.code == 11000:
            return CameraUploadAck(
                artifact_id=artifact_id,
                exam_id=exam_id,
                student_id=student_id,
                page_number=page_num,
                exam_type="pcr",
                routed_engine="pcr",
                deduplicated=True,
                accepted_at=now.isoformat(),
            )
        raise

    # Bridge to canonical ingest substrate
    try:
        from api.v1._exampen_imports import load_exampen
        ingest_mod = load_exampen("ingest.service")
        IngestService = ingest_mod.IngestService

        service = IngestService(tenant_db)
        await service.initialize()

        await service.ingest_submission(
            exam_id=exam_id,
            student_id=student_id,
            admin_id=current_user.get("user_id", "unknown"),
            source="camera",
            pen_mac=None,
            pages=[{
                "page_number": page_num,
                "raw_strokes": None,
                "raw_image_ref": artifact_id,
            }],
        )

        logger.info(
            "Camera page ingested: exam=%s student=%s page=%d",
            exam_id, student_id, page_num,
        )
    except (ImportError, AttributeError):
        logger.warning(
            "IngestService not available — camera artifact saved but not canonically ingested"
        )
    except Exception:
        logger.exception(
            "Canonical ingest failed for camera upload: exam=%s student=%s page=%d",
            exam_id, student_id, page_num,
        )

    return CameraUploadAck(
        artifact_id=artifact_id,
        exam_id=exam_id,
        student_id=student_id,
        page_number=page_num,
        exam_type="pcr",
        routed_engine="pcr",
        deduplicated=False,
        accepted_at=now.isoformat(),
    )


@router.get(
    "/{exam_id}/status",
    summary="Get camera upload status for an exam",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_camera_upload_status(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> CameraUploadStatus:
    """Get camera upload progress for an exam, broken down by student."""
    tenant_db = await _get_tenant_db(db, current_user)

    exam_doc = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    pipeline = [
        {"$match": {"exam_id": exam_id}},
        {
            "$group": {
                "_id": "$student_id",
                "pages_uploaded": {"$sum": 1},
                "last_upload": {"$max": "$uploaded_at"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    cursor = tenant_db["exampen_camera_uploads"].aggregate(pipeline)
    results = await cursor.to_list(length=500)

    by_student = [
        {
            "student_id": r["_id"],
            "pages_uploaded": r["pages_uploaded"],
            "last_upload": _fmt(r.get("last_upload")),
        }
        for r in results
    ]

    total = sum(r["pages_uploaded"] for r in by_student)

    return CameraUploadStatus(
        exam_id=exam_id,
        total_uploads=total,
        by_student=by_student,
    )
