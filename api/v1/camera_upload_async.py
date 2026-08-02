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
import uuid
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
    completion_required: bool = True


class CameraSubmissionAck(BaseModel):
    exam_id: str
    student_id: str
    submission_id: str
    page_count: int
    processing_job_id: Optional[str] = None
    processing_status: Optional[str] = None
    accepted_at: str


class CameraUploadStatus(BaseModel):
    exam_id: str
    total_uploads: int
    by_student: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

_indexed_collections: set[str] = set()


async def _ensure_indexes(collection) -> None:
    # Indexes need to exist in each tenant database, not just the first one
    # served by this application process.
    collection_key = str(getattr(collection, "full_name", "")) or repr(collection)
    if collection_key in _indexed_collections:
        return
    await collection.create_index(
        [("exam_id", 1), ("student_id", 1), ("page_number", 1)],
        unique=True,
    )
    await collection.create_index("content_hash")
    _indexed_collections.add(collection_key)


def _fmt(v) -> Optional[str]:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if v is not None:
        return str(v)
    return None


async def _require_camera_upload_context(
    tenant_db: Any,
    *,
    db: DatabaseManager,
    exam_id: str,
    student_id: str,
    current_user: Dict[str, Any],
    allow_in_progress: bool = False,
) -> Dict[str, Any]:
    """Validate a camera copy against its conducted-session boundary."""
    exam_doc = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam_doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Exam {exam_id} not found")

    from api.v1.exam_orch_async import _require_tutor_visibility

    await _require_tutor_visibility(exam_doc, current_user, db)
    if exam_doc.get("exam_type") != "pcr":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera fallback is available for PCR exams only",
        )
    configured_capture_mode = exam_doc.get("capture_mode")
    if configured_capture_mode is not None and str(configured_capture_mode) not in {"camera", "hybrid"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera capture was not enabled when this exam session was created",
        )
    lifecycle = exam_doc.get("lifecycle_state", "draft")
    allowed_lifecycles = {"collection_closed", "uploading"}
    if allow_in_progress:
        allowed_lifecycles.add("in_progress")
    if lifecycle not in allowed_lifecycles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Exam {exam_id} is in state '{lifecycle}' and is not accepting "
                "this answer-copy upload"
            ),
        )
    roster = [str(item) for item in (exam_doc.get("roster") or [])]
    if roster and student_id not in roster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student {student_id} not found in exam roster",
        )
    return exam_doc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{exam_id}/{student_id}/complete",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CameraSubmissionAck,
    summary="Finalize a student's uploaded camera copy and queue PCR processing",
    responses={
        400: {"description": "Camera fallback unavailable or exam not uploadable"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam, student, or uploaded pages not found"},
        422: {"description": "Exam has no canonical owner"},
    },
)
async def complete_camera_submission(
    exam_id: str,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> CameraSubmissionAck:
    """Turn all uploaded pages for one student into one immutable submission.

    Page uploads are deliberately not processed as they arrive: a multi-page
    answer must be complete before OCR/segmentation runs, otherwise a fast
    worker can mark only page one and never revisit later pages.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    exam_doc = await _require_camera_upload_context(
        tenant_db,
        db=db,
        exam_id=exam_id,
        student_id=student_id,
        current_user=current_user,
    )
    camera_col = tenant_db["exampen_camera_uploads"]
    await _ensure_indexes(camera_col)
    cursor = camera_col.find({"exam_id": exam_id, "student_id": student_id}).sort("page_number", 1)
    pages = await cursor.to_list(length=500)
    if not pages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload at least one camera page before completing this copy",
        )

    canonical_admin_id = str(exam_doc.get("admin_id") or "").strip()
    if not canonical_admin_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Exam is missing its canonical admin owner",
        )

    try:
        from api.v1._exampen_imports import load_exampen

        IngestService = load_exampen("ingest.service").IngestService
        service = IngestService(tenant_db)
        await service.initialize()
        result = await service.ingest_submission(
            exam_id=exam_id,
            student_id=student_id,
            admin_id=canonical_admin_id,
            source="camera",
            pen_mac=None,
            pages=[
                {
                    "page_number": page["page_number"],
                    "raw_strokes": None,
                    # OCR receives the protected local/S3 reference, not a
                    # public URL and not a mutable client payload.
                    "raw_image_ref": page.get("storage_path") or page.get("artifact_id"),
                    "asset_sha256": page.get("content_hash"),
                    "content_hash": page.get("content_hash"),
                    "content_type": page.get("content_type"),
                    "file_size_bytes": page.get("file_size_bytes"),
                    "original_filename": page.get("original_filename"),
                    "upload_id": page.get("upload_id"),
                }
                for page in pages
            ],
        )
    except ImportError as exc:
        logger.error("Camera canonical ingest is unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera ingestion is not available in this deployment",
        )
    except Exception as exc:
        logger.exception("Camera submission ingest failed: exam=%s student=%s", exam_id, student_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not finalize the uploaded camera copy",
        ) from exc

    await camera_col.update_many(
        {"exam_id": exam_id, "student_id": student_id},
        {
            "$set": {
                "submission_id": result.submission_id,
                "submission_completed_at": datetime.now(timezone.utc),
            }
        },
    )

    processing_job_id = None
    processing_status = None
    try:
        from services.exampen_workflow import schedule_submission_processing

        job = await schedule_submission_processing(
            tenant_db,
            db_name=str(current_user.get("db_name") or ""),
            exam_id=exam_id,
            submission_id=result.submission_id,
            student_id=student_id,
        )
        processing_job_id = job.get("job_id")
        processing_status = job.get("status")
    except Exception:
        # The canonical submission is durable and the reconciler can dispatch
        # its persisted job later.  Surface the copy as accepted rather than
        # inviting the invigilator to upload it again.
        logger.exception("Unable to schedule camera PCR job for %s", result.submission_id)

    return CameraSubmissionAck(
        exam_id=exam_id,
        student_id=student_id,
        submission_id=result.submission_id,
        page_count=len(pages),
        processing_job_id=processing_job_id,
        processing_status=processing_status,
        accepted_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/{exam_id}/{student_id}/answer-copy",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CameraSubmissionAck,
    summary="Upload and queue one student's complete answer copy as staff",
    responses={
        400: {"description": "Invalid PDF/images or exam not uploadable"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam or student not found"},
        409: {"description": "A canonical submission already exists"},
    },
)
async def upload_complete_answer_copy(
    exam_id: str,
    student_id: str,
    pages: Optional[List[UploadFile]] = File(None),
    answer_pdf: Optional[UploadFile] = File(None),
    confirm_submission: bool = Form(True),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> CameraSubmissionAck:
    """Let staff submit the same canonical PDF/image copy a student can submit."""

    tenant_db = await _get_tenant_db(db, current_user)
    exam_doc = await _require_camera_upload_context(
        tenant_db,
        db=db,
        exam_id=exam_id,
        student_id=student_id,
        current_user=current_user,
        allow_in_progress=True,
    )
    if not confirm_submission:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirm the final answer-copy upload before submitting it",
        )

    image_files = [item for item in (pages or []) if item and item.filename]
    has_pdf = bool(answer_pdf and answer_pdf.filename)
    if has_pdf and image_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload either one PDF or one or more JPG/PNG pages, not both",
        )
    if not has_pdf and not image_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload one PDF or at least one answer-page image",
        )

    existing = await tenant_db["evalpen_submissions"].find_one(
        {"exam_id": exam_id, "student_id": student_id},
        {"submission_id": 1},
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A final answer copy already exists for this student",
        )

    max_pages = max(1, min(50, int(exam_doc.get("student_submission_max_pages") or 20)))
    if len(image_files) > max_pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This exam allows at most {max_pages} answer pages",
        )
    canonical_admin_id = str(exam_doc.get("admin_id") or "").strip()
    if not canonical_admin_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Exam is missing its canonical admin owner",
        )

    from api.v1.evalpen_student_submission_async import (
        _canonical_ingest,
        _cleanup_released_student_copy_paths,
        _delete_private_student_copy_objects,
        _queue_pcr_processing,
        _secure_student_copy_pages,
    )

    attempt_id = f"staff-{uuid.uuid4().hex}"
    released_local_paths: List[str] = []
    uploaded_storage_paths: List[str] = []
    try:
        (
            page_records,
            source_format,
            _original_asset,
            released_local_paths,
            uploaded_storage_paths,
            _verdict_transfers,
        ) = await _secure_student_copy_pages(
            image_files=image_files,
            answer_pdf=answer_pdf if has_pdf else None,
            current_user=current_user,
            tenant_db=tenant_db,
            db_name=str(current_user.get("db_name") or ""),
            exam_id=exam_id,
            student_id=student_id,
            attempt_id=attempt_id,
            max_pages=max_pages,
            upload_actor_id=str(
                current_user.get("user_id")
                or current_user.get("tutor_id")
                or current_user.get("admin_id")
                or "staff"
            ),
            authorization_scope="staff-answer-copy",
        )
        result = await _canonical_ingest(
            tenant_db,
            exam_id=exam_id,
            student_id=student_id,
            admin_id=canonical_admin_id,
            pages=page_records,
            source="scan" if source_format == "pdf" else "camera",
        )
    except HTTPException:
        await _delete_private_student_copy_objects(uploaded_storage_paths)
        raise
    except Exception as exc:
        await _delete_private_student_copy_objects(uploaded_storage_paths)
        logger.exception(
            "Staff answer-copy ingest failed: exam=%s student=%s",
            exam_id,
            student_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create the canonical answer-copy submission",
        ) from exc
    finally:
        await _cleanup_released_student_copy_paths(released_local_paths)

    processing_job_id = None
    processing_status = None
    try:
        job = await _queue_pcr_processing(
            tenant_db,
            db_name=str(current_user.get("db_name") or ""),
            exam_id=exam_id,
            submission_id=result.submission_id,
            student_id=student_id,
        )
        processing_job_id = job.get("job_id")
        processing_status = job.get("status")
    except Exception:
        logger.exception(
            "Unable to schedule staff-uploaded PCR copy %s",
            result.submission_id,
        )

    return CameraSubmissionAck(
        exam_id=exam_id,
        student_id=student_id,
        submission_id=result.submission_id,
        page_count=len(page_records),
        processing_job_id=processing_job_id,
        processing_status=processing_status,
        accepted_at=datetime.now(timezone.utc).isoformat(),
    )


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

    await _require_camera_upload_context(
        tenant_db,
        db=db,
        exam_id=exam_id,
        student_id=student_id,
        current_user=current_user,
    )
    if source not in {"camera", "photographed_copy"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera source must be 'camera' or 'photographed_copy'",
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
        if str(existing.get("content_hash") or "") != content_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This page number already contains different image bytes. "
                    "Remove or explicitly replace the existing draft page before completing the copy."
                ),
            )
        # Same bytes for the same page — return an idempotent acknowledgement.
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

    return CameraUploadAck(
        artifact_id=artifact_id,
        exam_id=exam_id,
        student_id=student_id,
        page_number=page_num,
        exam_type="pcr",
        routed_engine="pcr",
        deduplicated=False,
        accepted_at=now.isoformat(),
        completion_required=True,
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
