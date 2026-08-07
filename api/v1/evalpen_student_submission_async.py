"""Student self-submission for PCR conducted exams.

This router is intentionally separate from ``camera_upload_async``.  The
camera router is an invigilator/admin tool that accepts a staff-selected
student id; this router derives the student identity from the authenticated
JWT and only allows that student to submit one final copy for an opt-in PCR
session.

The endpoint reuses the canonical ingest substrate and PCR processing job
workflow.  That keeps teacher collection/review/result screens unchanged:
they continue to read ``evalpen_submissions`` and PCR evaluation records.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from api.v1.auth_async import get_database
from api.v1.evalpen_student_bff_async import _get_student_identity_ids, require_student
from core.database import DatabaseManager
from core.upload_security.service import CleanUpload, secure_upload, secure_upload_many
from core.upload_security.storage import PrivateUploadStorage, safe_storage_segment
from utils.s3_storage import (
    PrivateObjectStorageError,
    delete_private_object,
    upload_private_object,
)


logger = logging.getLogger(__name__)
router = APIRouter()

STUDENT_COPY_UPLOADS_COLLECTION = "exampen_student_copy_uploads"
PROCESSING_JOBS_COLLECTION = "exampen_processing_jobs"
ALLOWED_UPLOAD_LIFECYCLE_STATES = {"in_progress", "collection_closed", "uploading"}
PRIVATE_STUDENT_COPY_S3_PREFIX = "private/exampen/student-answer-copies"
STUDENT_COPY_RECEIVING_LEASE = timedelta(minutes=15)


class StudentCopyStatus(BaseModel):
    """Student-safe progress state.  Scores are intentionally not exposed."""

    submission_id: Optional[str] = None
    status: str = "not_submitted"
    page_count: int = 0
    submitted_at: Optional[str] = None
    processing_status: Optional[str] = None
    publication_status: Optional[str] = None


class StudentCopyExamOption(BaseModel):
    exam_id: str
    title: str
    paper_title: Optional[str] = None
    subject: Optional[str] = None
    code: Optional[str] = None
    question_paper_available: bool = False
    lifecycle_state: str
    max_pages: int = Field(ge=1, le=50)
    can_submit: bool
    unavailable_reason: Optional[str] = None
    submission: StudentCopyStatus = Field(default_factory=StudentCopyStatus)


class StudentCopyExamOptionsResponse(BaseModel):
    items: List[StudentCopyExamOption] = Field(default_factory=list)


class StudentCopySubmissionAck(BaseModel):
    exam_id: str
    submission_id: str
    page_count: int
    processing_job_id: Optional[str] = None
    processing_status: Optional[str] = None
    accepted_at: str


def _student_copy_object_key(
    *,
    db_name: str,
    exam_id: str,
    attempt_id: str,
    filename: str,
) -> str:
    """Build an opaque, private S3 key for answer-copy evidence.

    A random attempt id, not a student name or browser filename, identifies the
    object.  The key is used only by authenticated backend services and never
    exposed as a public URL.
    """
    return "/".join(
        (
            PRIVATE_STUDENT_COPY_S3_PREFIX,
            safe_storage_segment(db_name),
            safe_storage_segment(exam_id),
            safe_storage_segment(attempt_id),
            safe_storage_segment(filename),
        )
    )


def _image_extension_for_content_type(content_type: str) -> str:
    normalized = (content_type or "").lower()
    return "png" if normalized == "image/png" else "jpg"


async def _store_student_copy_object(
    *,
    data: bytes,
    db_name: str,
    exam_id: str,
    attempt_id: str,
    filename: str,
    content_type: str,
    artifact_kind: str,
    page_number: int | None = None,
) -> str:
    """Persist one clean answer-copy artefact to private S3 only."""
    metadata = {
        "purpose": "exam_answer_copy",
        "artifact": artifact_kind,
        "tenant": safe_storage_segment(db_name),
        "exam": safe_storage_segment(exam_id),
        "attempt": safe_storage_segment(attempt_id),
    }
    if page_number is not None:
        metadata["page"] = str(page_number)
    try:
        return await upload_private_object(
            data,
            object_key=_student_copy_object_key(
                db_name=db_name,
                exam_id=exam_id,
                attempt_id=attempt_id,
                filename=filename,
            ),
            content_type=content_type,
            metadata=metadata,
        )
    except PrivateObjectStorageError as exc:
        logger.error(
            "Private S3 storage failed for student answer copy: exam=%s attempt=%s: %s",
            exam_id,
            attempt_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Private answer-copy storage is temporarily unavailable. "
                "No answer copy was submitted; please try again."
            ),
        ) from exc


async def _delete_private_student_copy_objects(storage_paths: List[str]) -> None:
    """Remove partially uploaded answer-copy objects after a failed handoff."""
    for storage_path in storage_paths:
        try:
            await delete_private_object(
                storage_path,
                allowed_key_prefix=PRIVATE_STUDENT_COPY_S3_PREFIX,
            )
        except PrivateObjectStorageError:
            logger.exception(
                "Could not clean up private answer-copy object after failed handoff: %s",
                storage_path,
            )


async def _cleanup_released_student_copy_paths(released_paths: List[str]) -> List[str]:
    """Delete scan-stage EC2 artefacts once S3 is the canonical copy."""
    storage = PrivateUploadStorage()
    failed_paths: List[str] = []
    for released_path in released_paths:
        try:
            deleted = await storage.delete_released_path(released_path)
            if not deleted:
                failed_paths.append(released_path)
        except Exception:
            logger.exception(
                "Could not clean local student answer-copy scan artefact: %s",
                released_path,
            )
            failed_paths.append(released_path)
    return failed_paths


async def _update_upload_verdict_storage(
    tenant_db: Any,
    *,
    transfers: List[Dict[str, str]],
    status_value: str,
) -> None:
    """Keep upload-security audit records pointing at the canonical S3 object."""
    if not transfers:
        return
    now = datetime.now(timezone.utc)
    for transfer in transfers:
        update: Dict[str, Any] = {
            "storage_backend": "s3",
            "storage_transfer_status": status_value,
            "storage_transfer_updated_at": now,
        }
        if status_value == "complete":
            update["released_storage_path"] = transfer["storage_path"]
        else:
            update["released_storage_path"] = None
        try:
            await tenant_db["upload_security_verdicts"].update_one(
                {"upload_id": transfer["upload_id"]},
                {"$set": update},
            )
        except Exception:
            logger.exception(
                "Could not update storage audit for student answer-copy upload %s",
                transfer["upload_id"],
            )


async def _get_tenant_db(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Any:
    db_name = str(current_user.get("db_name") or "").strip()
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


async def _ensure_student_copy_indexes(collection: Any) -> None:
    """Keep audit lookups fast without changing canonical submission indexes."""
    await collection.create_index("attempt_id", unique=True, name="uniq_student_copy_attempt")
    # This is a separate, early reservation from the canonical ingest index.
    # It prevents two browser tabs (or a slow network retry) from preparing
    # different files for the same student's one immutable submission.
    await collection.create_index(
        [("exam_id", 1), ("student_id", 1)],
        unique=True,
        name="uniq_student_copy_exam_student",
    )
    await collection.create_index(
        [("exam_id", 1), ("student_id", 1), ("created_at", -1)],
        name="idx_student_copy_exam_student",
    )


async def _reserve_student_copy_attempt(
    collection: Any,
    *,
    attempt_id: str,
    exam_id: str,
    student_id: str,
    admin_id: str,
) -> str:
    """Atomically reserve the student's single final-copy upload slot.

    The canonical ingest has its own unique ``(exam_id, student_id)`` index,
    but reserving before file processing closes the window in which two web
    requests could both scan/store a different final copy.  A failed upload
    can safely retry because no canonical evidence was written.
    """
    now = datetime.now(timezone.utc)
    lease_expires_at = now + STUDENT_COPY_RECEIVING_LEASE
    try:
        await collection.insert_one(
            {
                "attempt_id": attempt_id,
                "exam_id": exam_id,
                "student_id": student_id,
                "admin_id": admin_id,
                "submitted_by": student_id,
                "submission_channel": "student_web",
                "status": "receiving",
                "lease_expires_at": lease_expires_at,
                "upload_attempt_count": 1,
                "created_at": now,
                "updated_at": now,
            }
        )
        return attempt_id
    except Exception as exc:
        if getattr(exc, "code", None) != 11000:
            raise

    existing = await collection.find_one(
        {"exam_id": exam_id, "student_id": student_id},
        projection={
            "attempt_id": 1,
            "status": 1,
            "submission_id": 1,
            "created_at": 1,
            "updated_at": 1,
            "lease_expires_at": 1,
        },
    )
    existing_status = str((existing or {}).get("status") or "")
    stale_receiving = existing_status == "receiving" and _receiving_attempt_is_stale(
        existing,
        now=now,
    )
    if (
        existing
        and (existing_status == "upload_failed" or stale_receiving)
        and not existing.get("submission_id")
        and existing.get("attempt_id")
    ):
        existing_attempt_id = str(existing["attempt_id"])
        retry_filter: Dict[str, Any] = {
            "attempt_id": existing_attempt_id,
            "status": existing_status,
            "submission_id": {"$exists": False},
        }
        if stale_receiving:
            retry_filter["updated_at"] = existing.get("updated_at")
        retry = await collection.update_one(
            retry_filter,
            {
                "$set": {
                    "status": "receiving",
                    "updated_at": now,
                    "lease_expires_at": lease_expires_at,
                    "last_error": None,
                },
                "$inc": {"upload_attempt_count": 1},
            },
        )
        if retry.matched_count == 1:
            return existing_attempt_id

    if existing and existing.get("submission_id"):
        detail = "A final answer copy has already been submitted for this exam"
    elif existing and str(existing.get("status") or "") == "ingest_failed":
        detail = "Your earlier copy needs teacher support before another final copy can be submitted"
    else:
        detail = "An answer-copy submission is already in progress. Please wait before trying again"
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def _receiving_attempt_is_stale(
    attempt: Optional[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> bool:
    """Return whether a crashed pre-ingest upload reservation may be retried."""
    if not attempt or str(attempt.get("status") or "") != "receiving":
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    lease_expires_at = attempt.get("lease_expires_at")
    if isinstance(lease_expires_at, datetime):
        if lease_expires_at.tzinfo is None:
            lease_expires_at = lease_expires_at.replace(tzinfo=timezone.utc)
        return lease_expires_at <= current
    updated_at = attempt.get("updated_at") or attempt.get("created_at")
    if not isinstance(updated_at, datetime):
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return updated_at <= current - STUDENT_COPY_RECEIVING_LEASE


async def _mark_attempt_upload_failed(
    collection: Any,
    *,
    attempt_id: str,
    reason: str,
) -> None:
    """Release a pre-ingest reservation for a safe retry after upload failure."""
    await collection.update_one(
        {"attempt_id": attempt_id, "status": "receiving"},
        {
            "$set": {
                "status": "upload_failed",
                "last_error": reason[:500],
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )


def _fmt(value: Any) -> Optional[str]:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value) if value is not None else None


def _student_upload_availability(exam: Dict[str, Any], student_id: str) -> tuple[bool, Optional[str]]:
    """Return whether a student may make their one final submission now."""
    if str(exam.get("exam_type") or "") != "pcr":
        return False, "This exam does not use PCR copy evaluation"
    if not bool(exam.get("student_self_submission_enabled", False)):
        return False, "Your teacher has not enabled answer-copy upload for this exam"
    if str(exam.get("capture_mode") or "pen") not in {"camera", "hybrid"}:
        return False, "This exam is not configured for photographed or scanned copies"

    roster = {str(item) for item in (exam.get("roster") or []) if str(item)}
    if not roster or student_id not in roster:
        return False, "You are not on this exam's submission roster"
    absent = {str(item) for item in (exam.get("absent_student_ids") or []) if str(item)}
    if student_id in absent:
        return False, "You are marked absent for this exam"

    lifecycle = str(exam.get("lifecycle_state") or "draft")
    if lifecycle not in ALLOWED_UPLOAD_LIFECYCLE_STATES:
        if lifecycle in {"draft", "armed"}:
            return False, "Submission opens when your teacher starts the exam"
        if lifecycle == "ready_for_eval":
            return False, "This exam is already in teacher review"
        return False, "This exam is not accepting answer-copy uploads"
    return True, None


def _student_copy_status_from_records(
    submission: Optional[Dict[str, Any]],
    job: Optional[Dict[str, Any]] = None,
) -> StudentCopyStatus:
    """Build the student-safe status from records already fetched by a caller."""
    if submission is None:
        return StudentCopyStatus()

    submission_id = str(submission.get("submission_id") or "")
    processing_status = str((job or {}).get("status") or "") or None
    publication_status = str(submission.get("publication_status") or "") or None
    if publication_status == "published":
        overall_status = "published"
    elif processing_status:
        overall_status = processing_status
    else:
        overall_status = str(submission.get("segmentation_status") or "submitted")
    return StudentCopyStatus(
        submission_id=submission_id or None,
        status=overall_status,
        page_count=int(submission.get("page_count") or 0),
        submitted_at=_fmt(submission.get("submitted_at")),
        processing_status=processing_status,
        publication_status=publication_status,
    )


async def _get_submission_status(
    tenant_db: Any,
    *,
    exam_id: str,
    student_id: str,
) -> StudentCopyStatus:
    submission = await tenant_db["evalpen_submissions"].find_one(
        {"exam_id": exam_id, "student_id": student_id},
        projection={
            "submission_id": 1,
            "page_count": 1,
            "submitted_at": 1,
            "segmentation_status": 1,
            "publication_status": 1,
        },
    )
    if submission is None:
        return StudentCopyStatus()

    submission_id = str(submission.get("submission_id") or "")
    job = await tenant_db[PROCESSING_JOBS_COLLECTION].find_one(
        {"submission_id": submission_id},
        projection={"status": 1},
    )
    return _student_copy_status_from_records(submission, job)


async def _get_copy_attempt_state(
    tenant_db: Any,
    *,
    exam_id: str,
    student_id: str,
) -> Optional[Dict[str, Any]]:
    """Read the non-canonical reservation only when no submission exists."""
    return await tenant_db[STUDENT_COPY_UPLOADS_COLLECTION].find_one(
        {"exam_id": exam_id, "student_id": student_id},
        projection={
            "status": 1,
            "created_at": 1,
            "updated_at": 1,
            "lease_expires_at": 1,
            "page_count": 1,
        },
    )


async def _get_student_exam_or_404(
    tenant_db: Any,
    *,
    exam_id: str,
    student_id: str,
) -> Dict[str, Any]:
    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")

    # Keep the feature undiscoverable to students outside the roster.  The
    # endpoint never accepts a client-supplied student id.
    roster = {str(item) for item in (exam.get("roster") or []) if str(item)}
    if student_id not in roster:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    return exam


async def _canonical_ingest(
    tenant_db: Any,
    *,
    exam_id: str,
    student_id: str,
    admin_id: str,
    pages: List[Dict[str, Any]],
    source: str = "camera",
) -> Any:
    """Write the immutable canonical record used by the existing PCR engine."""
    from api.v1._exampen_imports import load_exampen

    normalized_source = str(source or "camera").strip().lower()
    if normalized_source in {
        "image",
        "pdf",
        "photographed_copy",
        "scan",
        "upload",
    }:
        normalized_source = "camera"

    IngestService = load_exampen("ingest.service").IngestService
    service = IngestService(tenant_db)
    await service.initialize()
    return await service.ingest_submission(
        exam_id=exam_id,
        student_id=student_id,
        admin_id=admin_id,
        source=normalized_source,
        pen_mac=None,
        pages=pages,
    )


async def _queue_pcr_processing(
    tenant_db: Any,
    *,
    db_name: str,
    exam_id: str,
    submission_id: str,
    student_id: str,
) -> Dict[str, Any]:
    from services.exampen_workflow import schedule_submission_processing

    return await schedule_submission_processing(
        tenant_db,
        db_name=db_name,
        exam_id=exam_id,
        submission_id=submission_id,
        student_id=student_id,
    )


def _render_pdf_pages_sync(pdf_bytes: bytes, *, max_pages: int) -> List[tuple[bytes, int, int]]:
    """Render a scanned PDF into protected PNG page artifacts for camera OCR."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - deployment configuration
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF answer-copy conversion is not available in this deployment",
        ) from exc

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded answer-copy PDF could not be read",
        ) from exc

    try:
        page_count = int(document.page_count or 0)
        if page_count < 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The uploaded answer-copy PDF has no pages",
            )
        if page_count > max_pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This exam allows at most {max_pages} answer pages",
            )

        rendered: List[tuple[bytes, int, int]] = []
        matrix = fitz.Matrix(180 / 72, 180 / 72)
        for page_index in range(page_count):
            page = document[page_index]
            estimated_width = max(1, int(round(page.rect.width * matrix.a)))
            estimated_height = max(1, int(round(page.rect.height * matrix.d)))
            if estimated_width * estimated_height > 25_000_000:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A rendered answer-copy page is too large to process safely",
                )
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            if pixmap.width * pixmap.height > 25_000_000:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A rendered answer-copy page is too large to process safely",
                )
            image_bytes = pixmap.tobytes("png")
            if len(image_bytes) > 25 * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A rendered answer-copy page is too large to process safely",
                )
            rendered.append((image_bytes, int(pixmap.width), int(pixmap.height)))
        return rendered
    finally:
        document.close()


async def _prepare_pdf_pages(
    *,
    clean_pdf: CleanUpload,
    db_name: str,
    exam_id: str,
    attempt_id: str,
    max_pages: int,
    uploaded_storage_paths: List[str],
) -> List[Dict[str, Any]]:
    if not clean_pdf.bytes:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The uploaded answer-copy PDF was not available for conversion",
        )

    rendered = await asyncio.to_thread(
        _render_pdf_pages_sync,
        clean_pdf.bytes,
        max_pages=max_pages,
    )
    # S3 writes are network-bound. Serial page uploads made a four-page copy
    # take more than a minute before the browser received its acknowledgement.
    # A small bound keeps memory/network pressure predictable for large papers.
    upload_slots = asyncio.Semaphore(4)

    async def _store_page(
        page_number: int,
        image_bytes: bytes,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        async with upload_slots:
            storage_path = await _store_student_copy_object(
                data=image_bytes,
                db_name=db_name,
                exam_id=exam_id,
                attempt_id=attempt_id,
                filename=f"page-{page_number}.png",
                content_type="image/png",
                artifact_kind="rendered_page",
                page_number=page_number,
            )
        return {
            "page_number": page_number,
            "raw_image_ref": storage_path,
            "image_width_px": width,
            "image_height_px": height,
            "original_filename": clean_pdf.original_filename,
            "upload_id": clean_pdf.upload_id,
            "content_hash": hashlib.sha256(image_bytes).hexdigest(),
            "storage_path": storage_path,
            "content_type": "image/png",
            "file_size_bytes": len(image_bytes),
        }

    results = await asyncio.gather(
        *(
            _store_page(page_number, image_bytes, width, height)
            for page_number, (image_bytes, width, height) in enumerate(rendered, start=1)
        ),
        return_exceptions=True,
    )
    pages: List[Dict[str, Any]] = []
    first_error: Optional[BaseException] = None
    for result in results:
        if isinstance(result, BaseException):
            first_error = first_error or result
            continue
        uploaded_storage_paths.append(str(result["storage_path"]))
        pages.append(result)
    if first_error is not None:
        raise first_error
    return sorted(pages, key=lambda page: int(page["page_number"]))


async def _secure_student_copy_pages(
    *,
    image_files: List[UploadFile],
    answer_pdf: Optional[UploadFile],
    current_user: Dict[str, Any],
    tenant_db: Any,
    db_name: str,
    exam_id: str,
    student_id: str,
    attempt_id: str,
    max_pages: int,
    upload_actor_id: Optional[str] = None,
    authorization_scope: str = "student-answer-copy",
) -> tuple[
    List[Dict[str, Any]],
    str,
    Optional[Dict[str, Any]],
    List[str],
    List[str],
    List[Dict[str, str]],
]:
    """Scan then transfer a full copy to private S3 before PCR ingest.

    The upload-security scanner has a short-lived local quarantine/release
    stage.  S3 is the only canonical evidence location returned from this
    function; callers remove those local scan artefacts after a successful
    handoff (and on failed handoffs as well).
    """
    uploaded_storage_paths: List[str] = []
    released_local_paths: List[str] = []
    verdict_transfers: List[Dict[str, str]] = []

    try:
        if answer_pdf is not None and answer_pdf.filename:
            clean_pdf = await secure_upload(
                file=answer_pdf,
                policy_id="student_answer_copy_pdf",
                actor=current_user,
                db=tenant_db,
                purpose_metadata={
                    "purpose": "student_answer_copy_pdf",
                    "collection": STUDENT_COPY_UPLOADS_COLLECTION,
                    "exam_id": exam_id,
                    "student_id": student_id,
                    "attempt_id": attempt_id,
                    "created_by": upload_actor_id or student_id,
                },
                authorization_subject=f"{authorization_scope}:{exam_id}:{student_id}:{attempt_id}:pdf",
                include_bytes=True,
            )
            released_local_paths.append(clean_pdf.released_storage_path)
            if not clean_pdf.bytes:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="The uploaded answer-copy PDF was not available for transfer",
                )

            original_storage_path = await _store_student_copy_object(
                data=clean_pdf.bytes,
                db_name=db_name,
                exam_id=exam_id,
                attempt_id=attempt_id,
                filename="original.pdf",
                content_type=clean_pdf.content_type or "application/pdf",
                artifact_kind="original_pdf",
            )
            uploaded_storage_paths.append(original_storage_path)
            verdict_transfers.append(
                {
                    "upload_id": clean_pdf.upload_id,
                    "storage_path": original_storage_path,
                }
            )
            page_records = await _prepare_pdf_pages(
                clean_pdf=clean_pdf,
                db_name=db_name,
                exam_id=exam_id,
                attempt_id=attempt_id,
                max_pages=max_pages,
                uploaded_storage_paths=uploaded_storage_paths,
            )
            original_asset: Optional[Dict[str, Any]] = {
                "upload_id": clean_pdf.upload_id,
                "storage_path": original_storage_path,
                "filename": clean_pdf.original_filename,
                "content_type": clean_pdf.content_type,
                "size_bytes": clean_pdf.size_bytes,
                "sha256": clean_pdf.sha256,
            }
            source_format = "pdf"
        else:
            clean_pages = await secure_upload_many(
                files=image_files,
                policy_id="student_answer_copy_image",
                actor=current_user,
                db=tenant_db,
                purpose_metadata_factory=lambda upload, index: {
                    "purpose": "student_answer_copy_image",
                    "collection": STUDENT_COPY_UPLOADS_COLLECTION,
                    "exam_id": exam_id,
                    "student_id": student_id,
                    "attempt_id": attempt_id,
                    "page_number": index + 1,
                    "created_by": upload_actor_id or student_id,
                },
                authorization_subject_factory=lambda upload, index: (
                    f"{authorization_scope}:{exam_id}:{student_id}:{attempt_id}:page-{index + 1}"
                ),
                include_bytes=True,
            )
            page_records = []
            for page_number, clean_upload in enumerate(clean_pages, start=1):
                released_local_paths.append(clean_upload.released_storage_path)
                if not clean_upload.bytes:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="The uploaded answer page was not available for transfer",
                    )
                content_type = clean_upload.content_type or "image/jpeg"
                storage_path = await _store_student_copy_object(
                    data=clean_upload.bytes,
                    db_name=db_name,
                    exam_id=exam_id,
                    attempt_id=attempt_id,
                    filename=(
                        f"page-{page_number}."
                        f"{_image_extension_for_content_type(content_type)}"
                    ),
                    content_type=content_type,
                    artifact_kind="answer_page",
                    page_number=page_number,
                )
                uploaded_storage_paths.append(storage_path)
                verdict_transfers.append(
                    {
                        "upload_id": clean_upload.upload_id,
                        "storage_path": storage_path,
                    }
                )
                page_records.append(
                    {
                        "page_number": page_number,
                        "raw_image_ref": storage_path,
                        "original_filename": clean_upload.original_filename,
                        "upload_id": clean_upload.upload_id,
                        "content_hash": clean_upload.sha256,
                        "storage_path": storage_path,
                        "content_type": content_type,
                        "file_size_bytes": clean_upload.size_bytes,
                    }
                )
            original_asset = None
            source_format = "images"

        if not page_records:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload at least one readable answer page",
            )
        await _update_upload_verdict_storage(
            tenant_db,
            transfers=verdict_transfers,
            status_value="complete",
        )
        return (
            page_records,
            source_format,
            original_asset,
            released_local_paths,
            uploaded_storage_paths,
            verdict_transfers,
        )
    except Exception:
        await _delete_private_student_copy_objects(uploaded_storage_paths)
        await _cleanup_released_student_copy_paths(released_local_paths)
        await _update_upload_verdict_storage(
            tenant_db,
            transfers=verdict_transfers,
            status_value="failed",
        )
        raise


@router.get(
    "/exams/answer-copy-options",
    response_model=StudentCopyExamOptionsResponse,
    summary="List PCR answer-copy submission options for the authenticated student",
)
async def list_answer_copy_options(
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
) -> StudentCopyExamOptionsResponse:
    tenant_db = await _get_tenant_db(db, current_user)
    student_ids = await _get_student_identity_ids(tenant_db, current_user)
    student_id = student_ids[0]
    cursor = tenant_db["exampen_exams"].find(
        {
            "exam_type": "pcr",
            "student_self_submission_enabled": True,
            "roster": student_id,
        },
        projection={
            "exam_id": 1,
            "title": 1,
            "lifecycle_state": 1,
            "student_submission_max_pages": 1,
            "student_self_submission_enabled": 1,
            "exam_type": 1,
            "capture_mode": 1,
            "prepared_document_id": 1,
            "subject": 1,
            "code": 1,
            "exam_code": 1,
            "roster": 1,
            "absent_student_ids": 1,
        },
    ).sort("created_at", -1)
    exams = await cursor.to_list(length=100)

    # Fetch status data in fixed-size batches. The previous implementation ran
    # two extra Mongo queries for every exam, so the 5-second student poll could
    # exceed its client timeout as the exam history grew.
    exam_ids = [str(exam.get("exam_id") or "") for exam in exams if exam.get("exam_id")]
    submissions = (
        await tenant_db["evalpen_submissions"]
        .find(
            {"exam_id": {"$in": exam_ids}, "student_id": student_id},
            projection={
                "exam_id": 1,
                "submission_id": 1,
                "page_count": 1,
                "submitted_at": 1,
                "segmentation_status": 1,
                "publication_status": 1,
            },
        )
        .to_list(length=max(1, len(exam_ids)))
        if exam_ids
        else []
    )
    submission_by_exam = {
        str(item.get("exam_id") or ""): item
        for item in submissions
        if item.get("exam_id")
    }
    submission_ids = [
        str(item.get("submission_id") or "")
        for item in submissions
        if item.get("submission_id")
    ]
    jobs = (
        await tenant_db[PROCESSING_JOBS_COLLECTION]
        .find(
            {"submission_id": {"$in": submission_ids}},
            projection={"submission_id": 1, "status": 1, "updated_at": 1, "created_at": 1},
        )
        .sort("updated_at", -1)
        .to_list(length=max(1, len(submission_ids) * 5))
        if submission_ids
        else []
    )
    job_by_submission: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        submission_id = str(job.get("submission_id") or "")
        if submission_id and submission_id not in job_by_submission:
            job_by_submission[submission_id] = job
    attempts = (
        await tenant_db[STUDENT_COPY_UPLOADS_COLLECTION]
        .find(
            {"exam_id": {"$in": exam_ids}, "student_id": student_id},
            projection={
                "exam_id": 1,
                "status": 1,
                "created_at": 1,
                "updated_at": 1,
                "lease_expires_at": 1,
                "page_count": 1,
            },
        )
        .to_list(length=max(1, len(exam_ids)))
        if exam_ids
        else []
    )
    attempt_by_exam = {
        str(item.get("exam_id") or ""): item
        for item in attempts
        if item.get("exam_id")
    }

    items: List[StudentCopyExamOption] = []
    for exam in exams:
        exam_id = str(exam.get("exam_id") or "")
        can_submit, unavailable_reason = _student_upload_availability(exam, student_id)
        submission_record = submission_by_exam.get(exam_id)
        submission_id = str((submission_record or {}).get("submission_id") or "")
        submission = _student_copy_status_from_records(
            submission_record,
            job_by_submission.get(submission_id),
        )
        if submission.submission_id:
            can_submit = False
            unavailable_reason = "Your final answer copy has already been submitted"
        else:
            attempt = attempt_by_exam.get(exam_id)
            attempt_status = str((attempt or {}).get("status") or "")
            if attempt_status == "receiving" and _receiving_attempt_is_stale(attempt):
                can_submit = True
                unavailable_reason = None
            elif attempt_status in {"receiving", "received"}:
                can_submit = False
                unavailable_reason = "Your answer copy is still being prepared. Please wait before trying again"
            elif attempt_status == "ingest_failed":
                can_submit = False
                unavailable_reason = "Your copy needs teacher support before another final copy can be submitted"
        items.append(
            StudentCopyExamOption(
                exam_id=exam_id,
                title=str(exam.get("title") or "PCR exam"),
                paper_title=str(exam.get("paper_title") or exam.get("title") or "PCR paper"),
                subject=exam.get("subject"),
                code=exam.get("code") or exam.get("exam_code"),
                question_paper_available=bool(exam.get("prepared_document_id")),
                lifecycle_state=str(exam.get("lifecycle_state") or "draft"),
                max_pages=int(exam.get("student_submission_max_pages") or 20),
                can_submit=can_submit,
                unavailable_reason=unavailable_reason,
                submission=submission,
            )
        )
    return StudentCopyExamOptionsResponse(items=items)


@router.get(
    "/exams/{exam_id}/answer-copy",
    response_model=StudentCopyStatus,
    summary="Get the authenticated student's private answer-copy processing status",
)
async def get_answer_copy_status(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
) -> StudentCopyStatus:
    tenant_db = await _get_tenant_db(db, current_user)
    student_ids = await _get_student_identity_ids(tenant_db, current_user)
    student_id = student_ids[0]
    exam = await _get_student_exam_or_404(tenant_db, exam_id=exam_id, student_id=student_id)
    if str(exam.get("exam_type") or "") != "pcr" or not bool(
        exam.get("student_self_submission_enabled", False)
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    submission = await _get_submission_status(tenant_db, exam_id=exam_id, student_id=student_id)
    if submission.submission_id:
        return submission
    attempt = await _get_copy_attempt_state(tenant_db, exam_id=exam_id, student_id=student_id)
    if attempt is None:
        return submission
    return StudentCopyStatus(
        status=str(attempt.get("status") or "not_submitted"),
        page_count=int(attempt.get("page_count") or 0),
        submitted_at=_fmt(attempt.get("created_at")),
        processing_status=None,
        publication_status=None,
    )


async def _read_student_download_asset(storage_path: str) -> bytes:
    """Read a server-owned private asset without exposing its storage location."""
    import inspect
    from pathlib import Path

    value = str(storage_path or "").strip()
    if not value:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    if value.startswith("s3://"):
        from utils.s3_storage import download_file

        payload = download_file(value)
        if inspect.isawaitable(payload):
            payload = await payload
        if isinstance(payload, bytes):
            return payload
        if hasattr(payload, "read"):
            content = payload.read()
            if inspect.isawaitable(content):
                content = await content
            if isinstance(content, bytes):
                return content
        if isinstance(payload, str):
            downloaded_path = Path(payload)
            if downloaded_path.is_file():
                return downloaded_path.read_bytes()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    local_path = Path(value)
    if not local_path.is_absolute():
        local_path = Path(__file__).resolve().parents[2] / local_path
    if not local_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return local_path.read_bytes()


@router.get(
    "/exams/{exam_id}/question-paper/download",
    summary="Download the authenticated student's question paper",
)
async def download_question_paper(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
):
    from fastapi.responses import Response

    tenant_db = await _get_tenant_db(db, current_user)
    student_ids = await _get_student_identity_ids(tenant_db, current_user)
    exam = await _get_student_exam_or_404(
        tenant_db,
        exam_id=exam_id,
        student_id=student_ids[0],
    )
    prepared_document_id = str(exam.get("prepared_document_id") or "").strip()
    if not prepared_document_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question paper not found")
    document = await tenant_db["documents"].find_one(
        {"document_id": prepared_document_id},
        projection={
            "_id": 0,
            "file_path": 1,
            "storage_path": 1,
            "source_storage_path": 1,
        },
    )
    storage_path = str(
        (document or {}).get("file_path")
        or (document or {}).get("storage_path")
        or (document or {}).get("source_storage_path")
        or ""
    )
    content = await _read_student_download_asset(storage_path)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{exam_id}-question-paper.pdf"'},
    )


@router.get(
    "/exams/{exam_id}/answer-copy/download",
    summary="Download the authenticated student's submitted answer copy",
)
async def download_answer_copy(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
):
    import io
    import zipfile
    from fastapi.responses import Response

    tenant_db = await _get_tenant_db(db, current_user)
    student_ids = await _get_student_identity_ids(tenant_db, current_user)
    student_id = student_ids[0]
    await _get_student_exam_or_404(tenant_db, exam_id=exam_id, student_id=student_id)
    attempt = await _get_copy_attempt_state(tenant_db, exam_id=exam_id, student_id=student_id)
    if not attempt:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Answer copy not found")

    original_asset = attempt.get("original_asset") or {}
    original_path = str(original_asset.get("storage_path") or "").strip()
    if original_path:
        content = await _read_student_download_asset(original_path)
        is_pdf = str(original_asset.get("content_type") or "").lower() == "application/pdf"
        suffix = "pdf" if is_pdf else "bin"
        media_type = "application/pdf" if is_pdf else "application/octet-stream"
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{exam_id}-answer-copy.{suffix}"'},
        )

    pages = [page for page in (attempt.get("pages") or []) if page.get("storage_path")]
    if not pages:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Answer copy not found")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for index, page in enumerate(pages, start=1):
            page_content = await _read_student_download_asset(str(page.get("storage_path") or ""))
            content_type = str(page.get("content_type") or "").lower()
            extension = "png" if "png" in content_type else "jpg"
            bundle.writestr(f"page-{index}.{extension}", page_content)
    return Response(
        content=archive.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{exam_id}-answer-copy.zip"'},
    )


@router.post(
    "/exams/{exam_id}/answer-copy",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=StudentCopySubmissionAck,
    summary="Submit one final handwritten PCR answer copy as images or one PDF",
    responses={
        400: {"description": "Invalid files, incomplete confirmation, or upload window is closed"},
        403: {"description": "Student self-submission is not enabled for this session"},
        409: {"description": "A final answer copy was already submitted"},
    },
)
async def submit_answer_copy(
    exam_id: str,
    pages: List[UploadFile] = File(default=[]),
    answer_pdf: Optional[UploadFile] = File(default=None),
    confirm_submission: bool = Form(False),
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
) -> StudentCopySubmissionAck:
    """Submit one complete copy, then queue the existing PCR evaluator.

    The final confirmation is intentional: OCR must not run on a partial copy.
    Canonical submissions are immutable, so a second different copy is rejected
    rather than silently overwriting the student's evidence.
    """
    if not confirm_submission:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirm that all answer pages are included before submitting",
        )

    tenant_db = await _get_tenant_db(db, current_user)
    student_ids = await _get_student_identity_ids(tenant_db, current_user)
    student_id = student_ids[0]
    exam = await _get_student_exam_or_404(tenant_db, exam_id=exam_id, student_id=student_id)
    can_submit, unavailable_reason = _student_upload_availability(exam, student_id)
    if not can_submit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=unavailable_reason or "This exam is not accepting answer-copy uploads",
        )

    existing_submission = await tenant_db["evalpen_submissions"].find_one(
        {"exam_id": exam_id, "student_id": student_id},
        projection={"submission_id": 1},
    )
    if existing_submission is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A final answer copy has already been submitted for this exam",
        )

    image_files = [upload for upload in pages if upload and upload.filename]
    has_pdf = bool(answer_pdf and answer_pdf.filename)
    if bool(image_files) == has_pdf:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload either one PDF or one or more JPG/PNG answer pages",
        )

    max_pages = int(exam.get("student_submission_max_pages") or 20)
    if len(image_files) > max_pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This exam allows at most {max_pages} answer pages",
        )

    admin_id = str(exam.get("admin_id") or "").strip()
    if not admin_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Exam is missing its canonical admin owner",
        )

    db_name = str(current_user.get("db_name") or "")
    attempt_id = f"student-copy-{uuid.uuid4().hex}"
    upload_collection = tenant_db[STUDENT_COPY_UPLOADS_COLLECTION]
    await _ensure_student_copy_indexes(upload_collection)
    attempt_id = await _reserve_student_copy_attempt(
        upload_collection,
        attempt_id=attempt_id,
        exam_id=exam_id,
        student_id=student_id,
        admin_id=admin_id,
    )

    released_local_paths: List[str] = []
    uploaded_storage_paths: List[str] = []
    verdict_transfers: List[Dict[str, str]] = []
    try:
        (
            page_records,
            source_format,
            original_asset,
            released_local_paths,
            uploaded_storage_paths,
            verdict_transfers,
        ) = await _secure_student_copy_pages(
            image_files=image_files,
            answer_pdf=answer_pdf if has_pdf else None,
            current_user=current_user,
            tenant_db=tenant_db,
            db_name=db_name,
            exam_id=exam_id,
            student_id=student_id,
            attempt_id=attempt_id,
            max_pages=max_pages,
        )
    except HTTPException as exc:
        await _mark_attempt_upload_failed(
            upload_collection,
            attempt_id=attempt_id,
            reason=str(exc.detail),
        )
        raise
    except Exception as exc:
        logger.exception("Student answer-copy upload failed: exam=%s student=%s", exam_id, student_id)
        await _mark_attempt_upload_failed(
            upload_collection,
            attempt_id=attempt_id,
            reason=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Your answer copy could not be safely prepared. Please try again.",
        ) from exc

    now = datetime.now(timezone.utc)
    attempt_doc = {
        "attempt_id": attempt_id,
        "exam_id": exam_id,
        "student_id": student_id,
        "admin_id": admin_id,
        "submitted_by": student_id,
        "submission_channel": "student_web",
        "source_format": source_format,
        "storage_backend": "s3",
        "storage_handoff_status": "complete",
        "original_asset": original_asset,
        "pages": [
            {
                "page_number": page["page_number"],
                "storage_path": page["storage_path"],
                "original_filename": page.get("original_filename"),
                "upload_id": page.get("upload_id"),
                "content_hash": page.get("content_hash"),
                "content_type": page.get("content_type", "image/png"),
                "file_size_bytes": page.get("file_size_bytes"),
            }
            for page in page_records
        ],
        "page_count": len(page_records),
        "status": "received",
        "created_at": now,
        "updated_at": now,
    }
    await upload_collection.update_one(
        {"attempt_id": attempt_id, "status": "receiving"},
        {"$set": attempt_doc},
    )

    # A staff camera/pen capture can arrive while the student file is being
    # scanned.  Recheck before canonical ingest so the student's files never
    # silently become a second version of the same immutable evidence.
    canonical_after_upload = await tenant_db["evalpen_submissions"].find_one(
        {"exam_id": exam_id, "student_id": student_id},
        projection={"submission_id": 1},
    )
    if canonical_after_upload is not None:
        await _delete_private_student_copy_objects(uploaded_storage_paths)
        await _cleanup_released_student_copy_paths(released_local_paths)
        await _update_upload_verdict_storage(
            tenant_db,
            transfers=verdict_transfers,
            status_value="failed",
        )
        await upload_collection.update_one(
            {"attempt_id": attempt_id},
            {
                "$set": {
                    "status": "superseded",
                    "submission_id": canonical_after_upload.get("submission_id"),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A final answer copy was received for this exam while your upload was being prepared",
        )

    try:
        result = await _canonical_ingest(
            tenant_db,
            exam_id=exam_id,
            student_id=student_id,
            admin_id=admin_id,
            pages=[
                {
                    "page_number": page["page_number"],
                    "raw_strokes": None,
                    "raw_image_ref": page["raw_image_ref"],
                }
                for page in page_records
            ],
        )
    except Exception as exc:
        logger.exception("Student answer-copy ingest failed: exam=%s student=%s", exam_id, student_id)
        await _delete_private_student_copy_objects(uploaded_storage_paths)
        await _cleanup_released_student_copy_paths(released_local_paths)
        await _update_upload_verdict_storage(
            tenant_db,
            transfers=verdict_transfers,
            status_value="failed",
        )
        await upload_collection.update_one(
            {"attempt_id": attempt_id},
            {"$set": {"status": "ingest_failed", "last_error": str(exc)[:500], "updated_at": datetime.now(timezone.utc)}},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Your copy was received but could not be prepared for evaluation. Please contact your teacher.",
        ) from exc

    if getattr(result, "already_existed", False):
        await _delete_private_student_copy_objects(uploaded_storage_paths)
        await _cleanup_released_student_copy_paths(released_local_paths)
        await _update_upload_verdict_storage(
            tenant_db,
            transfers=verdict_transfers,
            status_value="failed",
        )
        await upload_collection.update_one(
            {"attempt_id": attempt_id},
            {
                "$set": {
                    "status": "superseded",
                    "submission_id": result.submission_id,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A final answer copy was received for this exam while your upload was being prepared",
        )

    # The immutable submission now points exclusively to private S3 objects.
    # Remove the scanner's local release artefacts; OCR workers will only read
    # the durable S3 references persisted above.
    cleanup_failures = await _cleanup_released_student_copy_paths(released_local_paths)
    if cleanup_failures:
        logger.error(
            "Student answer-copy local cleanup incomplete: exam=%s attempt=%s count=%d",
            exam_id,
            attempt_id,
            len(cleanup_failures),
        )
        await upload_collection.update_one(
            {"attempt_id": attempt_id},
            {
                "$set": {
                    "local_scan_cleanup_status": "failed",
                    "local_scan_cleanup_failed_count": len(cleanup_failures),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
    else:
        await upload_collection.update_one(
            {"attempt_id": attempt_id},
            {
                "$set": {
                    "local_scan_cleanup_status": "complete",
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    processing_job: Optional[Dict[str, Any]] = None
    try:
        processing_job = await _queue_pcr_processing(
            tenant_db,
            db_name=db_name,
            exam_id=exam_id,
            submission_id=result.submission_id,
            student_id=student_id,
        )
    except Exception as exc:  # durable ingest succeeds even if the queue is briefly unavailable
        logger.exception("Could not queue PCR processing for student copy %s", result.submission_id)
        processing_job = {"status": "enqueue_failed", "last_error": str(exc)[:500]}

    await upload_collection.update_one(
        {"attempt_id": attempt_id},
        {
            "$set": {
                "status": "queued",
                "submission_id": result.submission_id,
                "processing_job_id": (processing_job or {}).get("job_id"),
                "processing_status": (processing_job or {}).get("status"),
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )

    return StudentCopySubmissionAck(
        exam_id=exam_id,
        submission_id=result.submission_id,
        page_count=len(page_records),
        processing_job_id=(processing_job or {}).get("job_id"),
        processing_status=(processing_job or {}).get("status"),
        accepted_at=datetime.now(timezone.utc).isoformat(),
    )
