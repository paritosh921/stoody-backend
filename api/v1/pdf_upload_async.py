"""
PDF upload endpoint.
"""

import asyncio
import base64
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import aiofiles
from bson import ObjectId as BsonObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_cache, get_database
from api.v1.pdf_dependencies import require_admin
from core.cache import CacheManager
from core.database import DatabaseManager
from services.pdf_data_access import find_one, insert_one, update_one
from services.pdf_processing_service import run_document_ocr_pipeline
from utils.s3_storage import upload_file as s3_upload_file, is_s3_enabled

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/upload")
@limiter.limit("10/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    document_id: str = Form(...),
    title: str = Form(...),
    document_type: str = Form(...),
    subject: str = Form(...),
    difficulty: Optional[str] = Form("medium"),
    course_plan: Optional[str] = Form("CBSE"),
    standard: Optional[str] = Form("11"),
    section: Optional[str] = Form(None),  # Section A-F for filtering
    teacher_ids: Optional[str] = Form(None),  # Comma-separated teacher IDs for filtering
    total_points: Optional[float] = Form(None),
    total_minutes: Optional[int] = Form(None),
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Upload PDF file and save metadata (without OCR processing).
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )

        if not document_id.isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document ID must be alphanumeric only (no spaces or special characters)"
            )

        is_b2c = current_user.get("user_type") == "b2c_admin"

        existing_doc = await find_one(db, "documents", {"document_id": document_id}, is_b2c)
        if existing_doc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document ID '{document_id}' already exists"
            )

        allowed_types = ["Practice Sets", "Test Series", "Chapter Notes"]
        if document_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document type. Allowed: {', '.join(allowed_types)}"
            )

        if len(title) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document title must not exceed 100 characters"
            )

        file_content = await file.read()
        file_size = len(file_content)

        pages_count = 0
        try:
            import io
            from pypdf import PdfReader
            pdf_reader = PdfReader(io.BytesIO(file_content))
            pages_count = len(pdf_reader.pages)
            logger.info("PDF %s has %s pages", document_id, pages_count)
        except Exception as pdf_err:
            logger.warning("Failed to count PDF pages for %s: %s", document_id, pdf_err)

        logger.info(
            "Uploading document: %s, Title: %s, Type: %s, Size: %s bytes",
            document_id,
            title,
            document_type,
            file_size
        )

        from pathlib import Path
        backend_dir = Path(os.getcwd())
        upload_dir = backend_dir / "uploads" / "documents" / document_type
        file_path = upload_dir / f"{document_id}.pdf"

        local_relative_path = f"uploads/documents/{document_type}/{document_id}.pdf"

        if is_s3_enabled():
            success, storage_path = await s3_upload_file(
                file_data=file_content,
                local_path=str(file_path),
                content_type="application/pdf"
            )
            if success:
                relative_path = storage_path
                logger.info("✅ Uploaded PDF to S3: %s", storage_path)
            else:
                logger.warning("S3 upload failed, falling back to local storage")
                upload_dir.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(str(file_path), "wb") as file_handle:
                    await file_handle.write(file_content)
                relative_path = local_relative_path
        else:
            upload_dir.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(str(file_path), "wb") as file_handle:
                await file_handle.write(file_content)
            relative_path = local_relative_path
            logger.info("Saved PDF locally: %s", file_path)

        if document_type == "Test Series" and total_points is not None and total_points <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Total points must be greater than 0"
            )

        if document_type == "Test Series" and total_minutes is not None and total_minutes <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Total minutes must be greater than 0"
            )

        teacher_ids_list = []
        if teacher_ids:
            teacher_ids_list = [tid.strip() for tid in teacher_ids.split(",") if tid.strip()]

        try:
            admin_oid = BsonObjectId(current_user.get("user_id"))
        except Exception:
            admin_oid = None

        document_metadata = {
            "document_id": document_id,
            "title": title,
            "document_type": document_type,
            "subject": subject or "General",
            "difficulty": difficulty or "medium",
            "course_plan": course_plan or "CBSE",
            "standard": standard or "11",
            "section": section,
            "teacher_ids": teacher_ids_list,
            "file_path": relative_path,
            "filename": file.filename,
            "file_size": file_size,
            "uploaded_by": current_user.get("user_id"),
            "admin_id": admin_oid,
            "uploaded_at": datetime.utcnow(),
            "ocr_status": "not_processed",
            "ocr_job_id": None,
            "extracted_questions_count": 0,
            "extracted_images_count": 0,
            "pages_count": pages_count,
            "total_points": total_points if document_type == "Test Series" else None,
            "total_minutes": total_minutes if document_type == "Test Series" else None,
            "is_validated": False,
            "is_active": True,
            "is_s3": is_s3_enabled()
        }

        await insert_one(db, "documents", document_metadata, is_b2c)

        logger.info(
            "Document %s uploaded successfully to %s database",
            document_id,
            "B2C" if is_b2c else "regular"
        )

        should_auto_ocr = document_type in ["Test Series", "Practice Sets"]
        ocr_status = "not_processed"
        ocr_job_id = None

        if should_auto_ocr:
            try:
                job_id = str(uuid.uuid4())
                ocr_job_id = job_id

                await update_one(
                    db,
                    "documents",
                    {"document_id": document_id},
                    {"$set": {"ocr_status": "processing", "ocr_job_id": job_id}},
                    is_b2c
                )
                ocr_status = "processing"

                pdf_base64 = base64.b64encode(file_content).decode('utf-8')

                processing_result = {
                    "job_id": job_id,
                    "status": "processing",
                    "progress": 20,
                    "extracted_questions": 0,
                    "extracted_images": 0,
                    "output_folder": f"extracted_{document_id}_{int(datetime.utcnow().timestamp())}",
                    "timestamp": datetime.utcnow()
                }

                await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

                async def background_ocr():
                    try:
                        await run_document_ocr_pipeline(
                            document=document_metadata,
                            pdf_base64=pdf_base64,
                            job_id=job_id,
                            processing_result=processing_result,
                            current_user=current_user,
                            db=db,
                            cache=cache
                        )
                        logger.info("Auto-OCR completed successfully for %s", document_id)
                    except Exception as ocr_exc:
                        logger.error("Auto-OCR failed for %s: %s", document_id, ocr_exc)
                        try:
                            await update_one(
                                db,
                                "documents",
                                {"document_id": document_id},
                                {"$set": {"ocr_status": "error"}},
                                is_b2c
                            )
                        except Exception:
                            pass

                asyncio.create_task(background_ocr())
                logger.info("Auto-OCR triggered for %s: %s", document_type, document_id)

            except Exception as auto_ocr_err:
                logger.warning("Failed to auto-trigger OCR for %s: %s", document_id, auto_ocr_err)
                try:
                    await update_one(
                        db,
                        "documents",
                        {"document_id": document_id},
                        {"$set": {"ocr_status": "not_processed", "ocr_job_id": None}},
                        is_b2c
                    )
                except Exception:
                    pass

        return {
            "message": "Document uploaded successfully" + (" - OCR processing started automatically" if should_auto_ocr else ""),
            "document_id": document_id,
            "file_path": relative_path,
            "ocr_status": ocr_status,
            "ocr_job_id": ocr_job_id,
            "pages_count": pages_count
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(exc)}"
        )
