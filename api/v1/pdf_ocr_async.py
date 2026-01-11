"""
OCR processing endpoints for PDFs.
"""

import asyncio
import base64
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_cache, get_current_user, get_database
from api.v1.pdf_dependencies import require_admin
from api.v1.pdf_schemas import PDFProcessingResult
from core.cache import CacheManager
from core.database import DatabaseManager
from services.async_mistral_ocr_service import call_mistral_ocr
from services.pdf_processing_service import run_document_ocr_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/documents/{document_id}/process-ocr", response_model=PDFProcessingResult)
@limiter.limit("5/minute")
async def process_document_ocr(
    request: Request,
    document_id: str,
    async_mode: bool = Query(True, description="Queue OCR and return immediately when true"),
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Trigger OCR processing on an existing uploaded document."""
    try:
        user_type = current_user.get("user_type")
        is_b2c = user_type in ["b2c_admin", "b2c_user"]

        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        if document.get("ocr_status") == "processing":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="OCR processing already in progress"
            )

        if document.get("ocr_status") == "completed":
            logger.info("Reprocessing document %s - cleaning up old data", document_id)

            if is_b2c:
                await db.b2c_delete_many("questions", {"document_id": document_id})
                logger.info("Deleted questions for document %s from B2C DB", document_id)
            else:
                questions_deleted = await db.mongo_delete_many("questions", {"document_id": document_id})
                logger.info("Deleted %s questions for document %s", questions_deleted, document_id)

            if is_b2c:
                images_result = await db.b2c_find("images", {"source_pdf": document["filename"]})
            else:
                images_result = await db.mongo_find("images", {"source_pdf": document["filename"]})

            for img in images_result:
                file_path = img.get("file_path")
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info("Deleted image file: %s", file_path)
                    except Exception as exc:
                        logger.error("Failed to delete image file %s: %s", file_path, exc)

            if is_b2c:
                await db.b2c_delete_many("images", {"source_pdf": document["filename"]})
            else:
                await db.mongo_delete_many("images", {"source_pdf": document["filename"]})

        from pathlib import Path as _Path
        backend_dir = _Path(os.getcwd())
        stored_path_raw = str(document.get("file_path", "")).replace("\\", "/")

        candidates: list[_Path] = []

        if stored_path_raw:
            sp = _Path(stored_path_raw)
            if sp.is_absolute():
                candidates.append(sp)
            candidates.append(backend_dir / stored_path_raw)

            if "uploads/" in stored_path_raw:
                try:
                    uploads_index = stored_path_raw.index("uploads/")
                    rel_after_uploads = stored_path_raw[uploads_index:]
                    candidates.append(backend_dir / rel_after_uploads)
                except ValueError:
                    pass

        canonical_fallback = backend_dir / f"uploads/documents/{document.get('document_type','')}/{document_id}.pdf"
        candidates.append(canonical_fallback)

        file_path = None
        for candidate in candidates:
            try:
                if candidate.exists():
                    file_path = candidate
                    break
            except Exception:
                continue

        if not file_path:
            logger.error(
                "PDF file not found for document %s. Checked: %s",
                document_id,
                ", ".join(str(candidate) for candidate in candidates)
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF file not found on server. Please re-upload this document from the Admin panel."
            )

        async with aiofiles.open(str(file_path), "rb") as file_handle:
            file_content = await file_handle.read()

        pdf_base64 = base64.b64encode(file_content).decode('utf-8')
        job_id = str(uuid.uuid4())

        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "processing", "ocr_job_id": job_id}}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "processing", "ocr_job_id": job_id}}
            )

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

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to prepare OCR job for %s: %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start OCR processing: {exc}"
        )

    async def execute_pipeline() -> PDFProcessingResult:
        return await run_document_ocr_pipeline(
            document=document,
            pdf_base64=pdf_base64,
            job_id=job_id,
            processing_result=processing_result,
            current_user=current_user,
            db=db,
            cache=cache
        )

    async def execute_with_semaphore() -> PDFProcessingResult:
        semaphore = getattr(request.app.state, "ocr_semaphore", None)
        if semaphore:
            async with semaphore:
                return await execute_pipeline()
        return await execute_pipeline()

    if async_mode:
        tasks = getattr(request.app.state, "ocr_tasks", None)

        async def background_runner():
            try:
                await execute_with_semaphore()
            except HTTPException:
                pass
            except Exception as exc:
                logger.error("Background OCR job %s failed: %s", job_id, exc, exc_info=True)

        task = asyncio.create_task(background_runner())
        if isinstance(tasks, dict):
            tasks[job_id] = task

            def _cleanup(_):
                tasks.pop(job_id, None)

            task.add_done_callback(_cleanup)

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=jsonable_encoder(PDFProcessingResult(**processing_result))
        )

    try:
        return await execute_with_semaphore()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("OCR processing failed for %s: %s", document_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {exc}"
        )


@router.post("/direct-ocr")
@limiter.limit("6/minute")
async def perform_direct_ocr(
    request: Request,
    file: UploadFile = File(...),
    subject: Optional[str] = Form("General"),
    difficulty: Optional[str] = Form("medium"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Direct OCR processing for authenticated users (no document persistence)."""
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )

        file_content = await file.read()
        pdf_base64 = base64.b64encode(file_content).decode("utf-8")

        async def _run_ocr() -> Dict[str, Any]:
            return await call_mistral_ocr(pdf_base64)

        semaphore = getattr(request.app.state, "ocr_semaphore", None)
        if semaphore:
            async with semaphore:
                ocr_result = await _run_ocr()
        else:
            ocr_result = await _run_ocr()

        return {
            "success": True,
            "filename": file.filename,
            "subject": subject or "General",
            "difficulty": difficulty or "medium",
            "pages": ocr_result.get("pages", []),
            "metadata": {
                "processed_by": current_user.get("user_id"),
                "processed_at": datetime.utcnow().isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Direct OCR processing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {exc}"
        )


@router.get("/status/{job_id}", response_model=PDFProcessingResult)
@limiter.limit("60/minute")
async def get_processing_status(
    request: Request,
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    cache: CacheManager = Depends(get_cache)
):
    """Get PDF processing job status."""
    try:
        cached_result = await cache.get(f"pdf_job:{job_id}", "admin")

        if not cached_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        return PDFProcessingResult(**cached_result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Get status error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get job status"
        )
