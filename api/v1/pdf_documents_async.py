"""
Document management endpoints for PDFs.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId as BsonObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_database
from api.v1.pdf_dependencies import require_admin, require_admin_or_tutor
from api.v1.pdf_schemas import DocumentListResponse, DocumentMetadata
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/documents", response_model=DocumentListResponse)
@limiter.limit("60/minute")
async def get_documents(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    document_type: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Get list of uploaded documents with pagination."""
    try:
        is_b2c = current_user.get("user_type") == "b2c_admin"

        user_type = current_user.get("user_type")
        filter_query: Dict[str, Any] = {}

        if is_b2c:
            pass
        elif user_type == "admin":
            try:
                filter_query["admin_id"] = BsonObjectId(current_user.get("admin_id", current_user["user_id"]))
            except Exception:
                pass
        else:
            admin_id = current_user.get("admin_id")
            if admin_id:
                try:
                    filter_query["admin_id"] = BsonObjectId(admin_id)
                except Exception:
                    pass
        if document_type:
            filter_query["document_type"] = document_type

        if user_type == "tutor":
            tutor_id = current_user.get("tutor_id")
            filter_query = {
                "$and": [
                    filter_query,
                    {"$or": [
                        {"teacher_ids": {"$in": [tutor_id]}},
                        {"teacher_ids": []},
                        {"teacher_ids": None},
                        {"teacher_ids": {"$exists": False}}
                    ]}
                ]
            }

        if is_b2c:
            total = len(await db.b2c_find("documents", filter_query))
            skip = (page - 1) * limit
            documents = await db.b2c_find(
                "documents",
                filter_query,
                skip=skip,
                limit=limit,
                sort=[("uploaded_at", -1)]
            )
        else:
            total = len(await db.mongo_find("documents", filter_query))
            skip = (page - 1) * limit
            documents = await db.mongo_find(
                "documents",
                filter_query,
                skip=skip,
                limit=limit,
                sort=[("uploaded_at", -1)]
            )

        from pathlib import Path
        document_list = []
        for doc in documents:
            file_path = Path(doc["file_path"])
            file_exists = file_path.exists()

            document_list.append(DocumentMetadata(
                document_id=doc["document_id"],
                title=doc["title"],
                document_type=doc["document_type"],
                subject=doc["subject"],
                difficulty=doc["difficulty"],
                course_plan=doc.get("course_plan"),
                standard=doc.get("standard"),
                file_path=doc["file_path"],
                filename=doc["filename"],
                uploaded_by=doc["uploaded_by"],
                uploaded_at=doc["uploaded_at"],
                ocr_status=doc["ocr_status"],
                ocr_job_id=doc.get("ocr_job_id"),
                extracted_questions_count=doc.get("extracted_questions_count", 0),
                extracted_images_count=doc.get("extracted_images_count", 0),
                pages_count=doc.get("pages_count", 0),
                total_points=doc.get("total_points"),
                total_minutes=doc.get("total_minutes"),
                file_exists=file_exists,
                is_active=doc.get("is_active", True)
            ))

        return DocumentListResponse(
            documents=document_list,
            total=total,
            page=page,
            limit=limit
        )

    except Exception as exc:
        logger.error("Get documents error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.get("/documents/{document_id}/file")
@limiter.limit("30/minute")
async def get_document_file(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Serve PDF file for viewing."""
    from fastapi.responses import FileResponse, Response

    try:
        logger.info("Attempting to fetch document with ID: %s", document_id)

        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            document = await db.b2c_find_one("documents", {"document_id": document_id})

        if not document:
            all_docs = await db.mongo_find("documents", {}, limit=10)
            available_ids = [doc.get('document_id', 'NO_ID') for doc in all_docs]
            logger.error("Document '%s' not found. Available IDs: %s", document_id, available_ids)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found in database. Available: {available_ids}"
            )

        if current_user.get("user_type") == "tutor":
            tutor_id = current_user.get("tutor_id")
            teacher_ids = document.get("teacher_ids")
            if teacher_ids and tutor_id not in teacher_ids:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tutor not authorized for this document")

        stored_path = document.get("file_path", "")
        logger.info("Document found. File path: %s", stored_path)

        if stored_path.startswith("s3://"):
            logger.info("Fetching PDF from S3: %s", stored_path)

            from utils.s3_storage import download_file as s3_download

            file_data = await s3_download(stored_path)

            if not file_data:
                logger.error("Failed to download PDF from S3: %s", stored_path)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="PDF file not found in S3"
                )

            return Response(
                content=file_data,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"inline; filename=\"{document.get('filename', 'document.pdf')}\""
                }
            )

        from pathlib import Path
        backend_dir = Path(os.getcwd())
        stored_path = stored_path.replace("\\", "/")
        file_path = backend_dir / stored_path
        logger.info("Full file path: %s", file_path)

        if not file_path.exists():
            logger.error("File does not exist at path: %s", file_path)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found on server at: {file_path}"
            )

        logger.info("Serving PDF file: %s", document['filename'])
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=document["filename"]
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Get document file error: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document file: {str(exc)}"
        )


@router.post("/documents/{document_id}/recalculate-points")
@limiter.limit("30/minute")
async def recalculate_document_points(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Recalculate total_points for a Test Series document based on question points."""
    try:
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        if document.get("document_type") != "Test Series":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only Test Series documents have total points"
            )

        questions = await db.mongo_find("questions", {"pdf_source": document_id})
        total_points = sum(q.get("points", 1.0) for q in questions)

        await db.mongo_update_one(
            "documents",
            {"document_id": document_id},
            {"$set": {"total_points": total_points}}
        )

        logger.info("Recalculated total_points for %s: %s", document_id, total_points)

        return {
            "message": "Total points recalculated successfully",
            "document_id": document_id,
            "total_points": total_points,
            "question_count": len(questions)
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Recalculate points error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to recalculate points: {str(exc)}"
        )


@router.patch("/documents/{document_id}/metadata")
@limiter.limit("30/minute")
async def update_document_metadata(
    request: Request,
    document_id: str,
    metadata: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Update document metadata (e.g., total_points)."""
    try:
        existing_doc = await db.mongo_find_one("documents", {"document_id": document_id})
        if not existing_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        update_data = {}
        if "total_points" in metadata:
            total_points = metadata["total_points"]
            if total_points < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Total points must be greater than or equal to 0"
                )
            update_data["total_points"] = total_points

        if "total_minutes" in metadata:
            total_minutes = metadata["total_minutes"]
            if total_minutes <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Total minutes must be greater than 0"
                )
            update_data["total_minutes"] = total_minutes

        if "is_active" in metadata:
            update_data["is_active"] = bool(metadata["is_active"])

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields to update"
            )

        logger.info("Updating document %s with metadata: %s", document_id, update_data)

        result = await db.mongo_update_one(
            "documents",
            {"document_id": document_id},
            {"$set": update_data}
        )

        logger.info("Update result for %s: %s", document_id, result)

        return {
            "message": "Document metadata updated successfully",
            "document_id": document_id,
            "updated_fields": update_data
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Update document metadata error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document metadata: {str(exc)}"
        )


@router.delete("/documents/{document_id}")
@limiter.limit("10/minute")
async def delete_document(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Delete document and all associated data (cascading delete)."""
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

        from pathlib import Path
        backend_dir = Path(os.getcwd())
        stored_path = document["file_path"].replace("\\", "/")
        file_path = backend_dir / stored_path
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted PDF file: %s", file_path)

        if is_b2c:
            questions = await db.b2c_find("questions", {"document_id": document_id})
        else:
            questions = await db.mongo_find("questions", {"document_id": document_id})

        logger.info("Found %s questions to delete for document %s", len(questions), document_id)

        for question in questions:
            try:
                await db.chroma_delete(question["id"])
                logger.debug("Deleted question %s from ChromaDB", question['id'])
            except Exception as exc:
                logger.warning("Failed to delete question %s from ChromaDB: %s", question['id'], str(exc))

        try:
            if is_b2c:
                await db.b2c_delete_many("questions", {"document_id": document_id})
            else:
                await db.mongo_delete_many("questions", {"document_id": document_id})
            logger.info("Deleted %s questions from MongoDB for document %s", len(questions), document_id)
        except Exception as exc:
            logger.error("Failed to delete questions from MongoDB: %s", str(exc))
            raise

        if is_b2c:
            images = await db.b2c_find("images", {"source_pdf": document["filename"]})
        else:
            images = await db.mongo_find("images", {"source_pdf": document["filename"]})

        logger.info("Found %s images to delete for document %s", len(images), document_id)

        for image in images:
            if "file_path" in image and os.path.exists(image["file_path"]):
                try:
                    os.remove(image["file_path"])
                    logger.debug("Deleted image file: %s", image['file_path'])
                except Exception as exc:
                    logger.warning("Failed to delete image file %s: %s", image['file_path'], str(exc))

        try:
            if is_b2c:
                await db.b2c_delete_many("images", {"source_pdf": document["filename"]})
            else:
                await db.mongo_delete_many("images", {"source_pdf": document["filename"]})
            logger.info("Deleted %s images from MongoDB for document %s", len(images), document_id)
        except Exception as exc:
            logger.error("Failed to delete images from MongoDB: %s", str(exc))
            raise

        try:
            if is_b2c:
                await db.b2c_delete_one("documents", {"document_id": document_id})
            else:
                await db.mongo_delete_one("documents", {"document_id": document_id})
            logger.info("Deleted document %s from MongoDB", document_id)
        except Exception as exc:
            logger.error("Failed to delete document from MongoDB: %s", str(exc))
            raise

        return {
            "message": f"Document {document_id} and all associated data deleted successfully",
            "deleted_questions": len(questions),
            "deleted_images": len(images)
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Delete document error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(exc)}"
        )
