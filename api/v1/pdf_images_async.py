"""
Image endpoints for PDF documents.
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
from api.v1.pdf_dependencies import require_admin
from api.v1.student_async import require_student_or_admin
from core.database import DatabaseManager
from services.pdf_data_access import find_many, find_one, is_b2c_user

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/documents/{document_id}/images")
@limiter.limit("60/minute")
async def get_document_images(
    request: Request,
    document_id: str,
    include_orphaned: Optional[bool] = Query(None, description="Include images that don't exist on disk"),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all images extracted from a specific document.
    If include_orphaned is not provided, defaults to:
    - admin/b2c_admin: False (filter missing files)
    - others: True (include all images)
    """
    try:
        from utils.image_validator import validate_image_exists

        # Check if B2C admin or B2C user
        user_type = current_user.get("user_type")
        is_b2c = is_b2c_user(user_type)

        if include_orphaned is None:
            include_orphaned = user_type not in ["admin", "b2c_admin"]

        # Verify document exists in appropriate database
        document = await find_one(db, "documents", {"document_id": document_id}, is_b2c)
            
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Access control
        if user_type == "student":
            student_admin_id = str(current_user.get("admin_id")) if current_user.get("admin_id") is not None else None
            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

            from config_async import DEBUG_MODE as _DEBUG_MODE
            if student_admin_id != document_admin_id_str and not _DEBUG_MODE:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this document"
                )
        elif not is_b2c:
            # Regular admin
            admin_id = str(current_user.get("user_id")) if current_user.get("user_id") is not None else None
            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

            from config_async import DEBUG_MODE as _DEBUG_MODE
            if admin_id != document_admin_id_str and not _DEBUG_MODE:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this document"
                )

        # Get images for this document
        images = await find_many(db, "images", {"source_pdf": document["filename"]}, is_b2c)

        serialized_images = []
        orphaned_count = 0
        for img in images:
            image_id = str(img.get("_id", ""))

            # Check if image exists (unless include_orphaned is True)
            if not include_orphaned:
                exists = await validate_image_exists(image_id, db, is_b2c=is_b2c)
                if not exists:
                    orphaned_count += 1
                    logger.debug(f"Skipping orphaned image {image_id}")
                    continue

            img_dict = {}
            for key, value in img.items():
                if isinstance(value, BsonObjectId):
                    img_dict[key] = str(value)
                elif isinstance(value, datetime):
                    img_dict[key] = value.isoformat()
                else:
                    img_dict[key] = value
            
            # Ensure url is present
            if "url" not in img_dict and "_id" in img_dict:
                img_dict["url"] = f"/api/v1/images/{img_dict['_id']}"
                
            serialized_images.append(img_dict)

        return {
            "document_id": document_id,
            "document_title": document["title"],
            "images_count": len(serialized_images),
            "total_in_db": len(images),
            "orphaned_count": orphaned_count,
            "images": serialized_images
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document images: {str(e)}"
        )


@router.post("/documents/{document_id}/clean-orphaned-images")
@limiter.limit("10/minute")
async def clean_document_orphaned_images(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Clean orphaned image references from all questions in a document.
    Removes image references that don't exist in database or filesystem.
    """
    try:
        from utils.image_validator import get_orphaned_images_in_document, clean_question_images

        # Verify document exists
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Find all orphaned images first
        orphaned_by_question = await get_orphaned_images_in_document(document_id, db)

        if not orphaned_by_question:
            return {
                "message": "No orphaned images found",
                "document_id": document_id,
                "questions_cleaned": 0,
                "total_images_removed": 0,
                "details": []
            }

        # Clean each affected question
        questions_cleaned = 0
        total_images_removed = 0
        details = []

        for question_id, orphaned_ids in orphaned_by_question.items():
            # Get question
            question = await db.mongo_find_one("questions", {"id": question_id})
            if not question:
                continue

            # Clean orphaned references
            cleaned_question, removed_count = await clean_question_images(question, db)

            if removed_count > 0:
                # Update question in database
                await db.mongo_update_one(
                    "questions",
                    {"id": question_id},
                    {"$set": {
                        "images": cleaned_question.get("images", []),
                        "question_figures": cleaned_question.get("question_figures", []),
                        "cleaned_at": datetime.utcnow(),
                        "cleaned_by": current_user.get("user_id")
                    }}
                )

                questions_cleaned += 1
                total_images_removed += removed_count

                details.append({
                    "question_id": question_id,
                    "removed_images": orphaned_ids,
                    "removed_count": removed_count
                })

                logger.info(f"Cleaned {removed_count} orphaned images from question {question_id}")

        return {
            "message": f"Successfully cleaned {total_images_removed} orphaned image references",
            "document_id": document_id,
            "questions_cleaned": questions_cleaned,
            "total_images_removed": total_images_removed,
            "details": details
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clean orphaned images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean orphaned images: {str(e)}"
        )


@router.post("/questions/{question_id}/clean-orphaned-images")
@limiter.limit("20/minute")
async def clean_question_orphaned_images(
    request: Request,
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Clean orphaned image references from a specific question.
    Removes image references that don't exist in database or filesystem.
    """
    try:
        from utils.image_validator import clean_question_images, get_orphaned_images_in_question

        # Get question
        question = await db.mongo_find_one("questions", {"id": question_id})
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )

        # Get orphaned images first
        orphaned_ids = await get_orphaned_images_in_question(question_id, db)

        if not orphaned_ids:
            return {
                "message": "No orphaned images found",
                "question_id": question_id,
                "removed_count": 0,
                "orphaned_images": []
            }

        # Clean orphaned references
        cleaned_question, removed_count = await clean_question_images(question, db)

        if removed_count > 0:
            # Update question in database
            await db.mongo_update_one(
                "questions",
                {"id": question_id},
                {"$set": {
                    "images": cleaned_question.get("images", []),
                    "question_figures": cleaned_question.get("question_figures", []),
                    "cleaned_at": datetime.utcnow(),
                    "cleaned_by": current_user.get("user_id")
                }}
            )

            logger.info(f"Cleaned {removed_count} orphaned images from question {question_id}")

        return {
            "message": f"Successfully removed {removed_count} orphaned image references",
            "question_id": question_id,
            "removed_count": removed_count,
            "orphaned_images": orphaned_ids
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clean question orphaned images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean orphaned images: {str(e)}"
        )


@router.get("/documents/{document_id}/orphaned-images")
@limiter.limit("30/minute")
async def get_document_orphaned_images(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all orphaned image references in a document without cleaning them.
    Useful for inspection before cleanup.
    """
    try:
        from utils.image_validator import get_orphaned_images_in_document

        # Verify document exists
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Find all orphaned images
        orphaned_by_question = await get_orphaned_images_in_document(document_id, db)

        total_orphaned = sum(len(ids) for ids in orphaned_by_question.values())

        return {
            "document_id": document_id,
            "document_title": document.get("title", ""),
            "total_orphaned_images": total_orphaned,
            "affected_questions": len(orphaned_by_question),
            "orphaned_by_question": orphaned_by_question
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get orphaned images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orphaned images: {str(e)}"
        )


