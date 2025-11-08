"""
Async Images API for SkillBot
Image management endpoints with authentication and rate limiting
"""

import logging
import os
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends, status, File, UploadFile, Query
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
import aiofiles

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user, get_database, get_cache
from config_async import settings
from utils.path_utils import get_absolute_path

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
class ImageMetadata(BaseModel):
    id: str
    filename: str
    original_filename: str
    size: int
    content_type: str
    uploaded_by: str
    uploaded_at: datetime
    is_processed: bool = False
    tags: List[str] = []

class ImageResponse(BaseModel):
    id: str
    filename: str
    url: str
    size: int
    content_type: str
    uploaded_at: datetime

class ImagesListResponse(BaseModel):
    images: List[ImageResponse]
    total: int
    page: int
    limit: int

def require_student_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require student or admin access"""
    user_type = current_user.get("user_type")
    logger.info(f"require_student_or_admin: user_type={user_type}, current_user keys={list(current_user.keys())}")

    if user_type not in ["student", "admin"]:
        logger.error(f"Access denied: user_type '{user_type}' not in ['student', 'admin']")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Student or admin access required. Current user_type: {user_type}"
        )
    return current_user

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx', '.txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@router.post("/upload", response_model=ImageResponse)
@limiter.limit("20/minute")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Upload an image file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )

        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # Check file size
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large. Maximum size is 10MB"
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        unique_filename = f"{file_id}{file_ext}"

        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.getcwd(), "uploads", "images")
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, unique_filename)

        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)

        # Save metadata to database
        image_metadata = {
            "_id": file_id,
            "filename": unique_filename,
            "original_filename": file.filename,
            "size": file_size,
            "content_type": file.content_type or "application/octet-stream",
            "uploaded_by": current_user["user_id"],
            "uploaded_at": datetime.utcnow(),
            "is_processed": False,
            "file_path": file_path,
            "tags": []
        }

        await db.mongo_insert_one("images", image_metadata)

        return ImageResponse(
            id=file_id,
            filename=unique_filename,
            url=f"/api/v1/images/{file_id}",
            size=file_size,
            content_type=file.content_type or "application/octet-stream",
            uploaded_at=image_metadata["uploaded_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload image error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload image"
        )

@router.get("/", response_model=ImagesListResponse)
@limiter.limit("60/minute")
async def get_images(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get paginated list of images"""
    try:
        user_id = current_user["user_id"]
        user_type = current_user["user_type"]

        # Build filter - students can only see their own images, admins see all
        filter_dict = {}
        if user_type == "student":
            filter_dict["uploaded_by"] = user_id

        # Get total count
        all_images = await db.mongo_find("images", filter_dict)
        total_images = len(all_images)

        # Get paginated results
        skip = (page - 1) * limit
        images_data = await db.mongo_find(
            "images",
            filter_dict,
            sort=[("uploaded_at", -1)],
            skip=skip,
            limit=limit
        )

        images = [
            ImageResponse(
                id=str(img["_id"]),
                filename=img["filename"],
                url=f"/api/v1/images/{img['_id']}",
                size=img["size"],
                content_type=img["content_type"],
                uploaded_at=img["uploaded_at"]
            )
            for img in images_data
        ]

        return ImagesListResponse(
            images=images,
            total=total_images,
            page=page,
            limit=limit
        )

    except Exception as e:
        logger.error(f"Get images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get images"
        )

@router.get("/{image_id}")
@limiter.limit("120/minute")
async def get_image(
    request: Request,
    image_id: str,
    db: DatabaseManager = Depends(get_database)
):
    """Get image file by ID"""
    try:
        # Get image metadata
        image_data = await db.mongo_find_one("images", {"_id": image_id})

        if not image_data:
            logger.error(f"Image not found in database: {image_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
            )

        # Note: Public access to image files to allow rendering in <img> tags
        # without requiring Authorization headers from the browser.

        # Check if file exists - convert relative path to absolute and normalize any Windows-style paths
        stored_path = image_data.get("file_path")
        if not stored_path:
            logger.error(f"Image {image_id} has no file_path in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image file path not found in database"
            )

        # Normalize path separators and handle Windows drive prefixes that were stored earlier
        from pathlib import Path
        stored_path_str = str(stored_path).replace("\\", "/")

        # If the path contains 'uploads/', prefer using the portion from 'uploads/' onwards
        # This makes Windows absolute paths portable on Linux servers
        uploads_idx = stored_path_str.lower().find("uploads/")
        if uploads_idx != -1:
            relative_from_uploads = stored_path_str[uploads_idx:]
            file_path = get_absolute_path(relative_from_uploads)
        else:
            # Treat POSIX absolute paths as-is; otherwise resolve relative to backend dir
            if Path(stored_path_str).is_absolute():
                file_path = Path(stored_path_str)
            else:
                file_path = get_absolute_path(stored_path_str)

        logger.info(
            f"Fetching image {image_id}: stored_path={stored_path_str}, resolved_path={file_path}, exists={file_path.exists()}"
        )

        if not file_path.exists():
            # Attempt fallback 1: strip Windows-style duplicate suffixes like " (1)" from any path segment
            from re import sub
            cleaned_rel = sub(r" \(\d+\)", "", stored_path_str)
            if cleaned_rel != stored_path_str:
                candidate = get_absolute_path(cleaned_rel)
                logger.info(f"Fallback path (strip copy index): {candidate} exists={candidate.exists()}")
                if candidate.exists():
                    file_path = candidate

        if not file_path.exists():
            # Attempt fallback 2: locate by filename anywhere under uploads/
            try:
                target_name = image_data.get("filename") or Path(stored_path_str).name
                from pathlib import Path as _P
                uploads_root = get_absolute_path("uploads")
                found = next((p for p in uploads_root.rglob(target_name) if p.is_file()), None)
                logger.info(
                    f"Fallback search for {target_name} under {uploads_root}: found={bool(found)}"
                )
                if found:
                    file_path = found
            except Exception as _e:
                logger.warning(f"Fallback search error: {str(_e)}")

        if not file_path.exists():
            logger.error(f"Image file not found on disk: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image file not found"
            )

        # Read and return file
        from fastapi.responses import FileResponse
        return FileResponse(
            str(file_path),
            media_type=image_data["content_type"],
            filename=image_data["original_filename"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get image error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get image"
        )

@router.delete("/{image_id}")
@limiter.limit("10/minute")
async def delete_image(
    request: Request,
    image_id: str,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Delete image by ID"""
    try:
        # Get image metadata
        image_data = await db.mongo_find_one("images", {"_id": image_id})

        if not image_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
            )

        # Check permissions - students can only delete their own images
        if (current_user["user_type"] == "student" and
            image_data["uploaded_by"] != current_user["user_id"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Delete file from filesystem
        stored_path = image_data.get("file_path")
        if stored_path:
            # Convert to absolute path (handles both old absolute and new relative paths)
            from pathlib import Path
            if Path(stored_path).is_absolute():
                file_path = Path(stored_path)
            else:
                file_path = get_absolute_path(stored_path)

            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
                except OSError as e:
                    logger.warning(f"Could not delete file {file_path}: {e}")

        # Delete from database
        result = await db.mongo_delete_one("images", {"_id": image_id})

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
            )

        return {"message": "Image deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete image error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete image"
        )# Force reload at Thu, Oct  2, 2025 11:37:18 PM