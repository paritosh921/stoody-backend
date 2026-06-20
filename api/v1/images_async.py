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
from core.upload_security.service import secure_upload
from config_async import settings
from utils.path_utils import get_absolute_path
from utils.s3_storage import download_file as s3_download_file, upload_file as s3_upload_file, delete_file as s3_delete_file, is_s3_enabled, get_public_url

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
    """Dependency to require student, B2C user, B2C admin, or admin access"""
    user_type = current_user.get("user_type")
    logger.info(f"require_student_or_admin: user_type={user_type}, current_user keys={list(current_user.keys())}")

    # b2c_user gets same access as student, b2c_admin gets admin access
    if user_type not in ["student", "admin", "b2c_user", "b2c_admin"]:
        logger.error(f"Access denied: user_type '{user_type}' not in ['student', 'admin', 'b2c_user', 'b2c_admin']")
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
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )

        clean_upload = await secure_upload(
            file=file,
            policy_id="generic_image_upload",
            actor=current_user,
            db=db,
            purpose_metadata={
                "purpose": "generic_image_upload",
                "collection": "images",
                "created_by": current_user.get("user_id"),
            },
            authorization_subject=f"image:{current_user.get('user_id', 'unknown')}",
        )

        file_ext = os.path.splitext(clean_upload.original_filename)[1].lower()
        file_id = clean_upload.upload_id
        unique_filename = f"{file_id}{file_ext}"

        # Save metadata to database
        image_metadata = {
            "_id": file_id,
            "filename": unique_filename,
            "original_filename": clean_upload.original_filename,
            "size": clean_upload.size_bytes,
            "content_type": clean_upload.content_type,
            "uploaded_by": current_user["user_id"],
            "uploaded_at": datetime.utcnow(),
            "is_processed": False,
            "file_path": clean_upload.released_storage_path,
            "is_s3": clean_upload.released_storage_path.startswith("s3://"),
            "upload_id": clean_upload.upload_id,
            "sha256": clean_upload.sha256,
            "tags": []
        }

        await db.mongo_insert_one("images", image_metadata)

        return ImageResponse(
            id=file_id,
            filename=unique_filename,
            url=f"/api/v1/images/{file_id}",
            size=clean_upload.size_bytes,
            content_type=clean_upload.content_type,
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

        # Build filter - students/b2c_users can only see their own images, admins see all
        filter_dict = {}
        if user_type in ["student", "b2c_user"]:
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
        # Get image metadata - try context DB first, then B2C
        image_data = await db.mongo_find_one("images", {"_id": image_id})
        
        if not image_data:
            # Try B2C database
            image_data = await db.b2c_find_one("images", {"_id": image_id})

        # If still not found and we have a mongo_client, search across all tenant databases
        # This handles the case where image was saved with tenant context but fetched without auth
        if not image_data and db.mongo_client is not None:
            try:
                master_db = await db.get_master_db()
                if master_db is not None:
                    tenants = await master_db["tenants"].find(
                        {"status": "active"},
                        {"db_name": 1}
                    ).to_list(length=100)
                    
                    for tenant in tenants:
                        tenant_db_name = tenant.get("db_name")
                        if not tenant_db_name:
                            continue
                        tenant_db = await db.get_tenant_db(tenant_db_name)
                        if tenant_db is not None:
                            image_data = await tenant_db["images"].find_one({"_id": image_id})
                            if image_data:
                                logger.info(f"Found image {image_id} in tenant DB: {tenant_db_name}")
                                break
            except Exception as tenant_search_err:
                logger.warning(f"Tenant search for image failed: {tenant_search_err}")

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

        # Check if this is an S3 path
        if str(stored_path).startswith("s3://"):
            logger.info(f"Fetching image from S3: {stored_path}")
            
            # Download from S3
            file_data = await s3_download_file(stored_path)
            
            if not file_data:
                logger.error(f"Failed to download image from S3: {stored_path}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Image file not found in S3"
                )
            
            # Return as streaming response
            from fastapi.responses import Response
            return Response(
                content=file_data,
                media_type=image_data.get("content_type", "image/jpeg"),
                headers={
                    "Content-Disposition": f"inline; filename=\"{image_data.get('original_filename', image_data.get('filename', 'image'))}\""
                }
            )

        # Local file handling (existing logic)
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

        # Check permissions - students/b2c_users can only delete their own images
        if (current_user["user_type"] in ["student", "b2c_user"] and
            image_data["uploaded_by"] != current_user["user_id"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Delete file from storage (S3 or local filesystem)
        stored_path = image_data.get("file_path")
        if stored_path:
            # Check if stored in S3
            if str(stored_path).startswith("s3://"):
                # Delete from S3
                deleted = await s3_delete_file(stored_path)
                if deleted:
                    logger.info(f"✅ Deleted file from S3: {stored_path}")
                else:
                    logger.warning(f"Failed to delete file from S3: {stored_path}")
            else:
                # Delete from local filesystem
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
