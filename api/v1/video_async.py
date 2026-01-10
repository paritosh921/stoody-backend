"""
Async Video Content API for SkillBot/Stoody
Video content management endpoints for YouTube video links
Supports both B2B (skillbot_db) and B2C (STOODY-b2c) databases
"""

import logging
import re
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class VideoCreate(BaseModel):
    """Model for creating a new video content"""
    video_id: str = Field(..., min_length=1, max_length=50, description="Unique video identifier")
    title: str = Field(..., min_length=1, max_length=200, description="Video title")
    youtube_url: str = Field(..., description="YouTube video URL")
    subject: str = Field(..., description="Subject (Physics, Chemistry, etc.)")
    standard: str = Field(..., description="Grade/Class (9, 10, 11, 12, etc.)")
    description: Optional[str] = Field(None, max_length=5000, description="Video description")
    course_plan: Optional[str] = Field("CBSE", description="Course plan (CBSE, JEE, NEET, etc.)")
    section: Optional[str] = Field(None, description="Section A-F for filtering")
    teacher_ids: Optional[List[str]] = Field(None, description="Teacher IDs for filtering")
    is_active: bool = Field(True, description="Whether video is visible to students")

    @validator('video_id')
    def validate_video_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Video ID must be alphanumeric with underscores/hyphens only')
        return v

    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        # More flexible patterns that use search instead of match
        youtube_patterns = [
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]+)',
            r'youtu\.be/([a-zA-Z0-9_-]+)',
            r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
        ]
        for pattern in youtube_patterns:
            if re.search(pattern, v):
                return v
        raise ValueError('Invalid YouTube URL format')


class VideoUpdate(BaseModel):
    """Model for updating video content"""
    title: Optional[str] = Field(None, max_length=200)
    youtube_url: Optional[str] = None
    subject: Optional[str] = None
    standard: Optional[str] = None
    description: Optional[str] = Field(None, max_length=5000)
    course_plan: Optional[str] = None
    section: Optional[str] = None
    teacher_ids: Optional[List[str]] = None
    is_active: Optional[bool] = None

    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        if v is None:
            return v
        # More flexible patterns that use search instead of match
        youtube_patterns = [
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]+)',
            r'youtu\.be/([a-zA-Z0-9_-]+)',
            r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
        ]
        for pattern in youtube_patterns:
            if re.search(pattern, v):
                return v
        raise ValueError('Invalid YouTube URL format')


class VideoResponse(BaseModel):
    """Response model for video content"""
    video_id: str
    title: str
    youtube_url: str
    youtube_video_id: str
    embed_url: str
    thumbnail_url: str
    subject: str
    standard: str
    description: Optional[str] = None
    course_plan: Optional[str] = None
    section: Optional[str] = None
    teacher_ids: Optional[List[str]] = None
    is_active: bool
    uploaded_by: str
    uploaded_at: datetime
    view_count: int = 0


def extract_youtube_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]+)',
        r'youtu\.be/([a-zA-Z0-9_-]+)',
        r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract YouTube video ID from URL")


def get_youtube_embed_url(video_id: str) -> str:
    """Get YouTube embed URL from video ID"""
    return f"https://www.youtube.com/embed/{video_id}"


def get_youtube_thumbnail_url(video_id: str) -> str:
    """Get YouTube thumbnail URL from video ID"""
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access (regular or B2C)"""
    if current_user.get("user_type") not in ["admin", "b2c_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Allow admin, B2C admin, and tutor roles"""
    if current_user.get("user_type") not in ["admin", "b2c_admin", "tutor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Tutor access required"
        )
    return current_user


def is_b2c_admin(current_user: Dict[str, Any]) -> bool:
    """Check if the current user is a B2C admin"""
    return current_user.get("user_type") == "b2c_admin"


@router.get("/test", response_model=dict)
async def test_video_route():
    """Simple test endpoint to verify routing works"""
    return {"success": True, "message": "Video API is working"}


@router.post("/videos/debug", response_model=dict)
async def debug_video_request(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor)
):
    """Debug endpoint to see raw request body"""
    try:
        body = await request.json()
        logger.info(f"DEBUG - Raw request body: {body}")
        return {"success": True, "received_data": body}
    except Exception as e:
        logger.error(f"DEBUG - Error parsing body: {str(e)}")
        return {"success": False, "error": str(e)}


@router.post("/videos", response_model=dict)
async def create_video(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Create a new video content entry.
    Admins and tutors can add YouTube video links.
    """
    try:
        # Parse and validate the request body manually for better error messages
        body = await request.json()
        logger.info(f"Create video request body: {body}")

        # Validate required fields
        required_fields = ['video_id', 'title', 'youtube_url', 'subject', 'standard']
        for field in required_fields:
            if field not in body or not body[field]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )

        # Create VideoCreate model
        try:
            video_data = VideoCreate(**body)
        except Exception as validation_error:
            logger.error(f"Validation error: {str(validation_error)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Validation error: {str(validation_error)}"
            )
        # Extract YouTube video ID
        youtube_video_id = extract_youtube_video_id(video_data.youtube_url)

        # Determine which database to use
        is_b2c = is_b2c_admin(current_user)

        # Check if video_id already exists
        if is_b2c:
            existing = await db.b2c_find_one("videos", {"video_id": video_data.video_id})
        else:
            existing = await db.mongo_find_one("videos", {"video_id": video_data.video_id})

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video with ID '{video_data.video_id}' already exists"
            )

        # Create video document
        video_doc = {
            "video_id": video_data.video_id,
            "title": video_data.title,
            "youtube_url": video_data.youtube_url,
            "youtube_video_id": youtube_video_id,
            "embed_url": get_youtube_embed_url(youtube_video_id),
            "thumbnail_url": get_youtube_thumbnail_url(youtube_video_id),
            "subject": video_data.subject,
            "standard": video_data.standard,
            "description": video_data.description,
            "course_plan": video_data.course_plan,
            "section": video_data.section,
            "teacher_ids": video_data.teacher_ids or [],
            "is_active": video_data.is_active,
            "uploaded_by": current_user.get("sub") or current_user.get("user_id"),
            "uploaded_at": datetime.utcnow(),
            "view_count": 0,
        }

        # Insert into appropriate database
        if is_b2c:
            await db.b2c_insert_one("videos", video_doc)
        else:
            await db.mongo_insert_one("videos", video_doc)

        logger.info(f"Video created: {video_data.video_id} by {current_user.get('sub')}")

        return {
            "success": True,
            "message": "Video added successfully",
            "video": video_doc
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create video: {str(e)}"
        )


@router.get("/videos", response_model=dict)
async def get_videos(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    subject: Optional[str] = None,
    standard: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get list of videos with pagination and filtering.
    For admin/tutor use.
    """
    try:
        is_b2c = is_b2c_admin(current_user)

        # Build filter
        filter_query = {}
        if subject:
            filter_query["subject"] = subject
        if standard:
            filter_query["standard"] = standard
        if is_active is not None:
            filter_query["is_active"] = is_active

        # Calculate skip
        skip = (page - 1) * limit

        # Get videos from appropriate database
        if is_b2c:
            # Get total count first (without pagination)
            all_videos_for_count = await db.b2c_find("videos", filter_query)
            total = len(all_videos_for_count)

            # Get paginated videos
            videos = await db.b2c_find(
                "videos",
                filter_query,
                sort=[("uploaded_at", -1)],
                skip=skip,
                limit=limit
            )
        else:
            # Get total count first (without pagination)
            all_videos_for_count = await db.mongo_find("videos", filter_query)
            total = len(all_videos_for_count)

            # Get paginated videos
            videos = await db.mongo_find(
                "videos",
                filter_query,
                sort=[("uploaded_at", -1)],
                skip=skip,
                limit=limit
            )

        # Convert ObjectId to string
        for video in videos:
            if "_id" in video:
                video["_id"] = str(video["_id"])

        return {
            "success": True,
            "videos": videos,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit
        }

    except Exception as e:
        logger.error(f"Failed to get videos: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get videos: {str(e)}"
        )


@router.get("/videos/{video_id}", response_model=dict)
async def get_video(
    video_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Get a specific video by ID"""
    try:
        is_b2c = is_b2c_admin(current_user) or current_user.get("user_type") == "b2c_user"

        if is_b2c:
            video = await db.b2c_find_one("videos", {"video_id": video_id})
        else:
            video = await db.mongo_find_one("videos", {"video_id": video_id})

        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )

        # Convert ObjectId to string
        if "_id" in video:
            video["_id"] = str(video["_id"])

        return {
            "success": True,
            "video": video
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get video: {str(e)}"
        )


@router.put("/videos/{video_id}", response_model=dict)
async def update_video(
    video_id: str,
    video_data: VideoUpdate,
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Update video content"""
    try:
        is_b2c = is_b2c_admin(current_user)

        # Check if video exists
        if is_b2c:
            existing = await db.b2c_find_one("videos", {"video_id": video_id})
        else:
            existing = await db.mongo_find_one("videos", {"video_id": video_id})

        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )

        # Build update document
        update_doc = {}
        if video_data.title is not None:
            update_doc["title"] = video_data.title
        if video_data.youtube_url is not None:
            youtube_video_id = extract_youtube_video_id(video_data.youtube_url)
            update_doc["youtube_url"] = video_data.youtube_url
            update_doc["youtube_video_id"] = youtube_video_id
            update_doc["embed_url"] = get_youtube_embed_url(youtube_video_id)
            update_doc["thumbnail_url"] = get_youtube_thumbnail_url(youtube_video_id)
        if video_data.subject is not None:
            update_doc["subject"] = video_data.subject
        if video_data.standard is not None:
            update_doc["standard"] = video_data.standard
        if video_data.description is not None:
            update_doc["description"] = video_data.description
        if video_data.course_plan is not None:
            update_doc["course_plan"] = video_data.course_plan
        if video_data.section is not None:
            update_doc["section"] = video_data.section
        if video_data.teacher_ids is not None:
            update_doc["teacher_ids"] = video_data.teacher_ids
        if video_data.is_active is not None:
            update_doc["is_active"] = video_data.is_active

        if not update_doc:
            return {
                "success": True,
                "message": "No changes to update"
            }

        update_doc["updated_at"] = datetime.utcnow()

        # Update in appropriate database
        if is_b2c:
            await db.b2c_update_one("videos", {"video_id": video_id}, {"$set": update_doc})
        else:
            await db.mongo_update_one("videos", {"video_id": video_id}, {"$set": update_doc})

        logger.info(f"Video updated: {video_id}")

        return {
            "success": True,
            "message": "Video updated successfully"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update video: {str(e)}"
        )


@router.delete("/videos/{video_id}", response_model=dict)
async def delete_video(
    video_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Delete a video content entry"""
    try:
        is_b2c = is_b2c_admin(current_user)

        # Check if video exists
        if is_b2c:
            existing = await db.b2c_find_one("videos", {"video_id": video_id})
        else:
            existing = await db.mongo_find_one("videos", {"video_id": video_id})

        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )

        # Delete from appropriate database
        if is_b2c:
            await db.b2c_delete_one("videos", {"video_id": video_id})
        else:
            await db.mongo_delete_one("videos", {"video_id": video_id})

        logger.info(f"Video deleted: {video_id}")

        return {
            "success": True,
            "message": "Video deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete video: {str(e)}"
        )


# Student endpoints
@router.get("/student/videos", response_model=dict)
async def get_student_videos(
    request: Request,
    subject: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get videos available to the current student.
    Filters by student's grade, section, and assigned teachers.
    """
    try:
        user_type = current_user.get("user_type")
        is_b2c = user_type == "b2c_user"

        # Build filter based on student's attributes
        filter_query = {"is_active": True}

        if subject:
            filter_query["subject"] = subject

        if is_b2c:
            # B2C user - filter by their standard/grade
            user_id = current_user.get("sub") or current_user.get("user_id")
            user_doc = await db.b2c_find_one("b2c_users", {"user_id": user_id})

            if user_doc:
                if user_doc.get("standard"):
                    filter_query["standard"] = user_doc["standard"]

            videos = await db.b2c_find(
                "videos",
                filter_query,
                sort=[("uploaded_at", -1)]
            )
        else:
            # B2B student
            student_id = current_user.get("sub") or current_user.get("user_id")
            student_doc = await db.mongo_find_one("students", {"_id": student_id})

            if not student_doc:
                # Try finding by student_id field
                student_doc = await db.mongo_find_one("students", {"student_id": student_id})

            if student_doc:
                # Filter by grade
                if student_doc.get("grade"):
                    filter_query["standard"] = student_doc["grade"]

                # Optional: filter by section
                student_section = student_doc.get("section")
                if student_section:
                    filter_query["$or"] = [
                        {"section": {"$exists": False}},
                        {"section": ""},
                        {"section": student_section}
                    ]

                # Optional: filter by teacher assignment
                student_teacher_ids = student_doc.get("teacher_ids", [])
                if student_teacher_ids:
                    # Include videos with no teacher restriction or matching teacher
                    if "$or" not in filter_query:
                        filter_query["$or"] = []
                    filter_query["$or"].extend([
                        {"teacher_ids": {"$exists": False}},
                        {"teacher_ids": []},
                        {"teacher_ids": {"$in": student_teacher_ids}}
                    ])

            videos = await db.mongo_find(
                "videos",
                filter_query,
                sort=[("uploaded_at", -1)]
            )

        # Convert ObjectId to string and group by subject
        videos_by_subject: Dict[str, List] = {}
        for video in videos:
            if "_id" in video:
                video["_id"] = str(video["_id"])

            subj = video.get("subject", "Other")
            if subj not in videos_by_subject:
                videos_by_subject[subj] = []
            videos_by_subject[subj].append(video)

        return {
            "success": True,
            "videos": videos,
            "videos_by_subject": videos_by_subject,
            "total": len(videos)
        }

    except Exception as e:
        logger.error(f"Failed to get student videos: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get videos: {str(e)}"
        )


@router.post("/student/videos/{video_id}/view", response_model=dict)
async def record_video_view(
    video_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """Record a video view by the student"""
    try:
        user_type = current_user.get("user_type")
        is_b2c = user_type == "b2c_user"
        user_id = current_user.get("sub") or current_user.get("user_id")

        # Update view count
        if is_b2c:
            await db.b2c_update_one(
                "videos",
                {"video_id": video_id},
                {"$inc": {"view_count": 1}}
            )

            # Record view activity
            await db.b2c_insert_one("video_views", {
                "video_id": video_id,
                "user_id": user_id,
                "viewed_at": datetime.utcnow()
            })
        else:
            await db.mongo_update_one(
                "videos",
                {"video_id": video_id},
                {"$inc": {"view_count": 1}}
            )

            # Record view activity
            await db.mongo_insert_one("video_views", {
                "video_id": video_id,
                "student_id": user_id,
                "viewed_at": datetime.utcnow()
            })

        return {
            "success": True,
            "message": "View recorded"
        }

    except Exception as e:
        logger.error(f"Failed to record video view: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record view: {str(e)}"
        )
