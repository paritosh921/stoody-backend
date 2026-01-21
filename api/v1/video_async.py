"""
Video Content Management API Endpoints (Async)
Handles video CRUD operations for admin and viewing for students
"""

import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from bson import ObjectId

from models.video import VideoSchema, CreateVideoRequest, UpdateVideoRequest
from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from api.v1.tutor_async import require_admin, require_admin_or_tutor
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# Helper function to extract YouTube Video ID
def extract_youtube_info(url: str):
    """Extract video ID and generate embed/thumbnail URLs"""
    video_id = None
    
    # Patterns to match various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'  # Just the ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
            
    if not video_id:
        raise ValueError("Invalid YouTube URL")
        
    return {
        "youtube_video_id": video_id,
        "embed_url": f"https://www.youtube.com/embed/{video_id}",
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
    }


@router.post("/videos", response_model=Dict[str, Any], status_code=201)
@limiter.limit("20/minute")
async def create_video(
    request: Request,
    video_data: CreateVideoRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Create a new video content
    """
    # Check if video_id already exists
    existing = await db.mongo_find_one("videos", {"video_id": video_data.video_id})
    if existing:
        raise HTTPException(status_code=400, detail="Video ID already exists")

    try:
        yt_info = extract_youtube_info(video_data.youtube_url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Prepare video document
    video_doc = video_data.dict()
    video_doc.update(yt_info)
    
    video_doc["uploaded_by"] = current_user.get("user_id")
    video_doc["uploaded_at"] = datetime.utcnow()
    video_doc["view_count"] = 0
    
    # If tutor, force teacher_ids to include self
    if current_user.get("user_type") == "tutor":
        tutor_id = current_user.get("tutor_id")
        if not video_doc.get("teacher_ids"):
            video_doc["teacher_ids"] = []
        if tutor_id and tutor_id not in video_doc["teacher_ids"]:
            video_doc["teacher_ids"].append(tutor_id)

    # Insert
    result = await db.mongo_insert_one("videos", video_doc)
    video_doc["_id"] = str(result)
    
    return {
        "success": True,
        "message": "Video added successfully",
        "video": video_doc
    }


@router.get("/videos", response_model=Dict[str, Any])
@limiter.limit("60/minute")
async def get_videos(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    subject: Optional[str] = None,
    standard: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get paginated list of videos
    """
    query = {}
    if subject:
        query["subject"] = subject
    if standard:
        query["standard"] = standard
        
    # If tutor, only show videos they uploaded OR are assigned to OR are global (no teacher_ids)
    if current_user.get("user_type") == "tutor":
        tutor_id = current_user.get("tutor_id")
        # Logic: (uploaded_by = me) OR (teacher_ids contains me) OR (teacher_ids is empty)
        # Note: This complexity might need simplification depending on business logic
        # For now, let's keep it simple: Tutors see all videos matching subject/standard
        pass 

    total = len(await db.mongo_find("videos", query))
    skip = (page - 1) * limit
    
    videos = await db.mongo_find(
        "videos", 
        query, 
        sort=[("uploaded_at", -1)],
        skip=skip,
        limit=limit
    )
    
    # Format _id
    for v in videos:
        v["_id"] = str(v["_id"])

    return {
        "success": True,
        "videos": videos,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": (total + limit - 1) // limit
    }


@router.get("/videos/{video_id}", response_model=Dict[str, Any])
@limiter.limit("60/minute")
async def get_video(
    request: Request,
    video_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get a single video
    """
    video = await db.mongo_find_one("videos", {"video_id": video_id})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    video["_id"] = str(video["_id"])
    return {
        "success": True,
        "video": video
    }


@router.put("/videos/{video_id}", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def update_video(
    request: Request,
    video_id: str,
    update_data: UpdateVideoRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Update video details
    """
    video = await db.mongo_find_one("videos", {"video_id": video_id})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    # Check permissions logic if needed (e.g. only uploader/admin can edit)
    
    updates = update_data.dict(exclude_unset=True)
    
    if "youtube_url" in updates:
        try:
            yt_info = extract_youtube_info(updates["youtube_url"])
            updates.update(yt_info)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            
    if not updates:
        return {"success": True, "message": "No changes made"}
        
    await db.mongo_update_one("videos", {"video_id": video_id}, {"$set": updates})
    
    return {"success": True, "message": "Video updated successfully"}


@router.delete("/videos/{video_id}", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def delete_video(
    request: Request,
    video_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Delete a video
    """
    result = await db.mongo_delete_one("videos", {"video_id": video_id})
    if result == 0:
        raise HTTPException(status_code=404, detail="Video not found")
        
    return {"success": True, "message": "Video deleted successfully"}


@router.get("/student/videos", response_model=Dict[str, Any])
@limiter.limit("60/minute")
async def get_student_videos(
    request: Request,
    subject: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user), # Any user (student)
    db: DatabaseManager = Depends(get_database)
):
    """
    Get videos for the student (filtered by their grade/section)
    """
    # Verify user type is student
    # if current_user.get("user_type") != "student": 
    #     # Allow admins/tutors to preview too? For now, assume mainly students or preview
    #     pass
        
    grade = current_user.get("grade")
    # section = current_user.get("section") 
    
    query = {"is_active": True}
    
    if grade:
        query["standard"] = grade
        
    # Filter by subject if provided
    if subject and subject != "all":
        query["subject"] = subject
        
    # TODO: Add section filtering and teacher assignment filtering logic here
    # For now, show all active videos for the grade
    
    videos = await db.mongo_find("videos", query, sort=[("uploaded_at", -1)])
    
    # Group by subject
    by_subject = {}
    for v in videos:
        v["_id"] = str(v["_id"])
        subj = v.get("subject", "Other")
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(v)
        
    return {
        "success": True,
        "videos": videos,
        "videos_by_subject": by_subject,
        "total": len(videos)
    }


@router.post("/student/videos/{video_id}/view", response_model=Dict[str, Any])
@limiter.limit("60/minute")
async def record_video_view(
    request: Request,
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Increment view count for a video
    """
    result = await db.mongo_update_one(
        "videos", 
        {"video_id": video_id}, 
        {"$inc": {"view_count": 1}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Video not found")
        
    return {"success": True, "message": "View recorded"}
