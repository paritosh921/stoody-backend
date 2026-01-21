from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class VideoSchema(BaseModel):
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
    teacher_ids: List[str] = []
    is_active: bool = True
    uploaded_by: str
    uploaded_at: datetime
    view_count: int = 0
    
class CreateVideoRequest(BaseModel):
    video_id: str = Field(..., min_length=2, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$')
    title: str = Field(..., min_length=2, max_length=200)
    youtube_url: str
    subject: str
    standard: str
    description: Optional[str] = None
    course_plan: Optional[str] = None
    section: Optional[str] = None
    teacher_ids: Optional[List[str]] = None
    is_active: bool = True

class UpdateVideoRequest(BaseModel):
    title: Optional[str] = None
    youtube_url: Optional[str] = None
    subject: Optional[str] = None
    standard: Optional[str] = None
    description: Optional[str] = None
    course_plan: Optional[str] = None
    section: Optional[str] = None
    teacher_ids: Optional[List[str]] = None
    is_active: Optional[bool] = None
