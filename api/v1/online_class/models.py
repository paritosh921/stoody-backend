from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class CreateLockRequest(BaseModel):
    question_text: Optional[str] = None
    question_image_id: Optional[str] = None
    question_bbox: Optional[Dict[str, Any]] = None
    duration_seconds: int = Field(120, ge=10, le=3600)


class LockResponse(BaseModel):
    lock_id: str
    meeting_id: str
    tutor_id: str
    question_text: Optional[str] = None
    question_image_id: Optional[str] = None
    question_bbox: Optional[Dict[str, Any]] = None
    duration_seconds: int
    start_ts: datetime
    end_ts: Optional[datetime] = None
    status: str
    created_at: datetime
    ended_at: Optional[datetime] = None


class CreateSubmissionRequest(BaseModel):
    canvas_pages: List[str] = Field(default_factory=list)
    question_page_refs: Optional[Dict[str, Any]] = None
    answer_text: Optional[str] = None
    time_spent: Optional[float] = None
    client_submitted_at: Optional[datetime] = None


class SubmissionResponse(BaseModel):
    submission_id: str
    meeting_id: str
    lock_id: str
    student_id: str
    canvas_pages: List[str] = Field(default_factory=list)
    question_page_refs: Optional[Dict[str, Any]] = None
    answer_text: Optional[str] = None
    time_spent: Optional[float] = None
    analysis_status: str
    score: Optional[float] = None
    is_correct: Optional[bool] = None
    student_answer: Optional[str] = None
    work_shown: Optional[str] = None
    what_went_wrong: Optional[str] = None
    correct_solution: Optional[str] = None
    analysis_error: Optional[str] = None
    analysis_completed_at: Optional[datetime] = None
    analysis_failed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class SubmissionResultItem(BaseModel):
    submission_id: str
    student_id: str
    student_name: Optional[str] = None
    canvas_pages: List[str] = Field(default_factory=list)
    answer_text: Optional[str] = None
    time_spent: Optional[float] = None
    analysis_status: str
    score: Optional[float] = None
    is_correct: Optional[bool] = None
    student_answer: Optional[str] = None
    work_shown: Optional[str] = None
    what_went_wrong: Optional[str] = None
    correct_solution: Optional[str] = None
    analysis_error: Optional[str] = None
    analysis_completed_at: Optional[datetime] = None
    analysis_failed_at: Optional[datetime] = None
    created_at: datetime
