from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr

# Pydantic models
class CreateStudentRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: str = Field(..., min_length=2, max_length=100)
    password: Optional[str] = Field(None, min_length=6)  # Optional - will auto-generate if not provided
    email: Optional[EmailStr] = None
    date_of_birth: Optional[str] = None  # Format: YYYY-MM-DD
    gender: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    stream: Optional[str] = None
    grade: Optional[str] = None
    phone: Optional[str] = None
    plan_types: Optional[List[str]] = None
    subjects: Optional[List[str]] = None

class UpdateStudentRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    grade: Optional[str] = Field(None, description="New grade/class for the student")
    section: Optional[str] = Field(None, description="New section for the student")
    is_active: Optional[bool] = None


class SessionPromotionRequest(BaseModel):
    """Request model for promoting students to next session"""
    new_session: str = Field(..., description="New academic session e.g., '2025-26'")
    grade_mappings: Dict[str, str] = Field(
        ..., 
        description="Mapping of current grade to new grade e.g., {'10': '11', '11': '12'}"
    )
    # Filters to select which students to promote
    grade_filter: Optional[List[str]] = Field(
        None,
        description="Only promote students from these grades. If empty/None, apply to all grades in mappings"
    )
    section_filter: Optional[List[str]] = Field(
        None,
        description="Only promote students from these sections. If empty/None, apply to all sections"
    )
    student_ids: Optional[List[str]] = Field(
        None,
        description="Specific student IDs to promote. If provided, ony these students are promoted (overrides grade/section filters)"
    )
    section_updates: Optional[Dict[str, str]] = Field(
        None,
        description="Optional section updates for specific students. Key is student_id, value is new section"
    )
    deactivate_old_content: bool = Field(
        True,
        description="Whether to mark old notes, assignments, tests as inactive"
    )
    preview_only: bool = Field(
        False,
        description="If true, only return preview without making changes"
    )


class SessionPromotionResponse(BaseModel):
    """Response model for session promotion"""
    success: bool
    message: str
    new_session: str
    students_promoted: int
    students_skipped: int
    content_deactivated: int
    details: Optional[List[Dict[str, Any]]] = None

class StudentResponse(BaseModel):
    id: str
    student_id: str
    username: str
    full_name: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    stream: Optional[str] = None
    grade: Optional[str] = None
    phone: Optional[str] = None
    plan_types: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    is_active: bool
    requires_password_change: Optional[bool] = None
    password_reset_requested: Optional[bool] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    generated_password: Optional[str] = None  # Only included on creation if auto-generated

class StudentsListResponse(BaseModel):
    students: List[StudentResponse]
    total: int
    page: int
    limit: int

class DashboardStats(BaseModel):
    total_students: int
    valid_students: int  # Students with is_active = true
    active_students: int  # Students who have logged in (have last_login)
    practice_sets_count: int
    test_series_count: int
    chapter_notes_count: int

class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)
