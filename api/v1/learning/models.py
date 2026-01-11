from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

class LearningDocument(BaseModel):
    """Document information for learning mode"""
    document_id: str
    title: str
    subject: str
    standard: str
    course_plan: str
    document_type: str
    difficulty: Optional[str] = None
    file_path: str
    ocr_status: str
    created_at: Optional[datetime] = None

class CourseStructure(BaseModel):
    """Course structure response"""
    standards: List[str]
    subjects: Dict[str, List[str]]
