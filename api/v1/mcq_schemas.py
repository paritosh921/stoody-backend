"""
Pydantic schemas for MCQ endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MCQOption(BaseModel):
    id: str
    text: str
    is_correct: bool = False


class MCQQuestion(BaseModel):
    id: Optional[str] = None
    question_text: str
    subject: str
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$")
    options: List[MCQOption] = Field(..., min_items=2, max_items=6)
    explanation: Optional[str] = None
    tags: List[str] = []
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    is_active: bool = True


class MCQResponse(BaseModel):
    id: str
    question_text: str
    subject: str
    difficulty: str
    options: List[MCQOption]
    explanation: Optional[str] = None
    tags: List[str] = []
    created_at: datetime


class MCQListResponse(BaseModel):
    questions: List[MCQResponse]
    total: int
    page: int
    limit: int


class MCQAttempt(BaseModel):
    question_id: str
    selected_option_id: str
    time_spent: int = Field(default=0, ge=0)


class MCQAttemptResponse(BaseModel):
    id: str
    question_id: str
    selected_option_id: str
    correct_option_id: str
    is_correct: bool
    time_spent: int
    submitted_at: datetime
    explanation: Optional[str] = None


class MCQStats(BaseModel):
    total_questions: int
    total_attempts: int
    correct_attempts: int
    accuracy_rate: float
    subject_breakdown: Dict[str, Dict[str, int]]
    difficulty_breakdown: Dict[str, Dict[str, int]]
