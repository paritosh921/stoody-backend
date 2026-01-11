"""
Practice Module - Pydantic Models
Request and response models for practice API endpoints
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator


class PracticeSession(BaseModel):
    """Model for practice session data"""
    id: Optional[str] = None
    student_id: str
    mode: str = Field(..., pattern="^(practice|exam|timed)$")
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    questions_attempted: int = Field(default=0, ge=0)
    correct_answers: int = Field(default=0, ge=0)
    total_time_spent: int = Field(default=0, ge=0)  # in seconds
    started_at: datetime
    completed_at: Optional[datetime] = None
    is_completed: bool = False


class SessionQuestion(BaseModel):
    """Model for a question attempt within a session"""
    question_id: str
    answer: str
    is_correct: bool
    time_spent: int = Field(ge=0)  # in seconds
    answered_at: datetime


class SessionAnswer(BaseModel):
    """Model for submitting an answer"""
    question_id: str
    answer: str
    time_spent: int = Field(default=0, ge=0)


class StartSessionRequest(BaseModel):
    """Request model for starting a practice session"""
    mode: str = Field(..., pattern="^(practice|exam|timed)$")
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    time_limit: Optional[int] = Field(None, ge=1)  # in minutes
    document_id: Optional[str] = None  # Practice set document ID


class SessionResponse(BaseModel):
    """Response model for session data"""
    id: str
    mode: str
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    questions_attempted: int
    correct_answers: int
    accuracy_rate: float
    total_time_spent: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    is_completed: bool


class SessionsListResponse(BaseModel):
    """Response model for paginated sessions list"""
    sessions: List[SessionResponse]
    total: int
    page: int
    limit: int


class PracticeStats(BaseModel):
    """Model for practice statistics"""
    total_sessions: int
    total_time_spent: int
    average_accuracy: float
    sessions_by_mode: Dict[str, int]
    recent_activity: List[Dict[str, Any]]


class NextQuestionRequest(BaseModel):
    """Request model for getting next practice question"""
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    excludeIds: Optional[List[str]] = None


class EvaluateRequest(BaseModel):
    """Request model for evaluating a student submission"""
    questionId: str
    answerText: Optional[str] = None
    canvasData: Optional[str] = None
    canvasPages: Optional[List[str]] = None

    @validator('canvasData', pre=True)
    def _normalize_canvas_data(cls, v):
        """Normalize canvas data to data URL format"""
        try:
            if v and isinstance(v, str) and not v.startswith('data:image'):
                return f"data:image/png;base64,{v}"
        except Exception:
            pass
        return v

    @validator('canvasPages', pre=True)
    def _normalize_canvas_pages(cls, v):
        """Normalize canvas pages to list of data URLs"""
        if v is None:
            return v
        try:
            if isinstance(v, list):
                out: List[str] = []
                for item in v:
                    s = None
                    if isinstance(item, str):
                        s = item
                    elif isinstance(item, dict):
                        s = (
                            item.get('dataUrl')
                            or item.get('url')
                            or item.get('data')
                            or item.get('image')
                            or item.get('src')
                        )
                    if s:
                        if not s.startswith('data:image'):
                            s = f"data:image/png;base64,{s}"
                        out.append(s)
                return out
            # If a single string is provided, wrap as list
            if isinstance(v, str):
                s = v
                if not s.startswith('data:image'):
                    s = f"data:image/png;base64,{s}"
                return [s]
        except Exception:
            return v
        return v

    @root_validator(pre=True)
    def _coerce_aliases(cls, values):
        """Accept snake_case aliases from frontend"""
        mapping = {
            'question_id': 'questionId',
            'answer_text': 'answerText',
            'canvas_data': 'canvasData',
            'canvas_pages': 'canvasPages'
        }
        for src, dst in mapping.items():
            if src in values and dst not in values:
                values[dst] = values[src]
        # If canvasPages provided as single object/string elsewhere, normalize to list
        cp = values.get('canvasPages')
        if isinstance(cp, str):
            values['canvasPages'] = [cp]
        return values


class EvaluateResponse(BaseModel):
    """Response model for evaluation results"""
    success: bool = True
    evaluation: Dict[str, Any]
