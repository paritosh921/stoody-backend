"""
Practice Module - Main Package
Refactored from practice_async.py (1,863 lines) into modular structure

This package provides:
- Practice session management
- Question fetching and evaluation
- AI-powered answer grading with OCR support
- Session statistics and analytics
"""

from .router import router
from .models import (
    PracticeSession,
    SessionQuestion,
    SessionAnswer,
    StartSessionRequest,
    SessionResponse,
    SessionsListResponse,
    PracticeStats,
    NextQuestionRequest,
    EvaluateRequest,
    EvaluateResponse,
)
from .dependencies import require_student_or_admin

__all__ = [
    "router",
    # Models
    "PracticeSession",
    "SessionQuestion",
    "SessionAnswer",
    "StartSessionRequest",
    "SessionResponse",
    "SessionsListResponse",
    "PracticeStats",
    "NextQuestionRequest",
    "EvaluateRequest",
    "EvaluateResponse",
    # Dependencies
    "require_student_or_admin",
]
