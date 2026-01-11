"""
Practice API - Re-export from modular implementation

This file maintains backward compatibility with existing imports.
The actual implementation is now in ./practice/

Original: 1,863 lines
Refactored: Split into 10+ modular files
"""

# Re-export router for backward compatibility
from .practice import router

# Re-export models
from .practice.models import (
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

# Re-export dependencies
from .practice.dependencies import require_student_or_admin

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
