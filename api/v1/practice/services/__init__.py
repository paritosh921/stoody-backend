"""
Practice Module - Services
"""

from .evaluation_service import (
    evaluate_student_submission,
    grade_student_submission,
)
from .session_service import (
    create_session,
    submit_answer,
    complete_session,
    get_sessions,
    get_stats,
)

__all__ = [
    "evaluate_student_submission",
    "grade_student_submission",
    "create_session",
    "submit_answer",
    "complete_session",
    "get_sessions",
    "get_stats",
]
