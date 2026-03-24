"""JWT validation adapter using exampen_common.auth.

Provides a thin re-export so routes import from adapters (per layer rules)
rather than directly from the shared library.
"""

from __future__ import annotations

from exampen_common.auth import (
    ExamPenUser,
    get_current_user,
    validate_token,
)

__all__ = [
    "ExamPenUser",
    "get_current_user",
    "validate_token",
]
