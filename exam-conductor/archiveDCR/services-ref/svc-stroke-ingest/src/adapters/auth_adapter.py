"""JWT validation adapter using ``exampen_common.auth``.

Re-exports the FastAPI dependency so route modules use a single import.
"""

from __future__ import annotations

from exampen_common.auth import ExamPenUser, get_current_user, validate_token

__all__ = ["ExamPenUser", "get_current_user", "validate_token"]
