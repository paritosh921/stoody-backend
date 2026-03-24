"""Auth adapter — re-exports exampen_common JWT dependency."""

from exampen_common.auth import ExamPenUser, get_current_user

__all__ = ["ExamPenUser", "get_current_user"]
