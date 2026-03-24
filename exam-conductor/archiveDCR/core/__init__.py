"""ExamPen DCR core — re-exports for convenience."""

from exampen.dcr.core.nats_client import NatsClient
from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
    require_exampen_role,
)
from exampen.dcr.core.indexes import ensure_exampen_indexes

__all__ = [
    "NatsClient",
    "ExamPenUser",
    "get_exampen_user",
    "require_exampen_role",
    "ensure_exampen_indexes",
]
