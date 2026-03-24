"""ExamPen shared Python utilities — auth, NATS, DB, logging."""

from exampen_common.auth import (
    ExamPenUser,
    JWKSManager,
    get_current_user,
    validate_token,
)
from exampen_common.db import (
    create_pool,
    get_health,
    rls_middleware,
)
from exampen_common.logging import (
    configure_logging,
    get_logger,
    RequestIdMiddleware,
)
from exampen_common.nats_client import (
    NatsClient,
    create_nats_client,
)

__all__ = [
    # auth
    "ExamPenUser",
    "JWKSManager",
    "get_current_user",
    "validate_token",
    # db
    "create_pool",
    "get_health",
    "rls_middleware",
    # logging
    "configure_logging",
    "get_logger",
    "RequestIdMiddleware",
    # nats
    "NatsClient",
    "create_nats_client",
]
