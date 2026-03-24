"""Consumer coordinator — starts all NATS background consumers.

Each consumer is registered as a durable JetStream subscription with
queue groups to prevent duplicate processing across worker instances.
All consumers are gracefully skipped if NATS is not available.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List

from . import (
    consumer_ai,
    consumer_analytics,
    consumer_page,
    consumer_plagiarism,
    consumer_rescore,
    consumer_score,
    consumer_stroke,
)

logger = logging.getLogger(__name__)

# Ordered list of consumer modules; each must expose a ``register(nats, db_manager)``
# coroutine that sets up the JetStream subscription.
_CONSUMER_MODULES = [
    consumer_stroke,
    consumer_page,
    consumer_ai,
    consumer_score,
    consumer_rescore,
    consumer_analytics,
    consumer_plagiarism,
]


async def start_all_consumers(
    nats: Any,
    db_manager: Any,
) -> List[asyncio.Task]:
    """Register every consumer and return a list of background tasks.

    Parameters
    ----------
    nats:
        A connected :class:`NatsClient` (or ``None`` to skip).
    db_manager:
        The application's database manager for obtaining per-tenant
        MongoDB connections.

    Returns
    -------
    List of ``asyncio.Task`` handles.  In practice, NATS push
    subscriptions run within the connection's internal loop, so the
    returned list is primarily for bookkeeping and graceful shutdown.
    If *nats* is ``None``, returns an empty list.
    """
    if nats is None:
        logger.warning(
            "NATS not available — all %d ExamPen consumers skipped",
            len(_CONSUMER_MODULES),
        )
        return []

    tasks: List[asyncio.Task] = []

    for module in _CONSUMER_MODULES:
        name = module.__name__.rsplit(".", 1)[-1]
        try:
            await module.register(nats, db_manager)
            logger.info("Consumer registered: %s", name)
        except Exception:
            logger.exception("Failed to register consumer: %s", name)

    logger.info(
        "All ExamPen consumers started (%d registered)", len(_CONSUMER_MODULES)
    )
    return tasks
