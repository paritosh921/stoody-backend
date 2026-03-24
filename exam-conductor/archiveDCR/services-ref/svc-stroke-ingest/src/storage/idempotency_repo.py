"""Redis-backed idempotency key store.

Keys are ``{exam_id}:{pen_mac}:{chunk_index}`` and expire after a
configurable TTL (default 7 days).
"""

from __future__ import annotations

from redis.asyncio import Redis

from exampen_common.logging import get_logger
from src.config import IDEMPOTENCY_TTL_SECONDS

_log = get_logger(__name__)


class RedisIdempotencyRepo:
    """Check-and-mark idempotency keys in Redis."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = IDEMPOTENCY_TTL_SECONDS,
    ) -> None:
        self._url = redis_url
        self._ttl = ttl_seconds
        self._redis: Redis | None = None

    async def connect(self) -> None:
        """Open a Redis connection pool."""
        self._redis = Redis.from_url(self._url, decode_responses=True)
        _log.info("Redis idempotency store connected: %s", self._url)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()

    async def check_and_mark(self, key: str) -> bool:
        """Atomically check whether *key* is new.

        Returns ``True`` if the key was newly inserted (first time),
        ``False`` if it already existed (duplicate).

        Uses ``SET NX EX`` for atomic insert-if-absent with TTL.
        """
        assert self._redis is not None, "call connect() first"
        result = await self._redis.set(
            f"idem:{key}",
            "1",
            nx=True,
            ex=self._ttl,
        )
        is_new = result is not None
        if not is_new:
            _log.debug("duplicate idempotency key: %s", key)
        return is_new
