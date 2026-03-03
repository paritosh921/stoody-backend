"""
Token Blacklist for JWT Revocation
Tracks revoked tokens to enable portal auto-logout.

Supports two levels of revocation:
  1. Token-level  – revoke(token)  blocks a single JWT string.
     In-memory per-worker (sufficient because the token is only
     used by the client that received it).
  2. User-level   – revoke_user_session() / is_user_session_revoked()
     Redis-backed so the flag is visible to ALL uvicorn workers.
     Cleared on next login via clear_user_session_revocation().
"""

from typing import Set
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Redis key prefix for user-level revocation
_USER_REVOKED_PREFIX = "user_revoked"
_USER_REVOKED_TTL = 86400  # 24 hours


class TokenBlacklist:
    """Manages revoked JWT tokens with automatic expiry cleanup."""

    def __init__(self):
        self._blacklist: Set[str] = set()
        self._expiry_times: dict[str, datetime] = {}
        logger.info("TokenBlacklist initialized")

    # ── Token-level (in-memory, per-worker) ────────────────────

    def revoke(self, token: str, expiry_seconds: int = 86400) -> None:
        """
        Add token to blacklist with expiry time.
        """
        self._blacklist.add(token)
        self._expiry_times[token] = datetime.utcnow() + timedelta(seconds=expiry_seconds)
        logger.info(f"Token revoked (expires in {expiry_seconds}s)")

    def is_revoked(self, token: str) -> bool:
        """
        Check if token is revoked.
        Also performs automatic cleanup of expired tokens.
        """
        if token in self._blacklist:
            if token in self._expiry_times:
                if datetime.utcnow() > self._expiry_times[token]:
                    self._blacklist.discard(token)
                    del self._expiry_times[token]
                    return False
            return True
        return False

    # ── Maintenance ─────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """Remove expired tokens from blacklist."""
        now = datetime.utcnow()
        expired = [
            token for token, expiry in self._expiry_times.items()
            if now > expiry
        ]
        for token in expired:
            self._blacklist.discard(token)
            del self._expiry_times[token]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired tokens from blacklist")
        return len(expired)

    def get_stats(self) -> dict:
        """Get blacklist statistics."""
        return {
            "total_revoked": len(self._blacklist),
            "oldest_expiry": min(self._expiry_times.values()) if self._expiry_times else None,
            "newest_expiry": max(self._expiry_times.values()) if self._expiry_times else None
        }


# Global instance
token_blacklist = TokenBlacklist()


# ── User-level revocation (Redis-backed, shared across workers) ──────
#
# These are module-level async functions (not methods on the class)
# because they need an async cache_manager that isn't available at
# import time.  Every call site already has auth_manager in scope.

async def revoke_user_session(cache_manager, user_id: str) -> None:
    """
    Mark a user as globally logged-out in Redis.

    All workers will see this flag on their next auth check,
    forcing the portal (and any other client) to log out.
    """
    if not cache_manager:
        logger.warning("No cache_manager — user-level revocation skipped")
        return
    await cache_manager.set(
        f"{_USER_REVOKED_PREFIX}:{user_id}",
        "1",
        ttl=_USER_REVOKED_TTL,
        prefix="auth",
    )
    logger.info(f"User-level revocation set in Redis for user {user_id}")


async def is_user_session_revoked(cache_manager, user_id: str) -> bool:
    """
    Check whether all sessions for this user have been revoked.

    Returns False (allow) if Redis is unavailable so that a cache
    outage doesn't lock every user out.
    """
    if not cache_manager:
        return False
    return await cache_manager.exists(
        f"{_USER_REVOKED_PREFIX}:{user_id}",
        prefix="auth",
    )


async def clear_user_session_revocation(cache_manager, user_id: str) -> None:
    """
    Remove the user-level revocation flag.

    Called during login so the freshly-issued token is accepted.
    """
    if not cache_manager:
        return
    deleted = await cache_manager.delete(
        f"{_USER_REVOKED_PREFIX}:{user_id}",
        prefix="auth",
    )
    if deleted:
        logger.info(f"User-level revocation cleared in Redis for user {user_id}")
