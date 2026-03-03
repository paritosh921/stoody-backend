"""
Token Blacklist for JWT Revocation
Tracks revoked tokens to enable portal auto-logout.

Supports two levels of revocation:
  1. Token-level  – revoke(token)  blocks a single JWT string.
  2. User-level   – revoke_user(user_id) blocks ALL tokens for a user,
     regardless of which client created them.  Cleared on next login
     via clear_user_revocation(user_id).
"""

from typing import Set
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TokenBlacklist:
    """Manages revoked JWT tokens with automatic expiry cleanup."""

    def __init__(self):
        self._blacklist: Set[str] = set()
        self._expiry_times: dict[str, datetime] = {}
        # User-level revocation: user_id → revoked_at timestamp
        self._user_revoked_at: dict[str, datetime] = {}
        logger.info("TokenBlacklist initialized")
    
    def revoke(self, token: str, expiry_seconds: int = 86400) -> None:
        """
        Add token to blacklist with expiry time.
        
        Args:
            token: JWT token to revoke
            expiry_seconds: How long to keep token in blacklist (default 24 hours)
        """
        self._blacklist.add(token)
        self._expiry_times[token] = datetime.utcnow() + timedelta(seconds=expiry_seconds)
        logger.info(f"Token revoked (expires in {expiry_seconds}s)")
    
    def is_revoked(self, token: str) -> bool:
        """
        Check if token is revoked.
        
        Also performs automatic cleanup of expired tokens.
        
        Args:
            token: JWT token to check
            
        Returns:
            True if token is revoked, False otherwise
        """
        if token in self._blacklist:
            # Check if token has expired from blacklist
            if token in self._expiry_times:
                if datetime.utcnow() > self._expiry_times[token]:
                    # Token expired from blacklist - remove it
                    self._blacklist.discard(token)
                    del self._expiry_times[token]
                    return False
            return True
        return False
    
    # ── User-level revocation ──────────────────────────────────────

    def revoke_user(self, user_id: str, expiry_seconds: int = 86400) -> None:
        """
        Revoke ALL sessions for a user regardless of which token they hold.

        Called from any logout endpoint so that every other client
        (portal, desktop agent, etc.) is forced out on the next auth check.

        Args:
            user_id: The user whose sessions should be invalidated.
            expiry_seconds: How long the revocation lasts (default 24 h).
                            A fresh login clears it immediately.
        """
        self._user_revoked_at[user_id] = datetime.utcnow()
        logger.info(f"User-level revocation set for user {user_id} "
                     f"(expires in {expiry_seconds}s)")

    def is_user_revoked(self, user_id: str) -> bool:
        """
        Check whether *all* sessions for this user have been revoked.

        Auto-expires after 24 hours so the dict doesn't grow unbounded.
        """
        if user_id not in self._user_revoked_at:
            return False
        revoked_at = self._user_revoked_at[user_id]
        if datetime.utcnow() - revoked_at > timedelta(hours=24):
            del self._user_revoked_at[user_id]
            return False
        return True

    def clear_user_revocation(self, user_id: str) -> None:
        """
        Clear user-level revocation.

        Called during login so the freshly-issued token is accepted.
        """
        if user_id in self._user_revoked_at:
            del self._user_revoked_at[user_id]
            logger.info(f"User-level revocation cleared for user {user_id}")

    # ── Maintenance ─────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """
        Remove expired tokens from blacklist.
        
        Returns:
            Number of tokens removed
        """
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
