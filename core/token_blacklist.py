"""
Token Blacklist for JWT Revocation
Tracks revoked tokens to enable portal auto-logout
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
