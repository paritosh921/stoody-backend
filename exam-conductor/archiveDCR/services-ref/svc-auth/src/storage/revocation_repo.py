"""PostgreSQL storage for token revocation records.

Table: ``revocations``
  - ``jti TEXT PRIMARY KEY`` — JWT token ID
  - ``tenant_id TEXT NOT NULL`` — tenant scope
  - ``subject_user_id TEXT`` — user whose token is revoked
  - ``revoked_at TIMESTAMPTZ`` — when the revocation was created
  - ``reason TEXT`` — mandatory reason for revocation
  - ``revoked_by TEXT`` — actor who performed the revocation
  - ``expires_at TIMESTAMPTZ`` — optional auto-expiry of the revocation
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from exampen_common.logging import get_logger

_log = get_logger(__name__)


class RevocationRepo:
    """CRUD operations for the ``revocations`` table."""

    def __init__(self, sf: async_sessionmaker[AsyncSession]) -> None:
        self._sf = sf

    async def revoke(
        self,
        jti: str,
        tenant_id: str,
        reason: str,
        revoked_by: str,
        subject_user_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Insert a revocation record. Returns the created record."""
        now = datetime.now(timezone.utc)
        async with self._sf() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO revocations
                        (jti, tenant_id, subject_user_id, revoked_at, reason, revoked_by, expires_at)
                    VALUES (:jti, :tenant_id, :subject_user_id, :revoked_at, :reason, :revoked_by, :expires_at)
                    ON CONFLICT (jti) DO UPDATE SET
                        reason = EXCLUDED.reason,
                        revoked_by = EXCLUDED.revoked_by,
                        revoked_at = EXCLUDED.revoked_at,
                        expires_at = EXCLUDED.expires_at
                    """
                ),
                {
                    "jti": jti,
                    "tenant_id": tenant_id,
                    "subject_user_id": subject_user_id,
                    "revoked_at": now,
                    "reason": reason,
                    "revoked_by": revoked_by,
                    "expires_at": expires_at,
                },
            )
            await session.commit()
        _log.info("Revoked jti=%s tenant=%s by=%s reason=%s", jti, tenant_id, revoked_by, reason)
        return {
            "jti": jti,
            "revoked": True,
            "revoked_at": now.isoformat(),
            "reason": reason,
        }

    async def is_revoked(self, jti: str) -> dict[str, Any]:
        """Check whether *jti* is currently revoked."""
        async with self._sf() as session:
            result = await session.execute(
                text(
                    """
                    SELECT jti, revoked_at, reason, expires_at
                    FROM revocations
                    WHERE jti = :jti
                    """
                ),
                {"jti": jti},
            )
            row = result.mappings().first()

        if row is None:
            return {"jti": jti, "revoked": False}

        # Check if revocation has expired
        expires_at = row["expires_at"]
        if expires_at is not None and expires_at < datetime.now(timezone.utc):
            return {"jti": jti, "revoked": False}

        return {
            "jti": jti,
            "revoked": True,
            "revoked_at": row["revoked_at"].isoformat() if row["revoked_at"] else None,
            "reason": row["reason"],
        }

    async def delete(self, jti: str) -> bool:
        """Remove a revocation record. Returns True if a row was deleted."""
        async with self._sf() as session:
            result = await session.execute(
                text("DELETE FROM revocations WHERE jti = :jti"),
                {"jti": jti},
            )
            await session.commit()
        deleted = result.rowcount > 0  # type: ignore[union-attr]
        if deleted:
            _log.info("Un-revoked jti=%s", jti)
        return deleted
