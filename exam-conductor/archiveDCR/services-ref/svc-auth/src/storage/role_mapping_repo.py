"""PostgreSQL storage for configurable role mappings.

Table: ``role_mappings``
  - ``stoody_role TEXT NOT NULL``
  - ``tenant_id TEXT NOT NULL``
  - ``exampen_roles TEXT[] NOT NULL``
  - ``updated_at TIMESTAMPTZ``
  - PRIMARY KEY (stoody_role, tenant_id)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from exampen_common.logging import get_logger

_log = get_logger(__name__)


class RoleMappingRepo:
    """CRUD operations for the ``role_mappings`` table."""

    def __init__(self, sf: async_sessionmaker[AsyncSession]) -> None:
        self._sf = sf

    async def get_all(self, tenant_id: str) -> dict[str, list[str]]:
        """Load role mappings for *tenant_id* plus global defaults.

        Tenant-specific overrides take precedence over global defaults
        (``tenant_id = ''``).  The RLS policy already filters rows, but
        we also ORDER BY tenant_id DESC so tenant-specific rows override
        globals during dict construction.
        """
        async with self._sf() as session:
            result = await session.execute(
                text(
                    """
                    SELECT stoody_role, exampen_roles, tenant_id
                    FROM role_mappings
                    WHERE tenant_id = :tid OR tenant_id = ''
                    ORDER BY tenant_id DESC
                    """
                ),
                {"tid": tenant_id},
            )
            rows = result.mappings().all()
        # Later rows (globals) only fill gaps — tenant-specific already set
        mappings: dict[str, list[str]] = {}
        for row in rows:
            if row["stoody_role"] not in mappings:
                mappings[row["stoody_role"]] = list(row["exampen_roles"])
        return mappings

    async def upsert(
        self,
        stoody_role: str,
        tenant_id: str,
        exampen_roles: list[str],
    ) -> dict[str, Any]:
        """Insert or update a tenant-scoped role mapping."""
        now = datetime.now(timezone.utc)
        async with self._sf() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO role_mappings (stoody_role, tenant_id, exampen_roles, updated_at)
                    VALUES (:stoody_role, :tenant_id, :exampen_roles, :updated_at)
                    ON CONFLICT (stoody_role, tenant_id) DO UPDATE SET
                        exampen_roles = EXCLUDED.exampen_roles,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "stoody_role": stoody_role,
                    "tenant_id": tenant_id,
                    "exampen_roles": exampen_roles,
                    "updated_at": now,
                },
            )
            await session.commit()
        _log.info("Upserted role mapping: %s/%s -> %s", stoody_role, tenant_id, exampen_roles)
        return {
            "stoody_role": stoody_role,
            "tenant_id": tenant_id,
            "exampen_roles": exampen_roles,
            "updated_at": now.isoformat(),
        }

    async def delete(self, stoody_role: str, tenant_id: str) -> bool:
        """Delete a tenant-scoped role mapping. Returns True if deleted."""
        async with self._sf() as session:
            result = await session.execute(
                text(
                    "DELETE FROM role_mappings WHERE stoody_role = :role AND tenant_id = :tid"
                ),
                {"role": stoody_role, "tid": tenant_id},
            )
            await session.commit()
        return result.rowcount > 0  # type: ignore[union-attr]
