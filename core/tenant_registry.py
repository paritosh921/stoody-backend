"""
Tenant registry helpers for DB-per-tenant resolution.
"""

import logging
from typing import Optional, Dict, Any

from core.database import DatabaseManager

logger = logging.getLogger(__name__)

TENANTS_COLLECTION = "tenants"
ACTIVE_STATUS = "active"


def normalize_institution_id(institution_id: str) -> str:
    """Normalize institution IDs for lookup (upper-case, trimmed)."""
    if not institution_id:
        return ""
    return institution_id.strip().upper()


def normalize_tenant_id(tenant_id: str) -> str:
    """Normalize tenant login IDs for lookup (upper-case, trimmed)."""
    if not tenant_id:
        return ""
    return tenant_id.strip().upper()


async def _find_tenant(
    db: DatabaseManager,
    filter_dict: Dict[str, Any],
    include_inactive: bool = False,
) -> Optional[Dict[str, Any]]:
    collection = await db.get_master_collection(TENANTS_COLLECTION)
    if collection is None:
        return None
    if not include_inactive:
        filter_dict = {**filter_dict, "status": ACTIVE_STATUS}
    return await collection.find_one(filter_dict)


async def get_tenant_by_subdomain(
    db: DatabaseManager,
    subdomain: str,
    include_inactive: bool = False,
) -> Optional[Dict[str, Any]]:
    if not subdomain:
        return None
    return await _find_tenant(db, {"subdomain": subdomain}, include_inactive)


async def get_tenant_by_institution_id(
    db: DatabaseManager,
    institution_id: str,
    include_inactive: bool = False,
) -> Optional[Dict[str, Any]]:
    normalized = normalize_institution_id(institution_id)
    if not normalized:
        return None
    return await _find_tenant(db, {"institution_id": normalized}, include_inactive)


async def get_tenant_by_tenant_id(
    db: DatabaseManager,
    tenant_id: str,
    include_inactive: bool = False,
) -> Optional[Dict[str, Any]]:
    normalized = normalize_tenant_id(tenant_id)
    if not normalized:
        return None
    return await _find_tenant(db, {"tenant_id": normalized}, include_inactive)
