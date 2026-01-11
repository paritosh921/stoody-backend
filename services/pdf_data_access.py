"""
Shared database helpers for PDF routes (B2C vs main DB).
"""

from typing import Any, Dict, List, Optional

from core.database import DatabaseManager


def is_b2c_user(user_type: Optional[str]) -> bool:
    return user_type in ["b2c_admin", "b2c_user"]


async def find_one(
    db: DatabaseManager,
    collection: str,
    query: Dict[str, Any],
    is_b2c: bool
) -> Optional[Dict[str, Any]]:
    if is_b2c:
        return await db.b2c_find_one(collection, query)
    return await db.mongo_find_one(collection, query)


async def find_many(
    db: DatabaseManager,
    collection: str,
    query: Dict[str, Any],
    is_b2c: bool
) -> List[Dict[str, Any]]:
    if is_b2c:
        return await db.b2c_find(collection, query)
    return await db.mongo_find(collection, query)


async def update_one(
    db: DatabaseManager,
    collection: str,
    query: Dict[str, Any],
    update: Dict[str, Any],
    is_b2c: bool,
    upsert: bool = False
) -> Any:
    if is_b2c:
        return await db.b2c_update_one(collection, query, update, upsert=upsert)
    return await db.mongo_update_one(collection, query, update, upsert=upsert)


async def delete_one(
    db: DatabaseManager,
    collection: str,
    query: Dict[str, Any],
    is_b2c: bool
) -> Any:
    if is_b2c:
        return await db.b2c_delete_one(collection, query)
    return await db.mongo_delete_one(collection, query)


async def insert_one(
    db: DatabaseManager,
    collection: str,
    document: Dict[str, Any],
    is_b2c: bool
) -> Any:
    if is_b2c:
        return await db.b2c_insert_one(collection, document)
    return await db.mongo_insert_one(collection, document)
