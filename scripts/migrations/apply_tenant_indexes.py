"""
Apply standard indexes to tenant databases (DB-per-tenant).

Usage:
    cd backend
    python scripts/migrations/apply_tenant_indexes.py --db-name skb_in_ABC12_DEF_34_001
    python scripts/migrations/apply_tenant_indexes.py --all-tenants
    python scripts/migrations/apply_tenant_indexes.py --all-tenants --dry-run
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Tuple

from pymongo.errors import OperationFailure
from motor.motor_asyncio import AsyncIOMotorClient

# Add parent directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_dir)

# Load .env file from backend directory
try:
    from dotenv import load_dotenv
    env_path = os.path.join(backend_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


IndexDef = Tuple[List[Tuple[str, int]], Dict[str, object]]


TENANT_INDEXES: Dict[str, List[IndexDef]] = {
    "students": [
        ([("username", 1)], {"unique": True, "name": "uniq_students_username"}),
        ([("email", 1)], {"sparse": True, "name": "idx_students_email"}),
    ],
    "tutors": [
        ([("username", 1)], {"unique": True, "name": "uniq_tutors_username"}),
        ([("tutor_id", 1)], {"unique": True, "name": "uniq_tutors_tutor_id"}),
    ],
    "admins": [
        ([("email", 1)], {"unique": True, "name": "uniq_admins_email"}),
    ],
    "strokes": [
        ([("user_id", 1), ("timestamp", -1)], {"name": "idx_strokes_user_ts"}),
    ],
    "smartboard_sessions": [
        ([("session_id", 1)], {"unique": True, "name": "uniq_smartboard_session_id"}),
        ([("tutor_id", 1), ("status", 1)], {"name": "idx_smartboard_tutor_status"}),
    ],
}


async def apply_indexes(tenant_db, dry_run: bool = False) -> None:
    """Apply tenant index template to a specific database."""
    for collection_name, indexes in TENANT_INDEXES.items():
        collection = tenant_db[collection_name]
        for keys, options in indexes:
            if dry_run:
                logger.info("DRY RUN: %s.%s create_index(%s, %s)", tenant_db.name, collection_name, keys, options)
                continue
            try:
                await collection.create_index(keys, **options)
                logger.info("Created index on %s.%s: %s", tenant_db.name, collection_name, options.get("name"))
            except OperationFailure as exc:
                logger.warning(
                    "Index creation failed for %s.%s (%s): %s",
                    tenant_db.name,
                    collection_name,
                    options.get("name"),
                    exc,
                )


async def get_tenant_db_names(master_db, only_active: bool = True) -> List[str]:
    query = {"status": "active"} if only_active else {}
    cursor = master_db["tenants"].find(query, {"db_name": 1})
    tenants = await cursor.to_list(length=None)
    names = [t.get("db_name") for t in tenants if t.get("db_name")]
    return names


async def run(args) -> None:
    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    master_db_name = args.master_db or os.getenv("MONGODB_DB_MASTER", "skb_master")

    client = AsyncIOMotorClient(mongo_uri)
    try:
        if args.db_name:
            tenant_db = client[args.db_name]
            await apply_indexes(tenant_db, dry_run=args.dry_run)
            return

        if not args.all_tenants:
            raise ValueError("Provide --db-name or --all-tenants")

        master_db = client[master_db_name]
        tenant_db_names = await get_tenant_db_names(master_db, only_active=not args.include_inactive)

        if not tenant_db_names:
            logger.warning("No tenant databases found in master registry")
            return

        for db_name in tenant_db_names:
            tenant_db = client[db_name]
            await apply_indexes(tenant_db, dry_run=args.dry_run)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply tenant index template")
    parser.add_argument("--db-name", help="Apply indexes to a single tenant DB")
    parser.add_argument("--all-tenants", action="store_true", help="Apply indexes to all tenants")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive tenants")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without applying")
    parser.add_argument("--mongo-uri", default=None, help="MongoDB URI (default: from env)")
    parser.add_argument("--master-db", default=None, help="Master DB name (default: from env)")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
