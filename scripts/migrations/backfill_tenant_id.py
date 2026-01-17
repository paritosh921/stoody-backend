"""
Backfill tenant_id (login code) for existing tenants in the master registry.

Usage:
    cd backend
    python scripts/migrations/backfill_tenant_id.py --dry-run
    python scripts/migrations/backfill_tenant_id.py
    python scripts/migrations/backfill_tenant_id.py --only-active
    python scripts/migrations/backfill_tenant_id.py --limit 10
    python scripts/migrations/backfill_tenant_id.py --keep-invalid
"""

import argparse
import asyncio
import logging
import os
import random
import string
import sys
import re
from datetime import datetime
from typing import Dict, Optional, Set

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
    pass  # python-dotenv not installed, rely on environment variables


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TENANT_ID_PATTERN = re.compile(r"^[A-Z]{4}[0-9]{4}$")


def _generate_tenant_id(existing: Set[str]) -> str:
    """Generate a unique tenant ID in AAAA0000 format."""
    while True:
        letters = "".join(random.choices(string.ascii_uppercase, k=4))
        digits = "".join(random.choices(string.digits, k=4))
        tenant_id = f"{letters}{digits}"
        if tenant_id not in existing:
            existing.add(tenant_id)
            return tenant_id


class TenantIdBackfill:
    def __init__(
        self,
        mongo_uri: str,
        master_db: str,
        dry_run: bool = False,
        only_active: bool = False,
        limit: Optional[int] = None,
        keep_invalid: bool = False,
    ):
        self.mongo_uri = mongo_uri
        self.master_db_name = master_db
        self.dry_run = dry_run
        self.only_active = only_active
        self.limit = limit
        self.keep_invalid = keep_invalid
        self.client: Optional[AsyncIOMotorClient] = None
        self.master_db = None
        self.stats: Dict[str, int] = {
            "scanned": 0,
            "updated": 0,
            "skipped_valid": 0,
            "skipped_invalid": 0,
        }

    async def connect(self) -> None:
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.master_db = self.client[self.master_db_name]
        logger.info("Connected to MongoDB master DB: %s", self.master_db_name)

    async def close(self) -> None:
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    async def _load_existing_ids(self) -> Set[str]:
        existing: Set[str] = set()
        cursor = self.master_db["tenants"].find({"tenant_id": {"$exists": True}}, {"tenant_id": 1})
        async for doc in cursor:
            tenant_id = (doc.get("tenant_id") or "").strip().upper()
            if tenant_id and TENANT_ID_PATTERN.match(tenant_id):
                existing.add(tenant_id)
        return existing

    async def run(self) -> None:
        await self.connect()
        try:
            existing_ids = await self._load_existing_ids()
            query = {}
            if self.only_active:
                query["status"] = "active"

            cursor = self.master_db["tenants"].find(query)
            if self.limit:
                cursor = cursor.limit(self.limit)

            async for tenant in cursor:
                self.stats["scanned"] += 1
                current = (tenant.get("tenant_id") or "").strip().upper()

                if current and TENANT_ID_PATTERN.match(current):
                    self.stats["skipped_valid"] += 1
                    continue

                if current and not TENANT_ID_PATTERN.match(current) and self.keep_invalid:
                    self.stats["skipped_invalid"] += 1
                    continue

                new_id = _generate_tenant_id(existing_ids)
                update = {
                    "tenant_id": new_id,
                    "tenant_id_assigned_at": datetime.utcnow(),
                }
                logger.info(
                    "Assigning tenant_id=%s to tenant=%s (db=%s, subdomain=%s)",
                    new_id,
                    str(tenant.get("_id")),
                    tenant.get("db_name"),
                    tenant.get("subdomain"),
                )

                if not self.dry_run:
                    await self.master_db["tenants"].update_one(
                        {"_id": tenant["_id"]},
                        {"$set": update},
                    )
                self.stats["updated"] += 1

            logger.info(
                "Backfill complete. scanned=%s updated=%s skipped_valid=%s skipped_invalid=%s",
                self.stats["scanned"],
                self.stats["updated"],
                self.stats["skipped_valid"],
                self.stats["skipped_invalid"],
            )
        finally:
            await self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill tenant_id for master tenants registry")
    parser.add_argument("--dry-run", action="store_true", help="Log changes without writing data")
    parser.add_argument("--only-active", action="store_true", help="Only update active tenants")
    parser.add_argument("--keep-invalid", action="store_true", help="Do not overwrite invalid tenant_id values")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tenants processed")
    parser.add_argument("--mongo-uri", default=None, help="MongoDB URI (default: from env)")
    parser.add_argument("--master-db", default=None, help="Master DB name (default: from env)")
    args = parser.parse_args()

    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    master_db = args.master_db or os.getenv("MONGODB_DB_MASTER", "skb_master")

    logger.info("MongoDB URI: %s...", mongo_uri[:50])
    logger.info("Master DB: %s", master_db)

    backfill = TenantIdBackfill(
        mongo_uri=mongo_uri,
        master_db=master_db,
        dry_run=args.dry_run,
        only_active=args.only_active,
        limit=args.limit,
        keep_invalid=args.keep_invalid,
    )
    asyncio.run(backfill.run())


if __name__ == "__main__":
    main()
