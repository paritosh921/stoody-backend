"""
Rename tenant databases to match new institution ID format and update master registry.

Institution ID format: AAAA-BBBB-0000
Tenant ID derived from institution ID suffix: BBBB-0000

Mapping file format (JSON array):
[
  {"old_db_name": "skb_in_4GZ4M_M30_DW_001", "institution_id": "INRJ-ABCD-1234"},
  {"old_db_name": "skb_in_564J5_PCE_BN_001", "institution_id": "INMH-WXYZ-5678"}
]

Usage:
  cd backend
  python scripts/migrations/rename_tenant_dbs.py --mapping mappings.json --dry-run
  python scripts/migrations/rename_tenant_dbs.py --mapping mappings.json
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List

from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_dir)

try:
    from dotenv import load_dotenv
    env_path = os.path.join(backend_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

INSTITUTION_ID_PATTERN = re.compile(r"^[A-Z]{4}-[A-Z]{4}-[0-9]{4}$")


def normalize_institution_id(value: str) -> str:
    return value.strip().upper()


def derive_tenant_id(institution_id: str) -> str:
    normalized = normalize_institution_id(institution_id)
    if not INSTITUTION_ID_PATTERN.match(normalized):
        raise ValueError("Institution ID must match AAAA-BBBB-0000 format")
    parts = normalized.split("-")
    return f"{parts[1]}-{parts[2]}"


def build_db_name(institution_id: str) -> str:
    normalized = normalize_institution_id(institution_id)
    return f"skb_{normalized.lower()}"


def load_mapping(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Mapping file must be a JSON array")
    return payload


def rename_database(client: MongoClient, old_db: str, new_db: str, dry_run: bool, drop_target: bool) -> None:
    if old_db == new_db:
        logger.info("DB name already matches: %s", old_db)
        return

    old = client[old_db]
    collections = old.list_collection_names()
    if not collections:
        logger.warning("No collections found in %s", old_db)
        return

    for name in collections:
        if name.startswith("system."):
            continue
        source = f"{old_db}.{name}"
        target = f"{new_db}.{name}"
        if dry_run:
            logger.info("DRY RUN: renameCollection %s -> %s", source, target)
            continue
        logger.info("Renaming %s -> %s", source, target)
        client.admin.command("renameCollection", source, to=target, dropTarget=drop_target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename tenant DBs and update tenant registry")
    parser.add_argument("--mapping", required=True, help="Path to JSON mapping file")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--master-db", default=os.getenv("MONGODB_DB_MASTER", "skb_master"))
    parser.add_argument("--dry-run", action="store_true", help="Log changes without writing")
    parser.add_argument("--drop-target", action="store_true", help="Drop target collections if they exist")
    args = parser.parse_args()

    mappings = load_mapping(args.mapping)
    client = MongoClient(args.mongo_uri)
    master_db = client[args.master_db]

    try:
        for entry in mappings:
            old_db_name = (entry.get("old_db_name") or "").strip()
            institution_id = (entry.get("institution_id") or "").strip()

            if not old_db_name or not institution_id:
                logger.error("Invalid mapping entry (missing old_db_name or institution_id): %s", entry)
                continue

            normalized_id = normalize_institution_id(institution_id)
            if not INSTITUTION_ID_PATTERN.match(normalized_id):
                logger.error("Invalid institution ID format for %s: %s", old_db_name, normalized_id)
                continue

            tenant_id = derive_tenant_id(normalized_id)
            new_db_name = build_db_name(normalized_id)

            tenant = master_db["tenants"].find_one({"db_name": old_db_name})
            if not tenant:
                logger.error("No tenant found with db_name=%s", old_db_name)
                continue

            logger.info("Mapping %s -> %s (tenant_id=%s)", old_db_name, new_db_name, tenant_id)

            if not args.dry_run:
                rename_database(client, old_db_name, new_db_name, dry_run=False, drop_target=args.drop_target)
                master_db["tenants"].update_one(
                    {"_id": tenant["_id"]},
                    {"$set": {
                        "institution_id": normalized_id,
                        "tenant_id": tenant_id,
                        "db_name": new_db_name
                    }}
                )
            else:
                rename_database(client, old_db_name, new_db_name, dry_run=True, drop_target=args.drop_target)

    finally:
        client.close()


if __name__ == "__main__":
    main()
