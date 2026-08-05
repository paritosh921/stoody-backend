"""
Data Migration Script: Backfill copy_sets for Copy Set Feature

This script:
1. Creates a default "Copy 1" copy_set for every user who has canvas_pages
2. Sets active_copy_id on the student document
3. Backfills copy_id on existing canvas_pages, note_classifications,
   and classification_queue documents
4. Deduplicates note_classifications where (user_id, copy_id, book_type, page_number)
   collides after removing pen_mac from the identity key
5. Drops old unique indexes and creates new ones that include copy_id
6. Claims one semantic Practice copy per user and creates its unique index

IMPORTANT: Run this script BEFORE deploying the new code that writes copy_id.

Usage:
    cd backend
    python scripts/migrations/backfill_copy_sets.py

    # Dry run (no changes):
    python scripts/migrations/backfill_copy_sets.py --dry-run

    # Verbose output:
    python scripts/migrations/backfill_copy_sets.py --verbose

    # Run for all tenant DBs in master registry:
    python scripts/migrations/backfill_copy_sets.py --all-tenants

    # Run for a specific tenant DB name:
    python scripts/migrations/backfill_copy_sets.py --db-name skb_indl-ciel-1001

    # Also migrate B2C database:
    python scripts/migrations/backfill_copy_sets.py --all-tenants --include-b2c
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

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

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError, OperationFailure

logger = logging.getLogger("backfill_copy_sets")

MONGODB_URL = os.getenv("MONGODB_URI") or os.getenv("MONGODB_URL")
MONGODB_DB_MASTER = os.getenv("MONGODB_DB_MASTER", "skb_master")
MONGODB_DB_B2C = os.getenv("MONGODB_DB_STOODY", "STOODY-b2c")


async def drop_matching_indexes(
    collection,
    *,
    key_specs: list[list[tuple[str, int]]],
    names: list[str] | None = None,
    unique: bool | None = None,
) -> int:
    """Drop legacy indexes by name or exact key spec."""
    index_info = await collection.index_information()
    target_specs = {tuple((str(k), int(v)) for k, v in spec) for spec in key_specs}
    dropped = 0

    for existing_name, existing_def in index_info.items():
        if existing_name == "_id_":
            continue
        existing_keys = tuple((str(k), int(v)) for k, v in existing_def.get("key", []))
        name_match = bool(names and existing_name in names)
        spec_match = existing_keys in target_specs if target_specs else False
        unique_match = unique is None or bool(existing_def.get("unique", False)) == unique
        if not ((name_match or spec_match) and unique_match):
            continue

        try:
            await collection.drop_index(existing_name)
            dropped += 1
            logger.info("Dropped legacy index %s on %s", existing_name, collection.full_name)
        except OperationFailure:
            pass

    return dropped


class MigrationStats:
    def __init__(self):
        self.copy_sets_created = 0
        self.active_copy_set = 0
        self.canvas_pages_updated = 0
        self.note_classifications_updated = 0
        self.note_classifications_deduped = 0
        self.classification_queue_updated = 0
        self.indexes_created = 0
        self.indexes_dropped = 0
        self.errors: List[str] = []

    def summary(self) -> str:
        lines = [
            f"  copy_sets created:                {self.copy_sets_created}",
            f"  active_copy_id set on users:      {self.active_copy_set}",
            f"  canvas_pages backfilled:           {self.canvas_pages_updated}",
            f"  note_classifications backfilled:   {self.note_classifications_updated}",
            f"  note_classifications deduped:      {self.note_classifications_deduped}",
            f"  classification_queue backfilled:   {self.classification_queue_updated}",
            f"  indexes created:                   {self.indexes_created}",
            f"  indexes dropped:                   {self.indexes_dropped}",
        ]
        if self.errors:
            lines.append(f"  ERRORS: {len(self.errors)}")
            for e in self.errors[:10]:
                lines.append(f"    - {e}")
        return "\n".join(lines)


async def migrate_database(
    client: AsyncIOMotorClient,
    db_name: str,
    *,
    dry_run: bool = False,
    verbose: bool = False,
    user_collection_name: str = "students",
) -> MigrationStats:
    """Run the copy_sets migration on a single database."""
    stats = MigrationStats()
    db = client[db_name]
    now = datetime.now(timezone.utc)

    canvas_pages = db["canvas_pages"]
    note_cls = db["note_classifications"]
    cls_queue = db["classification_queue"]
    copy_sets = db["copy_sets"]
    users = db[user_collection_name]

    # Step 1: Find all distinct user_ids that have canvas_pages
    logger.info("[%s] Step 1: Finding users with canvas pages...", db_name)
    user_ids: Set[str] = set()
    try:
        distinct_uids = await canvas_pages.distinct("user_id")
        for uid in distinct_uids:
            user_ids.add(str(uid))
    except Exception as e:
        stats.errors.append(f"distinct user_id failed: {e}")
        return stats

    logger.info("[%s] Found %d unique user_ids in canvas_pages", db_name, len(user_ids))

    # Step 2: For each user, create a copy_set and backfill copy_id
    for user_id in sorted(user_ids):
        if verbose:
            logger.info("[%s] Processing user %s", db_name, user_id)

        # Check if user already has a copy_set
        existing_copy = await copy_sets.find_one({"user_id": user_id})
        if existing_copy:
            copy_id = str(existing_copy["_id"])
            if verbose:
                logger.info("[%s]   User %s already has copy_set %s", db_name, user_id, copy_id)
        else:
            if dry_run:
                copy_id = "DRY_RUN_COPY_ID"
                logger.info("[%s]   [DRY RUN] Would create Copy 1 for user %s", db_name, user_id)
            else:
                doc = {
                    "user_id": user_id,
                    "title": "Copy 1",
                    "is_archived": False,
                    "created_at": now,
                    "updated_at": now,
                }
                result = await copy_sets.insert_one(doc)
                copy_id = str(result.inserted_id)
                stats.copy_sets_created += 1
                if verbose:
                    logger.info("[%s]   Created Copy 1 (%s) for user %s", db_name, copy_id, user_id)

        # Set active_copy_id on user document
        if not dry_run:
            # Try multiple user_id representations
            user_filter_parts = [{"user_id": user_id}]
            try:
                if ObjectId.is_valid(user_id):
                    user_filter_parts.append({"_id": ObjectId(user_id)})
            except Exception:
                pass
            user_filter_parts.append({"_id": user_id})

            for uf in user_filter_parts:
                result = await users.update_one(
                    {**uf, "active_copy_id": {"$exists": False}},
                    {"$set": {"active_copy_id": copy_id}},
                )
                if result.modified_count > 0:
                    stats.active_copy_set += 1
                    break

        # Backfill copy_id on canvas_pages for this user
        # Match all user_id variants
        uid_variants = [user_id]
        try:
            if ObjectId.is_valid(user_id):
                uid_variants.append(ObjectId(user_id))
        except Exception:
            pass

        if not dry_run:
            result = await canvas_pages.update_many(
                {"user_id": {"$in": uid_variants}, "copy_id": {"$exists": False}},
                {"$set": {"copy_id": copy_id}},
            )
            stats.canvas_pages_updated += result.modified_count
        else:
            count = await canvas_pages.count_documents(
                {"user_id": {"$in": uid_variants}, "copy_id": {"$exists": False}}
            )
            logger.info("[%s]   [DRY RUN] Would backfill %d canvas_pages", db_name, count)

        # Backfill copy_id on note_classifications
        if not dry_run:
            result = await note_cls.update_many(
                {"user_id": {"$in": uid_variants}, "copy_id": {"$exists": False}},
                {"$set": {"copy_id": copy_id}},
            )
            stats.note_classifications_updated += result.modified_count
        else:
            count = await note_cls.count_documents(
                {"user_id": {"$in": uid_variants}, "copy_id": {"$exists": False}}
            )
            logger.info("[%s]   [DRY RUN] Would backfill %d note_classifications", db_name, count)

        # Backfill copy_id on classification_queue
        if not dry_run:
            result = await cls_queue.update_many(
                {"user_id": {"$in": uid_variants}, "copy_id": {"$exists": False}},
                {"$set": {"copy_id": copy_id}},
            )
            stats.classification_queue_updated += result.modified_count

    # Step 3: Deduplicate note_classifications
    # After backfill, (user_id, copy_id, book_type, page_number) may collide
    # if the same page had different pen_mac entries. Keep the newest.
    logger.info("[%s] Step 3: Deduplicating note_classifications...", db_name)
    try:
        pipeline = [
            {"$group": {
                "_id": {
                    "user_id": "$user_id",
                    "copy_id": "$copy_id",
                    "book_type": "$book_type",
                    "page_number": "$page_number",
                },
                "doc_ids": {"$push": "$_id"},
                "count": {"$sum": 1},
            }},
            {"$match": {"count": {"$gt": 1}}},
        ]
        async for group in note_cls.aggregate(pipeline):
            doc_ids = group["doc_ids"]
            # Fetch full docs, keep the newest by updated_at
            docs = []
            async for d in note_cls.find({"_id": {"$in": doc_ids}}):
                docs.append(d)
            if not docs:
                continue
            docs.sort(
                key=lambda d: d.get("updated_at") or d.get("created_at") or datetime.min,
                reverse=True,
            )
            keep = docs[0]
            remove_ids = [d["_id"] for d in docs[1:]]
            if remove_ids:
                if not dry_run:
                    await note_cls.delete_many({"_id": {"$in": remove_ids}})
                    stats.note_classifications_deduped += len(remove_ids)
                else:
                    logger.info(
                        "[%s]   [DRY RUN] Would remove %d duplicate note_classifications for (%s)",
                        db_name, len(remove_ids), group["_id"],
                    )
    except Exception as e:
        stats.errors.append(f"note_classifications dedup failed: {e}")
        logger.warning("[%s] Dedup failed: %s", db_name, e)

    # Step 3b: Give exactly one legacy Practice copy per user the semantic
    # purpose used by web/mobile. Title remains display data only.
    try:
        practice_groups = copy_sets.aggregate(
            [
                {
                    "$match": {
                        "$or": [
                            {"title": "Practice"},
                            {"purpose": "practice"},
                        ]
                    }
                },
                {
                    "$addFields": {
                        "practice_purpose_rank": {
                            "$cond": [{"$eq": ["$purpose", "practice"]}, 0, 1]
                        }
                    }
                },
                {
                    "$sort": {
                        "practice_purpose_rank": 1,
                        "is_archived": 1,
                        "created_at": 1,
                        "_id": 1,
                    }
                },
                {
                    "$group": {
                        "_id": "$user_id",
                        "copy_ids": {"$push": "$_id"},
                    }
                },
            ]
        )
        async for group in practice_groups:
            copy_ids = group.get("copy_ids") or []
            if not copy_ids:
                continue
            canonical_id = copy_ids[0]
            duplicate_ids = copy_ids[1:]
            if dry_run:
                logger.info(
                    "[%s]   [DRY RUN] Would set Practice purpose for user %s on %s",
                    db_name,
                    group.get("_id"),
                    canonical_id,
                )
                continue
            if duplicate_ids:
                await copy_sets.update_many(
                    {"_id": {"$in": duplicate_ids}},
                    {"$unset": {"purpose": ""}},
                )
            await copy_sets.update_one(
                {"_id": canonical_id},
                {
                    "$set": {
                        "title": "Practice",
                        "purpose": "practice",
                        "is_archived": False,
                        "updated_at": now,
                    }
                },
            )
    except Exception as e:
        stats.errors.append(f"practice copy purpose backfill failed: {e}")
        logger.warning("[%s] Practice purpose backfill failed: %s", db_name, e)

    # Step 4: Update indexes
    if not dry_run:
        logger.info("[%s] Step 4: Updating indexes...", db_name)

        # canvas_pages: drop any legacy 3-field unique index, then create new
        stats.indexes_dropped += await drop_matching_indexes(
            canvas_pages,
            key_specs=[[("user_id", 1), ("book_type", 1), ("page_number", 1)]],
            names=["uniq_canvas_page", "uniq_canvas_pages_user_book_page"],
            unique=True,
        )

        try:
            await canvas_pages.create_index(
                [("user_id", 1), ("copy_id", 1), ("book_type", 1), ("page_number", 1)],
                unique=True,
                name="uniq_canvas_page_v2",
            )
            stats.indexes_created += 1
            logger.info("[%s]   Created uniq_canvas_page_v2 index", db_name)
        except OperationFailure as e:
            stats.errors.append(f"uniq_canvas_page_v2 creation failed: {e}")
            logger.warning("[%s]   Failed to create uniq_canvas_page_v2: %s", db_name, e)

        try:
            await copy_sets.create_index(
                [("user_id", 1), ("purpose", 1)],
                unique=True,
                partialFilterExpression={"purpose": {"$type": "string"}},
                name="uniq_copy_sets_user_purpose",
            )
            stats.indexes_created += 1
            logger.info("[%s]   Created uniq_copy_sets_user_purpose index", db_name)
        except OperationFailure as e:
            stats.errors.append(f"uniq_copy_sets_user_purpose creation failed: {e}")
            logger.warning(
                "[%s]   Failed to create uniq_copy_sets_user_purpose: %s",
                db_name,
                e,
            )

        # note_classifications: drop old, create new
        try:
            await note_cls.drop_index("uniq_note_page")
            stats.indexes_dropped += 1
            logger.info("[%s]   Dropped old uniq_note_page index", db_name)
        except OperationFailure:
            pass

        try:
            await note_cls.create_index(
                [("user_id", 1), ("copy_id", 1), ("book_type", 1), ("page_number", 1)],
                unique=True,
                name="uniq_note_page_v2",
            )
            stats.indexes_created += 1
            logger.info("[%s]   Created uniq_note_page_v2 index", db_name)
        except OperationFailure as e:
            stats.errors.append(f"uniq_note_page_v2 creation failed: {e}")
            logger.warning("[%s]   Failed to create uniq_note_page_v2: %s", db_name, e)

        try:
            await cls_queue.drop_index("uniq_cls_queue_page")
            stats.indexes_dropped += 1
            logger.info("[%s]   Dropped old uniq_cls_queue_page index", db_name)
        except OperationFailure:
            pass

        try:
            await cls_queue.create_index(
                [("user_id", 1), ("copy_id", 1), ("pen_mac", 1), ("book_type", 1), ("page_number", 1), ("db_name", 1)],
                unique=True,
                name="uniq_cls_queue_page_v2",
            )
            stats.indexes_created += 1
            logger.info("[%s]   Created uniq_cls_queue_page_v2 index", db_name)
        except OperationFailure as e:
            stats.errors.append(f"uniq_cls_queue_page_v2 creation failed: {e}")
            logger.warning("[%s]   Failed to create uniq_cls_queue_page_v2: %s", db_name, e)

        # copy_sets: user index
        try:
            await copy_sets.create_index(
                [("user_id", 1), ("is_archived", 1), ("created_at", 1)],
                name="idx_copy_sets_user",
            )
            stats.indexes_created += 1
        except OperationFailure:
            pass

        # Keep pen_mac as a non-unique lookup index on note_classifications
        try:
            await note_cls.create_index(
                [("user_id", 1), ("pen_mac", 1)],
                name="idx_note_user_pen_mac",
            )
        except OperationFailure:
            pass

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Backfill copy_sets for the Copy Set feature")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without making changes")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--all-tenants", action="store_true", help="Run for all tenant DBs in master registry")
    parser.add_argument("--db-name", type=str, help="Run for a specific tenant DB name")
    parser.add_argument("--include-b2c", action="store_true", help="Also migrate B2C database")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not MONGODB_URL:
        logger.error("MONGODB_URI or MONGODB_URL environment variable not set")
        sys.exit(1)

    client = AsyncIOMotorClient(MONGODB_URL)
    logger.info("Connected to MongoDB")

    db_names: List[str] = []
    b2c_names: List[str] = []

    if args.db_name:
        db_names.append(args.db_name)
    elif args.all_tenants:
        master_db = client[MONGODB_DB_MASTER]
        tenants = await master_db["tenants"].find(
            {"status": {"$ne": "deleted"}},
            {"db_name": 1},
        ).to_list(length=None)
        db_names = [t["db_name"] for t in tenants if t.get("db_name")]
        logger.info("Found %d tenant databases", len(db_names))

    if args.include_b2c:
        b2c_names.append(MONGODB_DB_B2C)

    if not db_names and not b2c_names:
        logger.error("No databases to migrate. Use --all-tenants, --db-name, or --include-b2c")
        sys.exit(1)

    total_stats = MigrationStats()

    for db_name in db_names:
        logger.info("=" * 60)
        logger.info("Migrating tenant database: %s", db_name)
        stats = await migrate_database(
            client, db_name,
            dry_run=args.dry_run,
            verbose=args.verbose,
            user_collection_name="students",
        )
        logger.info("[%s] Results:\n%s", db_name, stats.summary())
        # Accumulate
        total_stats.copy_sets_created += stats.copy_sets_created
        total_stats.active_copy_set += stats.active_copy_set
        total_stats.canvas_pages_updated += stats.canvas_pages_updated
        total_stats.note_classifications_updated += stats.note_classifications_updated
        total_stats.note_classifications_deduped += stats.note_classifications_deduped
        total_stats.classification_queue_updated += stats.classification_queue_updated
        total_stats.indexes_created += stats.indexes_created
        total_stats.indexes_dropped += stats.indexes_dropped
        total_stats.errors.extend(stats.errors)

    for db_name in b2c_names:
        logger.info("=" * 60)
        logger.info("Migrating B2C database: %s", db_name)
        stats = await migrate_database(
            client, db_name,
            dry_run=args.dry_run,
            verbose=args.verbose,
            user_collection_name="users",
        )
        logger.info("[%s] Results:\n%s", db_name, stats.summary())
        total_stats.copy_sets_created += stats.copy_sets_created
        total_stats.active_copy_set += stats.active_copy_set
        total_stats.canvas_pages_updated += stats.canvas_pages_updated
        total_stats.note_classifications_updated += stats.note_classifications_updated
        total_stats.note_classifications_deduped += stats.note_classifications_deduped
        total_stats.classification_queue_updated += stats.classification_queue_updated
        total_stats.indexes_created += stats.indexes_created
        total_stats.indexes_dropped += stats.indexes_dropped
        total_stats.errors.extend(stats.errors)

    logger.info("=" * 60)
    logger.info("TOTAL RESULTS:\n%s", total_stats.summary())

    if total_stats.errors:
        logger.warning("Migration completed with %d errors", len(total_stats.errors))
        sys.exit(1)
    else:
        logger.info("Migration completed successfully")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
