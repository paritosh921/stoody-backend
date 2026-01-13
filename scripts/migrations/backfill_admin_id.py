"""
Data Migration Script: Backfill admin_id for Tenant Isolation

This script adds admin_id to existing documents that were created before
tenant isolation was implemented.

Collections handled:
- smartboard_sessions: Links via tutor_id -> tutor -> created_by (admin_id)
- question_attempts: Links via session_id -> session -> admin_id

Usage:
    cd backend
    python scripts/migrations/backfill_admin_id.py

    # Dry run (no changes):
    python scripts/migrations/backfill_admin_id.py --dry-run

    # Verbose output:
    python scripts/migrations/backfill_admin_id.py --verbose
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime

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

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdminIdBackfill:
    """Backfill admin_id for tenant isolation"""

    def __init__(self, mongo_uri: str, db_name: str, dry_run: bool = False, verbose: bool = False):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.dry_run = dry_run
        self.verbose = verbose
        self.client = None
        self.db = None
        self.stats = {
            "smartboard_sessions": {"found": 0, "updated": 0, "skipped": 0, "errors": 0},
            "question_attempts": {"found": 0, "updated": 0, "skipped": 0, "errors": 0},
        }

    async def connect(self):
        """Connect to MongoDB"""
        logger.info(f"Connecting to MongoDB: {self.db_name}")
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client[self.db_name]

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    async def backfill_smartboard_sessions(self):
        """
        Backfill admin_id for smartboard_sessions.
        Links via: tutor_id -> tutors collection -> created_by (admin_id)
        """
        logger.info("=" * 60)
        logger.info("Backfilling smartboard_sessions...")
        logger.info("=" * 60)

        # Find sessions without admin_id
        cursor = self.db["smartboard_sessions"].find({"admin_id": {"$exists": False}})
        sessions = await cursor.to_list(length=None)

        self.stats["smartboard_sessions"]["found"] = len(sessions)
        logger.info(f"Found {len(sessions)} sessions without admin_id")

        for session in sessions:
            session_id = session.get("session_id", str(session.get("_id")))
            tutor_id = session.get("tutor_id")

            if not tutor_id:
                logger.warning(f"Session {session_id}: No tutor_id, skipping")
                self.stats["smartboard_sessions"]["skipped"] += 1
                continue

            # Look up tutor to get admin_id (created_by)
            tutor = await self.db["tutors"].find_one({"tutor_id": tutor_id})
            if not tutor:
                # Try by _id
                try:
                    tutor = await self.db["tutors"].find_one({"_id": ObjectId(tutor_id)})
                except Exception:
                    pass

            if not tutor:
                logger.warning(f"Session {session_id}: Tutor {tutor_id} not found, skipping")
                self.stats["smartboard_sessions"]["skipped"] += 1
                continue

            admin_id = tutor.get("created_by") or tutor.get("admin_id")
            if not admin_id:
                logger.warning(f"Session {session_id}: Tutor has no created_by/admin_id, skipping")
                self.stats["smartboard_sessions"]["skipped"] += 1
                continue

            # Ensure admin_id is ObjectId
            if isinstance(admin_id, str):
                admin_id = ObjectId(admin_id)

            if self.verbose:
                logger.info(f"Session {session_id}: Setting admin_id={admin_id}")

            if not self.dry_run:
                try:
                    result = await self.db["smartboard_sessions"].update_one(
                        {"_id": session["_id"]},
                        {"$set": {"admin_id": admin_id, "admin_id_backfilled_at": datetime.utcnow()}}
                    )
                    if result.modified_count > 0:
                        self.stats["smartboard_sessions"]["updated"] += 1
                    else:
                        self.stats["smartboard_sessions"]["skipped"] += 1
                except Exception as e:
                    logger.error(f"Session {session_id}: Update failed - {e}")
                    self.stats["smartboard_sessions"]["errors"] += 1
            else:
                self.stats["smartboard_sessions"]["updated"] += 1  # Would have updated

        logger.info(f"smartboard_sessions: {self.stats['smartboard_sessions']}")

    async def backfill_question_attempts(self):
        """
        Backfill admin_id for question_attempts.
        Links via: session_id -> smartboard_sessions -> admin_id
        """
        logger.info("=" * 60)
        logger.info("Backfilling question_attempts...")
        logger.info("=" * 60)

        # Find attempts without admin_id
        cursor = self.db["question_attempts"].find({"admin_id": {"$exists": False}})
        attempts = await cursor.to_list(length=None)

        self.stats["question_attempts"]["found"] = len(attempts)
        logger.info(f"Found {len(attempts)} attempts without admin_id")

        for attempt in attempts:
            attempt_id = attempt.get("attempt_id", str(attempt.get("_id")))
            session_id = attempt.get("session_id")

            if not session_id:
                logger.warning(f"Attempt {attempt_id}: No session_id, skipping")
                self.stats["question_attempts"]["skipped"] += 1
                continue

            # Look up session to get admin_id
            session = await self.db["smartboard_sessions"].find_one({"session_id": session_id})
            if not session:
                logger.warning(f"Attempt {attempt_id}: Session {session_id} not found, skipping")
                self.stats["question_attempts"]["skipped"] += 1
                continue

            admin_id = session.get("admin_id")
            if not admin_id:
                logger.warning(f"Attempt {attempt_id}: Session has no admin_id, skipping")
                self.stats["question_attempts"]["skipped"] += 1
                continue

            # Ensure admin_id is ObjectId
            if isinstance(admin_id, str):
                admin_id = ObjectId(admin_id)

            if self.verbose:
                logger.info(f"Attempt {attempt_id}: Setting admin_id={admin_id}")

            if not self.dry_run:
                try:
                    result = await self.db["question_attempts"].update_one(
                        {"_id": attempt["_id"]},
                        {"$set": {"admin_id": admin_id, "admin_id_backfilled_at": datetime.utcnow()}}
                    )
                    if result.modified_count > 0:
                        self.stats["question_attempts"]["updated"] += 1
                    else:
                        self.stats["question_attempts"]["skipped"] += 1
                except Exception as e:
                    logger.error(f"Attempt {attempt_id}: Update failed - {e}")
                    self.stats["question_attempts"]["errors"] += 1
            else:
                self.stats["question_attempts"]["updated"] += 1  # Would have updated

        logger.info(f"question_attempts: {self.stats['question_attempts']}")

    async def run(self):
        """Run all backfill operations"""
        try:
            await self.connect()

            if self.dry_run:
                logger.info("DRY RUN MODE - No changes will be made")

            # Run backfills in order (sessions first, then attempts which depend on sessions)
            await self.backfill_smartboard_sessions()
            await self.backfill_question_attempts()

            # Summary
            logger.info("=" * 60)
            logger.info("MIGRATION SUMMARY")
            logger.info("=" * 60)
            for collection, stats in self.stats.items():
                logger.info(f"{collection}:")
                logger.info(f"  Found:   {stats['found']}")
                logger.info(f"  Updated: {stats['updated']}")
                logger.info(f"  Skipped: {stats['skipped']}")
                logger.info(f"  Errors:  {stats['errors']}")

            if self.dry_run:
                logger.info("\nDRY RUN COMPLETE - Run without --dry-run to apply changes")

        finally:
            await self.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill admin_id for tenant isolation")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--mongo-uri", default=None, help="MongoDB URI (default: from env)")
    parser.add_argument("--db-name", default=None, help="Database name (default: from env)")

    args = parser.parse_args()

    # Get MongoDB settings from environment or arguments
    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = args.db_name or os.getenv("MONGODB_DB_NAME", "skillbot_local")

    logger.info(f"MongoDB URI: {mongo_uri[:50]}...")
    logger.info(f"Database: {db_name}")

    backfill = AdminIdBackfill(
        mongo_uri=mongo_uri,
        db_name=db_name,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    asyncio.run(backfill.run())


if __name__ == "__main__":
    main()
