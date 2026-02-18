"""
One-off script: Enable re-attempts on all student_test_attempts in a tenant DB.

Usage:
    python update_reattempts.py --db-name skb_indl-ciel-1001
    python update_reattempts.py --db-name skb_indl-ciel-1001 --mongo-uri mongodb+srv://...
"""

import argparse
import os
from pymongo import MongoClient


def main():
    parser = argparse.ArgumentParser(description="Enable re-attempts for all test attempts in a tenant DB")
    parser.add_argument("--db-name", required=True, help="Tenant database name (e.g. skb_indl-ciel-1001)")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
                        help="MongoDB connection URI (default: MONGODB_URI env or localhost)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without modifying")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]

    if args.dry_run:
        count = db.student_test_attempts.count_documents({"can_reattempt": {"$ne": True}})
        print(f"[DRY RUN] Would update {count} test attempts in {args.db_name}")
    else:
        result = db.student_test_attempts.update_many(
            {},
            {"$set": {"can_reattempt": True}}
        )
        print(f"Updated {result.modified_count} test attempts in {args.db_name}")
        print(f"Matched {result.matched_count} total attempts")

    client.close()


if __name__ == "__main__":
    main()
