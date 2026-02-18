"""
Fix orphan documents that are missing admin_id in a tenant database.

After the multi-tenant isolation changes, all tenant-scoped collections
require an admin_id field on every document. This script finds the admin
account in the target database and stamps admin_id on all documents that
are missing it.

Usage:
    python fix_admin_id_orphans.py --db-name skb_indl-ciel-1001 --admin-email cielknowledge@gmail.com

    # Dry-run (just report, don't modify):
    python fix_admin_id_orphans.py --db-name skb_indl-ciel-1001 --admin-email cielknowledge@gmail.com --dry-run
"""
import argparse
import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

# All tenant-scoped collections (must match core/tenant.py TENANT_SCOPED_COLLECTIONS)
TENANT_SCOPED_COLLECTIONS = [
    "students",
    "documents",
    "tutors",
    "questions",
    "question_attempts",
    "student_activity_log",
    "chat_sessions",
    "student_test_attempts",
    "assignments",
    "meetings",
    "notifications",
    "class_schedules",
    "smartboard_sessions",
]


def main():
    parser = argparse.ArgumentParser(
        description="Fix orphan documents missing admin_id in a tenant database"
    )
    parser.add_argument(
        "--db-name", required=True,
        help="Target database name (e.g. skb_indl-ciel-1001)"
    )
    parser.add_argument(
        "--admin-email", required=True,
        help="Email of the admin account to use as owner"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report orphans, don't modify anything"
    )
    args = parser.parse_args()

    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        print("[ERROR] MONGODB_URI not set in environment")
        sys.exit(1)

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
    db = client[args.db_name]

    print("=" * 60)
    print(f"FIX ADMIN_ID ORPHANS — database: {args.db_name}")
    if args.dry_run:
        print("MODE: DRY RUN (no changes will be made)")
    print("=" * 60)

    try:
        # Find the admin account
        admin = db.admins.find_one({"email": args.admin_email})
        if not admin:
            print(f"\n[ERROR] Admin not found: {args.admin_email}")
            print("Available admins in this database:")
            for a in db.admins.find({}, {"email": 1, "_id": 1, "name": 1, "full_name": 1}):
                name = a.get("name") or a.get("full_name") or "N/A"
                print(f"  - {a['email']} (ID: {a['_id']}, Name: {name})")
            sys.exit(1)

        admin_id = admin["_id"]
        admin_name = admin.get("name") or admin.get("full_name") or "N/A"
        print(f"\nAdmin: {args.admin_email}")
        print(f"Admin ID: {admin_id}")
        print(f"Admin Name: {admin_name}")

        orphan_filter = {
            "$or": [
                {"admin_id": {"$exists": False}},
                {"admin_id": None}
            ]
        }

        total_fixed = 0
        print(f"\n{'Collection':<30} {'Orphans':<10} {'Action'}")
        print("-" * 60)

        for collection_name in TENANT_SCOPED_COLLECTIONS:
            collection = db[collection_name]
            orphan_count = collection.count_documents(orphan_filter)
            total_count = collection.count_documents({})

            if orphan_count > 0:
                if args.dry_run:
                    action = f"WOULD FIX (of {total_count} total)"
                else:
                    result = collection.update_many(
                        orphan_filter,
                        {"$set": {"admin_id": admin_id}}
                    )
                    action = f"FIXED {result.modified_count} (of {total_count} total)"
                    total_fixed += result.modified_count
            else:
                action = f"OK ({total_count} total, all have admin_id)"

            print(f"{collection_name:<30} {orphan_count:<10} {action}")

        # Also check for documents with wrong admin_id type (string instead of ObjectId)
        print(f"\n{'='*60}")
        print("CHECKING FOR STRING admin_id (should be ObjectId)")
        print("-" * 60)

        for collection_name in TENANT_SCOPED_COLLECTIONS:
            collection = db[collection_name]
            # Find documents where admin_id is a string (not ObjectId)
            string_admin_filter = {
                "admin_id": {"$type": "string"}
            }
            string_count = collection.count_documents(string_admin_filter)
            if string_count > 0:
                if args.dry_run:
                    print(f"{collection_name:<30} {string_count} docs with string admin_id — WOULD CONVERT")
                else:
                    # Convert string admin_id to ObjectId
                    for doc in collection.find(string_admin_filter, {"admin_id": 1}):
                        try:
                            oid = ObjectId(doc["admin_id"])
                            collection.update_one(
                                {"_id": doc["_id"]},
                                {"$set": {"admin_id": oid}}
                            )
                        except Exception:
                            pass  # Skip invalid ObjectId strings
                    print(f"{collection_name:<30} {string_count} docs converted string → ObjectId")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print("=" * 60)
        if args.dry_run:
            print("No changes made (dry run)")
        else:
            print(f"Total documents fixed: {total_fixed}")

        # Final verification
        print(f"\nFinal document counts with admin_id = {admin_id}:")
        for collection_name in TENANT_SCOPED_COLLECTIONS:
            count = db[collection_name].count_documents({"admin_id": admin_id})
            if count > 0:
                print(f"  {collection_name}: {count}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    main()
