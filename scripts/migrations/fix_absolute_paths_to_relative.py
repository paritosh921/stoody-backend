"""
Migration Script: Convert Absolute File Paths to Relative Paths in MongoDB

This script fixes all documents in MongoDB that have absolute file paths
and converts them to relative paths for cross-platform compatibility.

Run this ONCE after deploying the path_utils fix.
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv(backend_dir / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "skillbot_db")

# Backend directory reference
BACKEND_DIR = backend_dir


def convert_absolute_to_relative(absolute_path: str) -> str:
    """
    Convert absolute path to relative path from backend root

    Handles both Unix (/home/ubuntu/backend/...) and Windows (D:\\...) paths
    """
    try:
        abs_path = Path(absolute_path)

        # Try to make it relative to backend dir
        try:
            rel_path = abs_path.relative_to(BACKEND_DIR)
            return str(rel_path).replace('\\', '/')
        except ValueError:
            # Path is not under backend dir
            # Extract just the relevant part (uploads/...)
            path_str = str(absolute_path)

            # Look for 'uploads' in the path
            if 'uploads' in path_str:
                idx = path_str.index('uploads')
                relative = path_str[idx:]
                return relative.replace('\\', '/')

            # If no uploads found, return as-is (will be logged as warning)
            print(f"  ⚠️ WARNING: Cannot convert path: {absolute_path}")
            return path_str.replace('\\', '/')

    except Exception as e:
        print(f"  ❌ ERROR converting path {absolute_path}: {e}")
        return str(absolute_path).replace('\\', '/')


async def migrate_collection(db, collection_name: str, path_field: str):
    """
    Migrate a collection by converting absolute paths to relative

    Args:
        db: MongoDB database instance
        collection_name: Name of collection to migrate
        path_field: Name of field containing file path
    """
    print(f"\n📦 Migrating collection: {collection_name}")
    print(f"   Field: {path_field}")

    collection = db[collection_name]

    # Find all documents with absolute paths
    # Absolute paths start with / (Unix) or drive letter (Windows)
    cursor = collection.find({
        path_field: {
            "$exists": True,
            "$ne": None,
            "$regex": "^(/|[A-Za-z]:)"
        }
    })

    documents = await cursor.to_list(length=None)
    total_docs = len(documents)

    if total_docs == 0:
        print(f"   ✅ No documents with absolute paths found")
        return 0

    print(f"   Found {total_docs} documents with absolute paths")

    updated_count = 0
    error_count = 0

    for doc in documents:
        try:
            doc_id = doc.get("_id")
            old_path = doc.get(path_field)

            if not old_path:
                continue

            new_path = convert_absolute_to_relative(old_path)

            if new_path != old_path:
                # Update document
                result = await collection.update_one(
                    {"_id": doc_id},
                    {"$set": {path_field: new_path}}
                )

                if result.modified_count > 0:
                    updated_count += 1
                    print(f"   ✓ Updated {doc_id}")
                    print(f"     Old: {old_path}")
                    print(f"     New: {new_path}")

        except Exception as e:
            error_count += 1
            print(f"   ❌ Error updating document {doc.get('_id')}: {e}")

    print(f"\n   📊 Results:")
    print(f"      Total documents: {total_docs}")
    print(f"      Updated: {updated_count}")
    print(f"      Errors: {error_count}")

    return updated_count


async def main():
    """Run the migration"""
    print("=" * 60)
    print("🔧 Absolute to Relative Path Migration")
    print("=" * 60)
    print(f"\nBackend Directory: {BACKEND_DIR}")
    print(f"MongoDB URI: {MONGODB_URI[:50]}...")
    print(f"Database: {MONGODB_DB_NAME}")

    if not MONGODB_URI:
        print("\n❌ ERROR: MONGODB_URI not configured in .env")
        return

    # Connect to MongoDB
    print("\n📡 Connecting to MongoDB...")
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]

    try:
        # Test connection
        await client.admin.command('ping')
        print("✅ Connected to MongoDB")

        # Migrate collections
        total_updated = 0

        # 1. Images collection (file_path field)
        total_updated += await migrate_collection(db, "images", "file_path")

        # 2. Documents collection (file_path field) if exists
        try:
            total_updated += await migrate_collection(db, "documents", "file_path")
        except Exception as e:
            print(f"\n⚠️ Skipping documents collection: {e}")

        # 3. Questions collection (if it has image paths)
        # Check if questions have direct file paths
        try:
            sample_question = await db.questions.find_one({})
            if sample_question and "file_path" in sample_question:
                total_updated += await migrate_collection(db, "questions", "file_path")
        except Exception as e:
            print(f"\n⚠️ Skipping questions collection: {e}")

        # Summary
        print("\n" + "=" * 60)
        print("✅ MIGRATION COMPLETE")
        print("=" * 60)
        print(f"\nTotal documents updated: {total_updated}")
        print("\n📝 Next Steps:")
        print("   1. Restart backend server")
        print("   2. Test image loading in frontend")
        print("   3. If issues persist, check backend logs")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        raise

    finally:
        client.close()
        print("\n🔌 Disconnected from MongoDB")


if __name__ == "__main__":
    print("\n⚠️  WARNING: This migration will modify data in MongoDB")
    print("   Make sure you have a backup before proceeding!")

    response = input("\n   Continue? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        asyncio.run(main())
    else:
        print("\n❌ Migration cancelled")
