"""
Add unique index on username field in students collection
Ensures globally unique usernames across all students
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING
from pymongo.errors import DuplicateKeyError
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_async import MONGODB_URL, MONGODB_DB_NAME

async def add_unique_index():
    """Add unique index on username field"""
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[MONGODB_DB_NAME]

    try:
        # Check for duplicate usernames before adding index
        print("Checking for duplicate usernames...")
        pipeline = [
            {
                "$group": {
                    "_id": "$username",
                    "count": {"$sum": 1},
                    "ids": {"$push": "$_id"}
                }
            },
            {
                "$match": {
                    "count": {"$gt": 1}
                }
            }
        ]

        duplicates = await db.students.aggregate(pipeline).to_list(length=None)

        if duplicates:
            print(f"\n⚠️  Found {len(duplicates)} duplicate username(s):")
            for dup in duplicates:
                print(f"   Username: '{dup['_id']}' appears {dup['count']} times")
                print(f"   Student IDs: {dup['ids']}")
            print("\n❌ Cannot add unique index. Please resolve duplicates first.")
            print("   You can manually rename duplicate usernames in the database.")
            return False

        print("✅ No duplicate usernames found")

        # Create unique index on username
        print("\nCreating unique index on username field...")
        index_model = IndexModel(
            [("username", ASCENDING)],
            name="username_unique",
            unique=True,
            background=True
        )

        result = await db.students.create_indexes([index_model])
        print(f"✅ Unique index created: {result}")

        # Also create compound index for admin panel filtering (admin_id + student_id)
        print("\nCreating compound index on admin_id + student_id...")
        admin_index = IndexModel(
            [("admin_id", ASCENDING), ("student_id", ASCENDING)],
            name="admin_student_id",
            background=True
        )

        result = await db.students.create_indexes([admin_index])
        print(f"✅ Compound index created: {result}")

        # List all indexes
        print("\nCurrent indexes on students collection:")
        indexes = await db.students.list_indexes().to_list(length=None)
        for idx in indexes:
            print(f"  - {idx['name']}: {idx.get('key', {})}")

        return True

    except DuplicateKeyError as e:
        print(f"❌ Duplicate key error: {e}")
        print("   There are duplicate usernames in the database.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Adding Unique Index on Username Field")
    print("=" * 60)

    success = asyncio.run(add_unique_index())

    if success:
        print("\n" + "=" * 60)
        print("✅ Migration completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Migration failed!")
        print("=" * 60)
        sys.exit(1)
