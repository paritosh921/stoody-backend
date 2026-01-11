"""
Add classroom-related collections and indexes for Online Class feature.
Run once: python -m scripts.migrations.add_classroom_collections
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from motor.motor_asyncio import AsyncIOMotorClient

# Get MongoDB URL from environment or use default
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "skillbot")


async def migrate():
    print("=" * 60)
    print("Online Classroom Feature - Database Migration")
    print("=" * 60)
    print(f"MongoDB URL: {MONGODB_URL}")
    print(f"Database: {DATABASE_NAME}")
    print()

    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    # =========================================================================
    # Step 1: Add class_mappings field to students collection
    # =========================================================================
    print("1. Adding class_mappings field to students...")

    # Check how many students don't have class_mappings
    students_without_mappings = await db.students.count_documents({
        "class_mappings": {"$exists": False}
    })

    if students_without_mappings > 0:
        # Initialize class_mappings as empty array for all students that don't have it
        result = await db.students.update_many(
            {"class_mappings": {"$exists": False}},
            {"$set": {"class_mappings": []}}
        )
        print(f"   Updated {result.modified_count} students with empty class_mappings")
    else:
        print("   All students already have class_mappings field")

    # Create index on class_mappings for efficient queries
    try:
        await db.students.create_index([
            ("class_mappings.standard", 1),
            ("class_mappings.section", 1),
            ("class_mappings.subject", 1)
        ], name="class_mappings_lookup", sparse=True)
        print("   Created index: class_mappings_lookup")
    except Exception as e:
        print(f"   Index may already exist: {e}")

    print()

    # =========================================================================
    # Step 2: Create online_classes collection
    # =========================================================================
    print("2. Creating online_classes collection...")

    # Check if collection exists
    collections = await db.list_collection_names()
    if "online_classes" not in collections:
        # Create the collection with a sample schema (MongoDB is schemaless)
        await db.create_collection("online_classes")
        print("   Created online_classes collection")
    else:
        print("   online_classes collection already exists")

    # Create indexes for online_classes
    indexes_to_create = [
        {"keys": [("tutor_id", 1), ("status", 1)], "name": "tutor_status"},
        {"keys": [("meet_code", 1)], "name": "meet_code_unique", "unique": True, "sparse": True},
        {"keys": [("standard", 1), ("section", 1), ("subject", 1)], "name": "class_filter"},
        {"keys": [("status", 1), ("scheduled_at", -1)], "name": "active_sessions"},
        {"keys": [("enrolled_students", 1)], "name": "enrolled_students_lookup"},
    ]

    for idx in indexes_to_create:
        try:
            await db.online_classes.create_index(
                idx["keys"],
                name=idx["name"],
                unique=idx.get("unique", False),
                sparse=idx.get("sparse", False)
            )
            print(f"   Created index: {idx['name']}")
        except Exception as e:
            print(f"   Index {idx['name']} may already exist: {e}")

    print()

    # =========================================================================
    # Step 3: Create smartboard_sessions collection
    # =========================================================================
    print("3. Creating smartboard_sessions collection...")

    if "smartboard_sessions" not in collections:
        await db.create_collection("smartboard_sessions")
        print("   Created smartboard_sessions collection")
    else:
        print("   smartboard_sessions collection already exists")

    # Create indexes for smartboard_sessions
    sb_indexes = [
        {"keys": [("tutor_id", 1), ("status", 1)], "name": "tutor_session_status"},
        {"keys": [("session_id", 1)], "name": "session_id_unique", "unique": True},
        {"keys": [("started_at", -1)], "name": "recent_sessions"},
    ]

    for idx in sb_indexes:
        try:
            await db.smartboard_sessions.create_index(
                idx["keys"],
                name=idx["name"],
                unique=idx.get("unique", False)
            )
            print(f"   Created index: {idx['name']}")
        except Exception as e:
            print(f"   Index {idx['name']} may already exist: {e}")

    print()

    # =========================================================================
    # Step 4: Summary
    # =========================================================================
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)

    # Count documents in new collections
    online_classes_count = await db.online_classes.count_documents({})
    smartboard_sessions_count = await db.smartboard_sessions.count_documents({})
    students_with_mappings = await db.students.count_documents({
        "class_mappings": {"$exists": True, "$ne": []}
    })
    total_students = await db.students.count_documents({})

    print(f"Total students: {total_students}")
    print(f"Students with class_mappings: {students_with_mappings}")
    print(f"Online classes: {online_classes_count}")
    print(f"SmartBoard sessions: {smartboard_sessions_count}")

    print()
    print("Migration completed successfully!")
    print()
    print("Next steps:")
    print("1. Map students to classes using the admin interface")
    print("2. Start the backend and test the /api/v1/classroom endpoints")

    client.close()


async def rollback():
    """Rollback migration (for testing)"""
    print("Rolling back migration...")

    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    # Remove class_mappings field from students
    await db.students.update_many(
        {},
        {"$unset": {"class_mappings": ""}}
    )
    print("   Removed class_mappings from students")

    # Drop collections
    await db.online_classes.drop()
    print("   Dropped online_classes collection")

    await db.smartboard_sessions.drop()
    print("   Dropped smartboard_sessions collection")

    client.close()
    print("Rollback completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rollback":
        asyncio.run(rollback())
    else:
        try:
            asyncio.run(migrate())
        except Exception as e:
            print(f"Migration failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
