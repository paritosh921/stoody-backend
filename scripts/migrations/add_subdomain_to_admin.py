"""
Add subdomain field to default admin and create indexes
Run once: python -m scripts.add_subdomain_to_admin
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Get MongoDB URL from environment or use default
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "skillbot"

async def migrate():
    print("🔄 Starting migration...")
    print(f"📦 MongoDB URL: {MONGODB_URL}")
    print(f"📊 Database: {DATABASE_NAME}")
    print()

    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    # Step 1: Find and update default admin
    print("1️⃣ Updating default admin...")
    admin = await db.admins.find_one({"email": "admin@skillbot.app"})

    if admin:
        # Add subdomain if not exists
        if not admin.get("subdomain"):
            result = await db.admins.update_one(
                {"_id": admin["_id"]},
                {
                    "$set": {
                        "subdomain": "demo",
                        "organization": "Demo School"
                    }
                }
            )
            print(f"   ✅ Added subdomain 'demo' to admin: {admin['email']}")
            print(f"      Matched: {result.matched_count}, Modified: {result.modified_count}")
        else:
            print(f"   ℹ️  Admin already has subdomain: {admin.get('subdomain')}")
    else:
        print("   ❌ Default admin not found. Creating one...")
        from models.admin import Admin
        admin = Admin.create_default_admin()
        # Update with subdomain
        await db.admins.update_one(
            {"email": "admin@skillbot.app"},
            {
                "$set": {
                    "subdomain": "demo",
                    "organization": "Demo School"
                }
            }
        )
        print("   ✅ Created default admin with subdomain 'demo'")

    print()

    # Step 2: Create unique index on admins.subdomain
    print("2️⃣ Creating index on admins.subdomain...")
    try:
        await db.admins.create_index("subdomain", unique=True, sparse=True)
        print("   ✅ Created unique index on admins.subdomain")
    except Exception as e:
        print(f"   ⚠️  Index may already exist: {e}")

    print()

    # Step 3: Create compound index on students (admin_id + username)
    print("3️⃣ Creating compound index on students...")
    try:
        await db.students.create_index([("admin_id", 1), ("username", 1)], unique=True, sparse=True)
        print("   ✅ Created compound index on students (admin_id, username)")
    except Exception as e:
        print(f"   ⚠️  Index may already exist: {e}")

    print()

    # Step 4: Update existing students to belong to default admin
    print("4️⃣ Updating existing students...")
    admin = await db.admins.find_one({"subdomain": "demo"})
    if admin:
        admin_id = admin["_id"]

        # Find students without admin_id
        students_without_admin = await db.students.count_documents({"admin_id": {"$exists": False}})

        if students_without_admin > 0:
            result = await db.students.update_many(
                {"admin_id": {"$exists": False}},
                {"$set": {"admin_id": admin_id}}
            )
            print(f"   ✅ Updated {result.modified_count} students with admin_id")
        else:
            print("   ℹ️  All students already have admin_id")
    else:
        print("   ❌ Could not find admin with subdomain 'demo'")

    print()

    # Step 5: Summary
    print("📊 Migration Summary:")
    total_admins = await db.admins.count_documents({})
    total_students = await db.students.count_documents({})
    students_with_admin = await db.students.count_documents({"admin_id": {"$exists": True}})

    print(f"   Total admins: {total_admins}")
    print(f"   Total students: {total_students}")
    print(f"   Students with admin_id: {students_with_admin}")

    print()
    print("✅ Migration completed successfully!")

    client.close()

if __name__ == "__main__":
    try:
        asyncio.run(migrate())
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
