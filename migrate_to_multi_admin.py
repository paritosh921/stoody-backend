"""
Migration script to assign admin_id to existing students and migrate questions to multi-admin setup
"""
import asyncio
from datetime import datetime

async def migrate_students_to_multi_admin():
    """Assign admin_id to existing students"""
    try:
        from core.database import DatabaseManager
        db = DatabaseManager()
        await db.initialize()

        # Get default admin (the existing admin@skillbot.app)
        default_admin = await db.mongo_find_one("admins", {"email": "admin@skillbot.app"})
        if not default_admin:
            print("❌ Default admin not found")
            return

        admin_id = default_admin["_id"]
        print(f"📋 Found default admin: {admin_id}")

        # Get MongoDB collection directly for operations not in DatabaseManager
        mongo_db = await db.get_mongo_db()
        if mongo_db is None:
            print("❌ MongoDB not available")
            return

        students_collection = mongo_db["students"]
        attempts_collection = mongo_db["question_attempts"]
        activity_collection = mongo_db["student_activity_log"]

        # Update all students without admin_id
        student_result = await students_collection.update_many(
            {"admin_id": {"$exists": False}},  # Students without admin_id
            {"$set": {"admin_id": admin_id}}
        )
        print(f"✅ Migrated {student_result.modified_count} students to default admin")

        # Update question attempts
        attempts_result = await attempts_collection.update_many(
            {"admin_id": {"$exists": False}},
            {"$set": {"admin_id": admin_id}}
        )
        print(f"✅ Migrated {attempts_result.modified_count} question attempts to default admin")

        # Update student activity logs
        activity_result = await activity_collection.update_many(
            {"admin_id": {"$exists": False}},
            {"$set": {"admin_id": admin_id}}
        )
        print(f"✅ Migrated {activity_result.modified_count} activity logs to default admin")

        await db.close()
        print("🎉 Student migration completed successfully!")

    except Exception as e:
        print(f"❌ Student migration failed: {str(e)}")

async def migrate_questions_to_admin_collection():
    """Migrate existing questions to default admin collection"""
    try:
        from core.database import DatabaseManager
        from models.chromadb_client import ChromaDBClient
        from services.question_service import QuestionService
        from models.question import Question

        db = DatabaseManager()
        await db.initialize()

        # Get default admin
        default_admin = await db.mongo_find_one("admins", {"email": "admin@skillbot.app"})
        if not default_admin:
            print("❌ Default admin not found")
            return

        admin_id = str(default_admin["_id"])
        print(f"📋 Migrating questions to admin: {admin_id}")

        # Get existing questions from old collection (no admin_id)
        old_client = ChromaDBClient()  # Uses default collection
        existing_questions = old_client.collection.get(include=["documents", "metadatas"])

        if not existing_questions['ids']:
            print("ℹ️ No existing questions to migrate")
            await db.close()
            return

        print(f"📋 Found {len(existing_questions['ids'])} questions to migrate")

        # Migrate to admin-specific collection
        question_service = QuestionService(admin_id)

        migrated_count = 0
        for i, question_id in enumerate(existing_questions['ids']):
            try:
                # Reconstruct question from ChromaDB data
                question = Question.from_chromadb_result(
                    document=existing_questions['documents'][i],
                    metadata=existing_questions['metadatas'][i],
                    id=question_id
                )

                # Save to admin-specific collection
                success, _, error = question_service.save_question(question.to_dict())
                if success:
                    migrated_count += 1
                    print(f"✅ Migrated question {question_id}")
                else:
                    print(f"❌ Failed to migrate question {question_id}: {error}")

            except Exception as e:
                print(f"❌ Error migrating question {question_id}: {str(e)}")

        await db.close()
        print(f"🎉 Migrated {migrated_count}/{len(existing_questions['ids'])} questions to admin collection")

    except Exception as e:
        print(f"❌ Question migration failed: {str(e)}")

async def verify_migration():
    """Verify that migration was successful"""
    try:
        from core.database import DatabaseManager
        db = DatabaseManager()
        await db.initialize()

        # Get MongoDB collections directly
        mongo_db = await db.get_mongo_db()
        if mongo_db is None:
            print("❌ MongoDB not available for verification")
            return

        students_collection = mongo_db["students"]
        attempts_collection = mongo_db["question_attempts"]
        activity_collection = mongo_db["student_activity_log"]

        # Check students
        students_with_admin = await students_collection.count_documents({"admin_id": {"$exists": True}})
        total_students = await students_collection.count_documents({})
        print(f"📊 Students: {students_with_admin}/{total_students} have admin_id")

        # Check question attempts
        attempts_with_admin = await attempts_collection.count_documents({"admin_id": {"$exists": True}})
        total_attempts = await attempts_collection.count_documents({})
        print(f"📊 Question attempts: {attempts_with_admin}/{total_attempts} have admin_id")

        # Check activity logs
        logs_with_admin = await activity_collection.count_documents({"admin_id": {"$exists": True}})
        total_logs = await activity_collection.count_documents({})
        print(f"📊 Activity logs: {logs_with_admin}/{total_logs} have admin_id")

        # Check admin-specific ChromaDB collections
        from models.chromadb_client import ChromaDBClient
        default_admin = await db.mongo_find_one("admins", {"email": "admin@skillbot.app"})
        if default_admin:
            admin_id = str(default_admin["_id"])
            admin_client = ChromaDBClient(admin_id)
            stats = admin_client.get_collection_stats()
            print(f"📊 Questions in admin collection: {stats.get('total_questions', 0)}")

        await db.close()

    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")

async def main():
    """Run all migrations"""
    print("🚀 Starting multi-admin migration...")

    print("\n1️⃣ Migrating students and related data...")
    await migrate_students_to_multi_admin()

    print("\n2️⃣ Migrating questions to admin-specific collections...")
    await migrate_questions_to_admin_collection()

    print("\n3️⃣ Verifying migration...")
    await verify_migration()

    print("\n✅ All migrations completed!")

if __name__ == "__main__":
    asyncio.run(main())
