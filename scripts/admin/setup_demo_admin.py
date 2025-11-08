"""
Setup Demo Admin Account and Associate Existing Data
This script creates a demo admin with subdomain 'demo' and associates all existing students and content.
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime

# Load environment
load_dotenv()

# Password hasher
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
db = client["skillbot_db"]

def setup_demo_admin():
    """Create demo admin and associate existing data"""

    print("=" * 60)
    print("DEMO ADMIN SETUP")
    print("=" * 60)

    try:
        # Check if demo admin already exists
        existing_demo = db.admins.find_one({"subdomain": "demo"})

        if existing_demo:
            demo_admin_id = existing_demo["_id"]
            print(f"\nDemo admin already exists with ID: {demo_admin_id}")
        else:
            # Create demo admin
            password_hash = pwd_context.hash("admin123")

            demo_admin = {
                "email": "admin@skillbot.app",
                "password_hash": password_hash,
                "name": "Demo Administrator",
                "subdomain": "demo",
                "organization": "SkillBot Demo School",
                "role": "admin",
                "is_active": True,
                "created_at": datetime.utcnow(),
                "google_id": None
            }

            result = db.admins.insert_one(demo_admin)
            demo_admin_id = result.inserted_id
            print(f"\nCreated demo admin with ID: {demo_admin_id}")
            print(f"  Email: admin@skillbot.app")
            print(f"  Password: admin123")
            print(f"  Subdomain: demo")
            print(f"  Organization: SkillBot Demo School")

        # Update all students without admin_id
        students_without_admin = db.students.count_documents({
            "$or": [
                {"admin_id": {"$exists": False}},
                {"admin_id": None}
            ]
        })

        if students_without_admin > 0:
            result = db.students.update_many(
                {
                    "$or": [
                        {"admin_id": {"$exists": False}},
                        {"admin_id": None}
                    ]
                },
                {"$set": {"admin_id": demo_admin_id}}
            )
            print(f"\nAssociated {result.modified_count} students with demo admin")
        else:
            print(f"\nAll students already have admin_id assigned")

        # Update all documents without admin_id
        documents_without_admin = db.documents.count_documents({
            "$or": [
                {"admin_id": {"$exists": False}},
                {"admin_id": None}
            ]
        })

        if documents_without_admin > 0:
            result = db.documents.update_many(
                {
                    "$or": [
                        {"admin_id": {"$exists": False}},
                        {"admin_id": None}
                    ]
                },
                {"$set": {"admin_id": demo_admin_id}}
            )
            print(f"Associated {result.modified_count} documents with demo admin")
        else:
            print(f"All documents already have admin_id assigned")

        # Update all questions without admin_id
        questions_without_admin = db.questions.count_documents({
            "$or": [
                {"admin_id": {"$exists": False}},
                {"admin_id": None}
            ]
        })

        if questions_without_admin > 0:
            result = db.questions.update_many(
                {
                    "$or": [
                        {"admin_id": {"$exists": False}},
                        {"admin_id": None}
                    ]
                },
                {"$set": {"admin_id": demo_admin_id}}
            )
            print(f"Associated {result.modified_count} questions with demo admin")
        else:
            print(f"All questions already have admin_id assigned")

        # Count final stats
        total_students = db.students.count_documents({"admin_id": demo_admin_id})
        total_documents = db.documents.count_documents({"admin_id": demo_admin_id})
        total_questions = db.questions.count_documents({"admin_id": demo_admin_id})

        print("\n" + "=" * 60)
        print("DEMO ADMIN STATISTICS")
        print("=" * 60)
        print(f"Admin ID: {demo_admin_id}")
        print(f"Total Students: {total_students}")
        print(f"Total Documents: {total_documents}")
        print(f"Total Questions: {total_questions}")
        print("\n" + "=" * 60)
        print("NEXT STEP: Update bypass login")
        print("=" * 60)
        print(f"\nCopy this admin_id to update auth_bypass.py:")
        print(f"{demo_admin_id}")
        print("\nSetup complete!")

    except Exception as e:
        print(f"\nError during setup: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    setup_demo_admin()
