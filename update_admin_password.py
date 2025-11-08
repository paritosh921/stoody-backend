"""
Update admin password directly in MongoDB
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime

# Load environment
load_dotenv()

# Password hasher - matching backend config
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
db = client["skillbot_db"]

# Find admin
admin = db.admins.find_one({"email": "admin@skillbot.app"})

if admin:
    print(f"[OK] Found admin: {admin['email']}")

    # Hash new password
    new_password_hash = pwd_context.hash("admin123")

    # Update admin
    result = db.admins.update_one(
        {"email": "admin@skillbot.app"},
        {
            "$set": {
                "password_hash": new_password_hash,
                "full_name": "System Administrator",
                "is_active": True,
                "updated_at": datetime.utcnow()
            }
        }
    )

    print(f"[OK] Admin password updated successfully")
    print(f"   Email: admin@skillbot.app")
    print(f"   Password: admin123")
    print(f"   Modified count: {result.modified_count}")

    # Verify the hash works
    admin_after = db.admins.find_one({"email": "admin@skillbot.app"})
    if admin_after and pwd_context.verify("admin123", admin_after["password_hash"]):
        print("[OK] Password verification successful!")
    else:
        print("[ERROR] Password verification failed!")
else:
    print("[ERROR] Admin not found!")

client.close()
