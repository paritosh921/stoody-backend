"""
Direct MongoDB script to create admin account
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

# Check if admin exists
existing = db.admins.find_one({"email": "admin@skillbot.app"})

if existing:
    print("[OK] Admin already exists")
    print(f"   ID: {existing['_id']}")
    print(f"   Email: {existing['email']}")
    print(f"   Name: {existing.get('full_name', 'N/A')}")
else:
    # Create admin
    admin_data = {
        "email": "admin@skillbot.app",
        "password_hash": pwd_context.hash("admin123"),
        "full_name": "System Administrator",
        "is_active": True,
        "created_at": datetime.utcnow()
    }

    result = db.admins.insert_one(admin_data)
    print("[OK] Admin created successfully")
    print(f"   ID: {result.inserted_id}")
    print(f"   Email: admin@skillbot.app")
    print(f"   Password: admin123")

client.close()
