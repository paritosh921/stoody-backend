"""
Direct MongoDB script to create admin account in a specific tenant database.

Usage:
    python init_admin_direct.py --db-name skb_XXXX-0000 --email admin@example.com --password <password>
"""
import argparse
import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def main():
    parser = argparse.ArgumentParser(description="Create admin account in tenant database")
    parser.add_argument("--db-name", required=True, help="Target database name (e.g. skb_XXXX-0000)")
    parser.add_argument("--email", required=True, help="Admin email address")
    parser.add_argument("--password", required=True, help="Admin password")
    parser.add_argument("--full-name", default="System Administrator", help="Admin full name")
    args = parser.parse_args()

    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        print("[ERROR] MONGODB_URI not set in environment")
        sys.exit(1)

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
    db = client[args.db_name]

    try:
        existing = db.admins.find_one({"email": args.email})
        if existing:
            print(f"[OK] Admin already exists in {args.db_name}")
            print(f"   ID: {existing['_id']}")
            print(f"   Email: {existing['email']}")
            print(f"   Name: {existing.get('full_name', 'N/A')}")
        else:
            admin_data = {
                "email": args.email,
                "password_hash": pwd_context.hash(args.password),
                "full_name": args.full_name,
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            result = db.admins.insert_one(admin_data)
            print(f"[OK] Admin created successfully in {args.db_name}")
            print(f"   ID: {result.inserted_id}")
            print(f"   Email: {args.email}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
