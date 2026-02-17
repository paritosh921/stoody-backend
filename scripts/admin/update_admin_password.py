"""
Update admin password directly in MongoDB.

Usage:
    python update_admin_password.py --db-name skb_XXXX-0000 --email admin@example.com --password <new_password>
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
    parser = argparse.ArgumentParser(description="Update admin password in tenant database")
    parser.add_argument("--db-name", required=True, help="Target database name (e.g. skb_XXXX-0000)")
    parser.add_argument("--email", required=True, help="Admin email address")
    parser.add_argument("--password", required=True, help="New admin password")
    args = parser.parse_args()

    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        print("[ERROR] MONGODB_URI not set in environment")
        sys.exit(1)

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
    db = client[args.db_name]

    try:
        admin = db.admins.find_one({"email": args.email})

        if admin:
            print(f"[OK] Found admin: {admin['email']} in {args.db_name}")

            new_password_hash = pwd_context.hash(args.password)

            result = db.admins.update_one(
                {"email": args.email},
                {
                    "$set": {
                        "password_hash": new_password_hash,
                        "is_active": True,
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            print(f"[OK] Admin password updated successfully")
            print(f"   Email: {args.email}")
            print(f"   Modified count: {result.modified_count}")

            # Verify the hash works
            admin_after = db.admins.find_one({"email": args.email})
            if admin_after and pwd_context.verify(args.password, admin_after["password_hash"]):
                print("[OK] Password verification successful!")
            else:
                print("[ERROR] Password verification failed!")
        else:
            print(f"[ERROR] Admin not found: {args.email} in {args.db_name}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
