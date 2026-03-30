"""
Create or update a super-admin account in the master database.

Features:
- Stores password as bcrypt hash
- Enforces unique email and authorization_code
- Enables first-login password reset flow (requires_password_change=True)
- Initializes 2FA policy as required but not enabled
"""

import argparse
import os
import secrets
import string
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from passlib.context import CryptContext
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

AUTH_CODE_ALPHABET = string.ascii_uppercase + string.digits
PASSWORD_ALPHABET = string.ascii_letters + string.digits + "!@#$%^&*"
AUTH_CODE_LENGTH = 6
DEFAULT_PASSWORD_LENGTH = 14

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def normalize_email(email: str) -> str:
    return email.strip().lower()


def normalize_auth_code(code: str) -> str:
    normalized = code.strip().upper()
    if len(normalized) != AUTH_CODE_LENGTH:
        raise ValueError("Authorization code must be exactly 6 characters.")
    if any(ch not in AUTH_CODE_ALPHABET for ch in normalized):
        raise ValueError("Authorization code must be uppercase alphanumeric.")
    return normalized


def generate_temp_password(length: int = DEFAULT_PASSWORD_LENGTH) -> str:
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(length))


def generate_auth_code() -> str:
    return "".join(secrets.choice(AUTH_CODE_ALPHABET) for _ in range(AUTH_CODE_LENGTH))


def ensure_indexes(collection) -> None:
    collection.create_index([("email", 1)], unique=True, name="uniq_superadmins_email")
    collection.create_index([("authorization_code", 1)], unique=True, name="uniq_superadmins_auth_code")


def resolve_unique_auth_code(collection, requested_code: Optional[str]) -> str:
    if requested_code:
        code = normalize_auth_code(requested_code)
        existing = collection.find_one({"authorization_code": code})
        if existing:
            raise ValueError(f"Authorization code already in use by {existing.get('email')}")
        return code

    for _ in range(20):
        code = generate_auth_code()
        if not collection.find_one({"authorization_code": code}):
            return code
    raise RuntimeError("Unable to generate unique authorization code after multiple attempts.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a super-admin user in MongoDB master DB.")
    parser.add_argument("--email", required=True, help="Super-admin email")
    parser.add_argument("--name", required=True, help="Super-admin full name")
    parser.add_argument("--password", default=None, help="Initial password. If omitted, a random one is generated.")
    parser.add_argument("--authorization-code", default=None, help="6-char uppercase alphanumeric code")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update existing super-admin if email already exists",
    )
    parser.add_argument("--mongo-uri", default=None, help="MongoDB URI (defaults to MONGODB_URI env)")
    parser.add_argument("--master-db", default=None, help="Master DB name (defaults to MONGODB_DB_MASTER env)")
    args = parser.parse_args()

    load_dotenv()
    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI")
    master_db_name = args.master_db or os.getenv("MONGODB_DB_MASTER", "skb_master")

    if not mongo_uri:
        print("ERROR: MONGODB_URI is not set.")
        return 1

    email = normalize_email(args.email)
    if "@" not in email:
        print("ERROR: Invalid email format.")
        return 1

    initial_password = args.password or generate_temp_password()

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
    try:
        master_db = client[master_db_name]
        super_admins = master_db["super_admins"]
        ensure_indexes(super_admins)

        existing = super_admins.find_one({"email": email})
        now = datetime.utcnow()

        if existing and not args.update_existing:
            print(f"ERROR: Super-admin already exists for {email}. Use --update-existing to modify it.")
            return 1

        if existing:
            if args.authorization_code:
                auth_code = resolve_unique_auth_code(super_admins, args.authorization_code)
            else:
                existing_code = existing.get("authorization_code")
                if existing_code:
                    auth_code = normalize_auth_code(existing_code)
                else:
                    auth_code = resolve_unique_auth_code(super_admins, None)
            update_doc = {
                "name": args.name.strip(),
                "password_hash": pwd_context.hash(initial_password),
                "authorization_code": auth_code,
                "requires_password_change": True,
                "password_reset_requested": False,
                "password_changed_at": None,
                "is_active": True,
                "updated_at": now,
                "two_fa": {
                    "enabled": False,
                    "required": True,
                    "secret_enc": None,
                    "temp_secret_enc": None,
                    "verified_at": None,
                    "last_verified_at": None,
                },
            }
            super_admins.update_one({"_id": existing["_id"]}, {"$set": update_doc})
            admin_id = str(existing["_id"])
            action = "updated"
        else:
            auth_code = resolve_unique_auth_code(super_admins, args.authorization_code)
            insert_doc = {
                "email": email,
                "name": args.name.strip(),
                "password_hash": pwd_context.hash(initial_password),
                "role": "super_admin",
                "permissions": ["all"],
                "is_active": True,
                "authorization_code": auth_code,
                "requires_password_change": True,
                "password_reset_requested": False,
                "password_changed_at": None,
                "created_at": now,
                "updated_at": now,
                "two_fa": {
                    "enabled": False,
                    "required": True,
                    "secret_enc": None,
                    "temp_secret_enc": None,
                    "verified_at": None,
                    "last_verified_at": None,
                },
            }
            result = super_admins.insert_one(insert_doc)
            admin_id = str(result.inserted_id)
            action = "created"

        print("SUCCESS: super-admin account provisioned")
        print(f"- action: {action}")
        print(f"- admin_id: {admin_id}")
        print(f"- email: {email}")
        print(f"- name: {args.name.strip()}")
        print(f"- authorization_code: {auth_code}")
        print(f"- temporary_password: {initial_password}")
        print("- requires_password_change: true")
        print("- two_fa_required: true")
        return 0
    except DuplicateKeyError as exc:
        print(f"ERROR: Duplicate key conflict: {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
