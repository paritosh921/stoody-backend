import secrets
import string
import bcrypt
from typing import Dict, Any
from core.database import DatabaseManager

def generate_secure_password(length: int = 12) -> str:
    """Generate a cryptographically secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    # Ensure password has at least one digit and one special char
    if not any(c.isdigit() for c in password):
        password = password[:-1] + secrets.choice(string.digits)
    if not any(c in "!@#$%^&*" for c in password):
        password = password[:-1] + secrets.choice("!@#$%^&*")
    return password

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    # Bcrypt has a 72 byte limit, truncate if necessary
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')

def is_b2c_admin(current_user: Dict[str, Any]) -> bool:
    """Check if the current user is a B2C admin"""
    return current_user.get("user_type") == "b2c_admin"

async def db_find_one(db: DatabaseManager, collection: str, query: dict, current_user: Dict[str, Any], **kwargs):
    """Route find_one to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_find_one(collection, query, **kwargs)
    return await db.mongo_find_one(collection, query, **kwargs)

async def db_find(db: DatabaseManager, collection: str, query: dict, current_user: Dict[str, Any], **kwargs):
    """Route find to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_find(collection, query, **kwargs)
    return await db.mongo_find(collection, query, **kwargs)

async def db_insert_one(db: DatabaseManager, collection: str, document: dict, current_user: Dict[str, Any]):
    """Route insert_one to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_insert_one(collection, document)
    return await db.mongo_insert_one(collection, document)

async def db_update_one(db: DatabaseManager, collection: str, query: dict, update: dict, current_user: Dict[str, Any], **kwargs):
    """Route update_one to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_update_one(collection, query, update, **kwargs)
    return await db.mongo_update_one(collection, query, update, **kwargs)

async def db_delete_one(db: DatabaseManager, collection: str, query: dict, current_user: Dict[str, Any]):
    """Route delete_one to B2C or regular database based on user type"""
    if is_b2c_admin(current_user):
        return await db.b2c_delete_one(collection, query)
    return await db.mongo_delete_one(collection, query)
