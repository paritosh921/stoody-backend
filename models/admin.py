"""
Admin Model for SkillBot Backend
Handles admin user data and authentication
"""

from datetime import datetime
from typing import Optional, Dict, Any
import bcrypt
from bson import ObjectId
from marshmallow import Schema, fields, validate, post_load
from .mongodb_client import get_collection

class Admin:
    """Admin user model for authentication and management"""

    def __init__(self, email: str, password_hash: str, name: str,
                 role: str = "admin", is_active: bool = True,
                 created_at: datetime = None, last_login: datetime = None,
                 google_id: str = None, subdomain: str = None,
                 organization: str = None,
                 _id: ObjectId = None):
        self._id = _id
        self.email = email
        self.password_hash = password_hash
        self.name = name
        self.role = role
        self.is_active = is_active
        self.created_at = created_at or datetime.utcnow()
        self.last_login = last_login
        self.google_id = google_id
        self.subdomain = subdomain
        self.organization = organization

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def to_dict(self, include_password: bool = False) -> Dict[str, Any]:
        """Convert admin object to dictionary"""
        data = {
            "_id": str(self._id) if self._id else None,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "google_id": self.google_id,
            "subdomain": self.subdomain,
            "organization": self.organization
        }

        if include_password:
            data["password_hash"] = self.password_hash

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Admin':
        """Create Admin object from dictionary"""
        return cls(
            _id=data.get('_id'),
            email=data['email'],
            password_hash=data.get('password_hash', ''),
            name=data['name'],
            role=data.get('role', 'admin'),
            is_active=data.get('is_active', True),
            created_at=data.get('created_at'),
            last_login=data.get('last_login'),
            google_id=data.get('google_id'),
            subdomain=data.get('subdomain'),
            organization=data.get('organization')
        )

    def save(self) -> ObjectId:
        """Save admin to database"""
        collection = get_collection('admins')

        admin_data = {
            "email": self.email,
            "password_hash": self.password_hash,
            "name": self.name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "google_id": self.google_id,
            "subdomain": self.subdomain,
            "organization": self.organization
        }

        if self._id:
            # Update existing admin
            collection.update_one(
                {"_id": self._id},
                {"$set": admin_data}
            )
            return self._id
        else:
            # Create new admin
            result = collection.insert_one(admin_data)
            self._id = result.inserted_id
            return self._id

    @classmethod
    def find_by_email(cls, email: str) -> Optional['Admin']:
        """Find admin by email"""
        collection = get_collection('admins')
        admin_data = collection.find_one({"email": email})

        if admin_data:
            return cls.from_dict(admin_data)
        return None

    @classmethod
    def find_by_id(cls, admin_id: str) -> Optional['Admin']:
        """Find admin by ID"""
        collection = get_collection('admins')
        try:
            admin_data = collection.find_one({"_id": ObjectId(admin_id)})
            if admin_data:
                return cls.from_dict(admin_data)
        except Exception:
            pass
        return None

    @classmethod
    def create_default_admin(cls) -> 'Admin':
        """Create default admin account"""
        email = "admin@skillbot.app"
        password = "admin123"
        name = "System Administrator"

        # Check if admin already exists
        existing_admin = cls.find_by_email(email)
        if existing_admin:
            return existing_admin

        # Create new admin
        admin = cls(
            email=email,
            password_hash=cls.hash_password(password),
            name=name,
            role="super_admin"
        )
        admin.save()
        return admin

    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        collection = get_collection('admins')
        collection.update_one(
            {"_id": self._id},
            {"$set": {"last_login": self.last_login}}
        )

class AdminSchema(Schema):
    """Marshmallow schema for Admin validation"""

    email = fields.Email(required=True, validate=validate.Length(max=255))
    password = fields.Str(required=True, validate=validate.Length(min=6, max=255), load_only=True)
    name = fields.Str(required=True, validate=validate.Length(min=2, max=255))
    role = fields.Str(validate=validate.OneOf(['admin', 'super_admin']), missing='admin')
    is_active = fields.Bool(missing=True)

    @post_load
    def make_admin(self, data, **kwargs):
        """Create Admin object from validated data"""
        password = data.pop('password', None)
        if password:
            data['password_hash'] = Admin.hash_password(password)
        return Admin(**data)

class AdminLoginSchema(Schema):
    """Schema for admin login validation"""

    email = fields.Email(required=True)
    password = fields.Str(required=True)