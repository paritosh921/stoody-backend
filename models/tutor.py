"""
Tutor Model for Stoody Backend
Handles tutor user data and authentication
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import random
import string
from bson import ObjectId
from marshmallow import Schema, fields, validate, post_load
from .mongodb_client import get_collection

class Tutor:
    """Tutor user model for authentication and student management"""

    def __init__(self, tutor_id: str, name: str, username: str,
                 password_hash: str, email: str = None, phone: str = None,
                 standards: List[str] = None, sections: List[str] = None,
                 subjects: List[str] = None, plan_types: List[str] = None,
                 can_edit_students: bool = False, is_active: bool = True,
                 assigned_student_ids: List[str] = None,
                 requires_password_change: bool = True,
                 password_reset_requested: bool = False,
                 created_by: str = None, created_at: datetime = None,
                 last_login: datetime = None, _id: ObjectId = None):
        self._id = _id
        self.tutor_id = tutor_id
        self.name = name
        self.username = username
        self.password_hash = password_hash
        self.email = email
        self.phone = phone
        self.standards = standards or []  # Multiple standards (e.g., ["9", "10", "11"])
        self.sections = sections or []  # Multiple sections (e.g., ["A", "B", "C"])
        self.subjects = subjects or []  # Multiple subjects
        self.plan_types = plan_types or []  # Multiple plan types (e.g., ["CBSE", "JEE"])
        self.can_edit_students = can_edit_students  # Permission to add/edit students
        self.is_active = is_active
        self.assigned_student_ids = assigned_student_ids or []  # Students assigned to this tutor
        self.requires_password_change = requires_password_change
        self.password_reset_requested = password_reset_requested
        self.created_by = created_by  # Admin ID who created this tutor
        self.created_at = created_at or datetime.utcnow()
        self.last_login = last_login

    @staticmethod
    def generate_tutor_id(prefix: str = "TUT") -> str:
        """Generate a unique tutor ID"""
        timestamp = datetime.now().strftime("%y%m")
        random_part = ''.join(random.choices(string.digits, k=4))
        return f"{prefix}{timestamp}{random_part}"

    @staticmethod
    def generate_password(length: int = 8) -> str:
        """Generate a random password"""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=length))

    def to_dict(self) -> Dict[str, Any]:
        """Convert tutor to dictionary for API response"""
        return {
            "id": str(self._id) if self._id else None,
            "tutor_id": self.tutor_id,
            "name": self.name,
            "username": self.username,
            "email": self.email,
            "phone": self.phone,
            "standards": self.standards,
            "sections": self.sections,
            "subjects": self.subjects,
            "plan_types": self.plan_types,
            "can_edit_students": self.can_edit_students,
            "is_active": self.is_active,
            "assigned_student_ids": self.assigned_student_ids,
            "requires_password_change": self.requires_password_change,
            "password_reset_requested": self.password_reset_requested,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Tutor':
        """Create Tutor instance from dictionary"""
        return Tutor(
            tutor_id=data.get('tutor_id'),
            name=data.get('name'),
            username=data.get('username'),
            password_hash=data.get('password_hash'),
            email=data.get('email'),
            phone=data.get('phone'),
            standards=data.get('standards', []),
            sections=data.get('sections', []),
            subjects=data.get('subjects', []),
            plan_types=data.get('plan_types', []),
            can_edit_students=data.get('can_edit_students', False),
            is_active=data.get('is_active', True),
            assigned_student_ids=data.get('assigned_student_ids', []),
            requires_password_change=data.get('requires_password_change', True),
            password_reset_requested=data.get('password_reset_requested', False),
            created_by=data.get('created_by'),
            created_at=data.get('created_at'),
            last_login=data.get('last_login'),
            _id=data.get('_id')
        )

    async def save(self):
        """Save tutor to MongoDB"""
        collection = await get_collection("tutors")
        tutor_dict = {
            "tutor_id": self.tutor_id,
            "name": self.name,
            "username": self.username,
            "password_hash": self.password_hash,
            "email": self.email,
            "phone": self.phone,
            "standards": self.standards,
            "sections": self.sections,
            "subjects": self.subjects,
            "plan_types": self.plan_types,
            "can_edit_students": self.can_edit_students,
            "is_active": self.is_active,
            "assigned_student_ids": self.assigned_student_ids,
            "requires_password_change": self.requires_password_change,
            "password_reset_requested": self.password_reset_requested,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "last_login": self.last_login,
        }

        if self._id:
            # Update existing tutor
            await collection.update_one(
                {"_id": self._id},
                {"$set": tutor_dict}
            )
        else:
            # Insert new tutor
            result = await collection.insert_one(tutor_dict)
            self._id = result.inserted_id

        return self._id

    @staticmethod
    async def find_by_username(username: str) -> Optional['Tutor']:
        """Find tutor by username"""
        collection = await get_collection("tutors")
        tutor_data = await collection.find_one({"username": username})
        if tutor_data:
            return Tutor.from_dict(tutor_data)
        return None

    @staticmethod
    async def find_by_tutor_id(tutor_id: str) -> Optional['Tutor']:
        """Find tutor by tutor_id"""
        collection = await get_collection("tutors")
        tutor_data = await collection.find_one({"tutor_id": tutor_id})
        if tutor_data:
            return Tutor.from_dict(tutor_data)
        return None

    @staticmethod
    async def find_by_id(tutor_id: ObjectId) -> Optional['Tutor']:
        """Find tutor by MongoDB _id"""
        collection = await get_collection("tutors")
        tutor_data = await collection.find_one({"_id": tutor_id})
        if tutor_data:
            return Tutor.from_dict(tutor_data)
        return None

    @staticmethod
    async def find_all(admin_id: str = None) -> List['Tutor']:
        """Find all tutors, optionally filtered by admin who created them"""
        collection = await get_collection("tutors")
        query = {}
        if admin_id:
            query["created_by"] = admin_id

        tutors_data = await collection.find(query).to_list(length=None)
        return [Tutor.from_dict(data) for data in tutors_data]

    async def update_last_login(self):
        """Update the last login timestamp"""
        self.last_login = datetime.utcnow()
        collection = await get_collection("tutors")
        await collection.update_one(
            {"_id": self._id},
            {"$set": {"last_login": self.last_login}}
        )

    async def update_password(self, new_password_hash: str, reset_required: bool = False):
        """Update tutor password"""
        self.password_hash = new_password_hash
        self.requires_password_change = reset_required
        self.password_reset_requested = False
        collection = await get_collection("tutors")
        await collection.update_one(
            {"_id": self._id},
            {"$set": {
                "password_hash": self.password_hash,
                "requires_password_change": self.requires_password_change,
                "password_reset_requested": self.password_reset_requested
            }}
        )

    async def request_password_reset(self):
        """Mark tutor as requesting password reset"""
        self.password_reset_requested = True
        collection = await get_collection("tutors")
        await collection.update_one(
            {"_id": self._id},
            {"$set": {"password_reset_requested": True}}
        )

    async def assign_student(self, student_id: str):
        """Assign a student to this tutor"""
        if student_id not in self.assigned_student_ids:
            self.assigned_student_ids.append(student_id)
            collection = await get_collection("tutors")
            await collection.update_one(
                {"_id": self._id},
                {"$addToSet": {"assigned_student_ids": student_id}}
            )

    async def unassign_student(self, student_id: str):
        """Unassign a student from this tutor"""
        if student_id in self.assigned_student_ids:
            self.assigned_student_ids.remove(student_id)
            collection = await get_collection("tutors")
            await collection.update_one(
                {"_id": self._id},
                {"$pull": {"assigned_student_ids": student_id}}
            )


# Marshmallow Schema for validation
class TutorSchema(Schema):
    tutor_id = fields.Str(required=True)
    name = fields.Str(required=True, validate=validate.Length(min=2, max=100))
    username = fields.Str(required=True, validate=validate.Length(min=3, max=50))
    email = fields.Email(allow_none=True)
    phone = fields.Str(validate=validate.Length(max=20), allow_none=True)
    standards = fields.List(fields.Str(), allow_none=True)
    sections = fields.List(fields.Str(validate=validate.OneOf(['A', 'B', 'C', 'D', 'E', 'F'])), allow_none=True)
    subjects = fields.List(fields.Str(), allow_none=True)
    plan_types = fields.List(fields.Str(), allow_none=True)
    can_edit_students = fields.Bool(missing=False)
    is_active = fields.Bool(missing=True)
    assigned_student_ids = fields.List(fields.Str(), allow_none=True)
    requires_password_change = fields.Bool(missing=True)
    password_reset_requested = fields.Bool(missing=False)

    @post_load
    def make_tutor(self, data, **kwargs):
        return data


class TutorUpdateSchema(Schema):
    name = fields.Str(validate=validate.Length(min=2, max=100))
    email = fields.Email(allow_none=True)
    phone = fields.Str(validate=validate.Length(max=20), allow_none=True)
    standards = fields.List(fields.Str(), allow_none=True)
    sections = fields.List(fields.Str(validate=validate.OneOf(['A', 'B', 'C', 'D', 'E', 'F'])), allow_none=True)
    subjects = fields.List(fields.Str(), allow_none=True)
    plan_types = fields.List(fields.Str(), allow_none=True)
    can_edit_students = fields.Bool()
    is_active = fields.Bool()


class TutorPasswordChangeSchema(Schema):
    old_password = fields.Str(required=True, validate=validate.Length(min=6))
    new_password = fields.Str(required=True, validate=validate.Length(min=6))