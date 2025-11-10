"""
Student Model for SkillBot Backend
Handles student user data and authentication
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import random
import string
from bson import ObjectId
from marshmallow import Schema, fields, validate, post_load
# from .mongodb_client import get_collection

class Student:
    """Student user model for authentication and progress tracking"""

    def __init__(self, student_id: str, name: str, username: str,
                 password_hash: str, email: str = None, age: int = None,
                 gender: str = None, location: str = None, school: str = None,
                 stream: str = None, grade: str = None, phone: str = None,
                 is_active: bool = True, created_by: str = None,
                 created_at: datetime = None, last_login: datetime = None,
                 _id: ObjectId = None):
        self._id = _id
        self.student_id = student_id
        self.name = name
        self.username = username
        self.password_hash = password_hash
        self.email = email
        self.age = age
        self.gender = gender
        self.location = location
        self.school = school
        self.stream = stream
        self.grade = grade
        self.phone = phone
        self.is_active = is_active
        self.created_by = created_by  # Admin ID who created this student
        self.created_at = created_at or datetime.utcnow()
        self.last_login = last_login

    @staticmethod
    def generate_student_id(prefix: str = "STU") -> str:
        """Generate unique student ID"""
        # Generate random 6-digit number
        random_digits = ''.join(random.choices(string.digits, k=6))
        return f"{prefix}{random_digits}"

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def to_dict(self, include_password: bool = False) -> Dict[str, Any]:
        """Convert student object to dictionary"""
        data = {
            "_id": str(self._id) if self._id else None,
            "student_id": self.student_id,
            "name": self.name,
            "username": self.username,
            "email": self.email,
            "age": self.age,
            "gender": self.gender,
            "location": self.location,
            "school": self.school,
            "stream": self.stream,
            "grade": self.grade,
            "phone": self.phone,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }

        if include_password:
            data["password_hash"] = self.password_hash

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Student':
        """Create Student object from dictionary"""
        return cls(
            _id=data.get('_id'),
            student_id=data.get('student_id', ''),
            name=data.get('name', ''),
            username=data.get('username', ''),  # Use .get() to handle missing username field
            password_hash=data.get('password_hash', ''),
            email=data.get('email'),
            age=data.get('age'),
            gender=data.get('gender'),
            location=data.get('location'),
            school=data.get('school'),
            stream=data.get('stream'),
            grade=data.get('grade'),
            phone=data.get('phone'),
            is_active=data.get('is_active', True),
            created_by=data.get('created_by'),
            created_at=data.get('created_at'),
            last_login=data.get('last_login')
        )

    def save(self) -> ObjectId:
        """Save student to database"""
        collection = get_collection('students')

        student_data = {
            "student_id": self.student_id,
            "name": self.name,
            "username": self.username,
            "password_hash": self.password_hash,
            "email": self.email,
            "age": self.age,
            "gender": self.gender,
            "location": self.location,
            "school": self.school,
            "stream": self.stream,
            "grade": self.grade,
            "phone": self.phone,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "last_login": self.last_login
        }

        if self._id:
            # Update existing student
            collection.update_one(
                {"_id": self._id},
                {"$set": student_data}
            )
            return self._id
        else:
            # Create new student
            result = collection.insert_one(student_data)
            self._id = result.inserted_id
            return self._id

    @classmethod
    def find_by_student_id(cls, student_id: str) -> Optional['Student']:
        """Find student by student ID"""
        collection = get_collection('students')
        student_data = collection.find_one({"student_id": student_id})

        if student_data:
            return cls.from_dict(student_data)
        return None

    @classmethod
    def find_by_id(cls, student_id: str) -> Optional['Student']:
        """Find student by MongoDB ObjectId"""
        collection = get_collection('students')
        try:
            student_data = collection.find_one({"_id": ObjectId(student_id)})
            if student_data:
                return cls.from_dict(student_data)
        except Exception:
            pass
        return None

    @classmethod
    def find_by_email(cls, email: str) -> Optional['Student']:
        """Find student by email"""
        collection = get_collection('students')
        student_data = collection.find_one({"email": email})

        if student_data:
            return cls.from_dict(student_data)
        return None

    @classmethod
    def find_by_username(cls, username: str) -> Optional['Student']:
        """Find student by username"""
        collection = get_collection('students')
        student_data = collection.find_one({"username": username})

        if student_data:
            return cls.from_dict(student_data)
        return None

    @classmethod
    def get_all_students(cls, page: int = 1, limit: int = 50, search: str = None) -> Dict[str, Any]:
        """Get all students with pagination and search"""
        collection = get_collection('students')

        # Build query
        query = {}
        if search:
            query = {
                "$or": [
                    {"name": {"$regex": search, "$options": "i"}},
                    {"student_id": {"$regex": search, "$options": "i"}},
                    {"username": {"$regex": search, "$options": "i"}},
                    {"email": {"$regex": search, "$options": "i"}},
                    {"school": {"$regex": search, "$options": "i"}}
                ]
            }

        # Get total count
        total = collection.count_documents(query)

        # Calculate pagination
        skip = (page - 1) * limit
        total_pages = (total + limit - 1) // limit

        # Get students
        students_data = collection.find(query).skip(skip).limit(limit).sort("created_at", -1)
        students = [cls.from_dict(student_data).to_dict() for student_data in students_data]

        return {
            "students": students,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }

    @classmethod
    def check_student_id_exists(cls, student_id: str) -> bool:
        """Check if student ID already exists"""
        collection = get_collection('students')
        return collection.find_one({"student_id": student_id}) is not None

    @classmethod
    def check_username_exists(cls, username: str) -> bool:
        """Check if username already exists"""
        collection = get_collection('students')
        return collection.find_one({"username": username}) is not None

    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        collection = get_collection('students')
        collection.update_one(
            {"_id": self._id},
            {"$set": {"last_login": self.last_login}}
        )

    def delete(self):
        """Soft delete student (mark as inactive)"""
        self.is_active = False
        collection = get_collection('students')
        collection.update_one(
            {"_id": self._id},
            {"$set": {"is_active": False}}
        )

class StudentSchema(Schema):
    """Marshmallow schema for Student validation"""

    student_id = fields.Str(validate=validate.Length(min=3, max=20))
    name = fields.Str(required=True, validate=validate.Length(min=2, max=255))
    username = fields.Str(required=True, validate=validate.Length(min=3, max=50))
    password = fields.Str(required=True, validate=validate.Length(min=6, max=255), load_only=True)
    email = fields.Email(validate=validate.Length(max=255), allow_none=True)
    age = fields.Int(validate=validate.Range(min=5, max=100), allow_none=True)
    gender = fields.Str(validate=validate.OneOf(['male', 'female', 'other', 'prefer-not-to-say']), allow_none=True)
    location = fields.Str(validate=validate.Length(max=255), allow_none=True)
    school = fields.Str(validate=validate.Length(max=255), allow_none=True)
    stream = fields.Str(validate=validate.OneOf(['science', 'arts', 'commerce', 'other']), allow_none=True)
    grade = fields.Str(validate=validate.Length(max=20), allow_none=True)
    phone = fields.Str(validate=validate.Length(max=20), allow_none=True)
    is_active = fields.Bool(missing=True)

    @post_load
    def make_student(self, data, **kwargs):
        """Create Student object from validated data"""
        # Generate student ID if not provided
        if 'student_id' not in data or not data['student_id']:
            data['student_id'] = Student.generate_student_id()

        # Hash password if provided
        password = data.pop('password', None)
        if password:
            data['password_hash'] = Student.hash_password(password)

        return Student(**data)

class StudentLoginSchema(Schema):
    """Schema for student login validation"""

    username = fields.Str(required=True)
    password = fields.Str(required=True)

class StudentUpdateSchema(Schema):
    """Schema for student update validation"""

    name = fields.Str(validate=validate.Length(min=2, max=255))
    username = fields.Str(validate=validate.Length(min=3, max=50))
    email = fields.Email(validate=validate.Length(max=255), allow_none=True)
    age = fields.Int(validate=validate.Range(min=5, max=100), allow_none=True)
    gender = fields.Str(validate=validate.OneOf(['male', 'female', 'other', 'prefer-not-to-say']), allow_none=True)
    location = fields.Str(validate=validate.Length(max=255), allow_none=True)
    school = fields.Str(validate=validate.Length(max=255), allow_none=True)
    stream = fields.Str(validate=validate.OneOf(['science', 'arts', 'commerce', 'other']), allow_none=True)
    grade = fields.Str(validate=validate.Length(max=20), allow_none=True)
    phone = fields.Str(validate=validate.Length(max=20), allow_none=True)
    is_active = fields.Bool()

class StudentPasswordChangeSchema(Schema):
    """Schema for student password change validation"""

    current_password = fields.Str(required=True)
    new_password = fields.Str(required=True, validate=validate.Length(min=6, max=255))

class StudentPasswordResetSchema(Schema):
    """Schema for admin password reset validation"""

    new_password = fields.Str(required=True, validate=validate.Length(min=6, max=255))