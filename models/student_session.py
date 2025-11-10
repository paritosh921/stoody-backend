"""
Student Session Model for SkillBot Backend
Tracks student login sessions and activity periods
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId
from marshmallow import Schema, fields, validate, post_load
# from .mongodb_client import get_collection

class StudentSession:
    """Model for tracking student login sessions and activity periods"""

    def __init__(self, student_id: str, session_id: str = None,
                 login_time: datetime = None, logout_time: datetime = None,
                 duration_seconds: int = None, is_active: bool = True,
                 ip_address: str = None, user_agent: str = None,
                 last_activity: datetime = None, activities_count: int = 0,
                 _id: ObjectId = None):
        self._id = _id
        self.student_id = student_id
        self.session_id = session_id or self._generate_session_id()
        self.login_time = login_time or datetime.utcnow()
        self.logout_time = logout_time
        self.duration_seconds = duration_seconds
        self.is_active = is_active
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.last_activity = last_activity or datetime.utcnow()
        self.activities_count = activities_count

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        import uuid
        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "_id": str(self._id) if self._id else None,
            "student_id": self.student_id,
            "session_id": self.session_id,
            "login_time": self.login_time.isoformat() if self.login_time else None,
            "logout_time": self.logout_time.isoformat() if self.logout_time else None,
            "duration_seconds": self.duration_seconds,
            "is_active": self.is_active,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "activities_count": self.activities_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudentSession':
        """Create StudentSession object from dictionary"""
        return cls(
            _id=data.get('_id'),
            student_id=data['student_id'],
            session_id=data.get('session_id'),
            login_time=data.get('login_time'),
            logout_time=data.get('logout_time'),
            duration_seconds=data.get('duration_seconds'),
            is_active=data.get('is_active', True),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            last_activity=data.get('last_activity'),
            activities_count=data.get('activities_count', 0)
        )

    def save(self) -> ObjectId:
        """Save session to database"""
        collection = get_collection('student_sessions')

        session_data = {
            "student_id": self.student_id,
            "session_id": self.session_id,
            "login_time": self.login_time,
            "logout_time": self.logout_time,
            "duration_seconds": self.duration_seconds,
            "is_active": self.is_active,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "last_activity": self.last_activity,
            "activities_count": self.activities_count
        }

        if self._id:
            # Update existing session
            collection.update_one(
                {"_id": self._id},
                {"$set": session_data}
            )
            return self._id
        else:
            # Create new session
            result = collection.insert_one(session_data)
            self._id = result.inserted_id
            return self._id

    def end_session(self):
        """End the current session"""
        if self.is_active:
            self.logout_time = datetime.utcnow()
            self.is_active = False
            if self.login_time:
                self.duration_seconds = int((self.logout_time - self.login_time).total_seconds())
            self.save()

    def update_activity(self):
        """Update last activity time and increment activity count"""
        self.last_activity = datetime.utcnow()
        self.activities_count += 1

        collection = get_collection('student_sessions')
        collection.update_one(
            {"_id": self._id},
            {
                "$set": {"last_activity": self.last_activity},
                "$inc": {"activities_count": 1}
            }
        )

    @classmethod
    def find_active_session(cls, student_id: str) -> Optional['StudentSession']:
        """Find active session for student"""
        collection = get_collection('student_sessions')
        session_data = collection.find_one({
            "student_id": student_id,
            "is_active": True
        })

        if session_data:
            return cls.from_dict(session_data)
        return None

    @classmethod
    def create_session(cls, student_id: str, ip_address: str = None,
                      user_agent: str = None) -> 'StudentSession':
        """Create new session for student"""
        # End any existing active sessions first
        existing_session = cls.find_active_session(student_id)
        if existing_session:
            existing_session.end_session()

        # Create new session
        session = cls(
            student_id=student_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        session.save()
        return session

    @classmethod
    def get_student_sessions(cls, student_id: str, limit: int = 10) -> List['StudentSession']:
        """Get recent sessions for student"""
        collection = get_collection('student_sessions')
        sessions_data = collection.find(
            {"student_id": student_id}
        ).sort("login_time", -1).limit(limit)

        return [cls.from_dict(session_data) for session_data in sessions_data]

    @classmethod
    def get_session_stats(cls, student_id: str) -> Dict[str, Any]:
        """Get session statistics for student"""
        collection = get_collection('student_sessions')

        # Get total sessions
        total_sessions = collection.count_documents({"student_id": student_id})

        # Get total time spent (sum of all session durations)
        pipeline = [
            {"$match": {"student_id": student_id, "duration_seconds": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "total_time_seconds": {"$sum": "$duration_seconds"},
                "avg_session_duration": {"$avg": "$duration_seconds"}
            }}
        ]

        result = list(collection.aggregate(pipeline))
        total_time_seconds = result[0]['total_time_seconds'] if result else 0
        avg_session_duration = result[0]['avg_session_duration'] if result else 0

        # Check if student is currently online
        is_online = cls.find_active_session(student_id) is not None

        return {
            "total_sessions": total_sessions,
            "total_time_seconds": total_time_seconds,
            "total_time_minutes": total_time_seconds // 60,
            "avg_session_duration_seconds": avg_session_duration,
            "avg_session_duration_minutes": avg_session_duration // 60 if avg_session_duration else 0,
            "is_online": is_online
        }

class StudentSessionSchema(Schema):
    """Marshmallow schema for StudentSession validation"""

    student_id = fields.Str(required=True)
    session_id = fields.Str()
    login_time = fields.DateTime()
    logout_time = fields.DateTime()
    duration_seconds = fields.Int()
    is_active = fields.Bool(missing=True)
    ip_address = fields.Str()
    user_agent = fields.Str()
    last_activity = fields.DateTime()
    activities_count = fields.Int(missing=0)

    @post_load
    def make_session(self, data, **kwargs):
        """Create StudentSession object from validated data"""
        return StudentSession(**data)