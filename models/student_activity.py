"""
Student Activity Model for SkillBot Backend
Tracks different student activities and actions
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId
from marshmallow import Schema, fields, validate, post_load
from .mongodb_client import get_collection

class StudentActivity:
    """Model for tracking student activities and actions"""

    def __init__(self, student_id: str, activity_type: str,
                 description: str = None, metadata: Dict[str, Any] = None,
                 score: float = None, duration_seconds: int = None,
                 session_id: str = None, created_at: datetime = None,
                 _id: ObjectId = None):
        self._id = _id
        self.student_id = student_id
        self.activity_type = activity_type  # 'login', 'logout', 'problem_solving', 'question_attempted', 'session_end', etc.
        self.description = description
        self.metadata = metadata or {}
        self.score = score
        self.duration_seconds = duration_seconds
        self.session_id = session_id
        self.created_at = created_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert activity to dictionary"""
        return {
            "_id": str(self._id) if self._id else None,
            "student_id": self.student_id,
            "activity_type": self.activity_type,
            "description": self.description,
            "metadata": self.metadata,
            "score": self.score,
            "duration_seconds": self.duration_seconds,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudentActivity':
        """Create StudentActivity object from dictionary"""
        return cls(
            _id=data.get('_id'),
            student_id=data['student_id'],
            activity_type=data['activity_type'],
            description=data.get('description'),
            metadata=data.get('metadata', {}),
            score=data.get('score'),
            duration_seconds=data.get('duration_seconds'),
            session_id=data.get('session_id'),
            created_at=data.get('created_at')
        )

    def save(self) -> ObjectId:
        """Save activity to database"""
        collection = get_collection('student_activities')

        activity_data = {
            "student_id": self.student_id,
            "activity_type": self.activity_type,
            "description": self.description,
            "metadata": self.metadata,
            "score": self.score,
            "duration_seconds": self.duration_seconds,
            "session_id": self.session_id,
            "created_at": self.created_at
        }

        if self._id:
            # Update existing activity
            collection.update_one(
                {"_id": self._id},
                {"$set": activity_data}
            )
            return self._id
        else:
            # Create new activity
            result = collection.insert_one(activity_data)
            self._id = result.inserted_id
            return self._id

    @classmethod
    def log_activity(cls, student_id: str, activity_type: str,
                    description: str = None, metadata: Dict[str, Any] = None,
                    score: float = None, duration_seconds: int = None,
                    session_id: str = None) -> 'StudentActivity':
        """Log a new student activity"""
        activity = cls(
            student_id=student_id,
            activity_type=activity_type,
            description=description,
            metadata=metadata,
            score=score,
            duration_seconds=duration_seconds,
            session_id=session_id
        )
        activity.save()
        return activity

    @classmethod
    def get_student_activities(cls, student_id: str, limit: int = 50,
                              activity_types: List[str] = None) -> List['StudentActivity']:
        """Get recent activities for student"""
        collection = get_collection('student_activities')

        query = {"student_id": student_id}
        if activity_types:
            query["activity_type"] = {"$in": activity_types}

        activities_data = collection.find(query).sort("created_at", -1).limit(limit)
        return [cls.from_dict(activity_data) for activity_data in activities_data]

    @classmethod
    def get_recent_activities(cls, limit: int = 100,
                            activity_types: List[str] = None) -> List[Dict[str, Any]]:
        """Get recent activities across all students with student names"""
        collection = get_collection('student_activities')
        students_collection = get_collection('students')

        # Build query
        query = {}
        if activity_types:
            query["activity_type"] = {"$in": activity_types}

        # Aggregation pipeline to join with student data
        pipeline = [
            {"$match": query},
            {"$sort": {"created_at": -1}},
            {"$limit": limit},
            {
                "$lookup": {
                    "from": "students",
                    "localField": "student_id",
                    "foreignField": "student_id",
                    "as": "student_info"
                }
            },
            {
                "$unwind": {
                    "path": "$student_info",
                    "preserveNullAndEmptyArrays": True
                }
            },
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "student_id": 1,
                    "student_name": "$student_info.name",
                    "activity_type": 1,
                    "description": 1,
                    "metadata": 1,
                    "score": 1,
                    "duration_seconds": 1,
                    "session_id": 1,
                    "created_at": 1
                }
            }
        ]

        activities = list(collection.aggregate(pipeline))

        # Convert datetime objects to ISO format for JSON serialization
        for activity in activities:
            if activity.get('created_at'):
                activity['created_at'] = activity['created_at'].isoformat()

        return activities

    @classmethod
    def get_activity_stats(cls, student_id: str) -> Dict[str, Any]:
        """Get activity statistics for student"""
        collection = get_collection('student_activities')

        # Get total activities count
        total_activities = collection.count_documents({"student_id": student_id})

        # Get problem solving activities count
        problem_solving_activities = collection.count_documents({
            "student_id": student_id,
            "activity_type": {"$in": ["problem_solving", "question_attempted"]}
        })

        # Get average score from scored activities
        pipeline = [
            {
                "$match": {
                    "student_id": student_id,
                    "score": {"$exists": True, "$ne": None}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$score"},
                    "max_score": {"$max": "$score"},
                    "min_score": {"$min": "$score"},
                    "scored_activities_count": {"$sum": 1}
                }
            }
        ]

        score_result = list(collection.aggregate(pipeline))
        avg_score = score_result[0]['avg_score'] if score_result else 0
        max_score = score_result[0]['max_score'] if score_result else 0
        min_score = score_result[0]['min_score'] if score_result else 0
        scored_activities = score_result[0]['scored_activities_count'] if score_result else 0

        # Get recent activity types breakdown
        type_breakdown_pipeline = [
            {"$match": {"student_id": student_id}},
            {"$group": {"_id": "$activity_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        type_breakdown = list(collection.aggregate(type_breakdown_pipeline))

        return {
            "total_activities": total_activities,
            "problem_solving_activities": problem_solving_activities,
            "avg_score": round(avg_score, 2) if avg_score else 0,
            "max_score": max_score,
            "min_score": min_score,
            "scored_activities_count": scored_activities,
            "activity_type_breakdown": type_breakdown
        }

class StudentActivitySchema(Schema):
    """Marshmallow schema for StudentActivity validation"""

    student_id = fields.Str(required=True)
    activity_type = fields.Str(required=True, validate=validate.OneOf([
        'login', 'logout', 'problem_solving', 'question_attempted',
        'session_end', 'canvas_drawing', 'chat_message', 'file_upload',
        'mode_switch', 'timer_start', 'timer_end'
    ]))
    description = fields.Str()
    metadata = fields.Dict()
    score = fields.Float(validate=validate.Range(min=0, max=100))
    duration_seconds = fields.Int(validate=validate.Range(min=0))
    session_id = fields.Str()
    created_at = fields.DateTime()

    @post_load
    def make_activity(self, data, **kwargs):
        """Create StudentActivity object from validated data"""
        return StudentActivity(**data)