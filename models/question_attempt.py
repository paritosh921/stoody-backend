"""
Question Attempt Model for SkillBot Backend
Tracks student question attempts and performance analytics
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId
from marshmallow import Schema, fields, validate, post_load
# from .mongodb_client import get_collection

class QuestionAttempt:
    """Model for tracking student question attempts and performance"""

    def __init__(self, student_id: str, question_id: str, question_type: str,
                 question_text: str, student_answer: str, correct_answer: str = None,
                 is_correct: bool = None, score: float = None, time_taken: int = None,
                 hints_used: int = 0, mode: str = "practice", difficulty: str = None,
                 subject: str = None, topic: str = None, created_at: datetime = None,
                 _id: ObjectId = None):
        self._id = _id
        self.student_id = student_id
        self.question_id = question_id
        self.question_type = question_type  # 'mcq', 'text', 'canvas_drawing', etc.
        self.question_text = question_text
        self.student_answer = student_answer
        self.correct_answer = correct_answer
        self.is_correct = is_correct
        self.score = score  # 0-100 score
        self.time_taken = time_taken  # in seconds
        self.hints_used = hints_used
        self.mode = mode  # 'practice', 'test', 'debugging'
        self.difficulty = difficulty  # 'easy', 'medium', 'hard'
        self.subject = subject
        self.topic = topic
        self.created_at = created_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert question attempt to dictionary"""
        return {
            "_id": str(self._id) if self._id else None,
            "student_id": self.student_id,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "student_answer": self.student_answer,
            "correct_answer": self.correct_answer,
            "is_correct": self.is_correct,
            "score": self.score,
            "time_taken": self.time_taken,
            "hints_used": self.hints_used,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "subject": self.subject,
            "topic": self.topic,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionAttempt':
        """Create QuestionAttempt object from dictionary"""
        return cls(
            _id=data.get('_id'),
            student_id=data['student_id'],
            question_id=data['question_id'],
            question_type=data['question_type'],
            question_text=data['question_text'],
            student_answer=data['student_answer'],
            correct_answer=data.get('correct_answer'),
            is_correct=data.get('is_correct'),
            score=data.get('score'),
            time_taken=data.get('time_taken'),
            hints_used=data.get('hints_used', 0),
            mode=data.get('mode', 'practice'),
            difficulty=data.get('difficulty'),
            subject=data.get('subject'),
            topic=data.get('topic'),
            created_at=data.get('created_at')
        )

    def save(self) -> ObjectId:
        """Save question attempt to database"""
        collection = get_collection('question_attempts')

        attempt_data = {
            "student_id": self.student_id,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "student_answer": self.student_answer,
            "correct_answer": self.correct_answer,
            "is_correct": self.is_correct,
            "score": self.score,
            "time_taken": self.time_taken,
            "hints_used": self.hints_used,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "subject": self.subject,
            "topic": self.topic,
            "created_at": self.created_at
        }

        if self._id:
            # Update existing attempt
            collection.update_one(
                {"_id": self._id},
                {"$set": attempt_data}
            )
            return self._id
        else:
            # Create new attempt
            result = collection.insert_one(attempt_data)
            self._id = result.inserted_id
            return self._id

    @classmethod
    def get_student_attempts(cls, student_id: str, page: int = 1, limit: int = 50) -> Dict[str, Any]:
        """Get all attempts by a student with pagination"""
        collection = get_collection('question_attempts')

        # Get total count
        total = collection.count_documents({"student_id": student_id})

        # Calculate pagination
        skip = (page - 1) * limit
        total_pages = (total + limit - 1) // limit

        # Get attempts
        attempts_data = collection.find({"student_id": student_id}).skip(skip).limit(limit).sort("created_at", -1)
        attempts = [cls.from_dict(attempt_data).to_dict() for attempt_data in attempts_data]

        return {
            "attempts": attempts,
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
    def get_student_performance(cls, student_id: str) -> Dict[str, Any]:
        """Get comprehensive performance analytics for a student"""
        collection = get_collection('question_attempts')

        # Basic stats
        total_attempts = collection.count_documents({"student_id": student_id})
        correct_attempts = collection.count_documents({"student_id": student_id, "is_correct": True})

        # Performance by difficulty
        difficulty_stats = {}
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_total = collection.count_documents({
                "student_id": student_id,
                "difficulty": difficulty
            })
            difficulty_correct = collection.count_documents({
                "student_id": student_id,
                "difficulty": difficulty,
                "is_correct": True
            })
            difficulty_stats[difficulty] = {
                "total": difficulty_total,
                "correct": difficulty_correct,
                "accuracy": (difficulty_correct / difficulty_total * 100) if difficulty_total > 0 else 0
            }

        # Performance by subject
        pipeline = [
            {"$match": {"student_id": student_id}},
            {"$group": {
                "_id": "$subject",
                "total": {"$sum": 1},
                "correct": {"$sum": {"$cond": ["$is_correct", 1, 0]}},
                "avg_score": {"$avg": "$score"},
                "avg_time": {"$avg": "$time_taken"}
            }}
        ]
        subject_stats = list(collection.aggregate(pipeline))

        # Recent activity (last 7 days)
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_attempts = collection.count_documents({
            "student_id": student_id,
            "created_at": {"$gte": week_ago}
        })

        # Average score and time
        avg_pipeline = [
            {"$match": {"student_id": student_id}},
            {"$group": {
                "_id": None,
                "avg_score": {"$avg": "$score"},
                "avg_time": {"$avg": "$time_taken"},
                "total_hints": {"$sum": "$hints_used"}
            }}
        ]
        avg_stats = list(collection.aggregate(avg_pipeline))
        avg_data = avg_stats[0] if avg_stats else {}

        return {
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "accuracy": (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0,
            "avg_score": avg_data.get("avg_score", 0),
            "avg_time_per_question": avg_data.get("avg_time", 0),
            "total_hints_used": avg_data.get("total_hints", 0),
            "recent_activity": recent_attempts,
            "difficulty_breakdown": difficulty_stats,
            "subject_performance": [
                {
                    "subject": stat["_id"],
                    "total": stat["total"],
                    "correct": stat["correct"],
                    "accuracy": (stat["correct"] / stat["total"] * 100) if stat["total"] > 0 else 0,
                    "avg_score": stat.get("avg_score", 0),
                    "avg_time": stat.get("avg_time", 0)
                }
                for stat in subject_stats if stat["_id"]
            ]
        }

    @classmethod
    def get_all_students_performance(cls, admin_id: str = None) -> List[Dict[str, Any]]:
        """Get performance summary for all students (optionally filtered by admin_id)"""
        collection = get_collection('question_attempts')

        # Build match stage for filtering
        match_stage = {}
        if admin_id:
            match_stage["admin_id"] = admin_id

        pipeline = []
        if match_stage:
            pipeline.append({"$match": match_stage})

        pipeline.extend([
            {"$group": {
                "_id": "$student_id",
                "total_attempts": {"$sum": 1},
                "correct_attempts": {"$sum": {"$cond": ["$is_correct", 1, 0]}},
                "avg_score": {"$avg": "$score"},
                "last_attempt": {"$max": "$created_at"},
                "total_time": {"$sum": "$time_taken"}
            }},
            {"$sort": {"last_attempt": -1}}
        ])

        results = list(collection.aggregate(pipeline))

        # Enrich with student data
        from .student import Student
        performance_data = []

        for result in results:
            student = Student.find_by_student_id(result["_id"])
            if student:
                performance_data.append({
                    "student_id": result["_id"],
                    "student_name": student.name,
                    "total_attempts": result["total_attempts"],
                    "correct_attempts": result["correct_attempts"],
                    "accuracy": (result["correct_attempts"] / result["total_attempts"] * 100) if result["total_attempts"] > 0 else 0,
                    "avg_score": result.get("avg_score", 0),
                    "total_time_spent": result.get("total_time", 0),
                    "last_attempt": result["last_attempt"].isoformat() if result["last_attempt"] else None
                })

        return performance_data

    @classmethod
    def get_question_analytics(cls, question_id: str) -> Dict[str, Any]:
        """Get analytics for a specific question"""
        collection = get_collection('question_attempts')

        total_attempts = collection.count_documents({"question_id": question_id})
        correct_attempts = collection.count_documents({"question_id": question_id, "is_correct": True})

        # Get common wrong answers
        wrong_answers_pipeline = [
            {"$match": {"question_id": question_id, "is_correct": False}},
            {"$group": {
                "_id": "$student_answer",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        wrong_answers = list(collection.aggregate(wrong_answers_pipeline))

        return {
            "question_id": question_id,
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "accuracy": (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0,
            "common_wrong_answers": [
                {"answer": ans["_id"], "count": ans["count"]}
                for ans in wrong_answers
            ]
        }

class QuestionAttemptSchema(Schema):
    """Marshmallow schema for QuestionAttempt validation"""

    student_id = fields.Str(required=True)
    question_id = fields.Str(required=True)
    question_type = fields.Str(required=True, validate=validate.OneOf(['mcq', 'text', 'canvas_drawing', 'file_upload']))
    question_text = fields.Str(required=True)
    student_answer = fields.Str(required=True)
    correct_answer = fields.Str(allow_none=True)
    is_correct = fields.Bool(allow_none=True)
    score = fields.Float(validate=validate.Range(min=0, max=100), allow_none=True)
    time_taken = fields.Int(validate=validate.Range(min=0), allow_none=True)
    hints_used = fields.Int(validate=validate.Range(min=0), missing=0)
    mode = fields.Str(validate=validate.OneOf(['practice', 'test', 'debugging']), missing='practice')
    difficulty = fields.Str(validate=validate.OneOf(['easy', 'medium', 'hard']), allow_none=True)
    subject = fields.Str(validate=validate.Length(max=100), allow_none=True)
    topic = fields.Str(validate=validate.Length(max=100), allow_none=True)

    @post_load
    def make_attempt(self, data, **kwargs):
        """Create QuestionAttempt object from validated data"""
        return QuestionAttempt(**data)