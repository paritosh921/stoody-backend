"""
Student Metrics Model for SkillBot Backend
Provides aggregated metrics and analytics for admin monitoring
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from bson import ObjectId
from .mongodb_client import get_collection
from .student import Student
from .student_session import StudentSession
from .student_activity import StudentActivity
from .question_attempt import QuestionAttempt

class StudentMetrics:
    """Service class for calculating and providing student metrics"""

    @staticmethod
    def get_student_progress_summary(student_id: str) -> Dict[str, Any]:
        """Get comprehensive progress summary for a student"""
        # Get basic student info
        student = Student.find_by_student_id(student_id)
        if not student:
            return None

        # Get session stats
        session_stats = StudentSession.get_session_stats(student_id)

        # Get activity stats
        activity_stats = StudentActivity.get_activity_stats(student_id)

        # Get question attempt stats
        attempt_stats = StudentMetrics._get_attempt_stats(student_id)

        # Calculate level based on problems solved
        level = StudentMetrics._calculate_level(attempt_stats.get('problems_solved', 0))

        # Calculate streak (simplified - days with activity)
        streak_days = StudentMetrics._calculate_streak(student_id)

        return {
            "student_id": student_id,
            "student_name": student.name,
            "email": student.email,
            "total_sessions": session_stats.get('total_sessions', 0),
            "total_time_spent": session_stats.get('total_time_minutes', 0),  # in minutes
            "problems_solved": attempt_stats.get('problems_solved', 0),
            "average_score": round(attempt_stats.get('avg_score', 0), 1),
            "last_active_at": StudentMetrics._get_last_active_time(student_id),
            "streak_days": streak_days,
            "level": level,
            "is_online": session_stats.get('is_online', False)
        }

    @staticmethod
    def get_all_student_progress() -> List[Dict[str, Any]]:
        """Get progress summary for all students"""
        students_collection = get_collection('students')
        all_students = students_collection.find({"is_active": True})

        progress_list = []
        for student_data in all_students:
            student_id = student_data.get('student_id')
            if student_id:
                progress = StudentMetrics.get_student_progress_summary(student_id)
                if progress:
                    progress_list.append(progress)

        # Sort by last activity (most recent first)
        progress_list.sort(key=lambda x: x.get('last_active_at', datetime.min), reverse=True)
        return progress_list

    @staticmethod
    def get_recent_activities_with_names(limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent activities across all students with proper formatting"""
        activities = StudentActivity.get_recent_activities(limit=limit)

        formatted_activities = []
        for activity in activities:
            # Map activity types to readable actions
            action_map = {
                'login': 'login',
                'logout': 'session_end',
                'problem_solving': 'problem_solving',
                'question_attempted': 'question_attempted',
                'session_end': 'session_end'
            }

            formatted_activity = {
                "id": activity.get('_id'),
                "student_id": activity.get('student_id'),
                "student_name": activity.get('student_name', 'Unknown Student'),
                "action": action_map.get(activity.get('activity_type'), activity.get('activity_type')),
                "timestamp": datetime.fromisoformat(activity['created_at'].replace('Z', '+00:00')) if isinstance(activity.get('created_at'), str) else activity.get('created_at', datetime.utcnow()),
                "score": activity.get('score'),
                "session_duration": activity.get('duration_seconds')
            }
            formatted_activities.append(formatted_activity)

        return formatted_activities

    @staticmethod
    def get_admin_dashboard_stats() -> Dict[str, Any]:
        """Get overall statistics for admin dashboard"""
        students_collection = get_collection('students')
        sessions_collection = get_collection('student_sessions')
        activities_collection = get_collection('student_activities')
        questions_collection = get_collection('questions')

        # Total students count
        total_students = students_collection.count_documents({"is_active": True})

        # Active students (with active sessions)
        active_students = sessions_collection.count_documents({"is_active": True})

        # Total questions
        total_questions = questions_collection.count_documents({})

        # Recent activities count (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_activities = activities_collection.count_documents({
            "created_at": {"$gte": one_hour_ago}
        })

        return {
            "total_students": total_students,
            "active_students": active_students,
            "total_questions": total_questions,
            "recent_activities": recent_activities
        }

    @staticmethod
    def _get_attempt_stats(student_id: str) -> Dict[str, Any]:
        """Get question attempt statistics for student"""
        attempts_collection = get_collection('question_attempts')

        # Total attempts
        total_attempts = attempts_collection.count_documents({"student_id": student_id})

        # Correct attempts (problems solved)
        problems_solved = attempts_collection.count_documents({
            "student_id": student_id,
            "is_correct": True
        })

        # Average score
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
                    "avg_score": {"$avg": "$score"}
                }
            }
        ]

        score_result = list(attempts_collection.aggregate(pipeline))
        avg_score = score_result[0]['avg_score'] if score_result else 0

        return {
            "total_attempts": total_attempts,
            "problems_solved": problems_solved,
            "avg_score": avg_score
        }

    @staticmethod
    def _calculate_level(problems_solved: int) -> int:
        """Calculate student level based on problems solved"""
        if problems_solved >= 200:
            return 5
        elif problems_solved >= 150:
            return 4
        elif problems_solved >= 100:
            return 3
        elif problems_solved >= 50:
            return 2
        else:
            return 1

    @staticmethod
    def _calculate_streak(student_id: str) -> int:
        """Calculate consecutive days with activity"""
        activities_collection = get_collection('student_activities')

        # Get activities from last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        activities = activities_collection.find({
            "student_id": student_id,
            "created_at": {"$gte": thirty_days_ago}
        }).sort("created_at", -1)

        # Group activities by date
        activity_dates = set()
        for activity in activities:
            date = activity['created_at'].date()
            activity_dates.add(date)

        # Calculate consecutive streak from today
        today = datetime.utcnow().date()
        streak = 0

        for i in range(30):  # Check last 30 days
            check_date = today - timedelta(days=i)
            if check_date in activity_dates:
                streak += 1
            else:
                break

        return streak

    @staticmethod
    def _get_last_active_time(student_id: str) -> datetime:
        """Get the last activity time for student"""
        activities_collection = get_collection('student_activities')

        latest_activity = activities_collection.find_one(
            {"student_id": student_id},
            sort=[("created_at", -1)]
        )

        if latest_activity:
            return latest_activity['created_at']

        # Fallback to last login if no activities
        sessions_collection = get_collection('student_sessions')
        latest_session = sessions_collection.find_one(
            {"student_id": student_id},
            sort=[("login_time", -1)]
        )

        if latest_session:
            return latest_session['login_time']

        return datetime.min

    @staticmethod
    def log_student_login(student_id: str, ip_address: str = None, user_agent: str = None):
        """Log student login and create session"""
        # Create new session
        session = StudentSession.create_session(student_id, ip_address, user_agent)

        # Log login activity
        StudentActivity.log_activity(
            student_id=student_id,
            activity_type='login',
            description='Student logged in',
            session_id=session.session_id
        )

        return session

    @staticmethod
    def log_student_logout(student_id: str):
        """Log student logout and end session"""
        # Find and end active session
        session = StudentSession.find_active_session(student_id)
        if session:
            session.end_session()

            # Log logout activity
            StudentActivity.log_activity(
                student_id=student_id,
                activity_type='logout',
                description='Student logged out',
                session_id=session.session_id,
                duration_seconds=session.duration_seconds
            )

    @staticmethod
    def log_problem_solving_activity(student_id: str, score: float = None,
                                   duration_seconds: int = None, metadata: Dict[str, Any] = None):
        """Log problem solving activity"""
        # Update session activity
        session = StudentSession.find_active_session(student_id)
        if session:
            session.update_activity()

        # Log activity
        StudentActivity.log_activity(
            student_id=student_id,
            activity_type='problem_solving',
            description='Student solved a problem',
            score=score,
            duration_seconds=duration_seconds,
            metadata=metadata,
            session_id=session.session_id if session else None
        )

    @staticmethod
    def log_question_attempt(student_id: str, question_id: str, score: float = None,
                           is_correct: bool = None, metadata: Dict[str, Any] = None):
        """Log question attempt activity"""
        # Update session activity
        session = StudentSession.find_active_session(student_id)
        if session:
            session.update_activity()

        # Log activity
        StudentActivity.log_activity(
            student_id=student_id,
            activity_type='question_attempted',
            description=f'Attempted question {question_id}',
            score=score,
            metadata=metadata,
            session_id=session.session_id if session else None
        )