"""
Meeting model for online classes.

Represents scheduled online class meetings created by tutors.
Students receive meeting invites based on their class/section/subject mapping.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId


class Meeting:
    """Online class meeting model."""

    def __init__(
        self,
        meeting_id: str,
        tutor_id: str,
        tutor_name: str,
        topic: str,
        subject: str,
        standard: str,  # Class/grade (e.g., "10", "11")
        section: Optional[str] = None,  # Section (e.g., "A", "B")
        course_type: Optional[str] = None,  # Plan type (e.g., "foundation", "advanced")
        scheduled_at: datetime = None,
        duration_minutes: int = 60,
        meet_link: Optional[str] = None,
        meet_code: Optional[str] = None,
        status: str = "scheduled",  # scheduled, active, ended, cancelled
        invited_student_ids: Optional[List[str]] = None,
        joined_student_ids: Optional[List[str]] = None,
        admin_id: Optional[str] = None,
        created_at: datetime = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        is_archived: bool = False,
        _id: ObjectId = None,
    ):
        self._id = _id
        self.meeting_id = meeting_id
        self.tutor_id = tutor_id
        self.tutor_name = tutor_name
        self.topic = topic
        self.subject = subject
        self.standard = standard
        self.section = section
        self.course_type = course_type
        self.scheduled_at = scheduled_at or datetime.utcnow()
        self.duration_minutes = duration_minutes
        self.meet_link = meet_link
        self.meet_code = meet_code
        self.status = status
        self.invited_student_ids = invited_student_ids or []
        self.joined_student_ids = joined_student_ids or []
        self.admin_id = admin_id
        self.created_at = created_at or datetime.utcnow()
        self.started_at = started_at
        self.ended_at = ended_at
        self.is_archived = is_archived

    @staticmethod
    def generate_meeting_id(prefix: str = "MTG") -> str:
        """Generate unique meeting ID"""
        import random
        import string
        random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        return f"{prefix}{random_chars}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert meeting object to dictionary"""
        return {
            "_id": str(self._id) if self._id else None,
            "meeting_id": self.meeting_id,
            "tutor_id": self.tutor_id,
            "tutor_name": self.tutor_name,
            "topic": self.topic,
            "subject": self.subject,
            "standard": self.standard,
            "section": self.section,
            "course_type": self.course_type,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "duration_minutes": self.duration_minutes,
            "meet_link": self.meet_link,
            "meet_code": self.meet_code,
            "status": self.status,
            "invited_student_ids": self.invited_student_ids,
            "joined_student_ids": self.joined_student_ids,
            "admin_id": self.admin_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "is_archived": self.is_archived,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Meeting':
        """Create Meeting object from dictionary"""
        # Handle datetime fields
        scheduled_at = data.get('scheduled_at')
        if isinstance(scheduled_at, str):
            scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))

        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

        started_at = data.get('started_at')
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))

        ended_at = data.get('ended_at')
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at.replace('Z', '+00:00'))

        return cls(
            _id=data.get('_id'),
            meeting_id=data.get('meeting_id', ''),
            tutor_id=data.get('tutor_id', ''),
            tutor_name=data.get('tutor_name', ''),
            topic=data.get('topic', ''),
            subject=data.get('subject', ''),
            standard=data.get('standard', ''),
            section=data.get('section'),
            course_type=data.get('course_type'),
            scheduled_at=scheduled_at,
            duration_minutes=data.get('duration_minutes', 60),
            meet_link=data.get('meet_link'),
            meet_code=data.get('meet_code'),
            status=data.get('status', 'scheduled'),
            invited_student_ids=data.get('invited_student_ids', []),
            joined_student_ids=data.get('joined_student_ids', []),
            admin_id=data.get('admin_id'),
            created_at=created_at,
            started_at=started_at,
            ended_at=ended_at,
            is_archived=bool(data.get('is_archived', False)),
        )

    def to_student_view(self) -> Dict[str, Any]:
        """Return meeting info visible to students"""
        return {
            "meeting_id": self.meeting_id,
            "topic": self.topic,
            "subject": self.subject,
            "standard": self.standard,
            "section": self.section,
            "tutor_name": self.tutor_name,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "duration_minutes": self.duration_minutes,
            "meet_link": self.meet_link,
            "meet_code": self.meet_code,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }
