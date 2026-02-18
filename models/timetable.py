"""
Timetable Models for Class Schedule & Teacher Calendar

Models:
- ClassTimetable: Master timetable definition for a grade+section
- TimetablePeriod: Individual period entry within a timetable
- AcademicCalendarEntry: Holidays, exams, events that affect scheduling
- PreRead: Pre-read material assigned by teacher for a specific period+date
"""

from datetime import datetime, time
from typing import Optional, Dict, Any, List
from bson import ObjectId
import random
import string


class ClassTimetable:
    """Master timetable for a grade+section (weekly recurring template)"""

    def __init__(
        self,
        grade: str,
        section: str,
        academic_session: str,
        class_teacher_id: Optional[str] = None,
        class_teacher_name: Optional[str] = None,
        period_timings: Optional[List[Dict[str, str]]] = None,
        weekdays: Optional[List[str]] = None,
        is_active: bool = True,
        admin_id: Optional[str] = None,
        created_by: Optional[str] = None,
        created_by_role: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None,
    ):
        self._id = _id
        self.grade = grade
        self.section = section
        self.academic_session = academic_session
        self.class_teacher_id = class_teacher_id
        self.class_teacher_name = class_teacher_name
        # List of {period_number, start_time, end_time, label}
        # e.g. [{"period_number": 1, "start_time": "08:00", "end_time": "08:40", "label": "Period 1"}]
        self.period_timings = period_timings or []
        # Active weekdays e.g. ["Monday", "Tuesday", ..., "Saturday"]
        self.weekdays = weekdays or [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
        ]
        self.is_active = is_active
        self.admin_id = admin_id
        self.created_by = created_by
        self.created_by_role = created_by_role  # "admin" or "tutor"
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": str(self._id) if self._id else None,
            "grade": self.grade,
            "section": self.section,
            "academic_session": self.academic_session,
            "class_teacher_id": self.class_teacher_id,
            "class_teacher_name": self.class_teacher_name,
            "period_timings": self.period_timings,
            "weekdays": self.weekdays,
            "is_active": self.is_active,
            "admin_id": self.admin_id,
            "created_by": self.created_by,
            "created_by_role": self.created_by_role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_mongo(self) -> Dict[str, Any]:
        """Convert to MongoDB document (without _id for inserts)"""
        doc = {
            "grade": self.grade,
            "section": self.section,
            "academic_session": self.academic_session,
            "class_teacher_id": self.class_teacher_id,
            "class_teacher_name": self.class_teacher_name,
            "period_timings": self.period_timings,
            "weekdays": self.weekdays,
            "is_active": self.is_active,
            "admin_id": self.admin_id,
            "created_by": self.created_by,
            "created_by_role": self.created_by_role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self._id:
            doc["_id"] = self._id
        return doc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassTimetable":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

        return cls(
            _id=data.get("_id"),
            grade=data.get("grade", ""),
            section=data.get("section", ""),
            academic_session=data.get("academic_session", ""),
            class_teacher_id=data.get("class_teacher_id"),
            class_teacher_name=data.get("class_teacher_name"),
            period_timings=data.get("period_timings", []),
            weekdays=data.get("weekdays", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
            is_active=data.get("is_active", True),
            admin_id=data.get("admin_id"),
            created_by=data.get("created_by"),
            created_by_role=data.get("created_by_role"),
            created_at=created_at,
            updated_at=updated_at,
        )


class TimetablePeriod:
    """Individual period entry within a class timetable"""

    def __init__(
        self,
        timetable_id: str,
        grade: str,
        section: str,
        day_of_week: str,
        period_number: int,
        start_time: str,
        end_time: str,
        subject: str,
        teacher_id: Optional[str] = None,
        teacher_name: Optional[str] = None,
        room: Optional[str] = None,
        period_type: str = "regular",
        admin_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None,
    ):
        self._id = _id
        self.timetable_id = timetable_id
        self.grade = grade
        self.section = section
        self.day_of_week = day_of_week  # "Monday", "Tuesday", etc.
        self.period_number = period_number
        self.start_time = start_time  # "08:00"
        self.end_time = end_time  # "08:40"
        self.subject = subject
        self.teacher_id = teacher_id
        self.teacher_name = teacher_name
        self.room = room
        # "regular", "break", "lunch", "assembly", "library", "sports"
        self.period_type = period_type
        self.admin_id = admin_id
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": str(self._id) if self._id else None,
            "timetable_id": self.timetable_id,
            "grade": self.grade,
            "section": self.section,
            "day_of_week": self.day_of_week,
            "period_number": self.period_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "subject": self.subject,
            "teacher_id": self.teacher_id,
            "teacher_name": self.teacher_name,
            "room": self.room,
            "period_type": self.period_type,
            "admin_id": self.admin_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_mongo(self) -> Dict[str, Any]:
        doc = {
            "timetable_id": self.timetable_id,
            "grade": self.grade,
            "section": self.section,
            "day_of_week": self.day_of_week,
            "period_number": self.period_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "subject": self.subject,
            "teacher_id": self.teacher_id,
            "teacher_name": self.teacher_name,
            "room": self.room,
            "period_type": self.period_type,
            "admin_id": self.admin_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self._id:
            doc["_id"] = self._id
        return doc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimetablePeriod":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

        return cls(
            _id=data.get("_id"),
            timetable_id=data.get("timetable_id", ""),
            grade=data.get("grade", ""),
            section=data.get("section", ""),
            day_of_week=data.get("day_of_week", ""),
            period_number=data.get("period_number", 0),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            subject=data.get("subject", ""),
            teacher_id=data.get("teacher_id"),
            teacher_name=data.get("teacher_name"),
            room=data.get("room"),
            period_type=data.get("period_type", "regular"),
            admin_id=data.get("admin_id"),
            created_at=created_at,
            updated_at=updated_at,
        )


class AcademicCalendarEntry:
    """Holiday, exam, or special event that affects scheduling"""

    VALID_TYPES = ["holiday", "exam", "event", "half_day", "cancelled"]

    def __init__(
        self,
        date: str,
        entry_type: str,
        title: str,
        description: Optional[str] = None,
        affects_grades: Optional[List[str]] = None,
        affects_sections: Optional[List[str]] = None,
        is_recurring_yearly: bool = False,
        admin_id: Optional[str] = None,
        created_by: Optional[str] = None,
        created_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None,
    ):
        self._id = _id
        self.date = date  # "YYYY-MM-DD"
        self.entry_type = entry_type
        self.title = title
        self.description = description
        # Empty list = affects ALL grades
        self.affects_grades = affects_grades or []
        self.affects_sections = affects_sections or []
        self.is_recurring_yearly = is_recurring_yearly
        self.admin_id = admin_id
        self.created_by = created_by
        self.created_at = created_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": str(self._id) if self._id else None,
            "date": self.date,
            "entry_type": self.entry_type,
            "title": self.title,
            "description": self.description,
            "affects_grades": self.affects_grades,
            "affects_sections": self.affects_sections,
            "is_recurring_yearly": self.is_recurring_yearly,
            "admin_id": self.admin_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_mongo(self) -> Dict[str, Any]:
        doc = {
            "date": self.date,
            "entry_type": self.entry_type,
            "title": self.title,
            "description": self.description,
            "affects_grades": self.affects_grades,
            "affects_sections": self.affects_sections,
            "is_recurring_yearly": self.is_recurring_yearly,
            "admin_id": self.admin_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }
        if self._id:
            doc["_id"] = self._id
        return doc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AcademicCalendarEntry":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        return cls(
            _id=data.get("_id"),
            date=data.get("date", ""),
            entry_type=data.get("entry_type", "holiday"),
            title=data.get("title", ""),
            description=data.get("description"),
            affects_grades=data.get("affects_grades", []),
            affects_sections=data.get("affects_sections", []),
            is_recurring_yearly=data.get("is_recurring_yearly", False),
            admin_id=data.get("admin_id"),
            created_by=data.get("created_by"),
            created_at=created_at,
        )

    def affects_class(self, grade: str, section: str) -> bool:
        """Check if this calendar entry affects a specific grade+section"""
        grade_match = not self.affects_grades or grade in self.affects_grades
        section_match = not self.affects_sections or section in self.affects_sections
        return grade_match and section_match


class PreRead:
    """Pre-read material for a specific period on a specific date"""

    def __init__(
        self,
        timetable_id: str,
        period_id: str,
        grade: str,
        section: str,
        subject: str,
        date: str,
        day_of_week: str,
        period_number: int,
        heading: str,
        description: Optional[str] = None,
        book_page_numbers: Optional[str] = None,
        things_required: Optional[List[str]] = None,
        teacher_id: Optional[str] = None,
        teacher_name: Optional[str] = None,
        admin_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None,
    ):
        self._id = _id
        self.timetable_id = timetable_id
        self.period_id = period_id
        self.grade = grade
        self.section = section
        self.subject = subject
        self.date = date  # "YYYY-MM-DD" (specific date, not recurring)
        self.day_of_week = day_of_week
        self.period_number = period_number
        self.heading = heading  # Required
        self.description = description  # Optional, 2-3 lines
        self.book_page_numbers = book_page_numbers  # Optional
        # Optional: ["Fevicol", "Stapler", "Compass", "Scissor", "Art sheets", "Props"]
        self.things_required = things_required or []
        self.teacher_id = teacher_id
        self.teacher_name = teacher_name
        self.admin_id = admin_id
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": str(self._id) if self._id else None,
            "timetable_id": self.timetable_id,
            "period_id": self.period_id,
            "grade": self.grade,
            "section": self.section,
            "subject": self.subject,
            "date": self.date,
            "day_of_week": self.day_of_week,
            "period_number": self.period_number,
            "heading": self.heading,
            "description": self.description,
            "book_page_numbers": self.book_page_numbers,
            "things_required": self.things_required,
            "teacher_id": self.teacher_id,
            "teacher_name": self.teacher_name,
            "admin_id": self.admin_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_mongo(self) -> Dict[str, Any]:
        doc = {
            "timetable_id": self.timetable_id,
            "period_id": self.period_id,
            "grade": self.grade,
            "section": self.section,
            "subject": self.subject,
            "date": self.date,
            "day_of_week": self.day_of_week,
            "period_number": self.period_number,
            "heading": self.heading,
            "description": self.description,
            "book_page_numbers": self.book_page_numbers,
            "things_required": self.things_required,
            "teacher_id": self.teacher_id,
            "teacher_name": self.teacher_name,
            "admin_id": self.admin_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self._id:
            doc["_id"] = self._id
        return doc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreRead":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

        return cls(
            _id=data.get("_id"),
            timetable_id=data.get("timetable_id", ""),
            period_id=data.get("period_id", ""),
            grade=data.get("grade", ""),
            section=data.get("section", ""),
            subject=data.get("subject", ""),
            date=data.get("date", ""),
            day_of_week=data.get("day_of_week", ""),
            period_number=data.get("period_number", 0),
            heading=data.get("heading", ""),
            description=data.get("description"),
            book_page_numbers=data.get("book_page_numbers"),
            things_required=data.get("things_required", []),
            teacher_id=data.get("teacher_id"),
            teacher_name=data.get("teacher_name"),
            admin_id=data.get("admin_id"),
            created_at=created_at,
            updated_at=updated_at,
        )
