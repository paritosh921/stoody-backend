"""
Meeting Management API Endpoints

Handles Google Meet integration for online classes.
- Tutors can create, start, end, and cancel meetings
- Students can view scheduled meetings for their class/section/subject
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from models.meeting import Meeting
from services.google_meet import google_meet_service
from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# Helper dependency functions
def require_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require tutor access"""
    if current_user.get("user_type") != "tutor":
        raise HTTPException(status_code=403, detail="Tutor access required")
    return current_user


def require_student(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require student access"""
    if current_user.get("user_type") != "student":
        raise HTTPException(status_code=403, detail="Student access required")
    return current_user


# Pydantic Models
class CreateMeetingRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200, description="Meeting topic")
    subject: str = Field(..., description="Subject for the class")
    standard: str = Field(..., description="Class/grade (e.g., '10', '11')")
    section: Optional[str] = Field(None, description="Section (e.g., 'A', 'B')")
    course_type: Optional[str] = Field(None, description="Plan type (e.g., 'foundation', 'advanced')")
    scheduled_at: datetime = Field(..., description="Scheduled start time")
    duration_minutes: int = Field(60, ge=15, le=480, description="Duration in minutes")


class MeetingResponse(BaseModel):
    meeting_id: str
    tutor_id: str
    tutor_name: str
    topic: str
    subject: str
    standard: str
    section: Optional[str] = None
    course_type: Optional[str] = None
    scheduled_at: datetime
    duration_minutes: int
    meet_link: Optional[str] = None
    meet_code: Optional[str] = None
    status: str
    invited_student_count: int = 0
    joined_student_count: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class StudentMeetingResponse(BaseModel):
    """Meeting info visible to students"""
    meeting_id: str
    topic: str
    subject: str
    standard: str
    section: Optional[str] = None
    tutor_name: str
    scheduled_at: datetime
    duration_minutes: int
    meet_link: Optional[str] = None
    meet_code: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None


async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


@router.post("/meetings", response_model=MeetingResponse, status_code=201)
@limiter.limit("10/minute")
async def create_meeting(
    request: Request,
    meeting_data: CreateMeetingRequest,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Create a new online class meeting with Google Meet link.
    Only tutors can create meetings.
    """
    tutor_id = current_user.get("tutor_id")
    tutor_name = current_user.get("name") or current_user.get("username", "Tutor")

    # Get tutor info for admin_id
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    admin_id = tutor.get("created_by")

    # Create Google Meet link
    description = f"Online class: {meeting_data.topic}\nSubject: {meeting_data.subject}\nClass: {meeting_data.standard}"
    if meeting_data.section:
        description += f" Section: {meeting_data.section}"

    meet_link, meet_code, google_event_id = await google_meet_service.create_meeting(
        topic=f"[{meeting_data.subject}] {meeting_data.topic}",
        description=description,
        scheduled_at=meeting_data.scheduled_at,
        duration_minutes=meeting_data.duration_minutes,
    )

    # Generate meeting ID
    meeting_id = Meeting.generate_meeting_id()

    # Find students to invite based on criteria
    invited_student_ids = await _find_eligible_students(
        db=db,
        tutor_id=tutor_id,
        standard=meeting_data.standard,
        section=meeting_data.section,
        subject=meeting_data.subject,
        course_type=meeting_data.course_type,
        admin_id=admin_id,
    )

    # Create meeting document
    meeting_doc = {
        "meeting_id": meeting_id,
        "tutor_id": tutor_id,
        "tutor_name": tutor_name,
        "topic": meeting_data.topic,
        "subject": meeting_data.subject,
        "standard": meeting_data.standard,
        "section": meeting_data.section,
        "course_type": meeting_data.course_type,
        "scheduled_at": meeting_data.scheduled_at,
        "duration_minutes": meeting_data.duration_minutes,
        "meet_link": meet_link,
        "meet_code": meet_code,
        "google_event_id": google_event_id,
        "status": "scheduled",
        "invited_student_ids": invited_student_ids,
        "joined_student_ids": [],
        "admin_id": admin_id,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "ended_at": None,
    }

    # Insert into database
    await db.mongo_insert_one("meetings", meeting_doc)

    logger.info(f"Created meeting {meeting_id} with {len(invited_student_ids)} invited students")

    return MeetingResponse(
        meeting_id=meeting_id,
        tutor_id=tutor_id,
        tutor_name=tutor_name,
        topic=meeting_data.topic,
        subject=meeting_data.subject,
        standard=meeting_data.standard,
        section=meeting_data.section,
        course_type=meeting_data.course_type,
        scheduled_at=meeting_data.scheduled_at,
        duration_minutes=meeting_data.duration_minutes,
        meet_link=meet_link,
        meet_code=meet_code,
        status="scheduled",
        invited_student_count=len(invited_student_ids),
        joined_student_count=0,
        created_at=meeting_doc["created_at"],
        started_at=None,
        ended_at=None,
    )


async def _find_eligible_students(
    db: DatabaseManager,
    tutor_id: str,
    standard: str,
    section: Optional[str],
    subject: str,
    course_type: Optional[str],
    admin_id: Optional[str],
) -> List[str]:
    """Find students eligible for the meeting based on criteria"""
    from bson import ObjectId

    # Build query for students
    query_conditions = []

    # Match by admin (same organization)
    if admin_id:
        try:
            admin_oid = ObjectId(admin_id)
            query_conditions.append({"admin_id": admin_oid})
        except Exception:
            pass

    # Match by standard/grade
    if standard:
        query_conditions.append({
            "$or": [
                {"grade": standard},
                {"standard": standard},
            ]
        })

    # Match by section (if specified)
    if section:
        query_conditions.append({"section": section})

    # Match by subject (if student has subjects list)
    if subject:
        query_conditions.append({
            "$or": [
                {"subjects": subject},
                {"subjects": {"$exists": False}},  # Include students without subjects filter
            ]
        })

    # Match by course type / plan type
    if course_type:
        query_conditions.append({
            "$or": [
                {"plan_types": course_type},
                {"plan_type": course_type},
                {"course_type": course_type},
                {"plan_types": {"$exists": False}},  # Include students without plan filter
            ]
        })

    # Also include students directly assigned to tutor
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    assigned_ids = tutor.get("assigned_student_ids", []) if tutor else []

    # Build final query
    if query_conditions:
        query = {"$and": query_conditions, "is_active": {"$ne": False}}
    else:
        query = {"is_active": {"$ne": False}}

    # Find matching students
    students = await db.mongo_find("students", query)

    # Extract student IDs and add assigned students
    student_ids = set()
    for student in students:
        student_ids.add(student.get("student_id"))

    # Add directly assigned students
    for sid in assigned_ids:
        student_ids.add(sid)

    return list(student_ids)


@router.get("/meetings", response_model=List[MeetingResponse])
@limiter.limit("30/minute")
async def get_tutor_meetings(
    request: Request,
    status: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all meetings for the current tutor.
    """
    tutor_id = current_user.get("tutor_id")

    # Build query
    query = {"tutor_id": tutor_id}
    if status:
        query["status"] = status

    # Get meetings
    meetings = await db.mongo_find("meetings", query)
    meetings = sorted(meetings, key=lambda m: m.get("scheduled_at", datetime.min), reverse=True)

    return [
        MeetingResponse(
            meeting_id=m.get("meeting_id"),
            tutor_id=m.get("tutor_id"),
            tutor_name=m.get("tutor_name"),
            topic=m.get("topic"),
            subject=m.get("subject"),
            standard=m.get("standard"),
            section=m.get("section"),
            course_type=m.get("course_type"),
            scheduled_at=m.get("scheduled_at"),
            duration_minutes=m.get("duration_minutes", 60),
            meet_link=m.get("meet_link"),
            meet_code=m.get("meet_code"),
            status=m.get("status"),
            invited_student_count=len(m.get("invited_student_ids", [])),
            joined_student_count=len(m.get("joined_student_ids", [])),
            created_at=m.get("created_at"),
            started_at=m.get("started_at"),
            ended_at=m.get("ended_at"),
        )
        for m in meetings
    ]


@router.get("/meetings/student", response_model=List[StudentMeetingResponse])
@limiter.limit("30/minute")
async def get_student_meetings(
    request: Request,
    status: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all meetings the current student is invited to.
    """
    student_id = current_user.get("student_id")

    # Build query - find meetings where student is invited
    query = {"invited_student_ids": student_id}
    if status:
        query["status"] = status
    else:
        # By default, only show scheduled or active meetings
        query["status"] = {"$in": ["scheduled", "active"]}

    # Get meetings
    meetings = await db.mongo_find("meetings", query)
    meetings = sorted(meetings, key=lambda m: m.get("scheduled_at", datetime.min))

    return [
        StudentMeetingResponse(
            meeting_id=m.get("meeting_id"),
            topic=m.get("topic"),
            subject=m.get("subject"),
            standard=m.get("standard"),
            section=m.get("section"),
            tutor_name=m.get("tutor_name"),
            scheduled_at=m.get("scheduled_at"),
            duration_minutes=m.get("duration_minutes", 60),
            meet_link=m.get("meet_link") if m.get("status") == "active" else None,  # Only show link if active
            meet_code=m.get("meet_code") if m.get("status") == "active" else None,
            status=m.get("status"),
            started_at=m.get("started_at"),
        )
        for m in meetings
    ]


@router.get("/meetings/available")
@limiter.limit("60/minute")
async def get_available_meetings_for_student(
    request: Request,
    student_id: str,
    db: DatabaseManager = Depends(get_database)
):
    """
    Get available meetings for a specific student.
    Used by the desktop agent to fetch meetings.
    No authentication required - student_id is used as identifier.
    """
    # Find meetings where student is invited and meeting is scheduled or active
    query = {
        "invited_student_ids": student_id,
        "status": {"$in": ["scheduled", "active"]},
    }

    # Get meetings
    meetings = await db.mongo_find("meetings", query)
    meetings = sorted(meetings, key=lambda m: m.get("scheduled_at", datetime.min))

    return [
        {
            "class_id": m.get("meeting_id"),
            "topic": m.get("topic"),
            "subject": m.get("subject"),
            "standard": m.get("standard"),
            "section": m.get("section"),
            "tutor_name": m.get("tutor_name"),
            "meet_link": m.get("meet_link") if m.get("status") == "active" else None,
            "meet_code": m.get("meet_code"),
            "status": m.get("status"),
            "student_count": len(m.get("invited_student_ids", [])),
            "started_at": m.get("started_at").isoformat() if m.get("started_at") else None,
        }
        for m in meetings
    ]


@router.post("/meetings/{meeting_id}/start")
@limiter.limit("10/minute")
async def start_meeting(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Start a scheduled meeting. Updates status to 'active'.
    """
    tutor_id = current_user.get("tutor_id")

    # Find meeting
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Verify tutor owns this meeting
    if meeting.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to start this meeting")

    if meeting.get("status") != "scheduled":
        raise HTTPException(status_code=400, detail=f"Meeting is already {meeting.get('status')}")

    # Update meeting status
    await db.mongo_update_one(
        "meetings",
        {"meeting_id": meeting_id},
        {"$set": {"status": "active", "started_at": datetime.utcnow()}}
    )

    logger.info(f"Meeting {meeting_id} started by tutor {tutor_id}")

    return {
        "message": "Meeting started",
        "meeting_id": meeting_id,
        "meet_link": meeting.get("meet_link"),
        "meet_code": meeting.get("meet_code"),
    }


@router.post("/meetings/{meeting_id}/end")
@limiter.limit("10/minute")
async def end_meeting(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    End an active meeting. Updates status to 'ended'.
    """
    tutor_id = current_user.get("tutor_id")

    # Find meeting
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Verify tutor owns this meeting
    if meeting.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to end this meeting")

    if meeting.get("status") == "ended":
        raise HTTPException(status_code=400, detail="Meeting is already ended")

    # Update meeting status
    await db.mongo_update_one(
        "meetings",
        {"meeting_id": meeting_id},
        {"$set": {"status": "ended", "ended_at": datetime.utcnow()}}
    )

    logger.info(f"Meeting {meeting_id} ended by tutor {tutor_id}")

    return {"message": "Meeting ended", "meeting_id": meeting_id}


@router.delete("/meetings/{meeting_id}")
@limiter.limit("10/minute")
async def cancel_meeting(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Cancel a scheduled meeting. Also cancels the Google Calendar event.
    """
    tutor_id = current_user.get("tutor_id")

    # Find meeting
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Verify tutor owns this meeting
    if meeting.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to cancel this meeting")

    if meeting.get("status") not in ["scheduled", "active"]:
        raise HTTPException(status_code=400, detail="Cannot cancel an ended meeting")

    # Cancel Google Calendar event
    google_event_id = meeting.get("google_event_id")
    if google_event_id:
        await google_meet_service.cancel_meeting(google_event_id)

    # Update meeting status
    await db.mongo_update_one(
        "meetings",
        {"meeting_id": meeting_id},
        {"$set": {"status": "cancelled", "ended_at": datetime.utcnow()}}
    )

    logger.info(f"Meeting {meeting_id} cancelled by tutor {tutor_id}")

    return {"message": "Meeting cancelled", "meeting_id": meeting_id}


@router.post("/meetings/{meeting_id}/join")
@limiter.limit("30/minute")
async def student_join_meeting(
    request: Request,
    meeting_id: str,
    student_id: str,
    db: DatabaseManager = Depends(get_database)
):
    """
    Record that a student joined a meeting.
    Used by the desktop agent when student clicks join.
    """
    # Find meeting
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    if meeting.get("status") != "active":
        raise HTTPException(status_code=400, detail="Meeting is not active")

    # Verify student was invited
    if student_id not in meeting.get("invited_student_ids", []):
        raise HTTPException(status_code=403, detail="Student not invited to this meeting")

    # Add student to joined list
    await db.mongo_update_one(
        "meetings",
        {"meeting_id": meeting_id},
        {"$addToSet": {"joined_student_ids": student_id}}
    )

    logger.info(f"Student {student_id} joined meeting {meeting_id}")

    return {
        "message": "Joined meeting",
        "meeting_id": meeting_id,
        "meet_link": meeting.get("meet_link"),
        "meet_code": meeting.get("meet_code"),
    }
