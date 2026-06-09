"""
Meeting Management API Endpoints

Handles Jitsi integration for online classes.
- Tutors can create, start, end, and cancel meetings
- Students can view scheduled meetings for their class/section/subject
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from models.meeting import Meeting
from services.online_class import jitsi_provider_service
from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from slowapi import Limiter
from slowapi.util import get_remote_address
from bson import ObjectId as BsonObjectId

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


async def resolve_business_student_id(
    current_user: Dict[str, Any], db: DatabaseManager
) -> Optional[str]:
    if current_user.get("user_type") != "student":
        return None
    raw = current_user.get("student_id") or current_user.get("user_id")
    if not raw:
        return None
    raw = str(raw)
    if raw.startswith("STU_"):
        return raw
    try:
        oid = BsonObjectId(raw)
    except Exception:
        return raw
    student = await db.mongo_find_one("students", {"_id": oid})
    if student and student.get("student_id"):
        return student["student_id"]
    return raw


async def resolve_notification_recipient_ids(
    db: DatabaseManager, invited_student_ids: List[str]
) -> List[str]:
    if not invited_student_ids:
        return []
    students = await db.mongo_find(
        "students", {"student_id": {"$in": invited_student_ids}}
    )
    oid_map: Dict[str, str] = {}
    for s in students:
        sid = s.get("student_id")
        if sid and sid in invited_student_ids:
            oid_map[sid] = str(s["_id"])
    return [oid_map.get(sid, sid) for sid in invited_student_ids]


# Pydantic Models
class CreateMeetingRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200, description="Meeting topic")
    subject: str = Field(..., description="Subject for the class")
    standard: str = Field(..., description="Class/grade (e.g., '10', '11')")
    section: Optional[str] = Field(None, description="Section (e.g., 'A', 'B')")
    course_type: Optional[str] = Field(None, description="Plan type (e.g., 'foundation', 'advanced')")
    scheduled_at: datetime = Field(..., description="Scheduled start time")
    duration_minutes: int = Field(60, ge=15, le=480, description="Duration in minutes")


class ProviderDetails(BaseModel):
    provider: Optional[str] = None
    domain: Optional[str] = None
    room_name: Optional[str] = None
    url: Optional[str] = None
    token_required: bool = False
    token: Optional[str] = None
    configured: bool = False


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
    provider_details: Optional[ProviderDetails] = None


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
    provider_details: Optional[ProviderDetails] = None


async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


def _build_provider_details(
    meeting_id: str,
    current_user: Optional[Dict[str, Any]] = None,
    moderator: bool = False,
) -> ProviderDetails:
    if jitsi_provider_service.configured:
        user_id = ""
        user_name = ""
        user_email = ""
        if current_user:
            user_id = current_user.get("user_id") or current_user.get("tutor_id") or current_user.get("student_id") or ""
            user_name = current_user.get("name") or current_user.get("username") or ""
            user_email = current_user.get("email") or ""
        details = jitsi_provider_service.get_provider_details(
            meeting_id=meeting_id,
            user_id=user_id,
            user_name=user_name,
            user_email=user_email,
            moderator=moderator,
        )
        return ProviderDetails(**details)
    return ProviderDetails(provider=None, configured=False)


def _require_provider_details(
    meeting_id: str,
    current_user: Optional[Dict[str, Any]],
    moderator: bool,
) -> ProviderDetails:
    provider = _build_provider_details(
        meeting_id=meeting_id,
        current_user=current_user,
        moderator=moderator,
    )
    if provider.configured:
        return provider
    raise HTTPException(
        status_code=503,
        detail="Online class video provider is not configured",
    )


def _provider_or_none(
    meeting_id: str,
    current_user: Optional[Dict[str, Any]],
    moderator: bool,
) -> Optional[ProviderDetails]:
    provider = _build_provider_details(
        meeting_id=meeting_id,
        current_user=current_user,
        moderator=moderator,
    )
    return provider if provider.configured else None


def _provider_video_fields(provider: Optional[ProviderDetails]) -> tuple[Optional[str], Optional[str]]:
    if provider and provider.configured:
        return provider.url, provider.room_name
    return None, None


def _public_video_fields(meeting_id: str) -> tuple[Optional[str], Optional[str]]:
    if not jitsi_provider_service.configured:
        return None, None
    room_name = jitsi_provider_service.generate_room_name(meeting_id)
    return jitsi_provider_service.get_room_url(room_name), room_name


@router.post("/meetings", response_model=MeetingResponse, status_code=201)
@limiter.limit("10/minute")
async def create_meeting(
    request: Request,
    meeting_data: CreateMeetingRequest,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Create a new online class meeting backed by Jitsi.
    Only tutors can create meetings.
    """
    tutor_id = current_user.get("tutor_id")
    tutor_name = current_user.get("name") or current_user.get("username", "Tutor")

    # Get tutor info for admin_id
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    admin_id = tutor.get("created_by")

    meeting_id = Meeting.generate_meeting_id()
    provider = _require_provider_details(meeting_id, current_user=current_user, moderator=True)
    meet_link, meet_code = _provider_video_fields(provider)

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
        "provider": "jitsi",
        "jitsi_room_name": provider.room_name,
        "jitsi_url": provider.url,
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

    if invited_student_ids:
        try:
            from api.v1.notifications_async import create_notifications_batch
            notif_recipient_ids = await resolve_notification_recipient_ids(
                db, invited_student_ids
            )
            await create_notifications_batch(
                db=db,
                admin_id=admin_id,
                recipient_ids=notif_recipient_ids,
                notif_type="info",
                category="online_class",
                title="Online Class Scheduled",
                message=f"{tutor_name} scheduled '{meeting_data.topic}' for {meeting_data.subject}",
                metadata={
                    "meeting_id": meeting_id,
                    "topic": meeting_data.topic,
                    "subject": meeting_data.subject,
                    "standard": meeting_data.standard,
                    "section": meeting_data.section,
                    "scheduled_at": meeting_data.scheduled_at.isoformat() if meeting_data.scheduled_at else None,
                    "status": "scheduled",
                },
                created_by=tutor_id,
                created_by_name=tutor_name,
            )
        except Exception as e:
            logger.warning(f"Failed to send online-class creation notifications: {e}")

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
        provider_details=provider,
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

    responses: List[MeetingResponse] = []
    for m in meetings:
        provider = _provider_or_none(
            m.get("meeting_id"),
            current_user=current_user,
            moderator=True,
        )
        meet_link, meet_code = _provider_video_fields(provider)
        responses.append(
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
            meet_link=meet_link,
            meet_code=meet_code,
            status=m.get("status"),
            invited_student_count=len(m.get("invited_student_ids", [])),
            joined_student_count=len(m.get("joined_student_ids", [])),
            created_at=m.get("created_at"),
            started_at=m.get("started_at"),
            ended_at=m.get("ended_at"),
            provider_details=provider,
            )
        )
    return responses


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
    student_id = await resolve_business_student_id(current_user, db)
    if not student_id:
        raise HTTPException(status_code=403, detail="Could not resolve student identity")

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

    responses: List[StudentMeetingResponse] = []
    for m in meetings:
        provider = (
            _provider_or_none(
                m.get("meeting_id"),
                current_user=current_user,
                moderator=False,
            )
            if m.get("status") == "active"
            else None
        )
        meet_link, meet_code = _provider_video_fields(provider)
        responses.append(
            StudentMeetingResponse(
            meeting_id=m.get("meeting_id"),
            topic=m.get("topic"),
            subject=m.get("subject"),
            standard=m.get("standard"),
            section=m.get("section"),
            tutor_name=m.get("tutor_name"),
            scheduled_at=m.get("scheduled_at"),
            duration_minutes=m.get("duration_minutes", 60),
            meet_link=meet_link,
            meet_code=meet_code,
            status=m.get("status"),
            started_at=m.get("started_at"),
            provider_details=provider,
            )
        )
    return responses


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

    response = []
    for m in meetings:
        meet_link, meet_code = (
            _public_video_fields(m.get("meeting_id"))
            if m.get("status") == "active"
            else (None, None)
        )
        response.append({
            "class_id": m.get("meeting_id"),
            "topic": m.get("topic"),
            "subject": m.get("subject"),
            "standard": m.get("standard"),
            "section": m.get("section"),
            "tutor_name": m.get("tutor_name"),
            "meet_link": meet_link,
            "meet_code": meet_code,
            "provider": "jitsi" if meet_link else None,
            "requires_authenticated_join": bool(meet_link),
            "status": m.get("status"),
            "student_count": len(m.get("invited_student_ids", [])),
            "started_at": m.get("started_at").isoformat() if m.get("started_at") else None,
        })
    return response


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

    provider = _require_provider_details(meeting_id, current_user=current_user, moderator=True)
    meet_link, meet_code = _provider_video_fields(provider)

    # Update meeting status
    await db.mongo_update_one(
        "meetings",
        {"meeting_id": meeting_id},
        {
            "$set": {
                "status": "active",
                "started_at": datetime.utcnow(),
                "provider": "jitsi",
                "jitsi_room_name": provider.room_name,
                "jitsi_url": provider.url,
                "meet_link": meet_link,
                "meet_code": meet_code,
            }
        }
    )

    logger.info(f"Meeting {meeting_id} started by tutor {tutor_id}")

    invited = meeting.get("invited_student_ids", [])
    if invited:
        try:
            from api.v1.notifications_async import create_notifications_batch
            notif_recipient_ids = await resolve_notification_recipient_ids(
                db, invited
            )
            await create_notifications_batch(
                db=db,
                admin_id=meeting.get("admin_id"),
                recipient_ids=notif_recipient_ids,
                notif_type="info",
                category="online_class",
                title="Online Class is Live!",
                message=f"'{meeting.get('topic', 'Class')}' has started. Join now!",
                metadata={
                    "meeting_id": meeting_id,
                    "topic": meeting.get("topic"),
                    "subject": meeting.get("subject"),
                    "standard": meeting.get("standard"),
                    "section": meeting.get("section"),
                    "scheduled_at": meeting.get("scheduled_at").isoformat() if meeting.get("scheduled_at") else None,
                    "status": "active",
                },
                created_by=tutor_id,
                created_by_name=meeting.get("tutor_name", "Tutor"),
            )
        except Exception as e:
            logger.warning(f"Failed to send online-class live notifications: {e}")

    return {
        "message": "Meeting started",
        "meeting_id": meeting_id,
        "meet_link": meet_link,
        "meet_code": meet_code,
        "provider_details": provider.dict(),
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
    Cancel a scheduled or active meeting.
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

    meet_link, meet_code = _public_video_fields(meeting_id)
    return {
        "message": "Joined meeting",
        "meeting_id": meeting_id,
        "meet_link": meet_link,
        "meet_code": meet_code,
        "provider": "jitsi" if meet_link else None,
        "requires_authenticated_join": bool(meet_link),
    }


@router.post("/meetings/{meeting_id}/join-auth")
@limiter.limit("30/minute")
async def student_join_meeting_auth(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    if meeting.get("status") != "active":
        raise HTTPException(status_code=400, detail="Meeting is not active")

    user_type = current_user.get("user_type")
    moderator = False
    if user_type == "tutor":
        tutor_id = current_user.get("tutor_id")
        if meeting.get("tutor_id") != tutor_id:
            raise HTTPException(status_code=403, detail="Not authorized to join this meeting")
        moderator = True
    elif user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        if not student_id:
            raise HTTPException(status_code=403, detail="Could not resolve student identity")
        if student_id not in meeting.get("invited_student_ids", []):
            raise HTTPException(status_code=403, detail="Student not invited to this meeting")

        await db.mongo_update_one(
            "meetings",
            {"meeting_id": meeting_id},
            {"$addToSet": {"joined_student_ids": student_id}}
        )
        logger.info(f"Student {student_id} joined meeting {meeting_id} via auth endpoint")
    else:
        raise HTTPException(status_code=403, detail="Online class access denied")

    provider = _require_provider_details(meeting_id, current_user=current_user, moderator=moderator)
    meet_link, meet_code = _provider_video_fields(provider)

    return {
        "message": "Joined meeting",
        "meeting_id": meeting_id,
        "meet_link": meet_link,
        "meet_code": meet_code,
        "provider_details": provider.dict(),
    }
