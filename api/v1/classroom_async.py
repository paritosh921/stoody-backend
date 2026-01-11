"""
Classroom API for Online Class Management

Handles:
- Online class CRUD with Meet links
- Student eligibility based on class mappings
- Integration with WebSocket sessions
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, validator

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Auth Dependencies
# =============================================================================

def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access (regular or B2C)"""
    if current_user.get("user_type") not in ["admin", "b2c_admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def require_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require tutor access"""
    if current_user.get("user_type") != "tutor":
        raise HTTPException(status_code=403, detail="Tutor access required")
    return current_user


def require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin, B2C admin, OR tutor access"""
    if current_user.get("user_type") not in ["admin", "b2c_admin", "tutor"]:
        raise HTTPException(status_code=403, detail="Admin or Tutor access required")
    return current_user


# =============================================================================
# Request/Response Models
# =============================================================================

class ClassMapping(BaseModel):
    """Student class mapping"""
    standard: str
    section: str
    subject: str
    course_plan: Optional[str] = None  # CBSE, JEE, NEET, etc.


class CreateOnlineClassRequest(BaseModel):
    """Request to create an online class"""
    meet_link: str = Field(..., description="Google Meet or other video call link")
    topic: str = Field(..., min_length=1, max_length=200)
    standard: str
    section: str
    subject: str
    course_plan: Optional[str] = None
    scheduled_at: Optional[datetime] = None

    @validator('meet_link')
    def validate_meet_link(cls, v):
        if not v:
            raise ValueError('Meet link is required')
        # Basic URL validation
        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError('Invalid URL format')
        except Exception:
            raise ValueError('Invalid meet link URL')
        return v


class UpdateOnlineClassRequest(BaseModel):
    """Request to update an online class"""
    meet_link: Optional[str] = None
    topic: Optional[str] = None
    scheduled_at: Optional[datetime] = None


class OnlineClassResponse(BaseModel):
    """Response model for online class"""
    class_id: str
    tutor_id: str
    tutor_name: Optional[str] = None
    meet_link: str
    meet_code: str
    topic: str
    standard: str
    section: str
    subject: str
    course_plan: Optional[str] = None
    status: str  # scheduled, active, ended
    enrolled_student_count: int = 0
    attended_student_count: int = 0
    created_via: str = "manual"
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_at: datetime


class EligibleStudentResponse(BaseModel):
    """Student eligible for a class"""
    student_id: str
    name: str
    username: Optional[str] = None
    grade: Optional[str] = None


class StudentAvailableClassResponse(BaseModel):
    """Class available for a student to join"""
    class_id: str
    topic: str
    subject: str
    standard: str
    section: str
    tutor_name: str
    meet_link: str
    meet_code: str
    status: str
    student_count: int
    started_at: Optional[datetime] = None


# =============================================================================
# Helper Functions
# =============================================================================

def get_db(request: Request) -> DatabaseManager:
    """Get database manager from app state"""
    return request.app.state.db


def extract_meet_code(meet_link: str) -> str:
    """Extract meeting code from Google Meet link or generate one"""
    # Try to extract from Google Meet URL: https://meet.google.com/xxx-yyyy-zzz
    if "meet.google.com" in meet_link:
        parsed = urlparse(meet_link)
        path = parsed.path.strip("/")
        if path and len(path) >= 10:
            return path

    # For other platforms, generate a unique code
    return str(uuid.uuid4())[:12].upper()


async def get_enrolled_students(
    db: DatabaseManager,
    standard: str,
    section: str,
    subject: str,
    course_plan: Optional[str] = None
) -> List[str]:
    """Get list of student IDs enrolled in a specific class"""
    query = {
        "class_mappings": {
            "$elemMatch": {
                "standard": standard,
                "section": section,
                "subject": subject
            }
        },
        "is_active": True
    }

    # Add course_plan filter if specified
    if course_plan:
        query["class_mappings"]["$elemMatch"]["course_plan"] = course_plan

    students = await db.mongo_find("students", query)
    return [s["student_id"] for s in students]


# =============================================================================
# Online Class Endpoints
# =============================================================================

@router.post("/online-class", response_model=OnlineClassResponse)
async def create_online_class(
    request_data: CreateOnlineClassRequest,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """Create a new online class with Meet link"""
    db = get_db(request)

    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    tutor_name = current_user.get("name") or current_user.get("full_name", "Teacher")

    # Extract or generate meet code
    meet_code = extract_meet_code(request_data.meet_link)

    # Check if meet_code already exists and is active
    existing = await db.mongo_find_one("online_classes", {
        "meet_code": meet_code,
        "status": {"$in": ["scheduled", "active"]}
    })
    if existing:
        raise HTTPException(
            status_code=400,
            detail="A class with this meeting link is already scheduled or active"
        )

    # Get enrolled students based on class filter
    enrolled_students = await get_enrolled_students(
        db,
        request_data.standard,
        request_data.section,
        request_data.subject,
        request_data.course_plan
    )

    # Generate unique class ID
    class_id = f"OC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    # Create the online class document
    online_class = {
        "class_id": class_id,
        "tutor_id": tutor_id,
        "tutor_name": tutor_name,
        "meet_link": request_data.meet_link,
        "meet_code": meet_code,
        "topic": request_data.topic,
        "standard": request_data.standard,
        "section": request_data.section,
        "subject": request_data.subject,
        "course_plan": request_data.course_plan,
        "status": "scheduled",
        "enrolled_students": enrolled_students,
        "attended_students": [],
        "created_via": "manual",
        "scheduled_at": request_data.scheduled_at,
        "started_at": None,
        "ended_at": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    await db.mongo_insert_one("online_classes", online_class)

    logger.info(f"Created online class {class_id} by tutor {tutor_id}")

    return OnlineClassResponse(
        class_id=class_id,
        tutor_id=tutor_id,
        tutor_name=tutor_name,
        meet_link=request_data.meet_link,
        meet_code=meet_code,
        topic=request_data.topic,
        standard=request_data.standard,
        section=request_data.section,
        subject=request_data.subject,
        course_plan=request_data.course_plan,
        status="scheduled",
        enrolled_student_count=len(enrolled_students),
        attended_student_count=0,
        created_via="manual",
        scheduled_at=request_data.scheduled_at,
        created_at=online_class["created_at"]
    )


@router.get("/online-classes", response_model=List[OnlineClassResponse])
async def get_online_classes(
    request: Request,
    status: Optional[str] = None,
    standard: Optional[str] = None,
    subject: Optional[str] = None,
    current_user: dict = Depends(require_tutor)
):
    """Get all online classes for the current tutor"""
    db = get_db(request)

    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    query = {"tutor_id": tutor_id}

    if status:
        query["status"] = status
    if standard:
        query["standard"] = standard
    if subject:
        query["subject"] = subject

    classes = await db.mongo_find(
        "online_classes",
        query,
        sort=[("created_at", -1)],
        limit=100
    )

    return [
        OnlineClassResponse(
            class_id=c["class_id"],
            tutor_id=c["tutor_id"],
            tutor_name=c.get("tutor_name"),
            meet_link=c["meet_link"],
            meet_code=c["meet_code"],
            topic=c["topic"],
            standard=c["standard"],
            section=c["section"],
            subject=c["subject"],
            course_plan=c.get("course_plan"),
            status=c["status"],
            enrolled_student_count=len(c.get("enrolled_students", [])),
            attended_student_count=len(c.get("attended_students", [])),
            created_via=c.get("created_via", "manual"),
            scheduled_at=c.get("scheduled_at"),
            started_at=c.get("started_at"),
            ended_at=c.get("ended_at"),
            created_at=c["created_at"]
        )
        for c in classes
    ]


@router.get("/online-class/{class_id}", response_model=OnlineClassResponse)
async def get_online_class(
    class_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get details of a specific online class"""
    db = get_db(request)

    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")

    return OnlineClassResponse(
        class_id=online_class["class_id"],
        tutor_id=online_class["tutor_id"],
        tutor_name=online_class.get("tutor_name"),
        meet_link=online_class["meet_link"],
        meet_code=online_class["meet_code"],
        topic=online_class["topic"],
        standard=online_class["standard"],
        section=online_class["section"],
        subject=online_class["subject"],
        course_plan=online_class.get("course_plan"),
        status=online_class["status"],
        enrolled_student_count=len(online_class.get("enrolled_students", [])),
        attended_student_count=len(online_class.get("attended_students", [])),
        created_via=online_class.get("created_via", "manual"),
        scheduled_at=online_class.get("scheduled_at"),
        started_at=online_class.get("started_at"),
        ended_at=online_class.get("ended_at"),
        created_at=online_class["created_at"]
    )


@router.put("/online-class/{class_id}/start")
async def start_online_class(
    class_id: str,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """Mark an online class as active (started)"""
    db = get_db(request)

    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")

    if online_class["tutor_id"] != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this class")

    if online_class["status"] == "ended":
        raise HTTPException(status_code=400, detail="Cannot start an ended class")

    await db.mongo_update_one(
        "online_classes",
        {"class_id": class_id},
        {
            "$set": {
                "status": "active",
                "started_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        }
    )

    logger.info(f"Started online class {class_id}")

    return {"status": "active", "started_at": datetime.utcnow().isoformat()}


@router.put("/online-class/{class_id}/end")
async def end_online_class(
    class_id: str,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """Mark an online class as ended"""
    db = get_db(request)

    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")

    if online_class["tutor_id"] != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this class")

    await db.mongo_update_one(
        "online_classes",
        {"class_id": class_id},
        {
            "$set": {
                "status": "ended",
                "ended_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        }
    )

    logger.info(f"Ended online class {class_id}")

    return {"status": "ended", "ended_at": datetime.utcnow().isoformat()}


@router.delete("/online-class/{class_id}")
async def delete_online_class(
    class_id: str,
    request: Request,
    current_user: dict = Depends(require_tutor)
):
    """Delete a scheduled online class (cannot delete active/ended)"""
    db = get_db(request)

    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")

    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")

    if online_class["tutor_id"] != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this class")

    if online_class["status"] != "scheduled":
        raise HTTPException(
            status_code=400,
            detail="Can only delete scheduled classes. End the class first."
        )

    await db.mongo_delete_one("online_classes", {"class_id": class_id})

    logger.info(f"Deleted online class {class_id}")

    return {"status": "deleted"}


# =============================================================================
# Student Eligibility Endpoints
# =============================================================================

@router.get("/eligible-students", response_model=List[EligibleStudentResponse])
async def get_eligible_students(
    request: Request,
    standard: str,
    section: str,
    subject: str,
    course_plan: Optional[str] = None,
    current_user: dict = Depends(require_tutor)
):
    """Get students eligible for a specific class configuration"""
    db = get_db(request)

    query = {
        "class_mappings": {
            "$elemMatch": {
                "standard": standard,
                "section": section,
                "subject": subject
            }
        },
        "is_active": True
    }

    if course_plan:
        query["class_mappings"]["$elemMatch"]["course_plan"] = course_plan

    students = await db.mongo_find("students", query)

    return [
        EligibleStudentResponse(
            student_id=s["student_id"],
            name=s.get("name", "Unknown"),
            username=s.get("username"),
            grade=s.get("grade")
        )
        for s in students
    ]


@router.post("/student-mapping/{student_id}")
async def update_student_class_mapping(
    student_id: str,
    mappings: List[ClassMapping],
    request: Request,
    current_user: dict = Depends(require_admin)
):
    """Update class mappings for a student (admin only)"""
    db = get_db(request)

    student = await db.mongo_find_one("students", {"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Convert to dict list
    mapping_dicts = [m.dict() for m in mappings]

    await db.mongo_update_one(
        "students",
        {"student_id": student_id},
        {"$set": {"class_mappings": mapping_dicts}}
    )

    logger.info(f"Updated class mappings for student {student_id}")

    return {"status": "updated", "mappings_count": len(mapping_dicts)}


@router.get("/student-mapping/{student_id}", response_model=List[ClassMapping])
async def get_student_class_mapping(
    student_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get class mappings for a student"""
    db = get_db(request)

    student = await db.mongo_find_one("students", {"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    mappings = student.get("class_mappings", [])
    return [ClassMapping(**m) for m in mappings]


# =============================================================================
# Student-facing Endpoints (for Desktop Client)
# =============================================================================

@router.get("/available-classes/{student_id}", response_model=List[StudentAvailableClassResponse])
async def get_available_classes_for_student(
    student_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Get active/scheduled online classes available for a student.
    Used by the desktop client to show available meetings.
    """
    db = get_db(request)

    # Verify student exists and get their class mappings
    student = await db.mongo_find_one("students", {"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    class_mappings = student.get("class_mappings", [])

    if not class_mappings:
        # Student has no class mappings, return empty list
        return []

    # Build query for online classes that match any of the student's mappings
    mapping_conditions = []
    for m in class_mappings:
        condition = {
            "standard": m.get("standard"),
            "section": m.get("section"),
            "subject": m.get("subject")
        }
        # Add course_plan if specified
        if m.get("course_plan"):
            condition["course_plan"] = m.get("course_plan")
        mapping_conditions.append(condition)

    # Find active or scheduled classes that match student's mappings
    query = {
        "$or": mapping_conditions,
        "status": {"$in": ["scheduled", "active"]}
    }

    classes = await db.mongo_find(
        "online_classes",
        query,
        sort=[("status", 1), ("scheduled_at", 1)]  # Active first, then by scheduled time
    )

    return [
        StudentAvailableClassResponse(
            class_id=c["class_id"],
            topic=c["topic"],
            subject=c["subject"],
            standard=c["standard"],
            section=c["section"],
            tutor_name=c.get("tutor_name", "Teacher"),
            meet_link=c["meet_link"],
            meet_code=c["meet_code"],
            status=c["status"],
            student_count=len(c.get("enrolled_students", [])),
            started_at=c.get("started_at")
        )
        for c in classes
    ]


@router.post("/join-class/{class_id}")
async def record_student_joining_class(
    class_id: str,
    student_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Record that a student is joining a class.
    Called when student clicks to join a meeting from desktop client.
    """
    db = get_db(request)

    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")

    # Check if student is enrolled
    if student_id not in online_class.get("enrolled_students", []):
        raise HTTPException(
            status_code=403,
            detail="Student is not enrolled in this class"
        )

    # Add to attended students if not already there
    if student_id not in online_class.get("attended_students", []):
        await db.mongo_update_one(
            "online_classes",
            {"class_id": class_id},
            {
                "$addToSet": {"attended_students": student_id},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

    logger.info(f"Student {student_id} joined class {class_id}")

    return {
        "status": "joined",
        "meet_link": online_class["meet_link"],
        "meet_code": online_class["meet_code"]
    }
