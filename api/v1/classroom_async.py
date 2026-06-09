"""
Classroom Management API - Online Class with Meet Links
Handles online class CRUD, student eligibility, and session management
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import secrets
import string

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class CreateOnlineClassRequest(BaseModel):
    """Request to create a new online class"""
    title: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = None
    subject: str = Field(..., description="Subject being taught")
    standard: str = Field(..., description="Class/grade level")
    section: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    meet_link: Optional[str] = None


class OnlineClassResponse(BaseModel):
    """Response model for online class"""
    id: str
    class_id: str
    title: str
    description: Optional[str] = None
    subject: str
    standard: str
    section: Optional[str] = None
    tutor_id: str
    tutor_name: Optional[str] = None
    meet_link: Optional[str] = None
    meet_code: Optional[str] = None  # Short code for students to join
    status: str  # scheduled, active, ended
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    student_count: int = 0
    created_at: datetime


class UpdateOnlineClassRequest(BaseModel):
    """Request to update an online class"""
    title: Optional[str] = None
    description: Optional[str] = None
    meet_link: Optional[str] = None
    scheduled_at: Optional[datetime] = None


# =============================================================================
# Helper Functions
# =============================================================================

def generate_meet_code(length: int = 6) -> str:
    """Generate a random alphanumeric code for students to join"""
    chars = string.ascii_uppercase + string.digits
    # Exclude confusing characters
    chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
    return ''.join(secrets.choice(chars) for _ in range(length))


async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


def require_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require tutor access"""
    if current_user.get("user_type") not in ["tutor", "admin", "b2c_admin"]:
        raise HTTPException(status_code=403, detail="Tutor access required")
    return current_user


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("", response_model=OnlineClassResponse, status_code=201)
async def create_online_class(
    request: Request,
    class_data: CreateOnlineClassRequest,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Create a new online class (Tutor only)
    Generates a unique meet_code for students to join
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    tutor_name = current_user.get("name") or current_user.get("username")
    
    # Generate unique class ID and meet code
    class_id = f"OC{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{secrets.token_hex(3).upper()}"
    meet_code = generate_meet_code()
    
    # Ensure meet_code is unique
    existing = await db.mongo_find_one("online_classes", {"meet_code": meet_code})
    while existing:
        meet_code = generate_meet_code()
        existing = await db.mongo_find_one("online_classes", {"meet_code": meet_code})
    
    new_class = {
        "class_id": class_id,
        "title": class_data.title,
        "description": class_data.description,
        "subject": class_data.subject,
        "standard": class_data.standard,
        "section": class_data.section,
        "tutor_id": tutor_id,
        "tutor_name": tutor_name,
        "meet_link": class_data.meet_link,
        "meet_code": meet_code,
        "status": "scheduled",
        "scheduled_at": class_data.scheduled_at,
        "started_at": None,
        "ended_at": None,
        "connected_students": [],
        "created_at": datetime.utcnow(),
    }
    
    result = await db.mongo_insert_one("online_classes", new_class)
    
    return OnlineClassResponse(
        id=str(result),
        class_id=class_id,
        title=class_data.title,
        description=class_data.description,
        subject=class_data.subject,
        standard=class_data.standard,
        section=class_data.section,
        tutor_id=tutor_id,
        tutor_name=tutor_name,
        meet_link=class_data.meet_link,
        meet_code=meet_code,
        status="scheduled",
        scheduled_at=class_data.scheduled_at,
        student_count=0,
        created_at=new_class["created_at"]
    )


@router.get("", response_model=List[OnlineClassResponse])
async def get_online_classes(
    request: Request,
    status: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all online classes for the current tutor
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    
    query = {"tutor_id": tutor_id}
    if status:
        query["status"] = status
    
    classes = await db.mongo_find("online_classes", query)
    
    return [
        OnlineClassResponse(
            id=str(c["_id"]),
            class_id=c.get("class_id"),
            title=c.get("title"),
            description=c.get("description"),
            subject=c.get("subject"),
            standard=c.get("standard"),
            section=c.get("section"),
            tutor_id=c.get("tutor_id"),
            tutor_name=c.get("tutor_name"),
            meet_link=c.get("meet_link"),
            meet_code=c.get("meet_code"),
            status=c.get("status", "scheduled"),
            scheduled_at=c.get("scheduled_at"),
            started_at=c.get("started_at"),
            ended_at=c.get("ended_at"),
            student_count=len(c.get("connected_students", [])),
            created_at=c.get("created_at")
        )
        for c in classes
    ]


@router.get("/{class_id}", response_model=OnlineClassResponse)
async def get_online_class(
    request: Request,
    class_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get a specific online class by ID
    """
    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")
    
    return OnlineClassResponse(
        id=str(online_class["_id"]),
        class_id=online_class.get("class_id"),
        title=online_class.get("title"),
        description=online_class.get("description"),
        subject=online_class.get("subject"),
        standard=online_class.get("standard"),
        section=online_class.get("section"),
        tutor_id=online_class.get("tutor_id"),
        tutor_name=online_class.get("tutor_name"),
        meet_link=online_class.get("meet_link"),
        meet_code=online_class.get("meet_code"),
        status=online_class.get("status", "scheduled"),
        scheduled_at=online_class.get("scheduled_at"),
        started_at=online_class.get("started_at"),
        ended_at=online_class.get("ended_at"),
        student_count=len(online_class.get("connected_students", [])),
        created_at=online_class.get("created_at")
    )


@router.post("/{class_id}/start")
async def start_online_class(
    request: Request,
    class_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Start an online class - sets status to 'active'
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    
    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")
    
    if online_class.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to start this class")
    
    if online_class.get("status") == "active":
        raise HTTPException(status_code=400, detail="Class is already active")
    
    await db.mongo_update_one(
        "online_classes",
        {"class_id": class_id},
        {"$set": {"status": "active", "started_at": datetime.utcnow()}}
    )
    
    return {"message": "Class started successfully", "meet_code": online_class.get("meet_code")}


@router.post("/{class_id}/end")
async def end_online_class(
    request: Request,
    class_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    End an online class - sets status to 'ended'
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    
    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")
    
    if online_class.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to end this class")
    
    await db.mongo_update_one(
        "online_classes",
        {"class_id": class_id},
        {"$set": {"status": "ended", "ended_at": datetime.utcnow()}}
    )
    
    return {"message": "Class ended successfully"}


@router.delete("/{class_id}")
async def delete_online_class(
    request: Request,
    class_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Delete a scheduled online class (cannot delete active classes)
    """
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    
    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")
    
    if online_class.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this class")
    
    if online_class.get("status") == "active":
        raise HTTPException(status_code=400, detail="Cannot delete an active class. End it first.")
    
    await db.mongo_delete_one("online_classes", {"class_id": class_id})
    
    return {"message": "Class deleted successfully"}


@router.get("/{class_id}/eligible-students")
async def get_eligible_students(
    request: Request,
    class_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get students eligible to join this class (based on standard/section)
    """
    online_class = await db.mongo_find_one("online_classes", {"class_id": class_id})
    if not online_class:
        raise HTTPException(status_code=404, detail="Online class not found")
    
    # Build query for eligible students
    query = {"grade": online_class.get("standard")}
    if online_class.get("section"):
        query["section"] = online_class.get("section")
    
    students = await db.mongo_find("students", query)
    
    return {
        "class_id": class_id,
        "eligible_count": len(students),
        "students": [
            {
                "student_id": s.get("student_id"),
                "name": s.get("name"),
                "grade": s.get("grade"),
                "section": s.get("section")
            }
            for s in students
        ]
    }


@router.get("/student/available")
async def get_available_classes_for_student(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get active classes available for the current student
    """
    if current_user.get("user_type") != "student":
        raise HTTPException(status_code=403, detail="Student access required")
    
    student_id = current_user.get("student_id")
    student = await db.mongo_find_one("students", {"student_id": student_id})
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Find active classes matching student's grade/section
    query = {
        "status": "active",
        "standard": student.get("grade")
    }
    
    # If student has a section, include it in the query
    if student.get("section"):
        query["$or"] = [
            {"section": student.get("section")},
            {"section": None}  # Classes open to all sections
        ]
    
    classes = await db.mongo_find("online_classes", query)
    
    return {
        "student_id": student_id,
        "available_classes": [
            {
                "class_id": c.get("class_id"),
                "title": c.get("title"),
                "subject": c.get("subject"),
                "tutor_name": c.get("tutor_name"),
                "meet_code": c.get("meet_code"),
                "meet_link": c.get("meet_link")
            }
            for c in classes
        ]
    }
