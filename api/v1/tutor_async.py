"""
Tutor Management API Endpoints (Async)
Handles tutor CRUD operations, authentication, and student assignments
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, EmailStr
import bcrypt

from models.tutor import Tutor, TutorSchema, TutorUpdateSchema, TutorPasswordChangeSchema
from models.student import Student
from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# Helper dependency functions
def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access"""
    if current_user.get("user_type") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user


def require_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require tutor access"""
    if current_user.get("user_type") != "tutor":
        raise HTTPException(
            status_code=403,
            detail="Tutor access required"
        )
    return current_user


def require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin OR tutor access"""
    if current_user.get("user_type") not in ["admin", "tutor"]:
        raise HTTPException(
            status_code=403,
            detail="Admin or Tutor access required"
        )
    return current_user


# Pydantic Models for Request/Response
class CreateTutorRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    standards: Optional[List[str]] = None  # Multiple standards
    sections: Optional[List[str]] = None  # Multiple sections (A-F)
    subjects: Optional[List[str]] = None  # Multiple subjects
    plan_types: Optional[List[str]] = None  # Multiple plan types
    can_edit_students: bool = False  # Permission to add/edit students


class TutorResponse(BaseModel):
    id: str
    tutor_id: str
    username: str
    full_name: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    standards: Optional[List[str]] = None
    sections: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    plan_types: Optional[List[str]] = None
    can_edit_students: bool
    is_active: bool
    assigned_student_ids: Optional[List[str]] = None
    requires_password_change: Optional[bool] = None
    password_reset_requested: Optional[bool] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    generated_password: Optional[str] = None  # Only included on creation


class AssignStudentRequest(BaseModel):
    student_id: str = Field(..., description="Student ID to assign to tutor")


class TutorLoginRequest(BaseModel):
    username: str
    password: str


# Helper function to get database
async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


@router.post("/tutors", response_model=TutorResponse, status_code=201)
@limiter.limit("10/minute")
async def create_tutor(
    request: Request,
    tutor_data: CreateTutorRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Create a new tutor account (Admin only)
    Auto-generates tutor_id and password
    """
    # Check if username already exists
    existing_tutor = await db.mongo_find_one("tutors", {"username": tutor_data.username})
    if existing_tutor:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Generate tutor ID and password
    auto_tutor_id = Tutor.generate_tutor_id()
    generated_password = Tutor.generate_password()
    password_hash = bcrypt.hashpw(generated_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Get admin ID
    admin_id = current_user.get("user_id")

    # Create tutor document
    new_tutor = {
        "tutor_id": auto_tutor_id,
        "name": tutor_data.full_name,
        "username": tutor_data.username,
        "password_hash": password_hash,
        "email": tutor_data.email,
        "phone": tutor_data.phone,
        "standards": tutor_data.standards or [],
        "sections": tutor_data.sections or [],
        "subjects": tutor_data.subjects or [],
        "plan_types": tutor_data.plan_types or [],
        "can_edit_students": tutor_data.can_edit_students,
        "is_active": True,
        "assigned_student_ids": [],
        "requires_password_change": True,  # Must change on first login
        "password_reset_requested": False,
        "created_by": admin_id,
        "created_at": datetime.utcnow(),
        "last_login": None,
    }

    # Insert into database
    result = await db.mongo_insert_one("tutors", new_tutor)
    new_tutor["_id"] = result

    # Return tutor data with generated password (only shown once)
    return TutorResponse(
        id=str(result),
        tutor_id=auto_tutor_id,
        username=tutor_data.username,
        full_name=tutor_data.full_name,
        name=tutor_data.full_name,
        email=tutor_data.email,
        phone=tutor_data.phone,
        standards=tutor_data.standards or [],
        sections=tutor_data.sections or [],
        subjects=tutor_data.subjects or [],
        plan_types=tutor_data.plan_types or [],
        can_edit_students=tutor_data.can_edit_students,
        is_active=True,
        assigned_student_ids=[],
        requires_password_change=True,
        password_reset_requested=False,
        created_at=new_tutor["created_at"],
        last_login=None,
        generated_password=generated_password  # Only shown on creation
    )


@router.get("/tutors", response_model=List[TutorResponse])
@limiter.limit("30/minute")
async def get_tutors(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all tutors (Admin only)
    """
    admin_id = current_user.get("user_id")

    # Get all tutors created by this admin
    tutors = await db.mongo_find("tutors", {"created_by": admin_id})

    return [
        TutorResponse(
            id=str(tutor["_id"]),
            tutor_id=tutor.get("tutor_id"),
            username=tutor.get("username"),
            name=tutor.get("name"),
            email=tutor.get("email"),
            phone=tutor.get("phone"),
            standards=tutor.get("standards", []),
            sections=tutor.get("sections", []),
            subjects=tutor.get("subjects", []),
            plan_types=tutor.get("plan_types", []),
            can_edit_students=tutor.get("can_edit_students", False),
            is_active=tutor.get("is_active", True),
            assigned_student_ids=tutor.get("assigned_student_ids", []),
            requires_password_change=tutor.get("requires_password_change"),
            password_reset_requested=tutor.get("password_reset_requested"),
            created_at=tutor.get("created_at"),
            last_login=tutor.get("last_login")
        )
        for tutor in tutors
    ]


@router.get("/tutors/{tutor_id}", response_model=TutorResponse)
@limiter.limit("30/minute")
async def get_tutor(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get a specific tutor by ID (Admin only)
    """
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    return TutorResponse(
        id=str(tutor["_id"]),
        tutor_id=tutor.get("tutor_id"),
        username=tutor.get("username"),
        name=tutor.get("name"),
        email=tutor.get("email"),
        phone=tutor.get("phone"),
        standards=tutor.get("standards", []),
        sections=tutor.get("sections", []),
        subjects=tutor.get("subjects", []),
        plan_types=tutor.get("plan_types", []),
        can_edit_students=tutor.get("can_edit_students", False),
        is_active=tutor.get("is_active", True),
        assigned_student_ids=tutor.get("assigned_student_ids", []),
        requires_password_change=tutor.get("requires_password_change"),
        password_reset_requested=tutor.get("password_reset_requested"),
        created_at=tutor.get("created_at"),
        last_login=tutor.get("last_login")
    )


@router.put("/tutors/{tutor_id}")
@limiter.limit("20/minute")
async def update_tutor(
    request: Request,
    tutor_id: str,
    updates: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Update tutor information (Admin only)
    """
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    # Update tutor
    await db.mongo_update_one(
        "tutors",
        {"tutor_id": tutor_id},
        {"$set": updates}
    )

    return {"message": "Tutor updated successfully"}


@router.delete("/tutors/{tutor_id}")
@limiter.limit("10/minute")
async def delete_tutor(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Delete a tutor (Admin only)
    """
    result = await db.mongo_delete_one("tutors", {"tutor_id": tutor_id})
    if result == 0:
        raise HTTPException(status_code=404, detail="Tutor not found")

    return {"message": "Tutor deleted successfully"}


@router.post("/tutors/{tutor_id}/assign-student")
@limiter.limit("20/minute")
async def assign_student_to_tutor(
    request: Request,
    tutor_id: str,
    assignment: AssignStudentRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Assign a student to a tutor
    When tutor assigns: adds current tutor_id to student's teacher_ids (non-mutable)
    When admin assigns: can select any tutor
    """
    # Get tutor
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    # Get student
    student = await db.mongo_find_one("students", {"student_id": assignment.student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # If current user is tutor, verify they're assigning to themselves
    if current_user.get("user_type") == "tutor":
        if current_user.get("tutor_id") != tutor_id:
            raise HTTPException(status_code=403, detail="Tutors can only assign students to themselves")

    # Add student to tutor's assigned_student_ids
    await db.mongo_update_one(
        "tutors",
        {"tutor_id": tutor_id},
        {"$addToSet": {"assigned_student_ids": assignment.student_id}}
    )

    # Add tutor to student's teacher_ids (if not already present)
    await db.mongo_update_one(
        "students",
        {"student_id": assignment.student_id},
        {"$addToSet": {"teacher_ids": tutor_id}}
    )

    return {"message": f"Student {assignment.student_id} assigned to tutor {tutor_id}"}


@router.delete("/tutors/{tutor_id}/unassign-student/{student_id}")
@limiter.limit("20/minute")
async def unassign_student_from_tutor(
    request: Request,
    tutor_id: str,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),  # Only admin can unassign
    db: DatabaseManager = Depends(get_database)
):
    """
    Unassign a student from a tutor (Admin only)
    """
    # Remove student from tutor's assigned_student_ids
    await db.mongo_update_one(
        "tutors",
        {"tutor_id": tutor_id},
        {"$pull": {"assigned_student_ids": student_id}}
    )

    # Remove tutor from student's teacher_ids
    await db.mongo_update_one(
        "students",
        {"student_id": student_id},
        {"$pull": {"teacher_ids": tutor_id}}
    )

    return {"message": f"Student {student_id} unassigned from tutor {tutor_id}"}


@router.get("/tutors/{tutor_id}/students", response_model=List[Dict[str, Any]])
@limiter.limit("30/minute")
async def get_tutor_students(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all students assigned to a tutor
    Tutors can only see their own students
    """
    # If tutor, verify they're requesting their own students
    if current_user.get("user_type") == "tutor":
        if current_user.get("tutor_id") != tutor_id:
            raise HTTPException(status_code=403, detail="Tutors can only view their own students")

    # Get tutor
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    assigned_student_ids = tutor.get("assigned_student_ids", []) or []

    students_union: List[Dict[str, Any]] = []

    # 1) Students explicitly assigned by id
    if assigned_student_ids:
        assigned_students = await db.mongo_find("students", {"student_id": {"$in": assigned_student_ids}})
        students_union.extend(assigned_students)

    # 2) Students mapped via teacher_ids
    teacher_mapped = await db.mongo_find("students", {"teacher_ids": {"$in": [tutor_id]}})
    students_union.extend(teacher_mapped)

    # 3) Criteria-based matching within same admin (OR across grade/section/subjects/plan_types)
    admin_id = tutor.get("created_by")
    try:
        from bson import ObjectId
        admin_oid = ObjectId(admin_id) if admin_id else None
    except Exception:
        admin_oid = None

    criteria = {}
    or_filters = []
    if admin_oid:
        criteria = {"admin_id": admin_oid}
    if tutor.get("standards"):
        or_filters.append({"grade": {"$in": tutor.get("standards")}})
    if tutor.get("sections"):
        or_filters.append({"section": {"$in": tutor.get("sections")}})
    if tutor.get("subjects"):
        or_filters.append({"subjects": {"$in": tutor.get("subjects")}})
    if tutor.get("plan_types"):
        or_filters.append({"plan_types": {"$in": tutor.get("plan_types")}})
    if or_filters:
        if criteria:
            criteria = {"$and": [criteria, {"$or": or_filters}]}
        else:
            criteria = {"$or": or_filters}

    if criteria:
        if admin_oid:
            criteria = {"$and": [{"admin_id": admin_oid}, criteria]}
        criteria_students = await db.mongo_find("students", criteria)
        students_union.extend(criteria_students)

    # Deduplicate by _id
    seen = set()
    students = []
    for s in students_union:
        sid = str(s.get("_id"))
        if sid not in seen:
            seen.add(sid)
            students.append(s)

    return [
        {
            "id": str(student["_id"]),
            "student_id": student.get("student_id"),
            "username": student.get("username"),
            "name": student.get("name"),
            "email": student.get("email"),
            "grade": student.get("grade"),
            "section": student.get("section"),
            "subjects": student.get("subjects", []),
            "is_active": student.get("is_active", True),
        }
        for student in students
    ]


@router.post("/tutors/{tutor_id}/reset-password-request")
@limiter.limit("5/minute")
async def request_tutor_password_reset(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Mark tutor as requesting password reset (Admin only)
    """
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    await db.mongo_update_one(
        "tutors",
        {"tutor_id": tutor_id},
        {"$set": {"password_reset_requested": True}}
    )

    return {"message": "Password reset requested for tutor"}


@router.post("/tutors/{tutor_id}/reset-password")
@limiter.limit("5/minute")
async def reset_tutor_password(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Reset tutor password to a new generated password (Admin only)
    """
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    # Generate new password
    new_password = Tutor.generate_password()
    password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Update password
    await db.mongo_update_one(
        "tutors",
        {"tutor_id": tutor_id},
        {"$set": {
            "password_hash": password_hash,
            "requires_password_change": True,
            "password_reset_requested": False
        }}
    )

    return {
        "message": "Password reset successfully",
        "new_password": new_password  # Return new password to admin
    }
