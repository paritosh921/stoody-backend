"""
Tutor Management API Endpoints (Async)
Handles tutor CRUD operations, authentication, and student assignments
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, EmailStr, ValidationError
import bcrypt

from models.tutor import (
    Tutor,
    TutorSchema,
    TutorUpdateSchema,
    TutorPasswordChangeSchema,
)
from models.student import Student
from core.database import DatabaseManager
from core.permissions import has_permission
from api.v1.auth_async import get_current_user, get_database
import logging

_logger = logging.getLogger(__name__)
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# Helper dependency functions
def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access (regular or B2C)"""
    if current_user.get("user_type") not in ["admin", "b2c_admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    if not has_permission(current_user, "manage_tutors"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
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
    if current_user.get("user_type") in ["admin", "b2c_admin"] and not has_permission(
        current_user, "manage_tutors"
    ):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return current_user


class TeachingAssignment(BaseModel):
    standard: str = Field(..., description="Class/grade for the assignment")
    subject: str = Field(..., description="Subject taught in the class")
    sections: List[str] = Field(
        default_factory=list, description="Sections within the class"
    )


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
    teaching_assignments: Optional[List[TeachingAssignment]] = (
        None  # Class / subject / section mapping
    )
    class_teacher_of: Optional[Dict[str, str]] = (
        None  # e.g. {"standard": "11", "section": "A"}
    )


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
    teaching_assignments: Optional[List[TeachingAssignment]] = None
    class_teacher_of: Optional[Dict[str, str]] = None


def derive_assignment_meta(assignments: Optional[List[TeachingAssignment]]):
    """
    Normalize teaching assignments and derive aggregate standards/sections/subjects.
    """
    normalized: List[Dict[str, Any]] = []
    standards: set[str] = set()
    sections: set[str] = set()
    subjects: set[str] = set()

    for assignment in assignments or []:
        normalized.append(assignment.dict())
        if assignment.standard:
            standards.add(assignment.standard)
        if assignment.subject:
            subjects.add(assignment.subject)
        for sec in assignment.sections or []:
            if sec:
                sections.add(sec)

    return normalized, sorted(standards), sorted(sections), sorted(subjects)


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
    db: DatabaseManager = Depends(get_database),
):
    """
    Create a new tutor account (Admin only)
    Auto-generates tutor_id and password
    """
    # Check registration limit before proceeding
    from api.v1.admin_async import check_registration_limit

    await check_registration_limit(db, current_user, "tutors", "max_tutors")

    normalized_username = tutor_data.username.strip()
    username_lower = normalized_username.lower()
    # Check if username already exists (case-insensitive)
    existing_tutor = await db.mongo_find_one(
        "tutors", {"username_lower": username_lower}
    )
    if not existing_tutor:
        existing_tutor = await db.mongo_find_one(
            "tutors",
            {"username": normalized_username},
            collation={"locale": "en", "strength": 2},
        )
    if existing_tutor:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Generate tutor ID and password
    auto_tutor_id = Tutor.generate_tutor_id()
    generated_password = Tutor.generate_password()
    password_hash = bcrypt.hashpw(
        generated_password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    # Get admin ID
    admin_id = current_user.get("user_id")

    # Normalize teaching assignments and derive aggregate fields
    (
        normalized_assignments,
        assignment_standards,
        assignment_sections,
        assignment_subjects,
    ) = derive_assignment_meta(tutor_data.teaching_assignments)
    standards = sorted(set((tutor_data.standards or []) + assignment_standards))
    sections = sorted(set((tutor_data.sections or []) + assignment_sections))
    subjects = sorted(set((tutor_data.subjects or []) + assignment_subjects))

    # Create tutor document
    new_tutor = {
        "tutor_id": auto_tutor_id,
        "name": tutor_data.full_name,
        "username": normalized_username,
        "username_lower": username_lower,
        "password_hash": password_hash,
        "email": tutor_data.email,
        "phone": tutor_data.phone,
        "standards": standards,
        "sections": sections,
        "subjects": subjects,
        "plan_types": tutor_data.plan_types or [],
        "can_edit_students": tutor_data.can_edit_students,
        "is_active": True,
        "assigned_student_ids": [],
        "requires_password_change": True,  # Must change on first login
        "password_reset_requested": False,
        "teaching_assignments": normalized_assignments,
        "class_teacher_of": tutor_data.class_teacher_of,
        "created_by": admin_id,
        "created_at": datetime.utcnow(),
        "last_login": None,
        # 2FA: Required for all new tutors
        "two_fa": {
            "enabled": False,
            "required": True,  # Force 2FA setup on first login
            "secret_enc": None,
            "verified_at": None,
        },
    }

    # Insert into database
    result = await db.mongo_insert_one("tutors", new_tutor)
    new_tutor["_id"] = result

    # Return tutor data with generated password (only shown once)
    return TutorResponse(
        id=str(result),
        tutor_id=auto_tutor_id,
        username=normalized_username,
        full_name=tutor_data.full_name,
        name=tutor_data.full_name,
        email=tutor_data.email,
        phone=tutor_data.phone,
        standards=standards,
        sections=sections,
        subjects=subjects,
        plan_types=tutor_data.plan_types or [],
        can_edit_students=tutor_data.can_edit_students,
        is_active=True,
        assigned_student_ids=[],
        requires_password_change=True,
        password_reset_requested=False,
        created_at=new_tutor["created_at"],
        last_login=None,
        generated_password=generated_password,  # Only shown on creation
        teaching_assignments=normalized_assignments,
    )


@router.get("/tutors", response_model=List[TutorResponse])
@limiter.limit("30/minute")
async def get_tutors(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
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
            last_login=tutor.get("last_login"),
            teaching_assignments=tutor.get("teaching_assignments", []),
            class_teacher_of=tutor.get("class_teacher_of"),
        )
        for tutor in tutors
    ]


@router.get("/tutors/{tutor_id}", response_model=TutorResponse)
@limiter.limit("30/minute")
async def get_tutor(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
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
        last_login=tutor.get("last_login"),
        teaching_assignments=tutor.get("teaching_assignments", []),
    )


@router.put("/tutors/{tutor_id}")
@limiter.limit("20/minute")
async def update_tutor(
    request: Request,
    tutor_id: str,
    updates: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """
    Update tutor information (Admin only)
    """
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    # Normalize teaching assignments if provided and derive aggregate fields
    if "teaching_assignments" in updates:
        assignments_payload = updates.get("teaching_assignments") or []
        validated_assignments = []
        assignment_standards: set[str] = set()
        assignment_sections: set[str] = set()
        assignment_subjects: set[str] = set()

        for item in assignments_payload:
            try:
                assignment = TeachingAssignment(**item)
            except ValidationError:
                # Skip invalid assignment entries instead of failing the whole request
                continue

            validated_assignments.append(assignment.dict())
            if assignment.standard:
                assignment_standards.add(assignment.standard)
            if assignment.subject:
                assignment_subjects.add(assignment.subject)
            for sec in assignment.sections or []:
                if sec:
                    assignment_sections.add(sec)

        updates["teaching_assignments"] = validated_assignments

        # Ensure lists before merging
        standards_from_updates = updates.get("standards", []) or []
        sections_from_updates = updates.get("sections", []) or []
        subjects_from_updates = updates.get("subjects", []) or []

        if not isinstance(standards_from_updates, list):
            standards_from_updates = [standards_from_updates]
        if not isinstance(sections_from_updates, list):
            sections_from_updates = [sections_from_updates]
        if not isinstance(subjects_from_updates, list):
            subjects_from_updates = [subjects_from_updates]

        updates["standards"] = sorted(
            set(standards_from_updates + list(assignment_standards))
        )
        updates["sections"] = sorted(
            set(sections_from_updates + list(assignment_sections))
        )
        updates["subjects"] = sorted(
            set(subjects_from_updates + list(assignment_subjects))
        )

    # Keep name and full_name in sync (frontend may send either)
    if "name" in updates:
        updates["full_name"] = updates["name"]
    elif "full_name" in updates:
        updates["name"] = updates["full_name"]

    # Update tutor
    await db.mongo_update_one("tutors", {"tutor_id": tutor_id}, {"$set": updates})

    return {"message": "Tutor updated successfully"}


class TutorSelfUpdateRequest(BaseModel):
    """Fields a tutor can update on their own profile."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=20)


@router.put("/tutors/me/profile")
@limiter.limit("10/minute")
async def update_tutor_self_profile(
    request: Request,
    profile_data: TutorSelfUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """
    Tutor updates their own profile (name, email, phone only).
    Admin-controlled fields (teaching_assignments, standards, sections, subjects,
    can_edit_students, username, tutor_id) cannot be changed here.
    """
    from bson import ObjectId

    tutor_id = ObjectId(current_user["user_id"])
    tutor = await db.mongo_find_one("tutors", {"_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    # Build update dict from non-None fields only
    updates: Dict[str, Any] = {}
    if profile_data.name is not None:
        updates["name"] = profile_data.name.strip()
        updates["full_name"] = profile_data.name.strip()
    if profile_data.email is not None:
        updates["email"] = profile_data.email.strip().lower()
    if profile_data.phone is not None:
        updates["phone"] = profile_data.phone.strip()

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    await db.mongo_update_one("tutors", {"_id": tutor_id}, {"$set": updates})

    _logger.info(f"Tutor {tutor.get('username')} updated their profile")

    # Return updated profile data so frontend can refresh
    updated_tutor = await db.mongo_find_one("tutors", {"_id": tutor_id})
    return {
        "success": True,
        "message": "Profile updated successfully",
        "user": {
            "name": updated_tutor.get("name") or updated_tutor.get("full_name"),
            "email": updated_tutor.get("email"),
            "phone": updated_tutor.get("phone"),
        },
    }


@router.delete("/tutors/{tutor_id}")
@limiter.limit("10/minute")
async def delete_tutor(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
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
    db: DatabaseManager = Depends(get_database),
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
            raise HTTPException(
                status_code=403, detail="Tutors can only assign students to themselves"
            )

    # Add student to tutor's assigned_student_ids
    await db.mongo_update_one(
        "tutors",
        {"tutor_id": tutor_id},
        {"$addToSet": {"assigned_student_ids": assignment.student_id}},
    )

    # Add tutor to student's teacher_ids (if not already present)
    await db.mongo_update_one(
        "students",
        {"student_id": assignment.student_id},
        {"$addToSet": {"teacher_ids": tutor_id}},
    )

    return {"message": f"Student {assignment.student_id} assigned to tutor {tutor_id}"}


@router.delete("/tutors/{tutor_id}/unassign-student/{student_id}")
@limiter.limit("20/minute")
async def unassign_student_from_tutor(
    request: Request,
    tutor_id: str,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),  # Only admin can unassign
    db: DatabaseManager = Depends(get_database),
):
    """
    Unassign a student from a tutor (Admin only)
    """
    # Remove student from tutor's assigned_student_ids
    await db.mongo_update_one(
        "tutors",
        {"tutor_id": tutor_id},
        {"$pull": {"assigned_student_ids": student_id}},
    )

    # Remove tutor from student's teacher_ids
    await db.mongo_update_one(
        "students", {"student_id": student_id}, {"$pull": {"teacher_ids": tutor_id}}
    )

    return {"message": f"Student {student_id} unassigned from tutor {tutor_id}"}


@router.get("/tutors/{tutor_id}/students", response_model=List[Dict[str, Any]])
@limiter.limit("30/minute")
async def get_tutor_students(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get all students assigned to a tutor
    Tutors can only see their own students
    """
    # If tutor, verify they're requesting their own students
    if current_user.get("user_type") == "tutor":
        if current_user.get("tutor_id") != tutor_id:
            raise HTTPException(
                status_code=403, detail="Tutors can only view their own students"
            )

    # Get tutor
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    from bson import ObjectId
    from utils.tutor_scoping import get_tutor_scoped_students

    # Get admin_id for data isolation
    admin_id = tutor.get("created_by")
    try:
        admin_oid = ObjectId(admin_id) if admin_id else None
    except Exception:
        admin_oid = None

    students = await get_tutor_scoped_students(
        tutor_id=tutor_id,
        admin_oid=admin_oid,
        db=db,
        tutor_doc=tutor,
    )

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
    db: DatabaseManager = Depends(get_database),
):
    """
    Mark tutor as requesting password reset (Admin only)
    """
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    await db.mongo_update_one(
        "tutors", {"tutor_id": tutor_id}, {"$set": {"password_reset_requested": True}}
    )

    return {"message": "Password reset requested for tutor"}


@router.post("/tutors/{tutor_id}/reset-password")
@limiter.limit("5/minute")
async def reset_tutor_password(
    request: Request,
    tutor_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """
    Reset tutor password to a new generated password (Admin only)
    """
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    # Generate new password
    new_password = Tutor.generate_password()
    password_hash = bcrypt.hashpw(
        new_password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    # Update password
    await db.mongo_update_one(
        "tutors",
        {"tutor_id": tutor_id},
        {
            "$set": {
                "password_hash": password_hash,
                "requires_password_change": True,
                "password_reset_requested": False,
            }
        },
    )

    return {
        "message": "Password reset successfully",
        "new_password": new_password,  # Return new password to admin
    }


# ---------------------------------------------------------------------------
# Analytics Endpoints
# ---------------------------------------------------------------------------


async def _get_tutor_visible_students(
    current_user: Dict[str, Any], db: DatabaseManager
) -> List[Dict[str, Any]]:
    """
    Reusable helper that returns the deduplicated list of student documents
    visible to the current tutor.  Combines:
      1. Students explicitly assigned  (assigned_student_ids)
      2. Students mapped via teacher_ids
      3. Students matching the tutor's teaching assignments (class/section)
    """
    from bson import ObjectId
    from utils.tutor_scoping import get_tutor_scoped_students

    tutor_id = current_user.get("tutor_id")
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    # Get admin_id for data isolation
    admin_id = tutor.get("created_by")
    try:
        admin_oid = ObjectId(admin_id) if admin_id else None
    except Exception:
        admin_oid = None

    return await get_tutor_scoped_students(
        tutor_id=tutor_id,
        admin_oid=admin_oid,
        db=db,
        tutor_doc=tutor,
    )


@router.get("/tutors/analytics/overview")
@limiter.limit("15/minute")
async def get_tutor_analytics_overview(
    request: Request,
    class_group: Optional[str] = Query(
        None, description="Filter by class (e.g. '10-A' or '10')"
    ),
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """
    Class-level aggregated analytics for the tutor's visible students.
    Returns summary stats, subject performance, daily activity,
    top performers and students needing attention.
    """
    try:
        students = await _get_tutor_visible_students(current_user, db)

        # Get the tutor's own subjects for filtering subject performance
        tutor_id = current_user.get("tutor_id")
        tutor_doc = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
        tutor_subjects = set(s for s in (tutor_doc.get("subjects") or []) if s) if tutor_doc else set()

        # Extract available classes from unfiltered visible students
        available_classes_set = set()
        for s in students:
            g = s.get("grade")
            sec = s.get("section")
            if g:
                if sec:
                    available_classes_set.add(f"{g}-{sec}")
                else:
                    available_classes_set.add(g)
        available_classes = sorted(list(available_classes_set))

        # Filter students if class_group is provided
        if class_group:
            parts = class_group.split("-", 1)
            filter_grade = parts[0]
            filter_section = parts[1] if len(parts) > 1 else None

            filtered = []
            for s in students:
                sg = s.get("grade")
                ss = s.get("section")
                if sg == filter_grade:
                    # If section is provided in filter, it must match. Otherwise, just grade needs to match.
                    if not filter_section or ss == filter_section:
                        filtered.append(s)
            students = filtered

        if not students:
            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_students": 0,
                        "active_students": 0,
                        "total_practice_attempts": 0,
                        "total_tests_taken": 0,
                        "overall_practice_accuracy": 0.0,
                        "overall_test_avg_score": 0.0,
                        "avg_time_per_question": 0.0,
                        "available_classes": available_classes,
                    },
                    "subject_performance": [],
                    "daily_activity": [],
                    "top_performers": [],
                    "needs_attention": [],
                },
            }

        # Build lookup structures
        student_oid_strings = [str(s["_id"]) for s in students]
        student_map: Dict[str, Dict[str, Any]] = {str(s["_id"]): s for s in students}

        now = datetime.utcnow()
        thirty_days_ago = now - timedelta(days=30)
        seven_days_ago = now - timedelta(days=7)

        # --- Active students (last_login within 7 days) ---
        active_students = sum(
            1
            for s in students
            if s.get("last_login") and s["last_login"] >= seven_days_ago
        )

        # ------------------------------------------------------------------
        # Practice attempts aggregation (all-time to match other stats)
        # ------------------------------------------------------------------
        practice_summary_pipeline = [
            {
                "$match": {
                    "student_id": {"$in": student_oid_strings},
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                    "total_time": {"$sum": {"$ifNull": ["$time_spent", 0]}},
                }
            },
        ]
        practice_summary_result = await db.mongo_aggregate(
            "practice_attempts", practice_summary_pipeline
        )
        practice_summary = (
            practice_summary_result[0]
            if practice_summary_result
            else {"total": 0, "correct": 0, "total_time": 0}
        )

        total_practice = practice_summary["total"]
        total_correct = practice_summary["correct"]
        total_time = practice_summary["total_time"]
        overall_practice_accuracy = round(
            (total_correct / total_practice * 100) if total_practice > 0 else 0.0, 1
        )
        avg_time_per_question = round(
            (total_time / total_practice) if total_practice > 0 else 0.0, 1
        )

        # ------------------------------------------------------------------
        # Test attempts aggregation
        # ------------------------------------------------------------------
        test_summary_pipeline = [
            {"$match": {"student_id": {"$in": student_oid_strings}}},
            {
                "$group": {
                    "_id": None,
                    "total_tests": {"$sum": 1},
                    "avg_score": {"$avg": "$percentage"},
                }
            },
        ]
        test_summary_result = await db.mongo_aggregate(
            "student_test_attempts", test_summary_pipeline
        )
        test_summary = (
            test_summary_result[0]
            if test_summary_result
            else {"total_tests": 0, "avg_score": 0.0}
        )

        total_tests = test_summary["total_tests"]
        overall_test_avg = round(test_summary["avg_score"] or 0.0, 1)

        # ------------------------------------------------------------------
        # Subject performance (filtered to subjects the teacher teaches)
        # ------------------------------------------------------------------
        practice_match: Dict[str, Any] = {
            "student_id": {"$in": student_oid_strings},
        }
        if tutor_subjects:
            practice_match["subject"] = {"$in": list(tutor_subjects)}

        subject_practice_pipeline = [
            {"$match": practice_match},
            {
                "$group": {
                    "_id": "$subject",
                    "practice_attempts": {"$sum": 1},
                    "correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                    "students": {"$addToSet": "$student_id"},
                }
            },
        ]
        subject_practice = await db.mongo_aggregate(
            "practice_attempts", subject_practice_pipeline
        )

        test_match: Dict[str, Any] = {
            "student_id": {"$in": student_oid_strings},
        }
        if tutor_subjects:
            test_match["subject"] = {"$in": list(tutor_subjects)}

        subject_test_pipeline = [
            {"$match": test_match},
            {
                "$group": {
                    "_id": "$subject",
                    "test_attempts": {"$sum": 1},
                    "avg_score": {"$avg": "$percentage"},
                }
            },
        ]
        subject_tests = await db.mongo_aggregate(
            "student_test_attempts", subject_test_pipeline
        )
        subject_test_map = {st["_id"]: st for st in subject_tests if st.get("_id")}

        subject_performance = []
        for sp in subject_practice:
            subj = sp.get("_id")
            if not subj:
                continue
            attempts = sp["practice_attempts"]
            correct = sp["correct"]
            accuracy = round((correct / attempts * 100) if attempts > 0 else 0.0, 1)
            test_info = subject_test_map.get(subj, {})
            subject_performance.append(
                {
                    "subject": subj,
                    "practice_attempts": attempts,
                    "practice_accuracy": accuracy,
                    "test_attempts": test_info.get("test_attempts", 0),
                    "test_avg_score": round(test_info.get("avg_score", 0) or 0, 1),
                    "student_count": len(sp.get("students", [])),
                }
            )

        # ------------------------------------------------------------------
        # Daily activity (last 30 days)
        # ------------------------------------------------------------------
        daily_practice_pipeline = [
            {
                "$match": {
                    "student_id": {"$in": student_oid_strings},
                    "created_at": {"$gte": thirty_days_ago},
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}
                    },
                    "practice_attempts": {"$sum": 1},
                    "practice_correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                    "active_students": {"$addToSet": "$student_id"},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        daily_practice = await db.mongo_aggregate(
            "practice_attempts", daily_practice_pipeline
        )

        daily_test_pipeline = [
            {
                "$match": {
                    "student_id": {"$in": student_oid_strings},
                    "submitted_at": {"$gte": thirty_days_ago},
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d", "date": "$submitted_at"}
                    },
                    "test_submissions": {"$sum": 1},
                }
            },
        ]
        daily_tests = await db.mongo_aggregate(
            "student_test_attempts", daily_test_pipeline
        )
        daily_test_map = {dt["_id"]: dt["test_submissions"] for dt in daily_tests}

        daily_activity = []
        for dp in daily_practice:
            date_str = dp["_id"]
            daily_activity.append(
                {
                    "date": date_str,
                    "practice_attempts": dp["practice_attempts"],
                    "practice_correct": dp["practice_correct"],
                    "test_submissions": daily_test_map.get(date_str, 0),
                    "active_students": len(dp.get("active_students", [])),
                }
            )

        # ------------------------------------------------------------------
        # Per-student combined accuracy for top / needs-attention
        # ------------------------------------------------------------------
        student_practice_pipeline = [
            {"$match": {"student_id": {"$in": student_oid_strings}}},
            {
                "$group": {
                    "_id": "$student_id",
                    "total": {"$sum": 1},
                    "correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                }
            },
        ]
        student_practice_stats = await db.mongo_aggregate(
            "practice_attempts", student_practice_pipeline
        )
        student_practice_map = {sp["_id"]: sp for sp in student_practice_stats}

        student_test_pipeline = [
            {"$match": {"student_id": {"$in": student_oid_strings}}},
            {
                "$group": {
                    "_id": "$student_id",
                    "test_count": {"$sum": 1},
                    "avg_score": {"$avg": "$percentage"},
                }
            },
        ]
        student_test_stats = await db.mongo_aggregate(
            "student_test_attempts", student_test_pipeline
        )
        student_test_stats_map = {st["_id"]: st for st in student_test_stats}

        # Build combined list
        combined_stats = []
        for sid_str, sdata in student_map.items():
            pstats = student_practice_map.get(sid_str, {})
            tstats = student_test_stats_map.get(sid_str, {})
            p_total = pstats.get("total", 0)
            p_correct = pstats.get("correct", 0)
            t_avg = tstats.get("avg_score", 0) or 0
            p_acc = round((p_correct / p_total * 100) if p_total > 0 else 0.0, 1)
            total_attempts = p_total + tstats.get("test_count", 0)
            combined_stats.append(
                {
                    "student_id": sdata.get("student_id"),
                    "name": sdata.get("name") or sdata.get("full_name", ""),
                    "grade": sdata.get("grade", ""),
                    "section": sdata.get("section", ""),
                    "practice_accuracy": p_acc,
                    "test_avg_score": round(t_avg, 1),
                    "total_attempts": total_attempts,
                }
            )

        # Sort by combined accuracy descending
        def _sort_key(entry):
            # average of practice_accuracy and test_avg_score (or just whichever is available)
            p = entry["practice_accuracy"]
            t = entry["test_avg_score"]
            if p > 0 and t > 0:
                return (p + t) / 2
            return p or t

        combined_stats.sort(key=_sort_key, reverse=True)
        top_performers = combined_stats[:5]

        # Needs attention: bottom 5 with at least 5 attempts
        with_attempts = [c for c in combined_stats if c["total_attempts"] >= 5]
        with_attempts.sort(key=_sort_key)
        needs_attention = with_attempts[:5]

        return {
            "success": True,
            "data": {
                "summary": {
                    "total_students": len(students),
                    "active_students": active_students,
                    "total_practice_attempts": total_practice,
                    "total_tests_taken": total_tests,
                    "overall_practice_accuracy": overall_practice_accuracy,
                    "overall_test_avg_score": overall_test_avg,
                    "avg_time_per_question": avg_time_per_question,
                    "available_classes": available_classes,
                },
                "subject_performance": subject_performance,
                "daily_activity": daily_activity,
                "top_performers": top_performers,
                "needs_attention": needs_attention,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error fetching tutor analytics overview: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to fetch analytics overview"
        )


@router.get("/tutors/analytics/student/{student_id}")
@limiter.limit("30/minute")
async def get_tutor_student_analytics(
    request: Request,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """
    Detailed analytics drill-down for a specific student visible to the tutor.
    ``student_id`` is the business student_id (e.g. "STU001"), **not** the
    MongoDB ObjectId.
    """
    try:
        from bson import ObjectId

        # ----- Verify student is visible to this tutor -----
        visible_students = await _get_tutor_visible_students(current_user, db)
        student = None
        for s in visible_students:
            if s.get("student_id") == student_id:
                student = s
                break

        if not student:
            raise HTTPException(
                status_code=404,
                detail="Student not found or not accessible by this tutor",
            )

        oid_str = str(student["_id"])

        # ----- Student info -----
        student_info = {
            "student_id": student.get("student_id"),
            "name": student.get("name") or student.get("full_name", ""),
            "grade": student.get("grade", ""),
            "section": student.get("section", ""),
            "subjects": student.get("subjects", []),
            "last_login": student.get("last_login").isoformat()
            if student.get("last_login")
            else None,
        }

        # ----- Practice summary -----
        practice_summary_pipeline = [
            {"$match": {"student_id": oid_str}},
            {
                "$group": {
                    "_id": None,
                    "total_attempted": {"$sum": 1},
                    "total_correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                    "total_time_spent": {"$sum": {"$ifNull": ["$time_spent", 0]}},
                    "hints_used": {"$sum": {"$ifNull": ["$hints_used", 0]}},
                }
            },
        ]
        practice_agg = await db.mongo_aggregate(
            "practice_attempts", practice_summary_pipeline
        )
        ps = (
            practice_agg[0]
            if practice_agg
            else {
                "total_attempted": 0,
                "total_correct": 0,
                "total_time_spent": 0,
                "hints_used": 0,
            }
        )
        p_total = ps["total_attempted"]
        p_correct = ps["total_correct"]
        practice_summary = {
            "total_attempted": p_total,
            "total_correct": p_correct,
            "accuracy": round((p_correct / p_total * 100) if p_total > 0 else 0.0, 1),
            "avg_time_per_question": round(
                (ps["total_time_spent"] / p_total) if p_total > 0 else 0.0, 1
            ),
            "total_time_spent": ps["total_time_spent"],
            "hints_used": ps["hints_used"],
        }

        # ----- Test summary -----
        test_summary_pipeline = [
            {"$match": {"student_id": oid_str}},
            {
                "$group": {
                    "_id": None,
                    "tests_taken": {"$sum": 1},
                    "avg_score": {"$avg": "$percentage"},
                    "best_score": {"$max": "$percentage"},
                    "worst_score": {"$min": "$percentage"},
                }
            },
        ]
        test_agg = await db.mongo_aggregate(
            "student_test_attempts", test_summary_pipeline
        )
        ts = (
            test_agg[0]
            if test_agg
            else {"tests_taken": 0, "avg_score": 0, "best_score": 0, "worst_score": 0}
        )
        test_summary = {
            "tests_taken": ts["tests_taken"],
            "avg_score": round(ts["avg_score"] or 0, 1),
            "best_score": round(ts["best_score"] or 0, 1),
            "worst_score": round(ts["worst_score"] or 0, 1),
            "avg_percentage": round(ts["avg_score"] or 0, 1),
        }

        # ----- Subject breakdown -----
        subject_practice_pipeline = [
            {"$match": {"student_id": oid_str}},
            {
                "$group": {
                    "_id": "$subject",
                    "practice_attempts": {"$sum": 1},
                    "correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                    "total_time": {"$sum": {"$ifNull": ["$time_spent", 0]}},
                }
            },
        ]
        subject_practice = await db.mongo_aggregate(
            "practice_attempts", subject_practice_pipeline
        )

        subject_test_pipeline = [
            {"$match": {"student_id": oid_str}},
            {
                "$group": {
                    "_id": "$subject",
                    "test_count": {"$sum": 1},
                    "avg_score": {"$avg": "$percentage"},
                }
            },
        ]
        subject_tests = await db.mongo_aggregate(
            "student_test_attempts", subject_test_pipeline
        )
        subject_test_map = {st["_id"]: st for st in subject_tests if st.get("_id")}

        subject_breakdown = []
        for sp in subject_practice:
            subj = sp.get("_id")
            if not subj:
                continue
            attempts = sp["practice_attempts"]
            correct = sp["correct"]
            total_time = sp["total_time"]
            test_info = subject_test_map.get(subj, {})
            subject_breakdown.append(
                {
                    "subject": subj,
                    "practice_attempts": attempts,
                    "practice_accuracy": round(
                        (correct / attempts * 100) if attempts > 0 else 0.0, 1
                    ),
                    "test_count": test_info.get("test_count", 0),
                    "test_avg_score": round(test_info.get("avg_score", 0) or 0, 1),
                    "avg_time": round(
                        (total_time / attempts) if attempts > 0 else 0.0, 1
                    ),
                }
            )

        # ----- Weekly trend (last 8 weeks) -----
        eight_weeks_ago = datetime.utcnow() - timedelta(weeks=8)

        weekly_practice_pipeline = [
            {
                "$match": {
                    "student_id": oid_str,
                    "created_at": {"$gte": eight_weeks_ago},
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%G-W%V", "date": "$created_at"}
                    },
                    "week_start": {"$min": "$created_at"},
                    "practice_attempts": {"$sum": 1},
                    "practice_correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]
        weekly_practice = await db.mongo_aggregate(
            "practice_attempts", weekly_practice_pipeline
        )

        weekly_test_pipeline = [
            {
                "$match": {
                    "student_id": oid_str,
                    "submitted_at": {"$gte": eight_weeks_ago},
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%G-W%V", "date": "$submitted_at"}
                    },
                    "tests_taken": {"$sum": 1},
                    "avg_score": {"$avg": "$percentage"},
                }
            },
        ]
        weekly_tests = await db.mongo_aggregate(
            "student_test_attempts", weekly_test_pipeline
        )
        weekly_test_map = {wt["_id"]: wt for wt in weekly_tests}

        weekly_trend = []
        for wp in weekly_practice:
            week_key = wp["_id"]
            attempts = wp["practice_attempts"]
            correct = wp["practice_correct"]
            wt_info = weekly_test_map.get(week_key, {})
            weekly_trend.append(
                {
                    "week": week_key,
                    "week_start": wp["week_start"].strftime("%Y-%m-%d")
                    if wp.get("week_start")
                    else None,
                    "practice_attempts": attempts,
                    "practice_correct": correct,
                    "accuracy": round(
                        (correct / attempts * 100) if attempts > 0 else 0.0, 1
                    ),
                    "tests_taken": wt_info.get("tests_taken", 0),
                    "test_avg_score": round(wt_info.get("avg_score", 0) or 0, 1),
                }
            )

        # ----- Difficulty breakdown -----
        difficulty_pipeline = [
            {"$match": {"student_id": oid_str}},
            {
                "$group": {
                    "_id": "$difficulty",
                    "attempts": {"$sum": 1},
                    "correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]
        difficulty_agg = await db.mongo_aggregate(
            "practice_attempts", difficulty_pipeline
        )
        difficulty_breakdown = []
        for d in difficulty_agg:
            diff = d.get("_id")
            if not diff:
                continue
            att = d["attempts"]
            cor = d["correct"]
            difficulty_breakdown.append(
                {
                    "difficulty": diff,
                    "attempts": att,
                    "correct": cor,
                    "accuracy": round((cor / att * 100) if att > 0 else 0.0, 1),
                }
            )

        # ----- Recent attempts (last 20) -----
        recent_raw = await db.mongo_find(
            "practice_attempts",
            {"student_id": oid_str},
            sort=[("created_at", -1)],
            limit=20,
        )
        recent_attempts = []
        for r in recent_raw:
            recent_attempts.append(
                {
                    "id": str(r.get("_id", "")),
                    "question_text": r.get("question_text", ""),
                    "subject": r.get("subject", ""),
                    "difficulty": r.get("difficulty", ""),
                    "is_correct": r.get("is_correct", False),
                    "student_answer": r.get("student_answer", ""),
                    "correct_answer": r.get("correct_answer", ""),
                    "time_spent": r.get("time_spent", 0),
                    "hints_used": r.get("hints_used", 0),
                    "created_at": r["created_at"].isoformat()
                    if r.get("created_at")
                    else None,
                }
            )

        return {
            "success": True,
            "data": {
                "student": student_info,
                "practice_summary": practice_summary,
                "test_summary": test_summary,
                "subject_breakdown": subject_breakdown,
                "weekly_trend": weekly_trend,
                "difficulty_breakdown": difficulty_breakdown,
                "recent_attempts": recent_attempts,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(
            f"Error fetching student analytics for {student_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to fetch student analytics")


@router.get("/tutors/analytics/attempt/{attempt_id}")
@limiter.limit("30/minute")
async def get_tutor_attempt_detail(
    request: Request,
    attempt_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """
    Full detail of a single practice attempt, including submission images
    and AI evaluation feedback.  Used by teachers to view student work.
    """
    try:
        from bson import ObjectId
        from utils.s3_storage import get_public_url

        # Fetch the attempt
        try:
            attempt_oid = ObjectId(attempt_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid attempt ID")

        attempt = await db.mongo_find_one(
            "practice_attempts", {"_id": attempt_oid}
        )
        if not attempt:
            raise HTTPException(status_code=404, detail="Attempt not found")

        # Verify the student belongs to this tutor's visible students
        visible_students = await _get_tutor_visible_students(current_user, db)
        attempt_student_id = attempt.get("student_id", "")
        student_visible = False
        for s in visible_students:
            sid = str(s.get("_id", ""))
            if sid == attempt_student_id or s.get("student_id") == attempt_student_id:
                student_visible = True
                break

        if not student_visible:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this student's data",
            )

        # Convert submission image storage paths to presigned URLs
        image_urls: list = []
        for path in attempt.get("submission_images") or []:
            try:
                url = get_public_url(path, expires_in=3600)
                if url:
                    image_urls.append(url)
            except Exception:
                pass  # Skip broken paths gracefully

        return {
            "success": True,
            "data": {
                "id": str(attempt["_id"]),
                "question_text": attempt.get("question_text", ""),
                "question_type": attempt.get("question_type", ""),
                "subject": attempt.get("subject", ""),
                "difficulty": attempt.get("difficulty", ""),
                "student_answer": attempt.get("student_answer", ""),
                "correct_answer": attempt.get("correct_answer", ""),
                "is_correct": attempt.get("is_correct", False),
                "score": attempt.get("score", 0),
                "evaluation_feedback": attempt.get("evaluation_feedback", ""),
                "evaluation_reasoning": attempt.get("evaluation_reasoning", ""),
                "work_shown": attempt.get("work_shown", ""),
                "what_went_wrong": attempt.get("what_went_wrong", ""),
                "correct_solution": attempt.get("correct_solution", ""),
                "submission_image_urls": image_urls,
                "time_spent": attempt.get("time_spent", 0),
                "hints_used": attempt.get("hints_used", 0),
                "created_at": attempt["created_at"].isoformat()
                if attempt.get("created_at")
                else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(
            f"Error fetching attempt detail {attempt_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to fetch attempt detail")


@router.get("/tutors/analytics/document/{document_id}/student/{student_id}/attempts")
@limiter.limit("20/minute")
async def get_student_document_attempts(
    request: Request,
    document_id: str,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """
    All practice attempts by a specific student for a specific document/paper,
    ordered by question sequence.  Each attempt includes submission image URLs
    so the teacher can review the student's handwritten work question by question.
    """
    try:
        from bson import ObjectId

        # Verify the student is visible to this tutor
        visible_students = await _get_tutor_visible_students(current_user, db)
        target_student = None
        for s in visible_students:
            if s.get("student_id") == student_id:
                target_student = s
                break
        if not target_student:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this student's data",
            )

        oid_str = str(target_student["_id"])

        # Fetch all attempts for this student + document, ordered by creation.
        # Keep only the LAST attempt per question_id (most recent submission).
        all_attempts = await db.mongo_find(
            "practice_attempts",
            {"student_id": oid_str, "document_id": document_id},
            sort=[("created_at", 1)],
        )
        last_by_question: Dict[str, Dict[str, Any]] = {}
        for a in all_attempts:
            qid_key = a.get("question_id", "")
            last_by_question[qid_key] = a  # later entries overwrite earlier
        raw_attempts = list(last_by_question.values())

        # Resolve canvas_pages collection for stroke lookup
        canvas_col = None
        try:
            db_name = current_user.get("db_name")
            if db_name:
                tenant_db = await db.get_tenant_db(db_name)
                if tenant_db is not None:
                    canvas_col = tenant_db["canvas_pages"]
        except Exception:
            pass

        # Build user_id variants for canvas_pages lookup (stores username)
        student_user_ids: list = []
        stu_username = target_student.get("username")
        if stu_username:
            student_user_ids.append(stu_username)
        student_user_ids.append(oid_str)
        try:
            student_user_ids.append(ObjectId(oid_str))
        except Exception:
            pass

        attempts = []
        for r in raw_attempts:
            # Fetch per-question strokes if question_page_refs exist
            question_strokes: list = []
            qpr = r.get("question_page_refs")
            if qpr and canvas_col is not None:
                active_pages = qpr.get("active_pages") or []
                book_type = qpr.get("book_type", "LS")
                copy_id_ref = qpr.get("copy_id")
                time_intervals = qpr.get("time_intervals") or []

                if active_pages:
                    try:
                        page_query: Dict[str, Any] = {
                            "user_id": {"$in": student_user_ids},
                            "page_number": {"$in": active_pages},
                            "book_type": book_type.upper(),
                        }
                        # Only filter by copy_id if it's a real ID
                        # (not "default" which is a frontend placeholder)
                        if copy_id_ref and copy_id_ref != "default":
                            page_query["copy_id"] = copy_id_ref

                        cursor = canvas_col.find(page_query).sort(
                            "page_number", 1
                        )
                        raw_pages = await cursor.to_list(length=20)

                        for pg in raw_pages:
                            raw_strokes = pg.get("strokes") or []

                            if raw_strokes:
                                question_strokes.append({
                                    "page_number": pg.get("page_number", 0),
                                    "book_type": pg.get("book_type", ""),
                                    "strokes": raw_strokes,
                                })
                    except Exception as fetch_err:
                        _logger.warning(
                            f"Failed to fetch canvas pages for attempt: {fetch_err}"
                        )

            attempts.append({
                "id": str(r.get("_id", "")),
                "question_id": r.get("question_id", ""),
                "question_text": r.get("question_text", ""),
                "question_type": r.get("question_type", ""),
                "subject": r.get("subject", ""),
                "difficulty": r.get("difficulty", ""),
                "student_answer": r.get("student_answer", ""),
                "correct_answer": r.get("correct_answer", ""),
                "is_correct": r.get("is_correct", False),
                "score": r.get("score", 0),
                "evaluation_feedback": r.get("evaluation_feedback", ""),
                "what_went_wrong": r.get("what_went_wrong", ""),
                "correct_solution": r.get("correct_solution", ""),
                "question_strokes": question_strokes,
                "time_spent": r.get("time_spent", 0),
                "hints_used": r.get("hints_used", 0),
                "created_at": r["created_at"].isoformat()
                if r.get("created_at")
                else None,
            })

        return {
            "success": True,
            "data": {
                "student": {
                    "student_id": target_student.get("student_id"),
                    "name": target_student.get("name")
                    or target_student.get("full_name", ""),
                    "grade": target_student.get("grade", ""),
                    "section": target_student.get("section", ""),
                },
                "document_id": document_id,
                "total_attempts": len(attempts),
                "attempts": attempts,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(
            f"Error fetching student document attempts "
            f"doc={document_id} student={student_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to fetch student attempts"
        )


@router.get("/tutors/analytics/documents")
@limiter.limit("15/minute")
async def get_tutor_document_analytics(
    request: Request,
    document_type: Optional[str] = None,
    subject: Optional[str] = None,
    standard: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """
    List all documents (Practice Sets + Test Series) visible to the tutor
    with aggregated attempt stats per document.
    """
    try:
        from bson import ObjectId

        tutor_id = current_user.get("tutor_id")
        admin_id = current_user.get("admin_id")

        # ----- Build document filter with tutor scoping -----
        try:
            admin_oid = ObjectId(admin_id)
            admin_matches = [admin_oid, str(admin_id)]
        except Exception:
            admin_matches = [admin_id]

        doc_filter: Dict[str, Any] = {
            "$and": [
                {"admin_id": {"$in": admin_matches}},
                {"document_type": {"$in": ["Practice Sets", "Test Series"]}},
                {
                    "$or": [
                        {"teacher_ids": {"$in": [tutor_id]}},
                        {"teacher_ids": []},
                        {"teacher_ids": None},
                        {"teacher_ids": {"$exists": False}},
                    ]
                },
            ]
        }

        # Apply optional filters
        if document_type:
            doc_filter["$and"].append({"document_type": document_type})
        if subject:
            doc_filter["$and"].append({"subject": subject})
        if standard:
            doc_filter["$and"].append({"standard": standard})

        # Fetch matching documents
        documents = await db.mongo_find(
            "documents", doc_filter, sort=[("uploaded_at", -1)]
        )

        if not documents:
            return {"success": True, "data": {"documents": [], "total": 0}}

        # ----- Get visible students -----
        visible_students = await _get_tutor_visible_students(current_user, db)
        visible_student_ids = [str(s["_id"]) for s in visible_students]
        total_visible_students = len(visible_student_ids)

        # Collect all document_ids
        doc_ids = []
        for d in documents:
            did = d.get("document_id") or str(d.get("_id", ""))
            if did:
                doc_ids.append(did)

        # ----- Aggregate practice_attempts per document -----
        practice_pipeline = [
            {
                "$match": {
                    "document_id": {"$in": doc_ids},
                    "student_id": {"$in": visible_student_ids},
                }
            },
            {
                "$group": {
                    "_id": "$document_id",
                    "total_attempts": {"$sum": 1},
                    "total_correct": {
                        "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                    },
                    "total_time": {"$sum": {"$ifNull": ["$time_spent", 0]}},
                    "students": {"$addToSet": "$student_id"},
                }
            },
        ]
        practice_agg = await db.mongo_aggregate("practice_attempts", practice_pipeline)
        practice_map: Dict[str, Dict[str, Any]] = {
            p["_id"]: p for p in practice_agg if p.get("_id")
        }

        # ----- Aggregate student_test_attempts per document -----
        test_pipeline = [
            {
                "$match": {
                    "document_id": {"$in": doc_ids},
                    "student_id": {"$in": visible_student_ids},
                }
            },
            {
                "$group": {
                    "_id": "$document_id",
                    "total_attempts": {"$sum": 1},
                    "avg_percentage": {"$avg": "$percentage"},
                    "total_time": {"$sum": {"$ifNull": ["$time_taken", 0]}},
                    "students": {"$addToSet": "$student_id"},
                }
            },
        ]
        test_agg = await db.mongo_aggregate("student_test_attempts", test_pipeline)
        test_map: Dict[str, Dict[str, Any]] = {
            t["_id"]: t for t in test_agg if t.get("_id")
        }

        # ----- Merge document metadata with attempt stats -----
        result_docs = []
        for doc in documents:
            did = doc.get("document_id") or str(doc.get("_id", ""))
            if not did:
                continue

            dtype = doc.get("document_type", "")
            doc_entry: Dict[str, Any] = {
                "document_id": did,
                "title": doc.get("title", ""),
                "document_type": dtype,
                "subject": doc.get("subject", ""),
                "standard": doc.get("standard", ""),
                "total_questions": doc.get("extracted_questions_count", 0),
                "is_active": doc.get("is_active", True),
                "uploaded_at": doc["uploaded_at"] if doc.get("uploaded_at") else None,
                "students_attempted": 0,
                "total_visible_students": total_visible_students,
                "completion_rate": 0.0,
                "avg_accuracy": None,
                "avg_score": None,
                "avg_time_spent": 0,
                "total_attempts": 0,
            }

            if dtype == "Practice Sets":
                stats = practice_map.get(did)
                if stats:
                    students_attempted = len(stats.get("students", []))
                    total_attempts = stats["total_attempts"]
                    total_correct = stats["total_correct"]
                    total_time = stats["total_time"]

                    doc_entry["students_attempted"] = students_attempted
                    doc_entry["completion_rate"] = round(
                        (students_attempted / total_visible_students * 100)
                        if total_visible_students > 0
                        else 0.0,
                        1,
                    )
                    doc_entry["avg_accuracy"] = round(
                        (total_correct / total_attempts * 100)
                        if total_attempts > 0
                        else 0.0,
                        1,
                    )
                    doc_entry["avg_score"] = None
                    doc_entry["avg_time_spent"] = round(
                        (total_time / students_attempted)
                        if students_attempted > 0
                        else 0,
                        1,
                    )
                    doc_entry["total_attempts"] = total_attempts

            elif dtype == "Test Series":
                stats = test_map.get(did)
                if stats:
                    students_attempted = len(stats.get("students", []))
                    total_attempts = stats["total_attempts"]
                    total_time = stats["total_time"]

                    doc_entry["students_attempted"] = students_attempted
                    doc_entry["completion_rate"] = round(
                        (students_attempted / total_visible_students * 100)
                        if total_visible_students > 0
                        else 0.0,
                        1,
                    )
                    doc_entry["avg_accuracy"] = None
                    doc_entry["avg_score"] = round(
                        stats.get("avg_percentage", 0) or 0, 1
                    )
                    doc_entry["avg_time_spent"] = round(
                        (total_time / students_attempted)
                        if students_attempted > 0
                        else 0,
                        1,
                    )
                    doc_entry["total_attempts"] = total_attempts

            result_docs.append(doc_entry)

        return {
            "success": True,
            "data": {"documents": result_docs, "total": len(result_docs)},
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error fetching document analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to fetch document analytics"
        )


@router.get("/tutors/analytics/documents/{document_id}")
@limiter.limit("30/minute")
async def get_tutor_document_detail_analytics(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """
    Detailed per-document analytics showing per-student results
    and per-question breakdown for a single document.
    """
    try:
        from bson import ObjectId

        tutor_id = current_user.get("tutor_id")
        admin_id = current_user.get("admin_id")

        try:
            admin_oid = ObjectId(admin_id)
            admin_matches = [admin_oid, str(admin_id)]
        except Exception:
            admin_matches = [admin_id]

        # ----- Fetch and verify document access -----
        doc_filter: Dict[str, Any] = {
            "$and": [
                {"document_id": document_id},
                {"admin_id": {"$in": admin_matches}},
                {
                    "$or": [
                        {"teacher_ids": {"$in": [tutor_id]}},
                        {"teacher_ids": []},
                        {"teacher_ids": None},
                        {"teacher_ids": {"$exists": False}},
                    ]
                },
            ]
        }
        doc = await db.mongo_find_one("documents", doc_filter)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail="Document not found or not accessible by this tutor",
            )

        dtype = doc.get("document_type", "")
        if dtype not in ("Practice Sets", "Test Series"):
            raise HTTPException(
                status_code=400,
                detail="Analytics are only available for Practice Sets and Test Series",
            )

        # ----- Get visible students -----
        visible_students = await _get_tutor_visible_students(current_user, db)
        visible_student_ids = [str(s["_id"]) for s in visible_students]
        total_visible_students = len(visible_student_ids)

        student_map: Dict[str, Dict[str, Any]] = {}
        for s in visible_students:
            student_map[str(s["_id"])] = s

        # ----- Document metadata -----
        doc_info = {
            "document_id": document_id,
            "title": doc.get("title", ""),
            "document_type": dtype,
            "subject": doc.get("subject", ""),
            "standard": doc.get("standard", ""),
            "total_questions": doc.get("extracted_questions_count", 0),
            "is_active": doc.get("is_active", True),
            "uploaded_at": doc["uploaded_at"].isoformat()
            if doc.get("uploaded_at")
            else None,
        }

        # ----- Fetch question metadata (preserve original document order) -----
        questions = await db.mongo_find(
            "questions", {"document_id": document_id}, sort=[("_id", 1)]
        )
        question_map: Dict[str, Dict[str, Any]] = {}
        question_order: Dict[str, int] = {}  # question_id → 1-based index
        for idx, q in enumerate(questions):
            qid = q.get("id") or str(q.get("_id", ""))
            question_map[qid] = q
            question_order[qid] = idx + 1

        # ----- Branch by document type -----
        student_results: List[Dict[str, Any]] = []
        question_analysis: List[Dict[str, Any]] = []

        summary: Dict[str, Any] = {
            "students_attempted": 0,
            "total_visible_students": total_visible_students,
            "completion_rate": 0.0,
            "avg_accuracy": None,
            "avg_score": None,
            "total_attempts": 0,
            "avg_time_per_student": 0,
        }

        if dtype == "Practice Sets":
            # --- Student-level aggregation ---
            student_pipeline = [
                {
                    "$match": {
                        "document_id": document_id,
                        "student_id": {"$in": visible_student_ids},
                    }
                },
                {
                    "$group": {
                        "_id": "$student_id",
                        "attempts": {"$sum": 1},
                        "correct": {
                            "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                        },
                        "time_spent": {"$sum": {"$ifNull": ["$time_spent", 0]}},
                        "last_attempted": {"$max": "$created_at"},
                    }
                },
            ]
            student_agg = await db.mongo_aggregate(
                "practice_attempts", student_pipeline
            )

            total_attempts_all = 0
            total_correct_all = 0
            total_time_all = 0

            for sa in student_agg:
                sid = sa["_id"]
                s_info = student_map.get(sid, {})
                attempts = sa["attempts"]
                correct = sa["correct"]
                time_spent = sa["time_spent"]

                total_attempts_all += attempts
                total_correct_all += correct
                total_time_all += time_spent

                student_results.append(
                    {
                        "student_id": s_info.get("student_id", sid),
                        "name": s_info.get("name") or s_info.get("full_name", ""),
                        "grade": s_info.get("grade", ""),
                        "section": s_info.get("section", ""),
                        "attempts": attempts,
                        "correct": correct,
                        "accuracy": round(
                            (correct / attempts * 100) if attempts > 0 else 0.0, 1
                        ),
                        "score": None,
                        "percentage": None,
                        "time_spent": time_spent,
                        "last_attempted": sa["last_attempted"].isoformat()
                        if sa.get("last_attempted")
                        else None,
                    }
                )

            # Sort by accuracy descending
            student_results.sort(key=lambda x: x["accuracy"], reverse=True)

            students_attempted = len(student_agg)
            summary["students_attempted"] = students_attempted
            summary["completion_rate"] = round(
                (students_attempted / total_visible_students * 100)
                if total_visible_students > 0
                else 0.0,
                1,
            )
            summary["avg_accuracy"] = round(
                (total_correct_all / total_attempts_all * 100)
                if total_attempts_all > 0
                else 0.0,
                1,
            )
            summary["avg_score"] = None
            summary["total_attempts"] = total_attempts_all
            summary["avg_time_per_student"] = round(
                (total_time_all / students_attempted) if students_attempted > 0 else 0,
                1,
            )

            # --- Question-level aggregation ---
            question_pipeline = [
                {
                    "$match": {
                        "document_id": document_id,
                        "student_id": {"$in": visible_student_ids},
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "question_id": "$question_id",
                            "student_id": "$student_id",
                        },
                        "attempts": {"$sum": 1},
                        "correct": {
                            "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                        },
                        "time_spent": {"$sum": {"$ifNull": ["$time_spent", 0]}},
                    }
                },
                {
                    "$group": {
                        "_id": "$_id.question_id",
                        "total_attempts": {"$sum": "$attempts"},
                        "correct_count": {"$sum": "$correct"},
                        "avg_time": {"$avg": {"$ifNull": ["$time_spent", 0]}},
                        "students": {
                            "$push": {
                                "student_id": "$_id.student_id",
                                "attempts": "$attempts",
                                "correct": "$correct",
                            }
                        },
                    }
                },
            ]
            question_agg = await db.mongo_aggregate(
                "practice_attempts", question_pipeline
            )

            for qa in question_agg:
                qid = qa["_id"]
                if not qid:
                    continue
                q_meta = question_map.get(qid, {})
                q_total = qa["total_attempts"]
                q_correct = qa["correct_count"]

                # Format student attempts
                student_attempts = []
                for s in qa.get("students", []):
                    sid = s.get("student_id")
                    if sid:
                        s_info = student_map.get(sid, {})
                        s_name = s_info.get("name") or s_info.get(
                            "full_name", "Unknown Student"
                        )
                        s_acc = round(
                            (s["correct"] / s["attempts"] * 100)
                            if s["attempts"] > 0
                            else 0,
                            1,
                        )
                        student_attempts.append(
                            {
                                "student_id": sid,
                                "name": s_name,
                                "attempts": s["attempts"],
                                "correct": s["correct"],
                                "accuracy": s_acc,
                            }
                        )
                student_attempts.sort(key=lambda x: x["accuracy"], reverse=True)

                question_analysis.append(
                    {
                        "question_id": qid,
                        "question_number": question_order.get(qid, 0),
                        "question_text": q_meta.get("text", ""),
                        "difficulty": q_meta.get("difficulty", ""),
                        "total_attempts": q_total,
                        "correct_count": q_correct,
                        "accuracy": round(
                            (q_correct / q_total * 100) if q_total > 0 else 0.0, 1
                        ),
                        "avg_time": round(qa.get("avg_time", 0) or 0, 1),
                        "students": student_attempts,
                    }
                )

            # Sort by accuracy ascending (hardest first) for the chart
            question_analysis.sort(key=lambda x: x["accuracy"])

        elif dtype == "Test Series":
            # --- Fetch test attempts ---
            test_attempts = await db.mongo_find(
                "student_test_attempts",
                {
                    "document_id": document_id,
                    "student_id": {"$in": visible_student_ids},
                },
            )

            total_percentage_sum = 0.0
            total_time_all = 0
            seen_students: Dict[str, Dict[str, Any]] = {}

            # Group attempts by student (handle reattempts)
            student_attempts_map: Dict[str, List[Dict[str, Any]]] = {}
            for ta in test_attempts:
                sid = ta.get("student_id", "")
                if sid not in student_attempts_map:
                    student_attempts_map[sid] = []
                student_attempts_map[sid].append(ta)

            for sid, attempts_list in student_attempts_map.items():
                s_info = student_map.get(sid, {})
                num_attempts = len(attempts_list)

                # Aggregate across all attempts for this student
                total_correct = sum(a.get("correct_count", 0) for a in attempts_list)
                total_questions = sum(
                    a.get("total_questions", 0) for a in attempts_list
                )
                avg_percentage = (
                    sum(a.get("percentage", 0) or 0 for a in attempts_list)
                    / num_attempts
                )
                avg_score = (
                    sum(a.get("score", 0) or 0 for a in attempts_list) / num_attempts
                )
                total_time = sum(a.get("time_taken", 0) or 0 for a in attempts_list)
                last_submitted = max(
                    (
                        a.get("submitted_at")
                        for a in attempts_list
                        if a.get("submitted_at")
                    ),
                    default=None,
                )

                total_percentage_sum += avg_percentage
                total_time_all += total_time

                student_results.append(
                    {
                        "student_id": s_info.get("student_id", sid),
                        "name": s_info.get("name")
                        or s_info.get("full_name", "")
                        or next((a.get("student_name", "") for a in attempts_list), ""),
                        "grade": s_info.get("grade", "")
                        or next(
                            (a.get("student_grade", "") for a in attempts_list), ""
                        ),
                        "section": s_info.get("section", ""),
                        "attempts": num_attempts,
                        "total_questions": total_questions,
                        "correct": total_correct,
                        "accuracy": round(
                            (total_correct / total_questions * 100)
                            if total_questions > 0
                            else 0.0,
                            1,
                        ),
                        "score": round(avg_score, 1),
                        "percentage": round(avg_percentage, 1),
                        "time_spent": total_time,
                        "last_attempted": last_submitted.isoformat()
                        if last_submitted
                        else None,
                    }
                )

            # Sort by accuracy descending
            student_results.sort(key=lambda x: x["accuracy"], reverse=True)

            students_attempted = len(student_attempts_map)
            total_attempts_count = len(test_attempts)

            summary["students_attempted"] = students_attempted
            summary["completion_rate"] = round(
                (students_attempted / total_visible_students * 100)
                if total_visible_students > 0
                else 0.0,
                1,
            )
            summary["avg_accuracy"] = None
            summary["avg_score"] = round(
                (total_percentage_sum / students_attempted)
                if students_attempted > 0
                else 0.0,
                1,
            )
            summary["total_attempts"] = total_attempts_count
            summary["avg_time_per_student"] = round(
                (total_time_all / students_attempted) if students_attempted > 0 else 0,
                1,
            )

            # --- Question-level analysis from question_results ---
            question_stats: Dict[str, Dict[str, Any]] = {}
            for ta in test_attempts:
                q_results = ta.get("question_results", []) or []
                for qr in q_results:
                    qid = qr.get("question_id")
                    if not qid:
                        continue
                    if qid not in question_stats:
                        question_stats[qid] = {
                            "total_attempts": 0,
                            "correct_count": 0,
                            "students_map": {},
                        }

                    # Ensure we only count actual attempts for accuracy
                    is_attempted = qr.get("is_attempted", True)
                    student_ans = str(qr.get("student_answer", "")).strip().upper()
                    if not student_ans or student_ans == "SKIPPED":
                        is_attempted = False

                    if is_attempted:
                        question_stats[qid]["total_attempts"] += 1
                        if qr.get("is_correct"):
                            question_stats[qid]["correct_count"] += 1

                        # Populate student metrics
                        sid = ta.get("student_id", "")
                        if sid:
                            if sid not in question_stats[qid]["students_map"]:
                                question_stats[qid]["students_map"][sid] = {
                                    "attempts": 0,
                                    "correct": 0,
                                }
                            question_stats[qid]["students_map"][sid]["attempts"] += 1
                            if qr.get("is_correct"):
                                question_stats[qid]["students_map"][sid]["correct"] += 1

            for qid, qs in question_stats.items():
                q_meta = question_map.get(qid, {})
                q_total = qs["total_attempts"]
                q_correct = qs["correct_count"]

                # Format student attempts
                student_attempts = []
                for sid, s_stats in qs.get("students_map", {}).items():
                    s_info = student_map.get(sid, {})
                    s_name = s_info.get("name") or s_info.get(
                        "full_name", "Unknown Student"
                    )
                    student_attempts.append(
                        {
                            "student_id": sid,
                            "name": s_name,
                            "attempts": s_stats["attempts"],
                            "correct": s_stats["correct"],
                            "accuracy": round(
                                (s_stats["correct"] / s_stats["attempts"] * 100)
                                if s_stats["attempts"] > 0
                                else 0,
                                1,
                            ),
                        }
                    )
                student_attempts.sort(key=lambda x: x["accuracy"], reverse=True)

                question_analysis.append(
                    {
                        "question_id": qid,
                        "question_number": question_order.get(qid, 0),
                        "question_text": q_meta.get("text", ""),
                        "difficulty": q_meta.get("difficulty", ""),
                        "total_attempts": q_total,
                        "correct_count": q_correct,
                        "accuracy": round(
                            (q_correct / q_total * 100) if q_total > 0 else 0.0, 1
                        ),
                        "avg_time": 0,
                        "students": student_attempts,
                    }
                )

            # Sort by accuracy ascending (hardest first) for the chart
            question_analysis.sort(key=lambda x: x["accuracy"])

        return {
            "success": True,
            "data": {
                "document": doc_info,
                "summary": summary,
                "student_results": student_results,
                "question_analysis": question_analysis,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(
            f"Error fetching document detail analytics for {document_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to fetch document detail analytics"
        )


# ---------------------------------------------------------------------------
# ExamPen metadata sync hooks (SWM-013)
# ---------------------------------------------------------------------------
# These helpers optionally push question metadata to the ExamPen
# ``evalpen_questions`` collection when a tutor creates or updates
# exam/question-paper data.
#
# Graceful degradation: if the exam-conductor package is not available
# (e.g. in deployments without ExamPen), all operations silently no-op.
#
# Usage from tutor or paper-builder flows:
#
#     from api.v1.tutor_async import sync_questions_to_exampen
#     await sync_questions_to_exampen(tenant_db, questions, exam_id, subject)
#
# Constraint C5: reuses existing tutor/backend question-paper data.
# ---------------------------------------------------------------------------


async def sync_questions_to_exampen(
    tenant_db,
    questions: List[Dict[str, Any]],
    exam_id: str,
    default_subject: Optional[str] = None,
) -> Optional[Dict[str, int]]:
    """Sync question metadata to ExamPen ``evalpen_questions`` via PCR adapter.

    Parameters
    ----------
    tenant_db
        Motor tenant database (``skb_<tenant>``).
    questions
        List of question dicts from the tutor/paper-builder workflow.
    exam_id
        Exam/paper identifier to tag questions with.
    default_subject
        Fallback subject when individual questions lack one.

    Returns
    -------
    dict or None
        ``{"inserted": N, "updated": M}`` on success, ``None`` if the
        exam-conductor module is not available.
    """
    try:
        from api.v1._exampen_imports import load_exampen

        adapter_mod = load_exampen("pcr.metadata_adapter")
        storage_mod = load_exampen("pcr.storage")

        adapt_question_to_pcr = adapter_mod.adapt_question_to_pcr
        QuestionRepository = storage_mod.QuestionRepository
    except ImportError:
        _logger.debug(
            "ExamPen modules not available; skipping evalpen_questions sync."
        )
        return None

    try:
        repo = QuestionRepository(tenant_db)

        pcr_docs = []
        for q in questions:
            pcr_doc = adapt_question_to_pcr(
                q,
                exam_id=exam_id,
                default_subject=default_subject,
            )
            if pcr_doc.get("question_id"):
                pcr_docs.append(pcr_doc)

        if not pcr_docs:
            _logger.debug(
                "No valid questions to sync to evalpen_questions for exam %s.",
                exam_id,
            )
            return {"inserted": 0, "updated": 0}

        inserted, updated = await repo.upsert_questions_bulk(pcr_docs)
        _logger.info(
            "ExamPen question metadata synced for exam %s: "
            "%d inserted, %d updated.",
            exam_id,
            inserted,
            updated,
        )
        return {"inserted": inserted, "updated": updated}

    except Exception as exc:
        _logger.warning(
            "Failed to sync question metadata to ExamPen for exam %s: %s",
            exam_id,
            exc,
        )
        return None


async def sync_paper_to_exampen(
    tenant_db,
    paper_doc: Dict[str, Any],
) -> Optional[Dict[str, int]]:
    """Sync a full question-paper document to ExamPen ``evalpen_questions``.

    Extracts questions from the paper's ``sections[].questions[]``
    structure and syncs them via the PCR metadata adapter.

    Parameters
    ----------
    tenant_db
        Motor tenant database (``skb_<tenant>``).
    paper_doc
        A question-paper document from the paper-builder collection.

    Returns
    -------
    dict or None
        ``{"inserted": N, "updated": M}`` on success, ``None`` if the
        exam-conductor module is not available.
    """
    try:
        from api.v1._exampen_imports import load_exampen

        adapter_mod = load_exampen("pcr.metadata_adapter")
        storage_mod = load_exampen("pcr.storage")

        adapt_paper_questions_to_pcr = adapter_mod.adapt_paper_questions_to_pcr
        QuestionRepository = storage_mod.QuestionRepository
    except ImportError:
        _logger.debug(
            "ExamPen modules not available; skipping paper sync."
        )
        return None

    try:
        repo = QuestionRepository(tenant_db)

        pcr_docs = adapt_paper_questions_to_pcr(paper_doc)
        pcr_docs = [d for d in pcr_docs if d.get("question_id")]

        if not pcr_docs:
            return {"inserted": 0, "updated": 0}

        inserted, updated = await repo.upsert_questions_bulk(pcr_docs)
        paper_id = paper_doc.get("id") or paper_doc.get("_id", "unknown")
        _logger.info(
            "ExamPen paper metadata synced for paper %s: "
            "%d inserted, %d updated.",
            paper_id,
            inserted,
            updated,
        )
        return {"inserted": inserted, "updated": updated}

    except Exception as exc:
        paper_id = paper_doc.get("id") or paper_doc.get("_id", "unknown")
        _logger.warning(
            "Failed to sync paper metadata to ExamPen for paper %s: %s",
            paper_id,
            exc,
        )
        return None


async def sync_answer_keys_to_dcr(
    questions: List[Dict[str, Any]],
    exam_doc: Optional[Dict[str, Any]] = None,
) -> Optional[List[Any]]:
    """Convert question data to DCR ``AnswerKey`` objects (in-memory only).

    This does NOT persist anything.  It produces the ``AnswerKey`` list
    that DCR's ``DCRService.evaluate()`` needs via the
    ``answer_key_loader`` callback.  The caller (typically the DCR
    evaluation route) can pass this list directly to the service.

    Parameters
    ----------
    questions
        List of question dicts from the tutor/paper-builder workflow.
    exam_doc
        Optional exam/paper document for context.

    Returns
    -------
    list[AnswerKey] or None
        ``None`` if the exam-conductor module is not available.
    """
    try:
        from api.v1._exampen_imports import load_exampen

        adapter_mod = load_exampen("dcr.metadata_adapter")
        adapt_exam_to_answer_keys = adapter_mod.adapt_exam_to_answer_keys
    except ImportError:
        _logger.debug(
            "ExamPen DCR modules not available; skipping answer key adaptation."
        )
        return None

    try:
        return adapt_exam_to_answer_keys(
            exam_doc or {},
            questions,
        )
    except Exception as exc:
        _logger.warning(
            "Failed to adapt answer keys for DCR: %s", exc
        )
        return None


async def sync_dcr_answer_keys(
    tenant_db,
    questions: List[Dict[str, Any]],
    exam_id: str,
    exam_doc: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, int]]:
    """Persist DCR answer keys to ``exampen_answer_keys`` collection.

    Converts question dicts to DCR ``AnswerKey`` objects via the
    DCR metadata adapter and upserts them into the tenant DB so that
    ``evalpen_dcr_async._make_answer_key_loader()`` can read them
    at evaluation time.

    Best-effort, non-blocking — callers should wrap in try/except.

    Parameters
    ----------
    tenant_db
        Motor tenant database (``skb_<tenant>``).
    questions
        List of question dicts from the tutor/paper-builder workflow.
    exam_id
        Exam/paper identifier to tag answer keys with.
    exam_doc
        Optional exam/paper document for context (passed through to
        ``adapt_exam_to_answer_keys``).

    Returns
    -------
    dict or None
        ``{"upserted": N}`` on success, ``None`` if the
        exam-conductor DCR module is not available.
    """
    try:
        from api.v1._exampen_imports import load_exampen

        adapter_mod = load_exampen("dcr.metadata_adapter")
        adapt_question_to_dcr = adapter_mod.adapt_question_to_dcr
    except ImportError:
        _logger.debug(
            "ExamPen DCR modules not available; skipping exampen_answer_keys sync."
        )
        return None

    try:
        collection = tenant_db["exampen_answer_keys"]

        upserted = 0
        for q in questions:
            ak = adapt_question_to_dcr(q)
            if ak is None:
                continue

            ak_doc = ak.model_dump(mode="json")
            ak_doc["exam_id"] = exam_id

            await collection.update_one(
                {"exam_id": exam_id, "question_id": ak_doc["question_id"]},
                {"$set": ak_doc},
                upsert=True,
            )
            upserted += 1

        if upserted:
            _logger.info(
                "DCR answer keys synced for exam %s: %d upserted.",
                exam_id,
                upserted,
            )
        else:
            _logger.debug(
                "No DCR-compatible answer keys to sync for exam %s.",
                exam_id,
            )

        # TODO: exampen_question_regions population.
        # The DCR _make_region_loader() reads bounding-box regions from
        # exampen_question_regions, but existing question data does not
        # carry bbox / region info.  The stub recognizer works without
        # regions (it processes the whole page), but the production
        # recognizer will need them.  Wire region sync here once the
        # exam paper template/layout provides bounding-box data.

        return {"upserted": upserted}

    except Exception as exc:
        _logger.warning(
            "Failed to sync DCR answer keys for exam %s: %s",
            exam_id,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# ExamPen evaluation status helper (SWM-014)
# ---------------------------------------------------------------------------
# Tutors can call this to get a quick status snapshot for their exams'
# ExamPen evaluation progress.  Graceful no-op if exam-conductor is
# unavailable.
#
# Usage from tutor flows:
#
#     from api.v1.tutor_async import get_exampen_eval_status
#     status = await get_exampen_eval_status(tenant_db, exam_id)
#
# Constraint C5: read-only access to PCR/DCR result collections.
# ---------------------------------------------------------------------------


async def get_exampen_eval_status(
    tenant_db,
    exam_id: str,
) -> Optional[Dict[str, Any]]:
    """Get ExamPen evaluation status summary for an exam.

    Returns a dict with counts of submissions, evaluated responses,
    blocked responses, and published submissions. Returns ``None`` if
    exam-conductor is not available (graceful degradation).

    Parameters
    ----------
    tenant_db
        Motor tenant database (``skb_<tenant>``).
    exam_id
        Exam identifier to check status for.

    Returns
    -------
    dict or None
        Status summary on success, ``None`` if exam-conductor unavailable.

        Shape::

            {
                "exam_id": str,
                "total_submissions": int,
                "published_submissions": int,
                "total_responses": int,
                "evaluated_responses": int,
                "blocked_responses": int,
                "pending_responses": int,
                "pcr_available": bool,
                "dcr_results_count": int,
            }
    """
    try:
        from api.v1._exampen_imports import load_exampen
        load_exampen("pcr.storage")
    except ImportError:
        _logger.debug(
            "ExamPen modules not available; skipping eval status check "
            "for exam %s.",
            exam_id,
        )
        return None

    try:
        # Count submissions
        submissions_cursor = tenant_db["evalpen_submissions"].find(
            {"exam_id": exam_id},
            projection={"submission_id": 1, "publication_status": 1},
        )
        submissions = await submissions_cursor.to_list(length=1000)
        total_submissions = len(submissions)
        published_submissions = sum(
            1
            for s in submissions
            if s.get("publication_status") == "published"
        )

        submission_ids = [s["submission_id"] for s in submissions]

        # Count responses by status
        total_responses = 0
        evaluated_responses = 0
        blocked_responses = 0
        pending_responses = 0

        if submission_ids:
            response_cursor = tenant_db[
                "evalpen_detected_responses"
            ].find(
                {"submission_id": {"$in": submission_ids}},
                projection={"eval_status": 1},
            )
            response_docs = await response_cursor.to_list(length=5000)
            total_responses = len(response_docs)

            for r in response_docs:
                es = r.get("eval_status", "pending")
                if es in ("evaluated", "evaluated_with_warnings", "manual_review"):
                    evaluated_responses += 1
                elif es == "blocked":
                    blocked_responses += 1
                elif es == "pending":
                    pending_responses += 1

        # Count DCR results
        dcr_count = await tenant_db["exampen_dcr_results"].count_documents(
            {"exam_id": exam_id}
        )

        return {
            "exam_id": exam_id,
            "total_submissions": total_submissions,
            "published_submissions": published_submissions,
            "total_responses": total_responses,
            "evaluated_responses": evaluated_responses,
            "blocked_responses": blocked_responses,
            "pending_responses": pending_responses,
            "pcr_available": True,
            "dcr_results_count": dcr_count,
        }

    except Exception as exc:
        _logger.warning(
            "Failed to get ExamPen eval status for exam %s: %s",
            exam_id,
            exc,
        )
        return None
