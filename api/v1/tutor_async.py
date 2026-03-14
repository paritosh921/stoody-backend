"""
Tutor Management API Endpoints (Async)
Handles tutor CRUD operations, authentication, and student assignments
"""

from fastapi import APIRouter, Depends, HTTPException, Request
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

    # Update tutor
    await db.mongo_update_one("tutors", {"tutor_id": tutor_id}, {"$set": updates})

    return {"message": "Tutor updated successfully"}


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

    assigned_student_ids = tutor.get("assigned_student_ids", []) or []

    # Get admin_id for data isolation
    admin_id = tutor.get("created_by")
    try:
        from bson import ObjectId

        admin_oid = ObjectId(admin_id) if admin_id else None
    except Exception:
        admin_oid = None

    students_union: List[Dict[str, Any]] = []

    # 1) Students explicitly assigned by id
    # IMPORTANT: Must include admin_id filter for data isolation
    if assigned_student_ids and admin_oid is not None:
        assigned_students = await db.mongo_find(
            "students",
            {"student_id": {"$in": assigned_student_ids}, "admin_id": admin_oid},
        )
        students_union.extend(assigned_students)

    # 2) Students mapped via teacher_ids
    # IMPORTANT: Must include admin_id filter for data isolation
    if admin_oid is not None:
        teacher_mapped = await db.mongo_find(
            "students", {"teacher_ids": {"$in": [tutor_id]}, "admin_id": admin_oid}
        )
        students_union.extend(teacher_mapped)

    # 3) Criteria-based matching within same admin (OR across grade/section/subjects/plan_types)
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
    visible to the current tutor.  Mirrors the scoping logic in
    ``get_tutor_students``.
    """
    from bson import ObjectId

    tutor_id = current_user.get("tutor_id")
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        raise HTTPException(status_code=404, detail="Tutor not found")

    assigned_student_ids = tutor.get("assigned_student_ids", []) or []

    # Get admin_id for data isolation
    admin_id = tutor.get("created_by")
    try:
        admin_oid = ObjectId(admin_id) if admin_id else None
    except Exception:
        admin_oid = None

    students_union: List[Dict[str, Any]] = []

    # 1) Students explicitly assigned by id
    if assigned_student_ids and admin_oid is not None:
        assigned_students = await db.mongo_find(
            "students",
            {"student_id": {"$in": assigned_student_ids}, "admin_id": admin_oid},
        )
        students_union.extend(assigned_students)

    # 2) Students mapped via teacher_ids
    if admin_oid is not None:
        teacher_mapped = await db.mongo_find(
            "students", {"teacher_ids": {"$in": [tutor_id]}, "admin_id": admin_oid}
        )
        students_union.extend(teacher_mapped)

    # 3) Criteria-based matching within same admin
    criteria: Dict[str, Any] = {}
    or_filters: List[Dict[str, Any]] = []
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
    seen: set = set()
    students: List[Dict[str, Any]] = []
    for s in students_union:
        sid = str(s.get("_id"))
        if sid not in seen:
            seen.add(sid)
            students.append(s)

    return students


@router.get("/tutors/analytics/overview")
@limiter.limit("15/minute")
async def get_tutor_analytics_overview(
    request: Request,
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
        # Subject performance
        # ------------------------------------------------------------------
        subject_practice_pipeline = [
            {"$match": {"student_id": {"$in": student_oid_strings}}},
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

        subject_test_pipeline = [
            {"$match": {"student_id": {"$in": student_oid_strings}}},
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

        # ----- Fetch question metadata -----
        questions = await db.mongo_find("questions", {"document_id": document_id})
        question_map: Dict[str, Dict[str, Any]] = {}
        for q in questions:
            qid = q.get("id") or str(q.get("_id", ""))
            question_map[qid] = q

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
                        "_id": "$question_id",
                        "total_attempts": {"$sum": 1},
                        "correct_count": {
                            "$sum": {"$cond": [{"$eq": ["$is_correct", True]}, 1, 0]}
                        },
                        "avg_time": {"$avg": {"$ifNull": ["$time_spent", 0]}},
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
                question_analysis.append(
                    {
                        "question_id": qid,
                        "question_text": q_meta.get("text", ""),
                        "difficulty": q_meta.get("difficulty", ""),
                        "total_attempts": q_total,
                        "correct_count": q_correct,
                        "accuracy": round(
                            (q_correct / q_total * 100) if q_total > 0 else 0.0, 1
                        ),
                        "avg_time": round(qa.get("avg_time", 0) or 0, 1),
                    }
                )

            # Sort by accuracy ascending (hardest first)
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
                        question_stats[qid] = {"total_attempts": 0, "correct_count": 0}
                    
                    # Ensure we only count actual attempts for accuracy
                    is_attempted = qr.get("is_attempted", True)
                    student_ans = str(qr.get("student_answer", "")).strip().upper()
                    if not student_ans or student_ans == "SKIPPED":
                        is_attempted = False
                        
                    if is_attempted:
                        question_stats[qid]["total_attempts"] += 1
                        if qr.get("is_correct"):
                            question_stats[qid]["correct_count"] += 1

            for qid, qs in question_stats.items():
                q_meta = question_map.get(qid, {})
                q_total = qs["total_attempts"]
                q_correct = qs["correct_count"]
                question_analysis.append(
                    {
                        "question_id": qid,
                        "question_text": q_meta.get("text", ""),
                        "difficulty": q_meta.get("difficulty", ""),
                        "total_attempts": q_total,
                        "correct_count": q_correct,
                        "accuracy": round(
                            (q_correct / q_total * 100) if q_total > 0 else 0.0, 1
                        ),
                        "avg_time": 0,
                    }
                )

            # Sort by accuracy ascending (hardest first)
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
