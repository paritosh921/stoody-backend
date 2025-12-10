"""
Async Admin API for SkillBot
Admin management endpoints with authentication and rate limiting
"""

import logging
import secrets
import string
import bcrypt
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user, get_database, get_cache
from config_async import settings, MONGODB_URL, DISABLE_MONGODB

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Utility functions
def generate_secure_password(length: int = 12) -> str:
    """Generate a cryptographically secure password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    # Ensure password has at least one digit and one special char
    if not any(c.isdigit() for c in password):
        password = password[:-1] + secrets.choice(string.digits)
    if not any(c in "!@#$%^&*" for c in password):
        password = password[:-1] + secrets.choice("!@#$%^&*")
    return password

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    # Bcrypt has a 72 byte limit, truncate if necessary
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')

# Pydantic models
class CreateStudentRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: str = Field(..., min_length=2, max_length=100)
    password: Optional[str] = Field(None, min_length=6)  # Optional - will auto-generate if not provided
    email: Optional[EmailStr] = None
    date_of_birth: Optional[str] = None  # Format: YYYY-MM-DD
    gender: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    stream: Optional[str] = None
    grade: Optional[str] = None
    phone: Optional[str] = None
    plan_types: Optional[List[str]] = None
    subjects: Optional[List[str]] = None

class UpdateStudentRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None

class StudentResponse(BaseModel):
    id: str
    student_id: str
    username: str
    full_name: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    stream: Optional[str] = None
    grade: Optional[str] = None
    phone: Optional[str] = None
    plan_types: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    is_active: bool
    requires_password_change: Optional[bool] = None
    password_reset_requested: Optional[bool] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    generated_password: Optional[str] = None  # Only included on creation if auto-generated

class StudentsListResponse(BaseModel):
    students: List[StudentResponse]
    total: int
    page: int
    limit: int

class DashboardStats(BaseModel):
    total_students: int
    valid_students: int  # Students with is_active = true
    active_students: int  # Students who have logged in (have last_login)
    practice_sets_count: int
    test_series_count: int
    chapter_notes_count: int

def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access"""
    if current_user.get("user_type") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Allow both admin and tutor roles"""
    if current_user.get("user_type") not in ["admin", "tutor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Tutor access required"
        )
    return current_user

async def calculate_streak_days(student_id: ObjectId, db: DatabaseManager) -> int:
    """Calculate consecutive login days for a student"""
    try:
        # Get all login activities sorted by timestamp descending
        login_activities = await db.mongo_find(
            "student_activity_log",
            {"student_id": student_id, "action": "login"},
            sort=[("timestamp", -1)],
            limit=365  # Only check last year
        )

        if not login_activities:
            return 0

        # Extract unique login dates (ignore time)
        login_dates = []
        for activity in login_activities:
            timestamp = activity.get("timestamp")
            if timestamp:
                login_date = timestamp.date()
                if not login_dates or login_dates[-1] != login_date:
                    login_dates.append(login_date)

        if not login_dates:
            return 0

        # Calculate streak from most recent date backwards
        today = datetime.utcnow().date()
        streak = 0

        # Check if user logged in today or yesterday (streak is still active)
        most_recent = login_dates[0]
        if most_recent == today or most_recent == today - timedelta(days=1):
            streak = 1
            expected_date = most_recent - timedelta(days=1)

            # Count consecutive days
            for i in range(1, len(login_dates)):
                if login_dates[i] == expected_date:
                    streak += 1
                    expected_date -= timedelta(days=1)
                elif login_dates[i] < expected_date:
                    # Gap in streak, stop counting
                    break

        return streak

    except Exception as e:
        logger.error(f"Calculate streak error: {str(e)}")
        return 0

async def calculate_student_level(student_id: ObjectId, db: DatabaseManager) -> tuple[int, int]:
    """Calculate student level and XP based on activities"""
    try:
        # Get all question attempts
        attempts = await db.mongo_find(
            "question_attempts",
            {"student_id": student_id}
        )

        # Calculate XP
        total_xp = 0
        for attempt in attempts:
            if attempt.get("is_correct"):
                # Award XP based on difficulty
                difficulty = attempt.get("metadata", {}).get("difficulty", "medium")
                if difficulty == "easy":
                    total_xp += 5
                elif difficulty == "medium":
                    total_xp += 10
                elif difficulty == "hard":
                    total_xp += 20
                else:
                    total_xp += 10  # Default
            else:
                # Small XP for attempting even if wrong
                total_xp += 1

        # Get chat sessions for bonus XP
        sessions = await db.mongo_find(
            "chat_sessions",
            {"student_id": student_id}
        )
        # 2 XP per chat session
        total_xp += len(sessions) * 2

        # Calculate level (100 XP per level)
        level = max(1, (total_xp // 100) + 1)

        return level, total_xp

    except Exception as e:
        logger.error(f"Calculate level error: {str(e)}")
        return 1, 0

@router.get("/students", response_model=StudentsListResponse)
@limiter.limit("30/minute")
async def get_students(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, max_length=100),
    is_active: Optional[bool] = Query(None),
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get paginated list of students"""
    try:
        # Get admin_id from JWT token - filter by tenant
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Build filter
        filter_dict = {"admin_id": admin_id}  # NEW - Multi-tenancy filter

        if search:
            filter_dict["$or"] = [
                {"student_id": {"$regex": search, "$options": "i"}},
                {"username": {"$regex": search, "$options": "i"}},
                {"full_name": {"$regex": search, "$options": "i"}},
                {"email": {"$regex": search, "$options": "i"}},
                {"phone": {"$regex": search, "$options": "i"}}
            ]
        if is_active is not None:
            filter_dict["is_active"] = is_active

        # Check cache first (include admin_id for tenant isolation)
        cache_key = f"students:{str(admin_id)}:{page}:{limit}:{search}:{is_active}"
        cached_result = await cache.get(cache_key, "admin")

        if cached_result:
            return StudentsListResponse(**cached_result)

        # Get total count
        total_students = len(await db.mongo_find("students", filter_dict))

        # Get paginated results
        skip = (page - 1) * limit
        students_data = await db.mongo_find(
            "students",
            filter_dict,
            projection={"password_hash": 0},  # Exclude password
            sort=[("created_at", -1)],
            skip=skip,
            limit=limit
        )

        students = []
        for student in students_data:
            students.append(
                StudentResponse(
                    id=str(student.get("_id") or student.get("id")),
                    student_id=str(student.get("student_id") or student.get("_id") or student.get("id")),
                    username=student.get("username", ""),
                    full_name=(
                        student.get("full_name")
                        or student.get("name")
                        or student.get("username", "")
                    ),
                    email=student.get("email"),
                    date_of_birth=student.get("date_of_birth"),
                    gender=student.get("gender"),
                    location=student.get("location"),
                    school=student.get("school"),
                    stream=student.get("stream"),
                    grade=student.get("grade"),
                    phone=student.get("phone"),
                    plan_types=student.get("plan_types"),
                    subjects=student.get("subjects"),
                    is_active=student.get("is_active", True),
                    requires_password_change=student.get("requires_password_change", False),
                    password_reset_requested=student.get("password_reset_requested", False),
                    created_at=student.get("created_at", datetime.utcnow()),
                    last_login=student.get("last_login")
                )
            )

        response_data = {
            "students": [s.dict() for s in students],
            "total": total_students,
            "page": page,
            "limit": limit
        }

        # Cache the result
        await cache.set(cache_key, response_data, 300, "admin")  # 5 minute cache

        return StudentsListResponse(**response_data)

    except Exception as e:
        logger.error(f"Get students error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get students"
        )

@router.put("/students/{student_id}", response_model=StudentResponse)
@limiter.limit("20/minute")
async def update_student(
    request: Request,
    student_id: str,
    update: UpdateStudentRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Update student details or status"""
    try:
        # Find by custom student_id first, then by ObjectId
        query: Dict[str, Any] = {"student_id": student_id}
        student = await db.mongo_find_one("students", query)
        if not student:
            try:
                oid = ObjectId(student_id)
                query = {"_id": oid}
                student = await db.mongo_find_one("students", query)
            except Exception:
                student = None

        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

        update_fields: Dict[str, Any] = {}
        if update.full_name is not None:
            update_fields["full_name"] = update.full_name
            # also keep legacy name in sync
            update_fields["name"] = update.full_name
        if update.email is not None:
            update_fields["email"] = update.email
        if update.is_active is not None:
            update_fields["is_active"] = update.is_active

        if update_fields:
            await db.mongo_update_one("students", query, {"$set": update_fields})

        # Return updated document
        updated = await db.mongo_find_one("students", query, projection={"password_hash": 0})
        if not updated:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update student")

        return StudentResponse(
            id=str(updated.get("_id")),
            username=updated.get("username", "unknown"),
            full_name=updated.get("full_name") or updated.get("name"),
            name=updated.get("name") or updated.get("full_name"),
            student_id=updated.get("student_id") or str(updated.get("_id")),
            email=updated.get("email"),
            is_active=updated.get("is_active", True),
            created_at=updated.get("created_at") or datetime.utcnow(),
            last_login=updated.get("last_login")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update student error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update student")

class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)

@router.post("/students/{student_id}/reset-password")
@limiter.limit("10/minute")
async def reset_student_password(
    request: Request,
    student_id: str,
    payload: ResetPasswordRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Reset student password"""
    try:
        from core.auth import AuthManager
        auth = AuthManager()

        # Find student
        query: Dict[str, Any] = {"student_id": student_id}
        student = await db.mongo_find_one("students", query)
        if not student:
            try:
                oid = ObjectId(student_id)
                query = {"_id": oid}
                student = await db.mongo_find_one("students", query)
            except Exception:
                student = None

        if not student:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

        # Truncate password to 72 bytes before hashing (bcrypt limit)
        new_password = payload.new_password
        password_bytes = new_password.encode('utf-8')
        if len(password_bytes) > 72:
            logger.warning(f"Password too long ({len(password_bytes)} bytes), truncating to 72 bytes")
            new_password = password_bytes[:72].decode('utf-8', errors='ignore')

        # Update password hash and reset flags
        password_hash = auth.get_password_hash(new_password)
        ok = await db.mongo_update_one(
            "students",
            query,
            {
                "$set": {
                    "password_hash": password_hash,
                    "requires_password_change": True,  # Force password change on next login
                    "password_reset_requested": False,  # Clear reset request flag
                    "password_reset_by_admin_at": datetime.utcnow()
                }
            }
        )
        if not ok:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reset password")

        return {"success": True, "message": "Password reset successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset student password error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reset password")
@router.post("/students", response_model=StudentResponse)
@limiter.limit("10/minute")
async def create_student(
    request: Request,
    student_data: CreateStudentRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Create a new student"""
    try:
        # Ensure MongoDB is available
        if await db.get_mongo_db() is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MongoDB is not configured or unavailable"
            )

        from core.auth import AuthManager
        auth_manager = AuthManager()

        # Get admin_id from JWT token
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Check if username already exists GLOBALLY (not just within admin's tenant)
        # This ensures globally unique usernames across entire platform
        if student_data.username:
            existing_student = await db.mongo_find_one("students", {
                "username": student_data.username
            })
            if existing_student:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists. Please choose a different username."
                )
        else:
            # Auto-generate username from full_name
            base_name = "".join(c for c in student_data.full_name.lower() if c.isalnum() or c == ' ').strip().replace(' ', '.')
            if not base_name:
                base_name = "student"
            
            # Find a unique username
            username = base_name
            counter = 1
            while True:
                existing = await db.mongo_find_one("students", {"username": username})
                if not existing:
                    student_data.username = username
                    break
                username = f"{base_name}{counter}"
                counter += 1

        # Check if email already exists within this admin's tenant
        # Email can be duplicated across different admins, but not within same admin
        if student_data.email:
            existing_email = await db.mongo_find_one("students", {
                "admin_id": admin_id,
                "email": student_data.email
            })
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already exists in your organization"
                )

        # Auto-generate student_id based on username and timestamp
        import time
        auto_student_id = f"STU_{student_data.username}_{int(time.time() * 1000) % 1000000}"

        # Auto-generate password if not provided
        plain_password = student_data.password or generate_secure_password()
        password_hash = hash_password(plain_password)

        # Create student document
        new_student = {
            "admin_id": admin_id,  # Links student to admin for multi-tenancy
            "student_id": auto_student_id,  # Auto-generated unique ID
            "username": student_data.username,
            "full_name": student_data.full_name,
            "name": student_data.full_name,  # Keep legacy field for compatibility
            "email": student_data.email,
            "password_hash": password_hash,
            "date_of_birth": student_data.date_of_birth,  # Store as string YYYY-MM-DD
            "gender": student_data.gender,
            "location": student_data.location,
            "school": student_data.school,
            "stream": student_data.stream,
            "grade": student_data.grade,
            "phone": student_data.phone,
            "plan_types": student_data.plan_types,
            "subjects": student_data.subjects,
            "is_active": True,
            "requires_password_change": True,  # NEW - Force password change on first login
            "created_at": datetime.utcnow(),
            "created_by": current_user["user_id"]
        }

        # Insert student
        inserted_id = await db.mongo_insert_one("students", new_student)
        if not inserted_id:
            # Distinguish likely causes
            if await db.get_mongo_db() is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database unavailable"
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create student"
            )

        # Invalidate cached students lists and dashboard stats
        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

        # Return created student with generated password (if applicable)
        return StudentResponse(
            id=inserted_id,
            student_id=auto_student_id,
            username=student_data.username,
            full_name=student_data.full_name,
            email=student_data.email,
            date_of_birth=student_data.date_of_birth,
            gender=student_data.gender,
            location=student_data.location,
            school=student_data.school,
            stream=student_data.stream,
            grade=student_data.grade,
            phone=student_data.phone,
            plan_types=student_data.plan_types,
            subjects=student_data.subjects,
            is_active=True,
            requires_password_change=True,
            password_reset_requested=False,
            created_at=new_student["created_at"],
            generated_password=plain_password if not student_data.password else None  # Only return if auto-generated
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create student error: {str(e)}", exc_info=True)

        # Check for duplicate key error from MongoDB unique index
        if "duplicate key error" in str(e).lower() or "E11000" in str(e):
            if "username" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists. Please choose a different username."
                )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create student"
        )

@router.get("/students/{student_id}", response_model=StudentResponse)
@limiter.limit("60/minute")
async def get_student(
    request: Request,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get student by ID"""
    try:
        # Canonical: look up by business student_id only

        student = await db.mongo_find_one(
            "students",
            {"student_id": student_id},
            projection={"password_hash": 0}
        )
        if not student:
            # Try by MongoDB ObjectId
            try:
                oid = ObjectId(student_id)
                student = await db.mongo_find_one(
                    "students",
                    {"_id": oid},
                    projection={"password_hash": 0}
                )
            except Exception:
                student = None

        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        return StudentResponse(
            id=str(student.get("_id")),
            username=student.get("username", "unknown"),
            full_name=student.get("full_name") or student.get("name"),
            name=student.get("name") or student.get("full_name"),
            student_id=student.get("student_id") or str(student.get("_id")),
            email=student.get("email"),
            is_active=student.get("is_active", True),
            created_at=student.get("created_at") or datetime.utcnow(),
            last_login=student.get("last_login")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get student error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get student"
        )

@router.put("/students/{student_id}", response_model=StudentResponse)
@limiter.limit("20/minute")
async def update_student(
    request: Request,
    student_id: str,
    update_data: UpdateStudentRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Update student by ID"""
    try:
        # Build update dict from provided fields only
        update_dict = {}
        if update_data.full_name is not None:
            update_dict["full_name"] = update_data.full_name
        if update_data.email is not None:
            update_dict["email"] = update_data.email
        if update_data.is_active is not None:
            update_dict["is_active"] = update_data.is_active

        if not update_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        # Update student by student_id
        updated = await db.mongo_update_one(
            "students",
            {"student_id": student_id},
            {"$set": update_dict}
        )


        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        # Get updated student
        student = await db.mongo_find_one("students", {"student_id": student_id})

        # Invalidate cached students lists and dashboard stats
        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

        # Format response
        return StudentResponse(
            id=str(student["_id"]),
            student_id=student["student_id"],
            username=student["username"],
            full_name=student.get("full_name", student.get("name", "")),
            email=student.get("email"),
            age=student.get("age"),
            gender=student.get("gender"),
            location=student.get("location"),
            school=student.get("school"),
            stream=student.get("stream"),
            grade=student.get("grade"),
            phone=student.get("phone"),
            plan_types=student.get("plan_types"),
            subjects=student.get("subjects"),
            is_active=student.get("is_active", True),
            created_at=student.get("created_at", datetime.utcnow()),
            last_login=student.get("last_login")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update student error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update student"
        )

@router.delete("/students/{student_id}")
@limiter.limit("10/minute")
async def delete_student(
    request: Request,
    student_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Delete student by ID"""
    try:
        # Get admin_id from JWT token for multi-tenancy
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Try to find and delete by student_id (business ID) within this admin's tenant
        query = {"admin_id": admin_id, "student_id": student_id}
        deleted = await db.mongo_delete_one("students", query)

        # If not found by student_id, try by ObjectId within this admin's tenant
        if not deleted:
            try:
                oid = ObjectId(student_id)
                query = {"admin_id": admin_id, "_id": oid}
                deleted = await db.mongo_delete_one("students", query)
            except Exception:
                pass

        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

        # Invalidate cached students lists and dashboard stats
        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

        logger.info(f"Student deleted successfully: {student_id} by admin {admin_id}")
        return {"message": "Student deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete student error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete student"
        )

@router.get("/dashboard/stats", response_model=DashboardStats)
@limiter.limit("30/minute")
async def get_dashboard_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get admin dashboard statistics"""
    try:
        # Get admin_id for data isolation
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Check cache first (include admin_id for isolation)
        cache_key = f"dashboard_stats_{admin_id}"
        cached_stats = await cache.get(cache_key, "admin")
        if cached_stats:
            return DashboardStats(**cached_stats)

        # Get statistics filtered by admin_id
        admin_students = await db.mongo_find("students", {"admin_id": admin_id})
        total_students = len(admin_students)

        # Valid students = students with is_active = true
        valid_students = len([s for s in admin_students if s.get("is_active", True)])

        # Active students = students who have logged in at least once (have last_login field)
        active_students = len([s for s in admin_students if s.get("last_login") is not None])

        # Get document counts by type (filtered by admin_id)
        practice_sets = await db.mongo_find("documents", {"document_type": "Practice Sets", "admin_id": admin_id})
        test_series = await db.mongo_find("documents", {"document_type": "Test Series", "admin_id": admin_id})
        chapter_notes = await db.mongo_find("documents", {"document_type": "Chapter Notes", "admin_id": admin_id})

        stats_data = {
            "total_students": total_students,
            "valid_students": valid_students,
            "active_students": active_students,
            "practice_sets_count": len(practice_sets),
            "test_series_count": len(test_series),
            "chapter_notes_count": len(chapter_notes)
        }

        # Cache for 5 minutes
        await cache.set(cache_key, stats_data, 300, "admin")

        return DashboardStats(**stats_data)

    except Exception as e:
        logger.error(f"Dashboard stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dashboard statistics"
        )


@router.get("/dashboard/school-stats")
@limiter.limit("30/minute")
async def get_school_dashboard_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Get school-level dashboard statistics with:
    1. Class/Division level breakdown
    2. Teacher summary (teachers per subject per class, documents uploaded)
    """
    try:
        # Get admin_id for data isolation
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Check cache first
        cache_key = f"school_dashboard_stats_{admin_id}"
        cached_stats = await cache.get(cache_key, "admin")
        if cached_stats:
            return cached_stats

        # Get all students for this admin
        admin_students = await db.mongo_find("students", {"admin_id": admin_id})

        # Get all tutors for this admin
        admin_tutors = await db.mongo_find("tutors", {"created_by": str(admin_id)})

        # Get all documents for this admin
        admin_documents = await db.mongo_find("documents", {"admin_id": admin_id})

        # ===== CLASS/DIVISION LEVEL BREAKDOWN =====
        class_division_stats = {}
        for student in admin_students:
            grade = student.get("grade", "Unknown")
            section = student.get("section", "Unknown")
            is_active = student.get("is_active", True)
            has_logged_in = student.get("last_login") is not None

            key = f"{grade}"
            if key not in class_division_stats:
                class_division_stats[key] = {
                    "grade": grade,
                    "sections": {},
                    "total_students": 0,
                    "active_students": 0,
                    "logged_in_students": 0
                }

            class_division_stats[key]["total_students"] += 1
            if is_active:
                class_division_stats[key]["active_students"] += 1
            if has_logged_in:
                class_division_stats[key]["logged_in_students"] += 1

            # Section-level stats
            if section not in class_division_stats[key]["sections"]:
                class_division_stats[key]["sections"][section] = {
                    "section": section,
                    "total_students": 0,
                    "active_students": 0,
                    "logged_in_students": 0,
                    "subjects": {}
                }

            class_division_stats[key]["sections"][section]["total_students"] += 1
            if is_active:
                class_division_stats[key]["sections"][section]["active_students"] += 1
            if has_logged_in:
                class_division_stats[key]["sections"][section]["logged_in_students"] += 1

            # Track subjects per section
            student_subjects = student.get("subjects", []) or []
            for subj in student_subjects:
                if subj not in class_division_stats[key]["sections"][section]["subjects"]:
                    class_division_stats[key]["sections"][section]["subjects"][subj] = 0
                class_division_stats[key]["sections"][section]["subjects"][subj] += 1

        # Convert to list format for frontend
        class_division_list = []
        for grade_key, grade_data in class_division_stats.items():
            sections_list = []
            for section_key, section_data in grade_data["sections"].items():
                sections_list.append({
                    "section": section_data["section"],
                    "total_students": section_data["total_students"],
                    "active_students": section_data["active_students"],
                    "logged_in_students": section_data["logged_in_students"],
                    "subjects": section_data["subjects"]
                })
            # Sort sections alphabetically
            sections_list.sort(key=lambda x: x["section"])
            class_division_list.append({
                "grade": grade_data["grade"],
                "total_students": grade_data["total_students"],
                "active_students": grade_data["active_students"],
                "logged_in_students": grade_data["logged_in_students"],
                "sections": sections_list
            })
        # Sort by grade
        class_division_list.sort(key=lambda x: str(x["grade"]))

        # ===== TEACHER SUMMARY =====
        teacher_summary = []
        for tutor in admin_tutors:
            tutor_id = tutor.get("tutor_id")
            tutor_name = tutor.get("name", tutor.get("username", "Unknown"))
            tutor_subjects = tutor.get("subjects", []) or []
            tutor_standards = tutor.get("standards", []) or []
            tutor_sections = tutor.get("sections", []) or []

            # Count documents uploaded by this teacher
            tutor_docs = [d for d in admin_documents if tutor_id in (d.get("teacher_ids", []) or [])]
            notes_count = len([d for d in tutor_docs if d.get("document_type") == "Chapter Notes"])
            tests_count = len([d for d in tutor_docs if d.get("document_type") == "Test Series"])
            practice_count = len([d for d in tutor_docs if d.get("document_type") == "Practice Sets"])

            # Count assigned students
            assigned_ids = tutor.get("assigned_student_ids", []) or []

            teacher_summary.append({
                "tutor_id": tutor_id,
                "name": tutor_name,
                "email": tutor.get("email"),
                "subjects": tutor_subjects,
                "standards": tutor_standards,
                "sections": tutor_sections,
                "is_active": tutor.get("is_active", True),
                "assigned_students_count": len(assigned_ids),
                "documents_uploaded": {
                    "notes": notes_count,
                    "tests": tests_count,
                    "practice": practice_count,
                    "total": notes_count + tests_count + practice_count
                }
            })

        # Sort teachers by name
        teacher_summary.sort(key=lambda x: x["name"].lower())

        # ===== DOCUMENT STATS BY CLASS =====
        docs_by_class = {}
        for doc in admin_documents:
            standard = doc.get("standard", "Unknown")
            doc_type = doc.get("document_type", "Unknown")

            if standard not in docs_by_class:
                docs_by_class[standard] = {
                    "grade": standard,
                    "Chapter Notes": 0,
                    "Test Series": 0,
                    "Practice Sets": 0
                }
            if doc_type in docs_by_class[standard]:
                docs_by_class[standard][doc_type] += 1

        docs_by_class_list = list(docs_by_class.values())
        docs_by_class_list.sort(key=lambda x: str(x["grade"]))

        result = {
            "success": True,
            "class_division_stats": class_division_list,
            "teacher_summary": teacher_summary,
            "documents_by_class": docs_by_class_list,
            "totals": {
                "total_classes": len(class_division_list),
                "total_teachers": len(teacher_summary),
                "total_students": len(admin_students),
                "total_documents": len(admin_documents)
            }
        }

        # Cache for 5 minutes
        await cache.set(cache_key, result, 300, "admin")

        return result

    except Exception as e:
        logger.error(f"School dashboard stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get school dashboard statistics"
        )


@router.get("/monitoring/class-section-stats")
@limiter.limit("30/minute")
async def get_class_section_monitoring_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Get class-section level monitoring statistics:
    - Number of students active
    - Notes/Assignments/Tests uploaded per class-section
    - Average usage in minutes
    - Average % completion
    - Average Score
    """
    try:
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Check cache first
        cache_key = f"class_section_monitoring_{admin_id}"
        cached_stats = await cache.get(cache_key, "admin")
        if cached_stats:
            return cached_stats

        # Get all students for this admin
        admin_students = await db.mongo_find("students", {"admin_id": admin_id})

        # Get all documents for this admin
        admin_documents = await db.mongo_find("documents", {"admin_id": admin_id})

        # Get all question attempts for students of this admin
        student_ids = [s["_id"] for s in admin_students]
        all_attempts = await db.mongo_find(
            "question_attempts",
            {"student_id": {"$in": student_ids}}
        ) if student_ids else []

        # Get all chat sessions for students of this admin
        all_sessions = await db.mongo_find(
            "chat_sessions",
            {"student_id": {"$in": student_ids}}
        ) if student_ids else []

        # Build class-section level stats
        class_section_stats = {}

        for student in admin_students:
            grade = student.get("grade", "Unknown")
            section = student.get("section", "Unknown")
            key = f"{grade}_{section}"

            if key not in class_section_stats:
                class_section_stats[key] = {
                    "grade": grade,
                    "section": section,
                    "total_students": 0,
                    "active_students": 0,  # has logged in
                    "online_students": 0,  # currently online
                    "documents": {
                        "notes": 0,
                        "tests": 0,
                        "practice": 0
                    },
                    "total_time_minutes": 0,
                    "total_problems_attempted": 0,
                    "total_correct": 0,
                    "total_score_sum": 0,
                    "score_count": 0,
                    "students_with_activity": 0
                }

            class_section_stats[key]["total_students"] += 1

            # Active = has logged in at least once
            if student.get("last_login") is not None:
                class_section_stats[key]["active_students"] += 1

            # Online = currently online
            if student.get("is_online", False):
                class_section_stats[key]["online_students"] += 1

            # Calculate student-specific metrics
            student_id = student["_id"]

            # Sessions and time
            student_sessions = [s for s in all_sessions if s.get("student_id") == student_id]
            student_time = sum(s.get("duration", 0) for s in student_sessions) / 60  # convert to minutes
            class_section_stats[key]["total_time_minutes"] += student_time

            # Attempts and scores
            student_attempts = [a for a in all_attempts if a.get("student_id") == student_id]
            if student_attempts:
                class_section_stats[key]["students_with_activity"] += 1
                class_section_stats[key]["total_problems_attempted"] += len(student_attempts)
                class_section_stats[key]["total_correct"] += sum(1 for a in student_attempts if a.get("is_correct", False))

                scores = [a.get("score", 0) for a in student_attempts if "score" in a]
                if scores:
                    class_section_stats[key]["total_score_sum"] += sum(scores)
                    class_section_stats[key]["score_count"] += len(scores)

        # Add document counts per class-section
        for doc in admin_documents:
            standard = doc.get("standard", "Unknown")
            section = doc.get("section", "Unknown")
            doc_type = doc.get("document_type", "")
            key = f"{standard}_{section}"

            if key in class_section_stats:
                if doc_type == "Chapter Notes":
                    class_section_stats[key]["documents"]["notes"] += 1
                elif doc_type == "Test Series":
                    class_section_stats[key]["documents"]["tests"] += 1
                elif doc_type == "Practice Sets":
                    class_section_stats[key]["documents"]["practice"] += 1

        # Convert to list and calculate averages
        result_list = []
        for key, stats in class_section_stats.items():
            total = stats["total_students"]
            with_activity = stats["students_with_activity"]
            score_count = stats["score_count"]

            avg_time = round(stats["total_time_minutes"] / total, 1) if total > 0 else 0
            avg_problems = round(stats["total_problems_attempted"] / total, 1) if total > 0 else 0

            # Completion % = (students with activity / total students) * 100
            completion_pct = round((with_activity / total) * 100, 1) if total > 0 else 0

            # Average score from all attempts
            avg_score = round(stats["total_score_sum"] / score_count, 1) if score_count > 0 else 0

            # Accuracy = correct / attempted
            accuracy = round(
                (stats["total_correct"] / stats["total_problems_attempted"]) * 100, 1
            ) if stats["total_problems_attempted"] > 0 else 0

            result_list.append({
                "grade": stats["grade"],
                "section": stats["section"],
                "total_students": stats["total_students"],
                "active_students": stats["active_students"],
                "online_students": stats["online_students"],
                "documents": stats["documents"],
                "avg_usage_minutes": avg_time,
                "avg_problems_per_student": avg_problems,
                "avg_completion_pct": completion_pct,
                "avg_score": avg_score,
                "avg_accuracy": accuracy
            })

        # Sort by grade, then section
        result_list.sort(key=lambda x: (str(x["grade"]), x["section"]))

        result = {
            "success": True,
            "class_section_stats": result_list,
            "totals": {
                "total_classes": len(set(s["grade"] for s in result_list)),
                "total_sections": len(result_list),
                "total_students": sum(s["total_students"] for s in result_list),
                "total_active": sum(s["active_students"] for s in result_list),
                "total_documents": len(admin_documents)
            }
        }

        # Cache for 5 minutes
        await cache.set(cache_key, result, 300, "admin")

        return result

    except Exception as e:
        logger.error(f"Class section monitoring stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get class section monitoring statistics"
        )


@router.get("/monitoring/student-progress")
@limiter.limit("30/minute")
async def get_student_progress(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Get student progress monitoring data"""
    try:
        user_type = current_user.get("user_type")
        students = []
        if user_type == "admin":
            admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
            students = await db.mongo_find(
                "students",
                {"admin_id": admin_id},
                projection={"password_hash": 0}
            )
        else:
            # Tutor scoping: assigned + teacher_ids + criteria-based match
            tutor_id = current_user.get("tutor_id")
            admin_id_str = current_user.get("admin_id")
            admin_oid = None
            try:
                admin_oid = ObjectId(admin_id_str) if admin_id_str else None
            except Exception:
                admin_oid = None

            # 1) Students explicitly mapped via teacher_ids
            mapped = await db.mongo_find(
                "students",
                {"teacher_ids": {"$in": [tutor_id]}},
                projection={"password_hash": 0}
            )

            # 2) Students listed in tutor.assigned_student_ids (business student_id)
            tutor_doc = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
            assigned = []
            if tutor_doc and tutor_doc.get("assigned_student_ids"):
                assigned_ids = tutor_doc.get("assigned_student_ids", [])
                if assigned_ids:
                    assigned = await db.mongo_find(
                        "students",
                        {"student_id": {"$in": assigned_ids}},
                        projection={"password_hash": 0}
                    )

            # 3) Criteria-based: same admin, OR overlap on standards/sections/subjects/plan_types
            criteria_matches = []
            if tutor_doc and admin_oid is not None:
                base = {"admin_id": admin_oid}
                or_filters = []
                standards = tutor_doc.get("standards") or []
                if standards:
                    or_filters.append({"grade": {"$in": standards}})
                sections = tutor_doc.get("sections") or []
                if sections:
                    or_filters.append({"section": {"$in": sections}})
                subjects = tutor_doc.get("subjects") or []
                if subjects:
                    or_filters.append({"subjects": {"$in": subjects}})
                plans = tutor_doc.get("plan_types") or []
                if plans:
                    or_filters.append({"plan_types": {"$in": plans}})

                # If no profile filters are set, default to all admin students
                if not or_filters:
                    criteria_matches = await db.mongo_find(
                        "students",
                        base,
                        projection={"password_hash": 0}
                    )
                else:
                    criteria_matches = await db.mongo_find(
                        "students",
                        {"$and": [base, {"$or": or_filters}]},
                        projection={"password_hash": 0}
                    )

            # Union and deduplicate by _id
            def _uniq(stus):
                seen = set()
                out = []
                for s in stus:
                    sid = str(s.get("_id"))
                    if sid not in seen:
                        seen.add(sid)
                        out.append(s)
                return out

            students = _uniq(mapped + assigned + criteria_matches)

        progress_data = []
        for student in students:
            student_oid = student["_id"]

            # Get student's chat sessions
            sessions = await db.mongo_find(
                "chat_sessions",
                {"student_id": student_oid}
            )

            # Get student's question attempts
            attempts = await db.mongo_find(
                "question_attempts",
                {"student_id": student_oid}
            )

            # Calculate total time spent (in minutes)
            total_time = sum(session.get("duration", 0) for session in sessions) / 60 if sessions else 0

            # Calculate average score from attempts
            scores = [attempt.get("score", 0) for attempt in attempts if "score" in attempt]
            avg_score = sum(scores) / len(scores) if scores else 0

            # Calculate problems solved (correct attempts)
            problems_solved = sum(1 for attempt in attempts if attempt.get("is_correct", False))

            # Calculate streak days
            streak_days = await calculate_streak_days(student_oid, db)

            # Calculate level and XP
            level, xp = await calculate_student_level(student_oid, db)

            progress_data.append({
                "student_id": str(student_oid),
                "student_name": student.get("full_name", student.get("name", "Unknown")),
                "email": student.get("email", ""),
                "grade": student.get("grade", "Unknown"),
                "section": student.get("section", "Unknown"),
                "total_sessions": len(sessions),
                "total_time_spent": int(total_time),
                "problems_solved": problems_solved,
                "average_score": round(avg_score, 1),
                "last_active_at": student.get("last_login", student.get("updated_at")),
                "streak_days": streak_days,
                "level": level,
                "xp": xp,
                "is_online": student.get("is_online", False)
            })

        return {
            "success": True,
            "data": progress_data
        }

    except Exception as e:
        logger.error(f"Get student progress error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get student progress"
        )

@router.get("/monitoring/recent-activities")
@limiter.limit("30/minute")
async def get_recent_activities(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Get recent student activities"""
    try:
        user_type = current_user.get("user_type")
        # Determine scoped student IDs
        scoped_student_ids = []
        if user_type == "admin":
            admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
            admin_students = await db.mongo_find("students", {"admin_id": admin_id}, projection={"_id": 1})
            scoped_student_ids = [s["_id"] for s in admin_students]
        else:
            tutor_id = current_user.get("tutor_id")
            admin_id_str = current_user.get("admin_id")
            admin_oid = None
            try:
                admin_oid = ObjectId(admin_id_str) if admin_id_str else None
            except Exception:
                admin_oid = None

            # teacher_ids mapping
            tutor_students = await db.mongo_find("students", {"teacher_ids": {"$in": [tutor_id]}}, projection={"_id": 1})

            # assigned_student_ids mapping
            tutor_doc = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
            assigned = []
            if tutor_doc and tutor_doc.get("assigned_student_ids"):
                assigned_ids = tutor_doc.get("assigned_student_ids", [])
                if assigned_ids:
                    assigned = await db.mongo_find(
                        "students",
                        {"student_id": {"$in": assigned_ids}},
                        projection={"_id": 1}
                    )

            # criteria matching (OR across attributes, scoped to admin)
            criteria_ids = []
            if tutor_doc and admin_oid is not None:
                base = {"admin_id": admin_oid}
                or_filters = []
                if tutor_doc.get("standards"):
                    or_filters.append({"grade": {"$in": tutor_doc.get("standards")}})
                if tutor_doc.get("sections"):
                    or_filters.append({"section": {"$in": tutor_doc.get("sections")}})
                if tutor_doc.get("subjects"):
                    or_filters.append({"subjects": {"$in": tutor_doc.get("subjects")}})
                if tutor_doc.get("plan_types"):
                    or_filters.append({"plan_types": {"$in": tutor_doc.get("plan_types")}})
                if not or_filters:
                    crit_students = await db.mongo_find("students", base, projection={"_id": 1})
                else:
                    crit_students = await db.mongo_find("students", {"$and": [base, {"$or": or_filters}]}, projection={"_id": 1})
                criteria_ids = [s["_id"] for s in crit_students]

            id_set = set([str(s["_id"]) for s in tutor_students] + [str(s["_id"]) for s in assigned] + [str(x) for x in criteria_ids])
            scoped_student_ids = [ObjectId(x) for x in id_set]

        # Get recent activities for scoped students
        recent_activity_logs = await db.mongo_find(
            "student_activity_log",
            {"student_id": {"$in": scoped_student_ids}},
            sort=[("timestamp", -1)],
            limit=limit * 2
        )

        # Get recent chat sessions for scoped students
        recent_sessions = await db.mongo_find(
            "chat_sessions",
            {"student_id": {"$in": scoped_student_ids}},
            sort=[("created_at", -1)],
            limit=limit
        )

        # Get recent question attempts for scoped students
        recent_attempts = await db.mongo_find(
            "question_attempts",
            {"student_id": {"$in": scoped_student_ids}},
            sort=[("created_at", -1)],
            limit=limit
        )

        activities = []

        # Process activity logs (login, logout, chat_message)
        for log in recent_activity_logs:
            student_id = log.get("student_id")
            student = await db.mongo_find_one("students", {"_id": student_id})

            activity = {
                "id": str(log.get("_id")),
                "student_id": str(student_id) if student_id else None,
                "student_name": student.get("full_name", student.get("name", "Unknown")) if student else "Unknown",
                "action": log.get("action"),
                "timestamp": log.get("timestamp"),
            }

            # Add metadata based on action type
            metadata = log.get("metadata", {})
            if log.get("action") == "login":
                activity["metadata"] = {
                    "ip_address": metadata.get("ip_address"),
                    "user_agent": metadata.get("user_agent")
                }
            elif log.get("action") == "session_end":
                activity["session_duration"] = metadata.get("session_duration")
            elif log.get("action") == "chat_message":
                activity["metadata"] = {
                    "session_id": metadata.get("session_id"),
                    "mode": metadata.get("mode")
                }

            activities.append(activity)

        # Process chat sessions
        for session in recent_sessions:
            student_id = session.get("student_id")
            student = await db.mongo_find_one("students", {"_id": student_id})
            activities.append({
                "id": str(session.get("_id")),
                "student_id": str(student_id) if student_id else None,
                "student_name": student.get("full_name", student.get("name", "Unknown")) if student else "Unknown",
                "action": "problem_solving",
                "timestamp": session.get("created_at"),
                "session_duration": session.get("duration"),
                "problem_details": {"session_id": str(session.get("_id"))}
            })

        # Process question attempts
        for attempt in recent_attempts:
            student_id = attempt.get("student_id")
            student = await db.mongo_find_one("students", {"_id": student_id})
            activities.append({
                "id": str(attempt.get("_id")),
                "student_id": str(student_id) if student_id else None,
                "student_name": student.get("full_name", student.get("name", "Unknown")) if student else "Unknown",
                "action": "question_attempted",
                "timestamp": attempt.get("created_at"),
                "score": attempt.get("score"),
                "problem_details": {
                    "correct": attempt.get("is_correct", False),
                    "question_id": str(attempt.get("question_id"))
                }
            })

        # Sort by timestamp descending
        activities.sort(key=lambda x: x["timestamp"] or datetime.min, reverse=True)

        return {
            "success": True,
            "data": activities[:limit]
        }

    except Exception as e:
        logger.error(f"Get recent activities error: {str(e)}")
        return {
            "success": True,
            "data": []
        }

@router.get("/database/status")
@limiter.limit("30/minute")
async def get_database_status(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Get database connection status"""
    try:
        # Determine MongoDB status without forcing a ping (to avoid SSL timeouts)
        mongo_connected = getattr(db, 'mongo_client', None) is not None and not DISABLE_MONGODB
        mongo_status = (
            "online" if mongo_connected else ("disabled" if DISABLE_MONGODB or not MONGODB_URL else "offline")
        )

        # Check ChromaDB connection
        chroma_healthy = False
        chroma_count = 0
        try:
            chroma_count = await db.chroma_count()
            chroma_healthy = True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {str(e)}")

        return {
            "success": True,
            "status": {
                "mongodb": {
                    "connected": mongo_connected,
                    "status": mongo_status
                },
                "chromadb": {
                    "connected": chroma_healthy,
                    "status": "online" if chroma_healthy else "offline",
                    "questions_count": chroma_count
                }
            }
        }

    except Exception as e:
        logger.error(f"Get database status error: {str(e)}")
        return {
            "success": False,
            "status": {
                "mongodb": {"connected": False, "status": "offline"},
                "chromadb": {"connected": False, "status": "offline", "questions_count": 0}
            }
        }


@router.post("/validate-test-series/{document_id}")
@limiter.limit("30/minute")
async def validate_test_series(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Validate test series before making it available to students:
    1. Total points matches sum of question points
    2. Total minutes > 0
    3. All questions have correct_answer set
    """
    try:
        # Get document
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.get("document_type") != "Test Series":
            raise HTTPException(status_code=400, detail="Only Test Series can be validated")

        # Get questions
        questions = await db.mongo_find("questions", {"document_id": document_id})

        if not questions or len(questions) == 0:
            raise HTTPException(
                status_code=400,
                detail="No questions found for this test series. Please process OCR first."
            )

        # Validation 1: Total points
        total_allocated = sum(q.get("points", 1) for q in questions)
        total_points = document.get("total_points", 0)
        if abs(total_allocated - total_points) >= 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Total points ({total_points}) doesn't match allocated points ({total_allocated})"
            )

        # Validation 2: Total minutes
        total_minutes = document.get("total_minutes", 0)
        if not total_minutes or total_minutes <= 0:
            raise HTTPException(status_code=400, detail="Total minutes must be greater than 0")

        # Validation 3: All questions have correct_answer
        questions_without_answer = [q for q in questions if not q.get("correct_answer") or q.get("correct_answer", "").strip() == ""]
        if questions_without_answer:
            raise HTTPException(
                status_code=400,
                detail=f"{len(questions_without_answer)} question(s) don't have correct answer set"
            )

        # Mark as validated
        await db.mongo_update_one(
            "documents",
            {"document_id": document_id},
            {"$set": {"is_validated": True, "validated_at": datetime.utcnow()}}
        )

        logger.info(f"Test series {document_id} validated successfully")

        return {
            "success": True,
            "message": "Test series validated successfully",
            "data": {
                "total_questions": len(questions),
                "total_points": total_points,
                "total_minutes": total_minutes
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate test series: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-attempts")
@limiter.limit("30/minute")
async def get_all_test_attempts(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all test attempts across all students for admin leaderboard
    """
    try:
        # Scope attempts to either admin's students or tutor's students
        from typing import List as _List
        user_type = current_user.get("user_type")
        student_ids: _List[str] = []

        if user_type == "admin":
            admin_oid = ObjectId(current_user.get("admin_id", current_user["user_id"]))
            admin_students = await db.mongo_find("students", {"admin_id": admin_oid}, projection={"_id": 1})
            student_ids = [str(s["_id"]) for s in admin_students]
        else:
            tutor_id = current_user.get("tutor_id")
            tutor_students = await db.mongo_find("students", {"teacher_ids": {"$in": [tutor_id]}}, projection={"_id": 1})
            student_ids = [str(s["_id"]) for s in tutor_students]

        attempts = await db.mongo_find(
            "student_test_attempts",
            {"student_id": {"$in": student_ids}},
            sort=[("submitted_at", -1)]
        )

        # Format response
        formatted_attempts = []
        for attempt in attempts:
            formatted_attempts.append({
                "attempt_id": str(attempt.get("_id")),
                "student_id": attempt.get("student_id"),
                "student_name": attempt.get("student_name"),
                "student_grade": attempt.get("student_grade"),
                "document_id": attempt.get("document_id"),
                "document_title": attempt.get("document_title"),
                "subject": attempt.get("subject"),
                "score": attempt.get("score", 0),
                "total_points": attempt.get("total_points", 0),
                "percentage": attempt.get("percentage", 0),
                "total_questions": attempt.get("total_questions", 0),
                "correct_count": attempt.get("correct_count", 0),
                "incorrect_count": attempt.get("incorrect_count", 0),
                "unanswered_count": attempt.get("unanswered_count", 0),
                "time_taken": attempt.get("time_taken", 0),
                "total_minutes": attempt.get("total_minutes", 0),
                "can_reattempt": attempt.get("can_reattempt", False),
                "submitted_at": attempt.get("submitted_at").isoformat() if attempt.get("submitted_at") else None
            })

        logger.info(f"Retrieved {len(formatted_attempts)} test attempts")

        return {
            "success": True,
            "data": {
                "attempts": formatted_attempts,
                "total": len(formatted_attempts)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get test attempts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-attempts/{attempt_id}/toggle-reattempt")
@limiter.limit("30/minute")
async def toggle_reattempt(
    request: Request,
    attempt_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """
    Toggle the can_reattempt flag for a test attempt
    Allows/disallows student to re-attempt the test
    """
    try:
        from bson import ObjectId

        # Get the attempt
        attempt = await db.mongo_find_one("student_test_attempts", {"_id": ObjectId(attempt_id)})
        if not attempt:
            raise HTTPException(status_code=404, detail="Test attempt not found")

        # Toggle the flag
        new_value = not attempt.get("can_reattempt", False)

        # Update in database
        await db.mongo_update_one(
            "student_test_attempts",
            {"_id": ObjectId(attempt_id)},
            {"$set": {"can_reattempt": new_value}}
        )

        logger.info(f"Toggled re-attempt for attempt {attempt_id} to {new_value}")

        return {
            "success": True,
            "message": f"Re-attempt {'enabled' if new_value else 'disabled'} successfully",
            "data": {
                "can_reattempt": new_value
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle re-attempt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
