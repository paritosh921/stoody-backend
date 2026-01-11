import logging
from typing import Dict, Any, Optional
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_database, get_cache
from .dependencies import require_admin
from .models import (
    StudentResponse, StudentsListResponse, CreateStudentRequest, 
    UpdateStudentRequest, ResetPasswordRequest
)
from .utils import (
    is_b2c_admin, generate_secure_password, hash_password
)

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

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
        is_b2c = is_b2c_admin(current_user)
        collection = "users" if is_b2c else "students"
        
        # Get admin_id from JWT token - filter by tenant
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Build filter
        if is_b2c:
            filter_dict = {}  # B2C admin sees all B2C users
        else:
            filter_dict = {"admin_id": admin_id}  # Multi-tenancy filter for regular admin

        if search:
            filter_dict["$or"] = [
                {"student_id": {"$regex": search, "$options": "i"}},
                {"username": {"$regex": search, "$options": "i"}},
                {"full_name": {"$regex": search, "$options": "i"}},
                {"name": {"$regex": search, "$options": "i"}},
                {"email": {"$regex": search, "$options": "i"}},
                {"phone": {"$regex": search, "$options": "i"}}
            ]
        if is_active is not None:
            filter_dict["is_active"] = is_active

        # Check cache first
        cache_key = f"students:{'b2c' if is_b2c else str(admin_id)}:{page}:{limit}:{search}:{is_active}"
        cached_result = await cache.get(cache_key, "admin")

        if cached_result:
            return StudentsListResponse(**cached_result)

        # Get data using appropriate database
        start_time = datetime.now()
        if is_b2c:
            all_students = await db.b2c_find(collection, filter_dict)
            total_students = len(all_students)
            skip = (page - 1) * limit
            students_data = await db.b2c_find(
                collection,
                filter_dict,
                projection={"password_hash": 0},
                sort=[("created_at", -1)],
                skip=skip,
                limit=limit
            )
        else:
            total_students = len(await db.mongo_find(collection, filter_dict))
            skip = (page - 1) * limit
            students_data = await db.mongo_find(
                collection,
                filter_dict,
                projection={"password_hash": 0},
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
        # AuthManager usage remains here as it might depend on other core modules

        # Get admin_id from JWT token
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))

        # Check if username already exists GLOBALLY
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

        import time
        auto_student_id = f"STU_{student_data.username}_{int(time.time() * 1000) % 1000000}"

        # Auto-generate password if not provided
        plain_password = student_data.password or generate_secure_password()
        password_hash = hash_password(plain_password)

        new_student = {
            "admin_id": admin_id,
            "student_id": auto_student_id,
            "username": student_data.username,
            "full_name": student_data.full_name,
            "name": student_data.full_name,
            "email": student_data.email,
            "password_hash": password_hash,
            "date_of_birth": student_data.date_of_birth,
            "gender": student_data.gender,
            "location": student_data.location,
            "school": student_data.school,
            "stream": student_data.stream,
            "grade": student_data.grade,
            "phone": student_data.phone,
            "plan_types": student_data.plan_types,
            "subjects": student_data.subjects,
            "is_active": True,
            "requires_password_change": True,
            "created_at": datetime.utcnow(),
            "created_by": current_user["user_id"]
        }

        inserted_id = await db.mongo_insert_one("students", new_student)
        if not inserted_id:
             if await db.get_mongo_db() is None:
                 raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create student")

        # Invalidate cached students lists and dashboard stats
        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

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
            generated_password=plain_password if not student_data.password else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create student error: {str(e)}", exc_info=True)
        if "duplicate key error" in str(e).lower() or "E11000" in str(e):
            if "username" in str(e):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create student")

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
        student = await db.mongo_find_one(
            "students",
            {"student_id": student_id},
            projection={"password_hash": 0}
        )
        if not student:
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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get student")

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
            update_fields["name"] = update.full_name
        if update.email is not None:
            update_fields["email"] = update.email
        if update.grade is not None:
            update_fields["grade"] = update.grade
        if update.section is not None:
            update_fields["section"] = update.section
        if update.is_active is not None:
            update_fields["is_active"] = update.is_active

        if update_fields:
            await db.mongo_update_one("students", query, {"$set": update_fields})

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
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
        query = {"admin_id": admin_id, "student_id": student_id}
        deleted = await db.mongo_delete_one("students", query)

        if not deleted:
            try:
                oid = ObjectId(student_id)
                query = {"admin_id": admin_id, "_id": oid}
                deleted = await db.mongo_delete_one("students", query)
            except Exception:
                pass

        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

        try:
            await cache.clear_pattern("students:*", "admin")
            await cache.delete("dashboard_stats", "admin")
        except Exception:
            pass

        return {"message": "Student deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete student error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete student")

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

        # Truncate password to 72 bytes before hashing
        new_password = payload.new_password
        password_bytes = new_password.encode('utf-8')
        if len(password_bytes) > 72:
            new_password = password_bytes[:72].decode('utf-8', errors='ignore')

        password_hash = auth.get_password_hash(new_password)
        ok = await db.mongo_update_one(
            "students",
            query,
            {
                "$set": {
                    "password_hash": password_hash,
                    "requires_password_change": True,
                    "password_reset_requested": False,
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
