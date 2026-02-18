"""
Timetable Management API Endpoints

Handles class timetable CRUD, teacher calendar generation,
academic calendar (holidays/exams), and pre-read management.

Access:
- Admin: Full CRUD on all timetables and academic calendar
- Teacher (class teacher): Edit timetable for their class, manage pre-reads
- Teacher (subject teacher): Manage pre-reads for their periods
- Student: Read-only timetable + pre-reads for their class
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from bson import ObjectId
import logging

from models.timetable import (
    ClassTimetable,
    TimetablePeriod,
    AcademicCalendarEntry,
    PreRead,
)
from core.database import DatabaseManager
from api.v1.auth_async import get_current_user
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Collection names (tenant-scoped)
# ---------------------------------------------------------------------------
COL_TIMETABLES = "class_timetables"
COL_PERIODS = "timetable_periods"
COL_CALENDAR = "academic_calendar"
COL_PREREADS = "pre_reads"

VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
VALID_PERIOD_TYPES = ["regular", "break", "lunch", "assembly", "library", "sports"]


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("user_type") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("user_type") not in ("admin", "tutor"):
        raise HTTPException(status_code=403, detail="Admin or Teacher access required")
    return current_user


def require_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("user_type") != "tutor":
        raise HTTPException(status_code=403, detail="Teacher access required")
    return current_user


def require_student(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("user_type") != "student":
        raise HTTPException(status_code=403, detail="Student access required")
    return current_user


# ---------------------------------------------------------------------------
# Helper: resolve admin_id for tutor users
# ---------------------------------------------------------------------------
async def resolve_admin_id(current_user: Dict[str, Any], db: DatabaseManager) -> str:
    """Get admin_id from current user context (admin directly, tutor via created_by)"""
    if current_user.get("user_type") == "admin":
        return current_user.get("user_id") or current_user.get("admin_id", "")
    # For tutors, lookup their created_by (admin_id)
    tutor_id = current_user.get("tutor_id")
    if tutor_id:
        tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
        if tutor:
            return tutor.get("created_by", "")
    return current_user.get("admin_id", "")


# ---------------------------------------------------------------------------
# Helper: check if tutor is class teacher for given grade+section
# ---------------------------------------------------------------------------
async def is_class_teacher(
    tutor_id: str, grade: str, section: str, db: DatabaseManager
) -> bool:
    tutor = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor:
        return False
    assignments = tutor.get("teaching_assignments", [])
    class_key = f"{grade}-{section}"
    return any(a.get("class_teacher_of") == class_key for a in assignments)


# ---------------------------------------------------------------------------
# Pydantic Request/Response Models
# ---------------------------------------------------------------------------

class PeriodTimingInput(BaseModel):
    period_number: int = Field(..., ge=1, le=15)
    start_time: str = Field(..., pattern=r"^\d{2}:\d{2}$", description="HH:MM format")
    end_time: str = Field(..., pattern=r"^\d{2}:\d{2}$", description="HH:MM format")
    label: Optional[str] = None  # "Period 1", "Lunch Break", etc.


class CreateTimetableRequest(BaseModel):
    grade: str = Field(..., description="Grade e.g. '10'")
    section: str = Field(..., description="Section e.g. 'A'")
    academic_session: str = Field(..., description="e.g. '2025-26'")
    class_teacher_id: Optional[str] = None
    period_timings: List[PeriodTimingInput] = Field(
        ..., min_length=1, max_length=15,
        description="Period timing slots for the day"
    )
    weekdays: Optional[List[str]] = Field(
        None,
        description="Active weekdays. Defaults to Mon-Fri"
    )


class UpdateTimetableRequest(BaseModel):
    class_teacher_id: Optional[str] = None
    period_timings: Optional[List[PeriodTimingInput]] = None
    weekdays: Optional[List[str]] = None
    is_active: Optional[bool] = None


class PeriodInput(BaseModel):
    day_of_week: str = Field(..., description="Monday, Tuesday, etc.")
    period_number: int = Field(..., ge=1, le=15)
    subject: str = Field(default="", max_length=100)
    teacher_id: Optional[str] = None
    teacher_name: Optional[str] = None
    room: Optional[str] = None
    period_type: str = Field("regular", description="regular, break, lunch, etc.")


class BulkPeriodInput(BaseModel):
    """Save multiple periods at once (full timetable grid)"""
    periods: List[PeriodInput]


class AcademicCalendarInput(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    entry_type: str = Field(..., description="holiday, exam, event, half_day, cancelled")
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=500)
    affects_grades: Optional[List[str]] = Field(None, description="Empty = all grades")
    affects_sections: Optional[List[str]] = None
    is_recurring_yearly: bool = False


class PreReadInput(BaseModel):
    period_id: str = Field(..., description="ID of the timetable_period")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    heading: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    book_page_numbers: Optional[str] = Field(None, max_length=200)
    things_required: Optional[List[str]] = None


class PreReadUpdateInput(BaseModel):
    heading: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    book_page_numbers: Optional[str] = Field(None, max_length=200)
    things_required: Optional[List[str]] = None


# ===========================================================================
# TIMETABLE CRUD (Admin + Class Teacher)
# ===========================================================================

@router.post("/timetables", status_code=201)
@limiter.limit("20/minute")
async def create_timetable(
    request: Request,
    data: CreateTimetableRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Create a new class timetable. Admin can create for any class.
    Class teacher can create for their own class."""
    admin_id = await resolve_admin_id(current_user, db)
    user_type = current_user.get("user_type")

    # If tutor, verify they are class teacher for this grade+section
    if user_type == "tutor":
        tutor_id = current_user.get("tutor_id")
        if not await is_class_teacher(tutor_id, data.grade, data.section, db):
            raise HTTPException(
                status_code=403,
                detail="Only class teachers can create timetables for their class"
            )

    # Validate weekdays
    weekdays = data.weekdays or ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day in weekdays:
        if day not in VALID_DAYS:
            raise HTTPException(status_code=400, detail=f"Invalid weekday: {day}")

    # Check for existing active timetable for this grade+section+session
    existing = await db.mongo_find_one(COL_TIMETABLES, {
        "grade": data.grade,
        "section": data.section,
        "academic_session": data.academic_session,
        "is_active": True,
        "admin_id": admin_id,
    })
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Active timetable already exists for {data.grade}-{data.section} ({data.academic_session}). "
                   "Deactivate the existing one first or update it."
        )

    # Resolve class teacher name
    class_teacher_name = None
    if data.class_teacher_id:
        teacher = await db.mongo_find_one("tutors", {"tutor_id": data.class_teacher_id})
        if teacher:
            class_teacher_name = teacher.get("name", "")

    timetable = ClassTimetable(
        grade=data.grade,
        section=data.section,
        academic_session=data.academic_session,
        class_teacher_id=data.class_teacher_id,
        class_teacher_name=class_teacher_name,
        period_timings=[pt.model_dump() for pt in data.period_timings],
        weekdays=weekdays,
        admin_id=admin_id,
        created_by=current_user.get("user_id") or current_user.get("tutor_id"),
        created_by_role=user_type,
    )

    inserted_id = await db.mongo_insert_one(COL_TIMETABLES, timetable.to_mongo())
    if not inserted_id:
        raise HTTPException(status_code=500, detail="Failed to create timetable")
    timetable._id = inserted_id

    logger.info(
        "Timetable created for %s-%s by %s (%s)",
        data.grade, data.section, current_user.get("user_id"), user_type,
    )

    return {"success": True, "timetable": timetable.to_dict()}


@router.get("/timetables")
@limiter.limit("60/minute")
async def list_timetables(
    request: Request,
    grade: Optional[str] = None,
    section: Optional[str] = None,
    academic_session: Optional[str] = None,
    is_active: Optional[bool] = True,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """List all timetables. Filterable by grade, section, session."""
    admin_id = await resolve_admin_id(current_user, db)
    query: Dict[str, Any] = {"admin_id": admin_id}
    if grade:
        query["grade"] = grade
    if section:
        query["section"] = section
    if academic_session:
        query["academic_session"] = academic_session
    if is_active is not None:
        query["is_active"] = is_active

    docs = await db.mongo_find(COL_TIMETABLES, query)
    timetables = [ClassTimetable.from_dict(d).to_dict() for d in docs]

    return {"success": True, "timetables": timetables, "total": len(timetables)}


@router.get("/timetables/{timetable_id}")
@limiter.limit("60/minute")
async def get_timetable(
    request: Request,
    timetable_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get a timetable with all its periods"""
    try:
        oid = ObjectId(timetable_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timetable ID")

    tt = await db.mongo_find_one(COL_TIMETABLES, {"_id": oid})
    if not tt:
        raise HTTPException(status_code=404, detail="Timetable not found")

    # Get all periods for this timetable
    periods_docs = await db.mongo_find(COL_PERIODS, {"timetable_id": timetable_id})
    periods = [TimetablePeriod.from_dict(p).to_dict() for p in periods_docs]

    timetable = ClassTimetable.from_dict(tt).to_dict()
    timetable["periods"] = periods

    return {"success": True, "timetable": timetable}


@router.put("/timetables/{timetable_id}")
@limiter.limit("20/minute")
async def update_timetable(
    request: Request,
    timetable_id: str,
    data: UpdateTimetableRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Update timetable metadata (class teacher, period timings, weekdays)"""
    try:
        oid = ObjectId(timetable_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timetable ID")

    tt = await db.mongo_find_one(COL_TIMETABLES, {"_id": oid})
    if not tt:
        raise HTTPException(status_code=404, detail="Timetable not found")

    # Tutor must be class teacher to update
    user_type = current_user.get("user_type")
    if user_type == "tutor":
        tutor_id = current_user.get("tutor_id")
        if not await is_class_teacher(tutor_id, tt["grade"], tt["section"], db):
            raise HTTPException(status_code=403, detail="Only class teachers can update their class timetable")

    update_fields: Dict[str, Any] = {"updated_at": datetime.utcnow()}

    if data.class_teacher_id is not None:
        update_fields["class_teacher_id"] = data.class_teacher_id
        teacher = await db.mongo_find_one("tutors", {"tutor_id": data.class_teacher_id})
        update_fields["class_teacher_name"] = teacher.get("name", "") if teacher else ""

    if data.period_timings is not None:
        update_fields["period_timings"] = [pt.model_dump() for pt in data.period_timings]

    if data.weekdays is not None:
        for day in data.weekdays:
            if day not in VALID_DAYS:
                raise HTTPException(status_code=400, detail=f"Invalid weekday: {day}")
        update_fields["weekdays"] = data.weekdays

    if data.is_active is not None:
        update_fields["is_active"] = data.is_active

    await db.mongo_update_one(COL_TIMETABLES, {"_id": oid}, {"$set": update_fields})

    updated = await db.mongo_find_one(COL_TIMETABLES, {"_id": oid})
    return {"success": True, "timetable": ClassTimetable.from_dict(updated).to_dict()}


@router.delete("/timetables/{timetable_id}")
@limiter.limit("10/minute")
async def delete_timetable(
    request: Request,
    timetable_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Delete a timetable and all associated periods + pre-reads. Admin only."""
    try:
        oid = ObjectId(timetable_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timetable ID")

    tt = await db.mongo_find_one(COL_TIMETABLES, {"_id": oid})
    if not tt:
        raise HTTPException(status_code=404, detail="Timetable not found")

    # Delete associated periods and pre-reads
    await db.mongo_delete_many(COL_PERIODS, {"timetable_id": timetable_id})
    await db.mongo_delete_many(COL_PREREADS, {"timetable_id": timetable_id})
    await db.mongo_delete_one(COL_TIMETABLES, {"_id": oid})

    logger.info("Timetable %s deleted by admin %s", timetable_id, current_user.get("user_id"))

    return {"success": True, "message": "Timetable and associated data deleted"}


# ===========================================================================
# PERIOD MANAGEMENT (within a timetable)
# ===========================================================================

@router.post("/timetables/{timetable_id}/periods", status_code=201)
@limiter.limit("30/minute")
async def save_periods(
    request: Request,
    timetable_id: str,
    data: BulkPeriodInput,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Save/replace all periods for a timetable (bulk upsert).
    Replaces existing periods for the given days."""
    try:
        oid = ObjectId(timetable_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timetable ID")

    tt = await db.mongo_find_one(COL_TIMETABLES, {"_id": oid})
    if not tt:
        raise HTTPException(status_code=404, detail="Timetable not found")

    admin_id = await resolve_admin_id(current_user, db)
    grade = tt["grade"]
    section = tt["section"]

    # Tutor must be class teacher
    user_type = current_user.get("user_type")
    if user_type == "tutor":
        tutor_id = current_user.get("tutor_id")
        if not await is_class_teacher(tutor_id, grade, section, db):
            raise HTTPException(status_code=403, detail="Only class teachers can edit periods")

    # Filter out periods with empty subjects (unfilled grid cells)
    valid_periods = [p for p in data.periods if p.subject and p.subject.strip()]

    # Validate inputs
    for p in valid_periods:
        if p.day_of_week not in VALID_DAYS:
            raise HTTPException(status_code=400, detail=f"Invalid day: {p.day_of_week}")
        if p.period_type not in VALID_PERIOD_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid period type: {p.period_type}")

    # Find the timing for each period_number
    timings_map = {
        pt["period_number"]: pt for pt in tt.get("period_timings", [])
    }

    # Delete existing periods for days that appear in input, then insert new ones
    days_in_input = list(set(p.day_of_week for p in valid_periods))
    if days_in_input:
        await db.mongo_delete_many(COL_PERIODS, {
            "timetable_id": timetable_id,
            "day_of_week": {"$in": days_in_input},
        })

    # Resolve teacher names for IDs
    teacher_cache: Dict[str, str] = {}

    periods_to_insert = []
    for p in valid_periods:
        teacher_name = p.teacher_name
        if p.teacher_id and not teacher_name:
            if p.teacher_id not in teacher_cache:
                t = await db.mongo_find_one("tutors", {"tutor_id": p.teacher_id})
                teacher_cache[p.teacher_id] = t.get("name", "") if t else ""
            teacher_name = teacher_cache[p.teacher_id]

        timing = timings_map.get(p.period_number, {})

        period = TimetablePeriod(
            timetable_id=timetable_id,
            grade=grade,
            section=section,
            day_of_week=p.day_of_week,
            period_number=p.period_number,
            start_time=timing.get("start_time", ""),
            end_time=timing.get("end_time", ""),
            subject=p.subject,
            teacher_id=p.teacher_id,
            teacher_name=teacher_name,
            room=p.room,
            period_type=p.period_type,
            admin_id=admin_id,
        )
        periods_to_insert.append(period.to_mongo())

    if periods_to_insert:
        await db.mongo_insert_many(COL_PERIODS, periods_to_insert)

    # Update timetable updated_at
    await db.mongo_update_one(
        COL_TIMETABLES, {"_id": oid}, {"$set": {"updated_at": datetime.utcnow()}}
    )

    logger.info(
        "Saved %d periods for timetable %s-%s", len(periods_to_insert), grade, section
    )

    return {
        "success": True,
        "message": f"{len(periods_to_insert)} periods saved",
        "periods_count": len(periods_to_insert),
    }


@router.get("/timetables/{timetable_id}/periods")
@limiter.limit("60/minute")
async def get_periods(
    request: Request,
    timetable_id: str,
    day_of_week: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get all periods for a timetable, optionally filtered by day"""
    query: Dict[str, Any] = {"timetable_id": timetable_id}
    if day_of_week:
        if day_of_week not in VALID_DAYS:
            raise HTTPException(status_code=400, detail=f"Invalid day: {day_of_week}")
        query["day_of_week"] = day_of_week

    docs = await db.mongo_find(COL_PERIODS, query)
    periods = sorted(
        [TimetablePeriod.from_dict(d).to_dict() for d in docs],
        key=lambda x: (VALID_DAYS.index(x["day_of_week"]) if x["day_of_week"] in VALID_DAYS else 99, x["period_number"]),
    )

    return {"success": True, "periods": periods, "total": len(periods)}


@router.get("/teacher-schedule")
@limiter.limit("30/minute")
async def get_teacher_schedule(
    request: Request,
    exclude_timetable_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Get all teacher-period assignments across all active timetables.
    Used for conflict detection when building timetables.
    Returns which teacher is teaching which class at each (day, period_number) slot.
    """
    admin_id = await resolve_admin_id(current_user, db)

    # Get all active timetable IDs
    active_tts = await db.mongo_find(
        COL_TIMETABLES,
        {"admin_id": admin_id, "is_active": True},
        projection={"_id": 1},
    )
    active_tt_ids = [str(tt["_id"]) for tt in active_tts]

    if not active_tt_ids:
        return {"success": True, "schedule": []}

    # Build query for periods that have a teacher assigned
    period_query: Dict[str, Any] = {
        "timetable_id": {"$in": active_tt_ids},
        "teacher_id": {"$ne": None},
    }

    # Optionally exclude periods from a specific timetable
    if exclude_timetable_id and exclude_timetable_id in active_tt_ids:
        remaining_ids = [tid for tid in active_tt_ids if tid != exclude_timetable_id]
        period_query["timetable_id"] = {"$in": remaining_ids}

    docs = await db.mongo_find(
        COL_PERIODS,
        period_query,
        projection={
            "day_of_week": 1,
            "period_number": 1,
            "teacher_id": 1,
            "teacher_name": 1,
            "grade": 1,
            "section": 1,
            "timetable_id": 1,
            "subject": 1,
        },
    )

    schedule = []
    for d in docs:
        if d.get("teacher_id"):
            schedule.append({
                "day_of_week": d.get("day_of_week"),
                "period_number": d.get("period_number"),
                "teacher_id": d.get("teacher_id"),
                "teacher_name": d.get("teacher_name", ""),
                "grade": d.get("grade", ""),
                "section": d.get("section", ""),
                "subject": d.get("subject", ""),
                "timetable_id": d.get("timetable_id", ""),
            })

    return {"success": True, "schedule": schedule, "total": len(schedule)}


# ===========================================================================
# CLASS TIMETABLE VIEW (Student)
# ===========================================================================

@router.get("/student/timetable")
@limiter.limit("60/minute")
async def get_student_timetable(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
):
    """Get the timetable for the student's class (grade+section)"""
    student_id = current_user.get("student_id")
    student = await db.mongo_find_one("students", {"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    grade = student.get("grade")
    section = student.get("section")
    if not grade or not section:
        raise HTTPException(status_code=400, detail="Student grade/section not set")

    admin_id = student.get("created_by", "")

    # Get active timetable
    tt = await db.mongo_find_one(COL_TIMETABLES, {
        "grade": grade,
        "section": section,
        "is_active": True,
        "admin_id": admin_id,
    })
    if not tt:
        return {"success": True, "timetable": None, "periods": [], "message": "No timetable set for your class"}

    timetable_id = str(tt["_id"])
    timetable = ClassTimetable.from_dict(tt).to_dict()

    # Get all periods
    periods_docs = await db.mongo_find(COL_PERIODS, {"timetable_id": timetable_id})
    periods = sorted(
        [TimetablePeriod.from_dict(p).to_dict() for p in periods_docs],
        key=lambda x: (VALID_DAYS.index(x["day_of_week"]) if x["day_of_week"] in VALID_DAYS else 99, x["period_number"]),
    )

    return {"success": True, "timetable": timetable, "periods": periods}


# ===========================================================================
# TEACHER CALENDAR (auto-generated from class timetables)
# ===========================================================================

@router.get("/teacher/calendar")
@limiter.limit("60/minute")
async def get_teacher_calendar(
    request: Request,
    week_offset: int = 0,
    current_user: Dict[str, Any] = Depends(require_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Get teacher's weekly calendar (auto-generated from all class timetables
    where they teach). Includes pre-reads and academic calendar entries.
    week_offset: 0 = current week, 1 = next week, -1 = last week"""
    tutor_id = current_user.get("tutor_id")
    admin_id = await resolve_admin_id(current_user, db)

    # Find all periods assigned to this teacher across all timetables
    periods_docs = await db.mongo_find(COL_PERIODS, {
        "teacher_id": tutor_id,
        "admin_id": admin_id,
    })

    if not periods_docs:
        return {"success": True, "calendar": [], "week_start": None, "week_end": None}

    # Calculate week dates
    today = datetime.utcnow().date()
    week_start = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
    week_end = week_start + timedelta(days=6)

    # Get academic calendar entries for this week
    calendar_entries = await db.mongo_find(COL_CALENDAR, {
        "admin_id": admin_id,
        "date": {
            "$gte": week_start.isoformat(),
            "$lte": week_end.isoformat(),
        },
    })
    holidays_map: Dict[str, List[Dict]] = {}
    for entry in calendar_entries:
        d = entry.get("date", "")
        if d not in holidays_map:
            holidays_map[d] = []
        holidays_map[d].append(AcademicCalendarEntry.from_dict(entry).to_dict())

    # Get pre-reads for this teacher's periods this week
    period_ids = [str(p["_id"]) for p in periods_docs]
    prereads_docs = await db.mongo_find(COL_PREREADS, {
        "teacher_id": tutor_id,
        "date": {
            "$gte": week_start.isoformat(),
            "$lte": week_end.isoformat(),
        },
    })
    prereads_map: Dict[str, Dict] = {}
    for pr in prereads_docs:
        key = f"{pr.get('period_id')}_{pr.get('date')}"
        prereads_map[key] = PreRead.from_dict(pr).to_dict()

    # Build calendar events
    calendar_events = []
    for period_doc in periods_docs:
        period = TimetablePeriod.from_dict(period_doc)
        day_name = period.day_of_week
        if day_name not in VALID_DAYS:
            continue

        day_index = VALID_DAYS.index(day_name)
        # Monday=0 in VALID_DAYS, weekday() Monday=0 too
        period_date = week_start + timedelta(days=day_index)

        # Skip weekends if not in weekdays
        date_str = period_date.isoformat()

        # Check if this date is a holiday
        is_holiday = False
        holiday_info = None
        if date_str in holidays_map:
            for h in holidays_map[date_str]:
                entry = AcademicCalendarEntry.from_dict(h) if isinstance(h, dict) else h
                # Check if this holiday affects this grade
                grade = period.grade
                section = period.section
                if isinstance(h, dict):
                    ag = h.get("affects_grades", [])
                    a_s = h.get("affects_sections", [])
                    if (not ag or grade in ag) and (not a_s or section in a_s):
                        is_holiday = True
                        holiday_info = h
                        break

        # Look up pre-read
        period_id_str = str(period_doc["_id"])
        preread_key = f"{period_id_str}_{date_str}"
        preread = prereads_map.get(preread_key)

        calendar_events.append({
            "date": date_str,
            "day_of_week": day_name,
            "period_number": period.period_number,
            "start_time": period.start_time,
            "end_time": period.end_time,
            "subject": period.subject,
            "grade": period.grade,
            "section": period.section,
            "room": period.room,
            "period_type": period.period_type,
            "period_id": period_id_str,
            "timetable_id": period.timetable_id,
            "is_holiday": is_holiday,
            "holiday_info": holiday_info,
            "pre_read": preread,
        })

    # Sort by date then period number
    calendar_events.sort(key=lambda e: (e["date"], e["period_number"]))

    return {
        "success": True,
        "calendar": calendar_events,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "total_periods": len(calendar_events),
    }


# ===========================================================================
# ACADEMIC CALENDAR (Holidays, Exams, Events) - Admin only
# ===========================================================================

@router.post("/academic-calendar", status_code=201)
@limiter.limit("20/minute")
async def create_calendar_entry(
    request: Request,
    data: AcademicCalendarInput,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Add a holiday, exam, or event to the academic calendar"""
    if data.entry_type not in AcademicCalendarEntry.VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entry type. Must be one of: {AcademicCalendarEntry.VALID_TYPES}"
        )

    admin_id = current_user.get("user_id") or current_user.get("admin_id", "")

    entry = AcademicCalendarEntry(
        date=data.date,
        entry_type=data.entry_type,
        title=data.title,
        description=data.description,
        affects_grades=data.affects_grades or [],
        affects_sections=data.affects_sections or [],
        is_recurring_yearly=data.is_recurring_yearly,
        admin_id=admin_id,
        created_by=admin_id,
    )

    inserted_id = await db.mongo_insert_one(COL_CALENDAR, entry.to_mongo())
    if not inserted_id:
        raise HTTPException(status_code=500, detail="Failed to create calendar entry")
    entry._id = inserted_id

    return {"success": True, "entry": entry.to_dict()}


@router.get("/academic-calendar")
@limiter.limit("60/minute")
async def get_academic_calendar(
    request: Request,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    entry_type: Optional[str] = None,
    grade: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get academic calendar entries, filterable by date range, type, grade"""
    admin_id = await resolve_admin_id(current_user, db)
    query: Dict[str, Any] = {"admin_id": admin_id}

    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}

    if entry_type:
        query["entry_type"] = entry_type

    docs = await db.mongo_find(COL_CALENDAR, query)
    entries = [AcademicCalendarEntry.from_dict(d).to_dict() for d in docs]

    # Filter by grade if specified (client-side since affects_grades can be empty = all)
    if grade:
        entries = [
            e for e in entries
            if not e.get("affects_grades") or grade in e.get("affects_grades", [])
        ]

    entries.sort(key=lambda e: e.get("date", ""))

    return {"success": True, "entries": entries, "total": len(entries)}


@router.put("/academic-calendar/{entry_id}")
@limiter.limit("20/minute")
async def update_calendar_entry(
    request: Request,
    entry_id: str,
    data: AcademicCalendarInput,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Update an academic calendar entry"""
    try:
        oid = ObjectId(entry_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid entry ID")

    existing = await db.mongo_find_one(COL_CALENDAR, {"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="Calendar entry not found")

    if data.entry_type not in AcademicCalendarEntry.VALID_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid entry type")

    update_fields = {
        "date": data.date,
        "entry_type": data.entry_type,
        "title": data.title,
        "description": data.description,
        "affects_grades": data.affects_grades or [],
        "affects_sections": data.affects_sections or [],
        "is_recurring_yearly": data.is_recurring_yearly,
    }

    await db.mongo_update_one(COL_CALENDAR, {"_id": oid}, {"$set": update_fields})
    updated = await db.mongo_find_one(COL_CALENDAR, {"_id": oid})

    return {"success": True, "entry": AcademicCalendarEntry.from_dict(updated).to_dict()}


@router.delete("/academic-calendar/{entry_id}")
@limiter.limit("10/minute")
async def delete_calendar_entry(
    request: Request,
    entry_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Delete an academic calendar entry"""
    try:
        oid = ObjectId(entry_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid entry ID")

    result = await db.mongo_delete_one(COL_CALENDAR, {"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Calendar entry not found")

    return {"success": True, "message": "Calendar entry deleted"}


# ===========================================================================
# PRE-READ MANAGEMENT (Teacher)
# ===========================================================================

@router.post("/pre-reads", status_code=201)
@limiter.limit("30/minute")
async def create_pre_read(
    request: Request,
    data: PreReadInput,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Create a pre-read for a specific period on a specific date.
    Teachers can only create pre-reads for their own periods."""
    admin_id = await resolve_admin_id(current_user, db)
    user_type = current_user.get("user_type")

    # Lookup the period
    try:
        period_oid = ObjectId(data.period_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid period ID")

    period_doc = await db.mongo_find_one(COL_PERIODS, {"_id": period_oid})
    if not period_doc:
        raise HTTPException(status_code=404, detail="Period not found")

    period = TimetablePeriod.from_dict(period_doc)

    # Tutors can only add pre-reads for their own periods
    if user_type == "tutor":
        tutor_id = current_user.get("tutor_id")
        if period.teacher_id != tutor_id:
            raise HTTPException(
                status_code=403,
                detail="You can only add pre-reads for your own periods"
            )

    # Check if pre-read already exists for this period+date
    existing = await db.mongo_find_one(COL_PREREADS, {
        "period_id": data.period_id,
        "date": data.date,
    })
    if existing:
        raise HTTPException(
            status_code=409,
            detail="Pre-read already exists for this period on this date. Use PUT to update."
        )

    teacher_id = current_user.get("tutor_id") if user_type == "tutor" else period.teacher_id
    teacher_name = current_user.get("name") or current_user.get("username", "")
    if user_type == "admin" and period.teacher_name:
        teacher_name = period.teacher_name

    preread = PreRead(
        timetable_id=period.timetable_id,
        period_id=data.period_id,
        grade=period.grade,
        section=period.section,
        subject=period.subject,
        date=data.date,
        day_of_week=period.day_of_week,
        period_number=period.period_number,
        heading=data.heading,
        description=data.description,
        book_page_numbers=data.book_page_numbers,
        things_required=data.things_required or [],
        teacher_id=teacher_id,
        teacher_name=teacher_name,
        admin_id=admin_id,
    )

    inserted_id = await db.mongo_insert_one(COL_PREREADS, preread.to_mongo())
    if not inserted_id:
        raise HTTPException(status_code=500, detail="Failed to create pre-read")
    preread._id = inserted_id

    return {"success": True, "pre_read": preread.to_dict()}


@router.put("/pre-reads/{preread_id}")
@limiter.limit("30/minute")
async def update_pre_read(
    request: Request,
    preread_id: str,
    data: PreReadUpdateInput,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Update an existing pre-read"""
    try:
        oid = ObjectId(preread_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid pre-read ID")

    existing = await db.mongo_find_one(COL_PREREADS, {"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="Pre-read not found")

    # Tutors can only update their own pre-reads
    user_type = current_user.get("user_type")
    if user_type == "tutor":
        if existing.get("teacher_id") != current_user.get("tutor_id"):
            raise HTTPException(status_code=403, detail="You can only update your own pre-reads")

    update_fields: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    if data.heading is not None:
        update_fields["heading"] = data.heading
    if data.description is not None:
        update_fields["description"] = data.description
    if data.book_page_numbers is not None:
        update_fields["book_page_numbers"] = data.book_page_numbers
    if data.things_required is not None:
        update_fields["things_required"] = data.things_required

    await db.mongo_update_one(COL_PREREADS, {"_id": oid}, {"$set": update_fields})
    updated = await db.mongo_find_one(COL_PREREADS, {"_id": oid})

    return {"success": True, "pre_read": PreRead.from_dict(updated).to_dict()}


@router.delete("/pre-reads/{preread_id}")
@limiter.limit("20/minute")
async def delete_pre_read(
    request: Request,
    preread_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Delete a pre-read"""
    try:
        oid = ObjectId(preread_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid pre-read ID")

    existing = await db.mongo_find_one(COL_PREREADS, {"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="Pre-read not found")

    user_type = current_user.get("user_type")
    if user_type == "tutor":
        if existing.get("teacher_id") != current_user.get("tutor_id"):
            raise HTTPException(status_code=403, detail="You can only delete your own pre-reads")

    await db.mongo_delete_one(COL_PREREADS, {"_id": oid})

    return {"success": True, "message": "Pre-read deleted"}


@router.get("/pre-reads")
@limiter.limit("60/minute")
async def get_pre_reads(
    request: Request,
    grade: Optional[str] = None,
    section: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    teacher_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get pre-reads filtered by class, date range, or teacher"""
    admin_id = await resolve_admin_id(current_user, db)
    query: Dict[str, Any] = {"admin_id": admin_id}

    if grade:
        query["grade"] = grade
    if section:
        query["section"] = section
    if teacher_id:
        query["teacher_id"] = teacher_id
    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}

    # For students, auto-filter to their class
    if current_user.get("user_type") == "student":
        student = await db.mongo_find_one("students", {
            "student_id": current_user.get("student_id")
        })
        if student:
            query["grade"] = student.get("grade", "")
            query["section"] = student.get("section", "")

    docs = await db.mongo_find(COL_PREREADS, query)
    prereads = [PreRead.from_dict(d).to_dict() for d in docs]
    prereads.sort(key=lambda x: (x.get("date", ""), x.get("period_number", 0)))

    return {"success": True, "pre_reads": prereads, "total": len(prereads)}


# ===========================================================================
# STUDENT PRE-READ VIEW (color-coded week view)
# ===========================================================================

@router.get("/student/pre-reads")
@limiter.limit("60/minute")
async def get_student_pre_reads(
    request: Request,
    week_offset: int = 0,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
):
    """Get pre-reads for the student's class for a given week.
    Returns periods with pre-read status for color coding."""
    student_id = current_user.get("student_id")
    student = await db.mongo_find_one("students", {"student_id": student_id})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    grade = student.get("grade")
    section = student.get("section")
    admin_id = student.get("created_by", "")

    # Get active timetable
    tt = await db.mongo_find_one(COL_TIMETABLES, {
        "grade": grade,
        "section": section,
        "is_active": True,
        "admin_id": admin_id,
    })
    if not tt:
        return {"success": True, "periods": [], "week_start": None, "week_end": None}

    timetable_id = str(tt["_id"])

    # Calculate week
    today = datetime.utcnow().date()
    week_start = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
    week_end = week_start + timedelta(days=6)

    # Get all periods
    periods_docs = await db.mongo_find(COL_PERIODS, {"timetable_id": timetable_id})

    # Get pre-reads for the week
    prereads_docs = await db.mongo_find(COL_PREREADS, {
        "timetable_id": timetable_id,
        "date": {"$gte": week_start.isoformat(), "$lte": week_end.isoformat()},
    })
    prereads_map: Dict[str, Dict] = {}
    for pr in prereads_docs:
        key = f"{pr.get('period_id')}_{pr.get('date')}"
        prereads_map[key] = PreRead.from_dict(pr).to_dict()

    # Get academic calendar entries
    calendar_docs = await db.mongo_find(COL_CALENDAR, {
        "admin_id": admin_id,
        "date": {"$gte": week_start.isoformat(), "$lte": week_end.isoformat()},
    })
    holidays_set = set()
    for entry in calendar_docs:
        e = AcademicCalendarEntry.from_dict(entry)
        if e.affects_class(grade, section):
            holidays_set.add(e.date)

    # Build response
    result_periods = []
    for p_doc in periods_docs:
        period = TimetablePeriod.from_dict(p_doc)
        day_name = period.day_of_week
        if day_name not in VALID_DAYS:
            continue

        day_index = VALID_DAYS.index(day_name)
        period_date = week_start + timedelta(days=day_index)
        date_str = period_date.isoformat()

        period_id_str = str(p_doc["_id"])
        preread_key = f"{period_id_str}_{date_str}"
        preread = prereads_map.get(preread_key)

        is_past = period_date < today
        is_today = period_date == today
        is_future = period_date > today
        is_holiday = date_str in holidays_set
        has_preread = preread is not None

        result_periods.append({
            **period.to_dict(),
            "date": date_str,
            "pre_read": preread,
            "has_pre_read": has_preread,
            "is_past": is_past,
            "is_today": is_today,
            "is_future": is_future,
            "is_holiday": is_holiday,
        })

    result_periods.sort(key=lambda x: (x["date"], x["period_number"]))

    return {
        "success": True,
        "periods": result_periods,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "today": today.isoformat(),
    }


# ===========================================================================
# TIMETABLE LOOKUP BY CLASS (for any authenticated user)
# ===========================================================================

@router.get("/class-timetable")
@limiter.limit("60/minute")
async def get_class_timetable(
    request: Request,
    grade: str,
    section: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get timetable for a specific class (used by admin/teacher to view any class)"""
    admin_id = await resolve_admin_id(current_user, db)

    tt = await db.mongo_find_one(COL_TIMETABLES, {
        "grade": grade,
        "section": section,
        "is_active": True,
        "admin_id": admin_id,
    })
    if not tt:
        return {"success": True, "timetable": None, "periods": []}

    timetable_id = str(tt["_id"])
    timetable = ClassTimetable.from_dict(tt).to_dict()

    periods_docs = await db.mongo_find(COL_PERIODS, {"timetable_id": timetable_id})
    periods = sorted(
        [TimetablePeriod.from_dict(p).to_dict() for p in periods_docs],
        key=lambda x: (VALID_DAYS.index(x["day_of_week"]) if x["day_of_week"] in VALID_DAYS else 99, x["period_number"]),
    )

    return {"success": True, "timetable": timetable, "periods": periods}


# ===========================================================================
# AVAILABLE TEACHERS (for dropdowns when building timetable)
# ===========================================================================

@router.get("/available-teachers")
@limiter.limit("60/minute")
async def get_available_teachers(
    request: Request,
    grade: Optional[str] = None,
    subject: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Get list of teachers available for timetable assignment.
    Optionally filter by grade (standard) and subject."""
    admin_id = await resolve_admin_id(current_user, db)
    query: Dict[str, Any] = {"created_by": admin_id, "is_active": True}

    if grade:
        query["standards"] = grade
    if subject:
        query["subjects"] = subject

    teachers_docs = await db.mongo_find("tutors", query)
    teachers = []
    for t in teachers_docs:
        teachers.append({
            "tutor_id": t.get("tutor_id"),
            "name": t.get("name", ""),
            "username": t.get("username", ""),
            "subjects": t.get("subjects", []),
            "standards": t.get("standards", []),
            "sections": t.get("sections", []),
            "teaching_assignments": t.get("teaching_assignments", []),
        })

    return {"success": True, "teachers": teachers, "total": len(teachers)}
