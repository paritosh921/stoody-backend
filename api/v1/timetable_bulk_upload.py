"""
Timetable Bulk Upload API

Allows admins to upload timetable data via CSV/Excel.
Supports preview (validation) and import (execution) flow.

CSV format:
grade,section,day,period_number,start_time,end_time,subject,teacher_username,room,period_type
10,A,Monday,1,08:00,08:40,Mathematics,amit_sharma,,regular
10,A,Monday,2,08:40,09:20,Physics,priya_singh,,regular
"""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
import logging
import io

from core.database import DatabaseManager
from core.upload_security.service import secure_upload
from api.v1.auth_async import get_current_user
from models.timetable import ClassTimetable, TimetablePeriod
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
VALID_PERIOD_TYPES = ["regular", "break", "lunch", "assembly", "library", "sports"]
REQUIRED_COLUMNS = {"grade", "section", "day", "period_number", "start_time", "end_time", "subject"}
OPTIONAL_COLUMNS = {"teacher_username", "teacher_name", "room", "period_type"}
COL_TIMETABLES = "class_timetables"
COL_PERIODS = "timetable_periods"


async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("user_type") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def _read_clean_timetable_upload(
    file: UploadFile,
    *,
    current_user: Dict[str, Any],
    db: DatabaseManager,
    purpose: str,
) -> tuple[bytes, str]:
    clean_upload = await secure_upload(
        file=file,
        policy_id="bulk_timetable",
        actor=current_user,
        db=db,
        purpose_metadata={
            "purpose": "bulk_timetable",
            "collection": COL_TIMETABLES,
            "operation": purpose,
            "created_by": current_user.get("user_id"),
        },
        authorization_subject=f"bulk_timetable:{purpose}:{current_user.get('user_id', 'unknown')}",
    )
    return clean_upload.bytes or b"", clean_upload.original_filename


def _parse_file(file_bytes: bytes, filename: str) -> List[Dict[str, str]]:
    """Parse CSV or Excel file into list of row dicts"""
    rows = []
    if filename.endswith((".xlsx", ".xls")):
        try:
            import pandas as pd
            df = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            rows = df.fillna("").to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {str(e)}")
    elif filename.endswith(".csv"):
        try:
            import pandas as pd
            # Try multiple encodings
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise HTTPException(status_code=400, detail="Could not decode CSV file")
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            rows = df.fillna("").to_dict(orient="records")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="File must be .csv, .xlsx, or .xls")

    return rows


def _validate_row(row: Dict[str, str], row_num: int, teacher_map: Dict[str, Dict]) -> Dict[str, Any]:
    """Validate a single row and return parsed data or errors"""
    errors = []

    grade = str(row.get("grade", "")).strip()
    section = str(row.get("section", "")).strip().upper()
    day = str(row.get("day", "")).strip()
    period_number_str = str(row.get("period_number", "")).strip()
    start_time = str(row.get("start_time", "")).strip()
    end_time = str(row.get("end_time", "")).strip()
    subject = str(row.get("subject", "")).strip()
    teacher_username = str(row.get("teacher_username", "")).strip()
    teacher_name = str(row.get("teacher_name", "")).strip()
    room = str(row.get("room", "")).strip()
    period_type = str(row.get("period_type", "regular")).strip().lower()

    if not grade:
        errors.append({"row": row_num, "field": "grade", "message": "Grade is required"})
    if not section:
        errors.append({"row": row_num, "field": "section", "message": "Section is required"})
    if not day:
        errors.append({"row": row_num, "field": "day", "message": "Day is required"})
    elif day not in VALID_DAYS:
        # Try case-insensitive match
        matched = [d for d in VALID_DAYS if d.lower() == day.lower()]
        if matched:
            day = matched[0]
        else:
            errors.append({"row": row_num, "field": "day", "message": f"Invalid day: {day}. Must be one of {VALID_DAYS}"})

    period_number = 0
    if not period_number_str:
        errors.append({"row": row_num, "field": "period_number", "message": "Period number is required"})
    else:
        try:
            period_number = int(float(period_number_str))
            if period_number < 1 or period_number > 15:
                errors.append({"row": row_num, "field": "period_number", "message": "Period number must be 1-15"})
        except ValueError:
            errors.append({"row": row_num, "field": "period_number", "message": "Period number must be a number"})

    if not start_time:
        errors.append({"row": row_num, "field": "start_time", "message": "Start time is required"})
    if not end_time:
        errors.append({"row": row_num, "field": "end_time", "message": "End time is required"})
    # Subject is required for regular periods; for break/lunch/assembly, auto-fill from period_type
    if not subject:
        if period_type in ("break", "lunch", "assembly"):
            subject = period_type.capitalize()
        else:
            errors.append({"row": row_num, "field": "subject", "message": "Subject is required"})
    if period_type and period_type not in VALID_PERIOD_TYPES:
        errors.append({"row": row_num, "field": "period_type", "message": f"Invalid period type: {period_type}"})

    # Resolve teacher
    resolved_teacher_id = None
    resolved_teacher_name = teacher_name
    if teacher_username and teacher_username in teacher_map:
        t = teacher_map[teacher_username]
        resolved_teacher_id = t.get("tutor_id")
        resolved_teacher_name = resolved_teacher_name or t.get("name", "")
    elif teacher_username:
        errors.append({"row": row_num, "field": "teacher_username", "message": f"Teacher '{teacher_username}' not found"})

    return {
        "row": row_num,
        "grade": grade,
        "section": section,
        "day": day,
        "period_number": period_number,
        "start_time": start_time,
        "end_time": end_time,
        "subject": subject,
        "teacher_id": resolved_teacher_id,
        "teacher_name": resolved_teacher_name,
        "teacher_username": teacher_username,
        "room": room,
        "period_type": period_type or "regular",
        "errors": errors,
    }


@router.post("/timetable/bulk-upload/preview")
@limiter.limit("10/minute")
async def preview_timetable_upload(
    request: Request,
    file: UploadFile = File(...),
    academic_session: str = "2025-26",
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Preview timetable bulk upload. Validates data without importing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_bytes, clean_filename = await _read_clean_timetable_upload(
        file,
        current_user=current_user,
        db=db,
        purpose="preview",
    )
    rows = _parse_file(file_bytes, clean_filename)
    if not rows:
        raise HTTPException(status_code=400, detail="File is empty")

    # Validate required columns
    first_row_keys = set(rows[0].keys())
    missing_cols = REQUIRED_COLUMNS - first_row_keys
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing_cols)}. "
                   f"Required: {', '.join(REQUIRED_COLUMNS)}"
        )

    # Build teacher lookup map
    admin_id = current_user.get("user_id") or current_user.get("admin_id", "")
    teachers_docs = await db.mongo_find("tutors", {"created_by": admin_id, "is_active": True})
    teacher_map = {t.get("username"): t for t in teachers_docs if t.get("username")}

    # Validate each row
    all_errors = []
    valid_rows = []
    for i, row in enumerate(rows, start=1):
        parsed = _validate_row(row, i, teacher_map)
        if parsed["errors"]:
            all_errors.extend(parsed["errors"])
        else:
            valid_rows.append(parsed)

    # Group by grade+section for summary
    classes_summary: Dict[str, Dict] = {}
    for r in valid_rows:
        key = f"{r['grade']}-{r['section']}"
        if key not in classes_summary:
            classes_summary[key] = {"grade": r["grade"], "section": r["section"], "period_count": 0, "days": set()}
        classes_summary[key]["period_count"] += 1
        classes_summary[key]["days"].add(r["day"])

    for k in classes_summary:
        classes_summary[k]["days"] = sorted(list(classes_summary[k]["days"]))

    return {
        "success": True,
        "file_name": clean_filename,
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "error_rows": len(rows) - len(valid_rows),
        "errors": all_errors,
        "classes_summary": list(classes_summary.values()),
        "preview_data": valid_rows[:20],
        "academic_session": academic_session,
    }


@router.post("/timetable/bulk-upload/import")
@limiter.limit("5/minute")
async def import_timetable_upload(
    request: Request,
    file: UploadFile = File(...),
    academic_session: str = "2025-26",
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """Import timetable from CSV/Excel. Creates timetables and periods."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_bytes, clean_filename = await _read_clean_timetable_upload(
        file,
        current_user=current_user,
        db=db,
        purpose="import",
    )
    rows = _parse_file(file_bytes, clean_filename)
    if not rows:
        raise HTTPException(status_code=400, detail="File is empty")

    admin_id = current_user.get("user_id") or current_user.get("admin_id", "")
    teachers_docs = await db.mongo_find("tutors", {"created_by": admin_id, "is_active": True})
    teacher_map = {t.get("username"): t for t in teachers_docs if t.get("username")}

    # Parse and validate
    all_errors = []
    valid_rows = []
    for i, row in enumerate(rows, start=1):
        parsed = _validate_row(row, i, teacher_map)
        if parsed["errors"]:
            all_errors.extend(parsed["errors"])
        else:
            valid_rows.append(parsed)

    if not valid_rows:
        raise HTTPException(status_code=400, detail="No valid rows to import")

    # Group by grade+section
    classes: Dict[str, List[Dict]] = {}
    for r in valid_rows:
        key = f"{r['grade']}-{r['section']}"
        if key not in classes:
            classes[key] = []
        classes[key].append(r)

    timetables_created = 0
    periods_created = 0

    for class_key, class_rows in classes.items():
        grade = class_rows[0]["grade"]
        section = class_rows[0]["section"]

        # Create or find timetable
        existing_tt = await db.mongo_find_one(COL_TIMETABLES, {
            "grade": grade,
            "section": section,
            "academic_session": academic_session,
            "is_active": True,
            "admin_id": admin_id,
        })

        if existing_tt:
            timetable_id = str(existing_tt["_id"])
        else:
            # Build period timings from the data
            timings_map: Dict[int, Dict] = {}
            for r in class_rows:
                pn = r["period_number"]
                if pn not in timings_map:
                    timings_map[pn] = {
                        "period_number": pn,
                        "start_time": r["start_time"],
                        "end_time": r["end_time"],
                        "label": f"Period {pn}",
                    }

            period_timings = sorted(timings_map.values(), key=lambda x: x["period_number"])
            weekdays = sorted(list(set(r["day"] for r in class_rows)),
                              key=lambda d: VALID_DAYS.index(d) if d in VALID_DAYS else 99)

            tt = ClassTimetable(
                grade=grade,
                section=section,
                academic_session=academic_session,
                period_timings=period_timings,
                weekdays=weekdays,
                admin_id=admin_id,
                created_by=admin_id,
                created_by_role="admin",
            )
            timetable_id = await db.mongo_insert_one(COL_TIMETABLES, tt.to_mongo())
            if not timetable_id:
                raise HTTPException(status_code=500, detail=f"Failed to create timetable for {grade}-{section}")
            timetables_created += 1

        # Delete existing periods for this timetable's days and insert new ones
        days_in_upload = list(set(r["day"] for r in class_rows))
        await db.mongo_delete_many(COL_PERIODS, {
            "timetable_id": timetable_id,
            "day_of_week": {"$in": days_in_upload},
        })

        periods_to_insert = []
        for r in class_rows:
            period = TimetablePeriod(
                timetable_id=timetable_id,
                grade=grade,
                section=section,
                day_of_week=r["day"],
                period_number=r["period_number"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                subject=r["subject"],
                teacher_id=r["teacher_id"],
                teacher_name=r["teacher_name"],
                room=r["room"],
                period_type=r["period_type"],
                admin_id=admin_id,
            )
            periods_to_insert.append(period.to_mongo())

        if periods_to_insert:
            await db.mongo_insert_many(COL_PERIODS, periods_to_insert)
            periods_created += len(periods_to_insert)

    logger.info(
        "Timetable bulk import: %d timetables, %d periods by admin %s",
        timetables_created, periods_created, admin_id,
    )

    return {
        "success": True,
        "message": f"Import complete: {timetables_created} timetables created, {periods_created} periods added",
        "timetables_created": timetables_created,
        "periods_created": periods_created,
        "errors": all_errors,
        "classes_processed": len(classes),
    }


@router.get("/timetable/bulk-upload/template")
@limiter.limit("10/minute")
async def download_template(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
):
    """Download a CSV template for timetable bulk upload"""
    from fastapi.responses import StreamingResponse

    template = "grade,section,day,period_number,start_time,end_time,subject,teacher_username,room,period_type\n"
    template += "10,A,Monday,1,08:00,08:40,Mathematics,amit_sharma,,regular\n"
    template += "10,A,Monday,2,08:40,09:20,Physics,priya_singh,,regular\n"
    template += "10,A,Monday,3,09:20,09:30,Break,,,break\n"
    template += "10,A,Monday,4,09:30,10:10,Chemistry,rahul_verma,,regular\n"
    template += "10,A,Tuesday,1,08:00,08:40,English,neha_gupta,,regular\n"

    return StreamingResponse(
        io.BytesIO(template.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=timetable_template.csv"},
    )
