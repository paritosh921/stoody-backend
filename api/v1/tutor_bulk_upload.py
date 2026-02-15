"""
Teacher/Tutor Bulk Upload API
Supports CSV and Excel file uploads for creating multiple teacher accounts at once.
Follows the same pattern as student_bulk_upload.py.
"""

import logging
import io
import re
import secrets
import string
import random
import bcrypt
from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId

import pandas as pd
from fastapi import APIRouter, Request, HTTPException, Depends, status, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user, get_database, get_cache
from api.v1.admin_async import require_admin_permission, get_tenant_db_or_403
from models.tutor import Tutor

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Template columns for teacher bulk upload
TEMPLATE_COLUMNS = {
    "full_name": {"required": True, "description": "Teacher's full name (2-100 characters)", "example": "Amit Sharma"},
    "username": {"required": False, "description": "Custom username (optional - auto-generated if empty)", "example": ""},
    "password": {"required": False, "description": "Login password (min 6 chars). Leave empty to auto-generate.", "example": ""},
    "email": {"required": False, "description": "Email address (unique)", "example": "amit@school.com"},
    "phone": {"required": False, "description": "Phone number", "example": "9876543210"},
    "standard": {"required": True, "description": "Class/grade to teach", "example": "10"},
    "subject": {"required": True, "description": "Subject to teach", "example": "Mathematics"},
    "sections": {"required": False, "description": "Comma-separated sections", "example": "A,B,C"},
    "can_edit_students": {"required": False, "description": "Self-manage students (true/false)", "example": "false"},
    "class_teacher_of": {"required": False, "description": "Class teacher of (format: standard-section, e.g. 11-A)", "example": ""},
}


# Pydantic models
class TeacherBulkUploadError(BaseModel):
    row: int
    field: str
    value: Optional[str] = None
    message: str


class TeacherBulkPreviewTeacher(BaseModel):
    row: int
    full_name: str
    username: str
    email: Optional[str] = None
    standards: List[str] = []
    subjects: List[str] = []


class TeacherBulkPreviewResponse(BaseModel):
    success: bool
    file_name: str
    file_type: str
    total_rows: int
    valid_rows: int
    error_rows: int
    errors: List[TeacherBulkUploadError]
    warnings: List[TeacherBulkUploadError]
    preview_data: List[TeacherBulkPreviewTeacher]


class TeacherCredential(BaseModel):
    full_name: str
    username: str
    password: str
    email: Optional[str] = None
    standards: List[str] = []
    subjects: List[str] = []


class TeacherBulkImportResponse(BaseModel):
    success: bool
    message: str
    total_imported: int
    total_skipped: int
    credentials: List[TeacherCredential]
    errors: List[TeacherBulkUploadError]


# Utility functions
def generate_secure_password(length: int = 10) -> str:
    """Generate a cryptographically secure password"""
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    if not any(c.isdigit() for c in password):
        password = password[:-1] + secrets.choice(string.digits)
    return password


def generate_teacher_username(full_name: str) -> str:
    """Generate username from full name: {firstname}{4-digit-random}"""
    words = full_name.strip().split()
    first_name = re.sub(r'[^a-zA-Z0-9]', '', words[0] if words else "teacher").lower()
    short_name = first_name[:8] if first_name else "teacher"
    random_suffix = random.randint(1000, 9999)
    return f"{short_name}{random_suffix}"


def validate_email(email: str) -> bool:
    """Simple email validation"""
    if not email:
        return True
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def parse_sections(value: str) -> List[str]:
    """Parse comma-separated sections string"""
    if not value or pd.isna(value):
        return []
    return [s.strip().upper() for s in str(value).split(',') if s.strip()]


async def parse_upload_file(file: UploadFile) -> pd.DataFrame:
    """Parse uploaded CSV or Excel file into DataFrame"""
    content = await file.read()
    file_name = file.filename.lower()

    try:
        if file_name.endswith('.csv'):
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=encoding, dtype=str)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file. Please use UTF-8 encoding.")
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content), dtype=str)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel (.xlsx, .xls) file.")

        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df = df.fillna('')
        return df
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        raise ValueError(f"Failed to parse file: {str(e)}")


@router.get("/tutors/bulk/template")
@limiter.limit("30/minute")
async def download_teacher_bulk_template(
    request: Request,
    format: str = Query("xlsx", description="Template format: xlsx or csv"),
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_tutors")),
    db: DatabaseManager = Depends(get_database)
):
    """
    Download a template file for bulk teacher upload.
    Supports CSV and Excel formats.
    """
    # Fetch school settings for dynamic values
    try:
        tenant_db = await get_tenant_db_or_403(db, current_user)
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
        settings_doc = await tenant_db["school_settings"].find_one({"admin_id": admin_id})

        valid_classes = settings_doc.get("classes", ["6", "7", "8", "9", "10", "11", "12"]) if settings_doc else ["6", "7", "8", "9", "10", "11", "12"]
        valid_sections = settings_doc.get("sections", ["A", "B", "C", "D", "E", "F"]) if settings_doc else ["A", "B", "C", "D", "E", "F"]
        valid_subjects = settings_doc.get("subjects", ["Mathematics", "Physics", "Chemistry", "Biology", "English"]) if settings_doc else ["Mathematics", "Physics", "Chemistry", "Biology", "English"]
    except Exception as e:
        logger.warning(f"Could not fetch school settings: {e}")
        valid_classes = ["6", "7", "8", "9", "10", "11", "12"]
        valid_sections = ["A", "B", "C", "D", "E", "F"]
        valid_subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "English"]

    # Sample data with 3 example rows
    sample_data = [
        {
            "full_name": "Amit Sharma",
            "username": "",
            "password": "mypass123",
            "email": "amit@school.com",
            "phone": "9876543210",
            "standard": "10",
            "subject": "Mathematics",
            "sections": "A,B,C",
            "can_edit_students": "false",
            "class_teacher_of": "10-A",
        },
        {
            "full_name": "Priya Gupta",
            "username": "",
            "password": "",
            "email": "priya@school.com",
            "phone": "9876543211",
            "standard": "11",
            "subject": "Physics",
            "sections": "A,B",
            "can_edit_students": "true",
            "class_teacher_of": "",
        },
        {
            "full_name": "Raj Kumar",
            "username": "rajkumar2025",
            "password": "raj@2025",
            "email": "",
            "phone": "9876543212",
            "standard": "12",
            "subject": "Chemistry",
            "sections": "A",
            "can_edit_students": "false",
            "class_teacher_of": "",
        },
    ]

    df = pd.DataFrame(sample_data)

    # Instructions
    instructions_data = [
        {"Column": "full_name", "Required": "YES", "Description": "Teacher's full name (2-100 characters)"},
        {"Column": "username", "Required": "NO", "Description": "Custom username (3-50 chars). Leave empty to auto-generate."},
        {"Column": "password", "Required": "NO", "Description": "Login password (min 6 chars). Leave empty to auto-generate a secure password."},
        {"Column": "email", "Required": "NO", "Description": "Email address (must be unique)"},
        {"Column": "phone", "Required": "NO", "Description": "Phone number"},
        {"Column": "standard", "Required": "YES", "Description": f"Class/grade to teach. Allowed: {', '.join(valid_classes)}. For multiple classes, add separate rows."},
        {"Column": "subject", "Required": "YES", "Description": f"Subject to teach. Allowed: {', '.join(valid_subjects)}. For multiple subjects, add separate rows."},
        {"Column": "sections", "Required": "NO", "Description": f"Comma-separated sections. Allowed: {', '.join(valid_sections)}"},
        {"Column": "can_edit_students", "Required": "NO", "Description": "Self-manage students (true/false). Default: false"},
        {"Column": "class_teacher_of", "Required": "NO", "Description": "Class teacher of (format: standard-section, e.g. 11-A). Leave empty if not a class teacher."},
    ]
    instructions_df = pd.DataFrame(instructions_data)

    if format.lower() == "csv":
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=teacher_bulk_upload_template.csv"}
        )
    else:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Teachers', index=False)
            instructions_df.to_excel(writer, sheet_name='Instructions', index=False)

            workbook = writer.book
            teachers_sheet = writer.sheets['Teachers']
            instructions_sheet = writer.sheets['Instructions']

            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4A90D9',
                'font_color': 'white',
                'border': 1
            })

            for col_num, value in enumerate(df.columns.values):
                teachers_sheet.write(0, col_num, value, header_format)
                teachers_sheet.set_column(col_num, col_num, 20)

            for col_num, value in enumerate(instructions_df.columns.values):
                instructions_sheet.write(0, col_num, value, header_format)

            instructions_sheet.set_column(0, 0, 18)
            instructions_sheet.set_column(1, 1, 10)
            instructions_sheet.set_column(2, 2, 70)

        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=teacher_bulk_upload_template.xlsx"}
        )


@router.post("/tutors/bulk/preview", response_model=TeacherBulkPreviewResponse)
@limiter.limit("20/minute")
async def preview_teacher_bulk_upload(
    request: Request,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_tutors")),
    db: DatabaseManager = Depends(get_database)
):
    """
    Preview and validate a teacher bulk upload file before importing.
    Multiple rows for the same teacher (by username or full_name+email) are
    merged into multiple teaching assignments.
    """
    file_name = file.filename.lower()
    if not file_name.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload CSV or Excel (.xlsx, .xls) file."
        )

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 5MB."
        )
    await file.seek(0)

    try:
        df = await parse_upload_file(file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if len(df) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many rows ({len(df)}). Maximum allowed is 500 rows per upload."
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty. Please add teacher data."
        )

    # Get existing usernames and emails
    admin_id = current_user.get("user_id")
    existing_tutors = await db.mongo_find("tutors", {"created_by": admin_id}, projection={"email": 1, "username_lower": 1})
    existing_emails = {t.get("email", "").lower() for t in existing_tutors if t.get("email")}
    existing_usernames = {t.get("username_lower", "").lower() for t in existing_tutors if t.get("username_lower")}

    errors: List[TeacherBulkUploadError] = []
    warnings: List[TeacherBulkUploadError] = []

    # First pass: group rows by teacher identity (username or full_name+email)
    # Each teacher can have multiple rows = multiple teaching assignments
    teacher_groups: Dict[str, Dict[str, Any]] = {}
    file_emails: Dict[str, int] = {}
    file_usernames: Dict[str, int] = {}

    for idx, row in df.iterrows():
        row_num = idx + 2
        full_name = str(row.get('full_name', '')).strip().title()
        custom_username = str(row.get('username', '')).strip() if row.get('username') else ''
        email = str(row.get('email', '')).strip().lower() if row.get('email') else ''
        phone = str(row.get('phone', '')).strip() if row.get('phone') else ''
        standard = str(row.get('standard', '')).strip()
        subject = str(row.get('subject', '')).strip()
        sections_str = str(row.get('sections', '')).strip() if row.get('sections') else ''
        can_edit = str(row.get('can_edit_students', 'false')).strip().lower()
        class_teacher_of_raw = str(row.get('class_teacher_of', '')).strip() if row.get('class_teacher_of') else ''

        row_valid = True

        # Validate required fields
        if not full_name or len(full_name) < 2:
            errors.append(TeacherBulkUploadError(row=row_num, field="full_name", value=full_name, message="Full name is required (min 2 characters)"))
            row_valid = False
        elif len(full_name) > 100:
            errors.append(TeacherBulkUploadError(row=row_num, field="full_name", value=full_name[:50]+"...", message="Full name too long (max 100 characters)"))
            row_valid = False

        if not standard:
            errors.append(TeacherBulkUploadError(row=row_num, field="standard", value=standard, message="Standard/class is required"))
            row_valid = False

        if not subject:
            errors.append(TeacherBulkUploadError(row=row_num, field="subject", value=subject, message="Subject is required"))
            row_valid = False

        # Validate custom username if provided
        if custom_username:
            if len(custom_username) < 3 or len(custom_username) > 50:
                errors.append(TeacherBulkUploadError(row=row_num, field="username", value=custom_username, message="Username must be 3-50 characters"))
                row_valid = False
            elif custom_username.lower() in existing_usernames:
                errors.append(TeacherBulkUploadError(row=row_num, field="username", value=custom_username, message="Username already exists"))
                row_valid = False

        # Validate email
        if email:
            if not validate_email(email):
                errors.append(TeacherBulkUploadError(row=row_num, field="email", value=email, message="Invalid email format"))
                row_valid = False
            elif email in existing_emails:
                errors.append(TeacherBulkUploadError(row=row_num, field="email", value=email, message="Email already exists"))
                row_valid = False

        if not row_valid:
            continue

        # Determine teacher identity key for grouping
        # If username provided, group by username; otherwise by full_name+email
        if custom_username:
            key = f"user:{custom_username.lower()}"
        else:
            key = f"name:{full_name.lower()}|{email}"

        # Parse class_teacher_of (format: standard-section, e.g. 11-A)
        class_teacher_of = None
        if class_teacher_of_raw:
            ct_match = re.match(r'^(\w+)-([A-Za-z]+)$', class_teacher_of_raw)
            if ct_match:
                class_teacher_of = {"standard": ct_match.group(1), "section": ct_match.group(2).upper()}
            else:
                warnings.append(TeacherBulkUploadError(
                    row=row_num, field="class_teacher_of", value=class_teacher_of_raw,
                    message="Invalid format for class_teacher_of. Expected: standard-section (e.g. 11-A). Ignoring."
                ))

        sections = parse_sections(sections_str)
        assignment = {"standard": standard, "subject": subject, "sections": sections}

        if key not in teacher_groups:
            # Track unique usernames/emails
            if custom_username:
                if custom_username.lower() in file_usernames:
                    # Same username in multiple rows = same teacher, merge assignments
                    pass
                else:
                    file_usernames[custom_username.lower()] = row_num

            if email and email not in file_emails:
                file_emails[email] = row_num

            teacher_groups[key] = {
                "first_row": row_num,
                "full_name": full_name,
                "username": custom_username,
                "email": email,
                "phone": phone,
                "can_edit_students": can_edit in ('true', '1', 'yes'),
                "class_teacher_of": class_teacher_of,
                "assignments": [assignment],
            }
        else:
            # Merge assignment into existing teacher
            # class_teacher_of is teacher-level, only use from first row
            teacher_groups[key]["assignments"].append(assignment)

    # Build preview data
    preview_data: List[TeacherBulkPreviewTeacher] = []
    valid_count = len(teacher_groups)

    for key, teacher in teacher_groups.items():
        if len(preview_data) >= 20:
            break

        username = teacher["username"]
        if not username:
            first_word = teacher["full_name"].split()[0] if teacher["full_name"].split() else "teacher"
            username = generate_teacher_username(first_word)

        standards = list(set(a["standard"] for a in teacher["assignments"]))
        subjects = list(set(a["subject"] for a in teacher["assignments"]))

        preview_data.append(TeacherBulkPreviewTeacher(
            row=teacher["first_row"],
            full_name=teacher["full_name"],
            username=username,
            email=teacher["email"] if teacher["email"] else None,
            standards=standards,
            subjects=subjects,
        ))

    file_type = "Excel" if file_name.endswith(('.xlsx', '.xls')) else "CSV"

    return TeacherBulkPreviewResponse(
        success=len(errors) == 0,
        file_name=file.filename,
        file_type=file_type,
        total_rows=len(df),
        valid_rows=valid_count,
        error_rows=len(df) - sum(len(t["assignments"]) for t in teacher_groups.values()),
        errors=errors,
        warnings=warnings,
        preview_data=preview_data,
    )


@router.post("/tutors/bulk/import", response_model=TeacherBulkImportResponse)
@limiter.limit("10/minute")
async def import_bulk_teachers(
    request: Request,
    file: UploadFile = File(...),
    skip_errors: bool = Query(True, description="Skip rows with errors and import valid ones"),
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_tutors")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Import teachers from a bulk upload file.
    Creates teacher accounts with auto-generated passwords.
    Multiple rows for the same teacher are merged into multiple teaching assignments.
    """
    file_name = file.filename.lower()
    if not file_name.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload CSV or Excel (.xlsx, .xls) file."
        )

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 5MB."
        )
    await file.seek(0)

    try:
        df = await parse_upload_file(file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if len(df) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many rows ({len(df)}). Maximum allowed is 500 rows per upload."
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty. Please add teacher data."
        )

    admin_id = current_user.get("user_id")

    # Get existing usernames and emails
    existing_tutors = await db.mongo_find("tutors", {"created_by": admin_id}, projection={"email": 1, "username_lower": 1})
    existing_emails = {t.get("email", "").lower() for t in existing_tutors if t.get("email")}
    existing_usernames = {t.get("username_lower", "").lower() for t in existing_tutors if t.get("username_lower")}

    errors: List[TeacherBulkUploadError] = []
    used_usernames = set(existing_usernames)

    # Group rows by teacher identity
    teacher_groups: Dict[str, Dict[str, Any]] = {}

    for idx, row in df.iterrows():
        row_num = idx + 2
        full_name = str(row.get('full_name', '')).strip().title()
        custom_username = str(row.get('username', '')).strip() if row.get('username') else ''
        custom_password = str(row.get('password', '')).strip() if row.get('password') else ''
        email = str(row.get('email', '')).strip().lower() if row.get('email') else ''
        phone = str(row.get('phone', '')).strip() if row.get('phone') else ''
        standard = str(row.get('standard', '')).strip()
        subject = str(row.get('subject', '')).strip()
        sections_str = str(row.get('sections', '')).strip() if row.get('sections') else ''
        can_edit = str(row.get('can_edit_students', 'false')).strip().lower()
        class_teacher_of_raw = str(row.get('class_teacher_of', '')).strip() if row.get('class_teacher_of') else ''

        row_valid = True

        if not full_name or len(full_name) < 2 or len(full_name) > 100:
            errors.append(TeacherBulkUploadError(row=row_num, field="full_name", message="Invalid full name"))
            row_valid = False

        if not standard:
            errors.append(TeacherBulkUploadError(row=row_num, field="standard", message="Standard/class is required"))
            row_valid = False

        if not subject:
            errors.append(TeacherBulkUploadError(row=row_num, field="subject", message="Subject is required"))
            row_valid = False

        if custom_username:
            if len(custom_username) < 3 or len(custom_username) > 50:
                errors.append(TeacherBulkUploadError(row=row_num, field="username", message=f"Username must be 3-50 characters: {custom_username}"))
                row_valid = False
            elif custom_username.lower() in existing_usernames:
                errors.append(TeacherBulkUploadError(row=row_num, field="username", message=f"Username already exists: {custom_username}"))
                row_valid = False

        if email and not validate_email(email):
            errors.append(TeacherBulkUploadError(row=row_num, field="email", message=f"Invalid email: {email}"))
            row_valid = False

        if email and email in existing_emails:
            errors.append(TeacherBulkUploadError(row=row_num, field="email", message=f"Email already exists: {email}"))
            row_valid = False

        if not row_valid:
            if not skip_errors:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Validation error in row {row_num}. Use skip_errors=true to skip invalid rows."
                )
            continue

        # Parse class_teacher_of (format: standard-section, e.g. 11-A)
        class_teacher_of = None
        if class_teacher_of_raw:
            ct_match = re.match(r'^(\w+)-([A-Za-z]+)$', class_teacher_of_raw)
            if ct_match:
                class_teacher_of = {"standard": ct_match.group(1), "section": ct_match.group(2).upper()}

        sections = parse_sections(sections_str)
        assignment = {"standard": standard, "subject": subject, "sections": sections}

        if custom_username:
            key = f"user:{custom_username.lower()}"
        else:
            key = f"name:{full_name.lower()}|{email}"

        if key not in teacher_groups:
            teacher_groups[key] = {
                "full_name": full_name,
                "username": custom_username,
                "password": custom_password,
                "email": email,
                "phone": phone,
                "can_edit_students": can_edit in ('true', '1', 'yes'),
                "class_teacher_of": class_teacher_of,
                "assignments": [assignment],
            }
        else:
            # class_teacher_of is teacher-level, only use from first row
            teacher_groups[key]["assignments"].append(assignment)

    # Create teacher documents
    credentials: List[TeacherCredential] = []
    teachers_to_insert: List[Dict[str, Any]] = []

    for key, teacher in teacher_groups.items():
        # Determine username
        if teacher["username"]:
            username = teacher["username"]
        else:
            first_word = teacher["full_name"].split()[0] if teacher["full_name"].split() else "teacher"
            max_attempts = 100
            for _ in range(max_attempts):
                username = generate_teacher_username(first_word)
                if username.lower() not in used_usernames:
                    break
            else:
                username = f"{first_word[:8].lower()}{int(datetime.utcnow().timestamp()) % 10000}"

        used_usernames.add(username.lower())

        # Use password from sheet, or auto-generate
        plain_password = teacher["password"] if teacher["password"] and len(teacher["password"]) >= 6 else generate_secure_password()
        password_hash = bcrypt.hashpw(plain_password.encode('utf-8')[:72], bcrypt.gensalt()).decode('utf-8')

        # Generate tutor_id
        auto_tutor_id = Tutor.generate_tutor_id()

        # Build teaching assignments and derive aggregate fields
        assignments = teacher["assignments"]
        standards = sorted(set(a["standard"] for a in assignments))
        sections = sorted(set(s for a in assignments for s in a.get("sections", [])))
        subjects = sorted(set(a["subject"] for a in assignments))

        normalized_assignments = [
            {"standard": a["standard"], "subject": a["subject"], "sections": a.get("sections", [])}
            for a in assignments
        ]

        teacher_doc = {
            "tutor_id": auto_tutor_id,
            "name": teacher["full_name"],
            "username": username,
            "username_lower": username.lower(),
            "password_hash": password_hash,
            "email": teacher["email"] if teacher["email"] else None,
            "phone": teacher["phone"] if teacher["phone"] else None,
            "standards": standards,
            "sections": sections,
            "subjects": subjects,
            "plan_types": [],
            "can_edit_students": teacher["can_edit_students"],
            "class_teacher_of": teacher["class_teacher_of"],
            "is_active": True,
            "assigned_student_ids": [],
            "requires_password_change": True,
            "password_reset_requested": False,
            "teaching_assignments": normalized_assignments,
            "created_by": admin_id,
            "created_at": datetime.utcnow(),
            "created_via": "bulk_upload",
            "last_login": None,
            "two_fa": {
                "enabled": False,
                "required": True,
                "secret_enc": None,
                "verified_at": None
            }
        }

        teachers_to_insert.append(teacher_doc)
        credentials.append(TeacherCredential(
            full_name=teacher["full_name"],
            username=username,
            password=plain_password,
            email=teacher["email"] if teacher["email"] else None,
            standards=standards,
            subjects=subjects,
        ))

    # Bulk insert using tenant-aware method
    total_imported = 0
    if teachers_to_insert:
        try:
            result = await db.mongo_insert_many("tutors", teachers_to_insert)
            total_imported = len(result) if result else 0
        except Exception as e:
            logger.error(f"Bulk teacher insert error: {str(e)}")
            if hasattr(e, 'details') and 'nInserted' in e.details:
                total_imported = e.details['nInserted']
            errors.append(TeacherBulkUploadError(row=0, field="database", message=f"Database error: {str(e)[:100]}"))

    # Invalidate cache
    try:
        await cache.clear_pattern("tutors:*", "admin")
    except Exception:
        pass

    total_skipped = len(teacher_groups) - total_imported

    return TeacherBulkImportResponse(
        success=total_imported > 0,
        message=f"Successfully imported {total_imported} teachers. {total_skipped} skipped.",
        total_imported=total_imported,
        total_skipped=max(0, total_skipped),
        credentials=credentials[:total_imported],
        errors=errors,
    )
