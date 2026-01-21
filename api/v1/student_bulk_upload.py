"""
Student Bulk Upload API
Supports CSV and Excel file uploads for creating multiple students at once
"""

import logging
import io
import re
import secrets
import string
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

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Valid grades for students
VALID_GRADES = ["6", "7", "8", "9", "10", "11", "12"]

# Template columns configuration
TEMPLATE_COLUMNS = {
    "full_name": {"required": True, "description": "Student's full name (2-100 characters)", "example": "John Doe"},
    "grade": {"required": True, "description": "Grade/Class (6, 7, 8, 9, 10, 11, or 12)", "example": "10"},
    "section": {"required": False, "description": "Section (A, B, C, D, E, F)", "example": "A"},
    "email": {"required": False, "description": "Email address (unique within institute)", "example": "john@example.com"},
    "phone": {"required": False, "description": "Phone number", "example": "9876543210"},
    "stream": {"required": False, "description": "Stream (science, commerce, arts, other)", "example": "science"},
    "gender": {"required": False, "description": "Gender (male, female, other)", "example": "male"},
    "date_of_birth": {"required": False, "description": "Date of birth (YYYY-MM-DD format)", "example": "2008-05-15"},
    "school": {"required": False, "description": "School name", "example": "ABC School"},
    "location": {"required": False, "description": "City/Location", "example": "Mumbai"},
    "plan_types": {"required": False, "description": "Plan types comma-separated (CBSE, JEE, NEET, CUET)", "example": "JEE,NEET"},
    "subjects": {"required": False, "description": "Subjects comma-separated (Physics, Chemistry, Mathematics, Biology)", "example": "Physics,Chemistry,Mathematics"},
}


# Pydantic models for bulk upload
class BulkUploadError(BaseModel):
    row: int
    field: str
    value: Optional[str] = None
    message: str


class BulkPreviewStudent(BaseModel):
    row: int
    full_name: str
    grade: str
    section: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    stream: Optional[str] = None
    auto_username: str  # Generated username preview


class BulkPreviewResponse(BaseModel):
    success: bool
    file_name: str
    file_type: str
    total_rows: int
    valid_rows: int
    error_rows: int
    errors: List[BulkUploadError]
    warnings: List[BulkUploadError]
    preview_data: List[BulkPreviewStudent]
    duplicate_emails: List[str]
    duplicate_in_file: List[str]


class StudentCredential(BaseModel):
    full_name: str
    username: str
    password: str
    grade: str
    section: Optional[str] = None


class BulkImportResponse(BaseModel):
    success: bool
    message: str
    total_imported: int
    total_skipped: int
    credentials: List[StudentCredential]
    errors: List[BulkUploadError]


# Utility functions
def generate_secure_password(length: int = 10) -> str:
    """Generate a cryptographically secure password"""
    # Use simpler charset for easy typing
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    # Ensure password has at least one digit
    if not any(c.isdigit() for c in password):
        password = password[:-1] + secrets.choice(string.digits)
    return password


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')


def generate_username(full_name: str) -> str:
    """Generate username from full name"""
    # Clean and normalize the name
    base_name = re.sub(r'[^a-zA-Z0-9\s]', '', full_name.lower().strip())
    base_name = base_name.replace(' ', '.')
    if not base_name:
        base_name = "student"
    # Add random suffix for uniqueness
    random_suffix = ''.join(secrets.choice(string.digits) for _ in range(4))
    return f"{base_name}.{random_suffix}"


def validate_email(email: str) -> bool:
    """Simple email validation"""
    if not email:
        return True  # Email is optional
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_grade(grade: str) -> bool:
    """Validate grade is in allowed list"""
    return str(grade).strip() in VALID_GRADES


def validate_section(section: str, valid_sections: List[str]) -> bool:
    """Validate section is in allowed list"""
    if not section:
        return True  # Section is optional
    return section.strip().upper() in [s.upper() for s in valid_sections]


def parse_list_field(value: str) -> List[str]:
    """Parse comma-separated string into list"""
    if not value or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(',') if item.strip()]


async def parse_upload_file(file: UploadFile) -> pd.DataFrame:
    """Parse uploaded CSV or Excel file into DataFrame"""
    content = await file.read()
    file_name = file.filename.lower()
    
    try:
        if file_name.endswith('.csv'):
            # Try different encodings for CSV
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
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Replace NaN with empty strings
        df = df.fillna('')
        
        return df
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        raise ValueError(f"Failed to parse file: {str(e)}")


@router.get("/students/bulk/template")
@limiter.limit("30/minute")
async def download_bulk_template(
    request: Request,
    format: str = Query("xlsx", description="Template format: xlsx or csv"),
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database)
):
    """
    Download a template file for bulk student upload.
    Supports CSV and Excel formats.
    """
    # Fetch school settings for dynamic dropdowns
    try:
        tenant_db = await get_tenant_db_or_403(db, current_user)
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
        settings_doc = await tenant_db["school_settings"].find_one({"admin_id": admin_id})
        
        # Get valid sections from settings
        valid_sections = settings_doc.get("sections", ["A", "B", "C", "D", "E", "F"]) if settings_doc else ["A", "B", "C", "D", "E", "F"]
        valid_classes = settings_doc.get("classes", VALID_GRADES) if settings_doc else VALID_GRADES
        valid_streams = settings_doc.get("streams", ["science", "commerce", "arts", "other"]) if settings_doc else ["science", "commerce", "arts", "other"]
        valid_plan_types = settings_doc.get("plan_types", ["CBSE", "JEE", "NEET", "CUET"]) if settings_doc else ["CBSE", "JEE", "NEET", "CUET"]
        valid_subjects = settings_doc.get("subjects", ["Physics", "Chemistry", "Mathematics", "Biology"]) if settings_doc else ["Physics", "Chemistry", "Mathematics", "Biology"]
    except Exception as e:
        logger.warning(f"Could not fetch school settings: {e}")
        valid_sections = ["A", "B", "C", "D", "E", "F"]
        valid_classes = VALID_GRADES
        valid_streams = ["science", "commerce", "arts", "other"]
        valid_plan_types = ["CBSE", "JEE", "NEET", "CUET"]
        valid_subjects = ["Physics", "Chemistry", "Mathematics", "Biology"]
    
    # Create sample data with 3 example rows
    sample_data = [
        {
            "full_name": "Rahul Sharma",
            "grade": "10",
            "section": "A",
            "email": "rahul@example.com",
            "phone": "9876543210",
            "stream": "science",
            "gender": "male",
            "date_of_birth": "2008-05-15",
            "school": "",
            "location": "Mumbai",
            "plan_types": "JEE,NEET",
            "subjects": "Physics,Chemistry,Mathematics"
        },
        {
            "full_name": "Priya Patel",
            "grade": "11",
            "section": "B",
            "email": "priya@example.com",
            "phone": "9876543211",
            "stream": "science",
            "gender": "female",
            "date_of_birth": "2007-03-20",
            "school": "",
            "location": "Delhi",
            "plan_types": "JEE",
            "subjects": "Physics,Chemistry,Mathematics"
        },
        {
            "full_name": "Amit Kumar",
            "grade": "12",
            "section": "A",
            "email": "",
            "phone": "9876543212",
            "stream": "commerce",
            "gender": "male",
            "date_of_birth": "2006-11-10",
            "school": "",
            "location": "Bangalore",
            "plan_types": "",
            "subjects": ""
        }
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Create instructions DataFrame
    instructions_data = [
        {"Column": "full_name", "Required": "YES", "Description": "Student's full name (2-100 characters)"},
        {"Column": "grade", "Required": "YES", "Description": f"Grade/Class. Allowed values: {', '.join(valid_classes)}"},
        {"Column": "section", "Required": "NO", "Description": f"Section. Allowed values: {', '.join(valid_sections)}"},
        {"Column": "email", "Required": "NO", "Description": "Email address (must be unique within your institute)"},
        {"Column": "phone", "Required": "NO", "Description": "Phone number"},
        {"Column": "stream", "Required": "NO", "Description": f"Stream. Allowed values: {', '.join(valid_streams)}"},
        {"Column": "gender", "Required": "NO", "Description": "Gender: male, female, other"},
        {"Column": "date_of_birth", "Required": "NO", "Description": "Date of birth in YYYY-MM-DD format (e.g., 2008-05-15)"},
        {"Column": "school", "Required": "NO", "Description": "School name"},
        {"Column": "location", "Required": "NO", "Description": "City/Location"},
        {"Column": "plan_types", "Required": "NO", "Description": f"Comma-separated plan types. Allowed: {', '.join(valid_plan_types)}"},
        {"Column": "subjects", "Required": "NO", "Description": f"Comma-separated subjects. Allowed: {', '.join(valid_subjects)}"},
    ]
    instructions_df = pd.DataFrame(instructions_data)
    
    if format.lower() == "csv":
        # For CSV, just return the main data with headers
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=student_bulk_upload_template.csv"}
        )
    else:
        # For Excel, create multiple sheets with instructions
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write main data sheet
            df.to_excel(writer, sheet_name='Students', index=False)
            
            # Write instructions sheet
            instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
            
            # Get workbook and worksheet objects for formatting
            workbook = writer.book
            students_sheet = writer.sheets['Students']
            instructions_sheet = writer.sheets['Instructions']
            
            # Format header row
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4A90D9',
                'font_color': 'white',
                'border': 1
            })
            
            # Apply header format to Students sheet
            for col_num, value in enumerate(df.columns.values):
                students_sheet.write(0, col_num, value, header_format)
                students_sheet.set_column(col_num, col_num, 20)
            
            # Apply header format to Instructions sheet
            for col_num, value in enumerate(instructions_df.columns.values):
                instructions_sheet.write(0, col_num, value, header_format)
            
            instructions_sheet.set_column(0, 0, 15)  # Column
            instructions_sheet.set_column(1, 1, 10)  # Required
            instructions_sheet.set_column(2, 2, 60)  # Description
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=student_bulk_upload_template.xlsx"}
        )


@router.post("/students/bulk/preview", response_model=BulkPreviewResponse)
@limiter.limit("20/minute")
async def preview_bulk_upload(
    request: Request,
    file: UploadFile = File(...),
    default_grade: Optional[str] = Query(None, description="Default grade for rows without grade"),
    default_section: Optional[str] = Query(None, description="Default section for rows without section"),
    default_stream: Optional[str] = Query(None, description="Default stream for rows without stream"),
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database)
):
    """
    Preview and validate a bulk upload file before importing.
    Parses the file, validates each row, and returns a summary with errors.
    """
    # Validate file type
    file_name = file.filename.lower()
    if not file_name.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload CSV or Excel (.xlsx, .xls) file."
        )
    
    # Check file size (max 5MB)
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 5MB."
        )
    await file.seek(0)  # Reset file pointer
    
    # Parse file
    try:
        df = await parse_upload_file(file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    # Check row limit
    if len(df) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many rows ({len(df)}). Maximum allowed is 500 students per upload."
        )
    
    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty. Please add student data."
        )
    
    # Get school settings
    try:
        tenant_db = await get_tenant_db_or_403(db, current_user)
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
        settings_doc = await tenant_db["school_settings"].find_one({"admin_id": admin_id})
        
        valid_sections = settings_doc.get("sections", ["A", "B", "C", "D", "E", "F"]) if settings_doc else ["A", "B", "C", "D", "E", "F"]
    except Exception:
        valid_sections = ["A", "B", "C", "D", "E", "F"]
    
    # Get existing emails for this admin
    try:
        existing_students = await db.mongo_find("students", {"admin_id": admin_id}, projection={"email": 1, "username_lower": 1})
        existing_emails = {s.get("email", "").lower() for s in existing_students if s.get("email")}
        existing_usernames = {s.get("username_lower", "").lower() for s in existing_students if s.get("username_lower")}
    except Exception:
        existing_emails = set()
        existing_usernames = set()
    
    errors: List[BulkUploadError] = []
    warnings: List[BulkUploadError] = []
    preview_data: List[BulkPreviewStudent] = []
    duplicate_emails: List[str] = []
    duplicate_in_file: List[str] = []
    
    # Track emails within the file for duplicate detection
    file_emails: Dict[str, int] = {}
    file_names: Dict[str, int] = {}
    
    valid_count = 0
    
    for idx, row in df.iterrows():
        row_num = idx + 2  # Excel/CSV row number (1-indexed + header)
        row_valid = True
        
        # Get values with defaults
        full_name = str(row.get('full_name', '')).strip()
        grade = str(row.get('grade', default_grade or '')).strip()
        section = str(row.get('section', default_section or '')).strip().upper() if row.get('section') or default_section else ''
        email = str(row.get('email', '')).strip().lower() if row.get('email') else ''
        phone = str(row.get('phone', '')).strip()
        stream = str(row.get('stream', default_stream or '')).strip().lower() if row.get('stream') or default_stream else ''
        gender = str(row.get('gender', '')).strip().lower() if row.get('gender') else ''
        date_of_birth = str(row.get('date_of_birth', '')).strip() if row.get('date_of_birth') else ''
        
        # Validate required fields
        if not full_name or len(full_name) < 2:
            errors.append(BulkUploadError(row=row_num, field="full_name", value=full_name, message="Full name is required (min 2 characters)"))
            row_valid = False
        elif len(full_name) > 100:
            errors.append(BulkUploadError(row=row_num, field="full_name", value=full_name[:50]+"...", message="Full name too long (max 100 characters)"))
            row_valid = False
        
        if not grade:
            errors.append(BulkUploadError(row=row_num, field="grade", value=grade, message="Grade is required"))
            row_valid = False
        elif not validate_grade(grade):
            errors.append(BulkUploadError(row=row_num, field="grade", value=grade, message=f"Invalid grade. Must be one of: {', '.join(VALID_GRADES)}"))
            row_valid = False
        
        # Validate optional fields
        if section and not validate_section(section, valid_sections):
            warnings.append(BulkUploadError(row=row_num, field="section", value=section, message=f"Invalid section. Allowed: {', '.join(valid_sections)}"))
        
        if email:
            if not validate_email(email):
                errors.append(BulkUploadError(row=row_num, field="email", value=email, message="Invalid email format"))
                row_valid = False
            elif email in existing_emails:
                errors.append(BulkUploadError(row=row_num, field="email", value=email, message="Email already exists in your organization"))
                duplicate_emails.append(email)
                row_valid = False
            elif email in file_emails:
                errors.append(BulkUploadError(row=row_num, field="email", value=email, message=f"Duplicate email in file (also in row {file_emails[email]})"))
                duplicate_in_file.append(email)
                row_valid = False
            else:
                file_emails[email] = row_num
        
        if stream and stream not in ['science', 'commerce', 'arts', 'other']:
            warnings.append(BulkUploadError(row=row_num, field="stream", value=stream, message="Invalid stream. Will default to empty."))
        
        if gender and gender not in ['male', 'female', 'other']:
            warnings.append(BulkUploadError(row=row_num, field="gender", value=gender, message="Invalid gender. Will default to empty."))
        
        # Validate date format
        if date_of_birth:
            try:
                datetime.strptime(date_of_birth, '%Y-%m-%d')
            except ValueError:
                warnings.append(BulkUploadError(row=row_num, field="date_of_birth", value=date_of_birth, message="Invalid date format. Use YYYY-MM-DD."))
        
        if row_valid:
            valid_count += 1
            # Generate preview username
            auto_username = generate_username(full_name)
            
            # Only show first 20 valid rows in preview
            if len(preview_data) < 20:
                preview_data.append(BulkPreviewStudent(
                    row=row_num,
                    full_name=full_name,
                    grade=grade,
                    section=section if section else None,
                    email=email if email else None,
                    phone=phone if phone else None,
                    stream=stream if stream else None,
                    auto_username=auto_username
                ))
    
    file_type = "Excel" if file_name.endswith(('.xlsx', '.xls')) else "CSV"
    
    return BulkPreviewResponse(
        success=len(errors) == 0,
        file_name=file.filename,
        file_type=file_type,
        total_rows=len(df),
        valid_rows=valid_count,
        error_rows=len(df) - valid_count,
        errors=errors,
        warnings=warnings,
        preview_data=preview_data,
        duplicate_emails=list(set(duplicate_emails)),
        duplicate_in_file=list(set(duplicate_in_file))
    )


@router.post("/students/bulk/import", response_model=BulkImportResponse)
@limiter.limit("10/minute")
async def import_bulk_students(
    request: Request,
    file: UploadFile = File(...),
    default_grade: Optional[str] = Query(None, description="Default grade for rows without grade"),
    default_section: Optional[str] = Query(None, description="Default section for rows without section"),
    default_stream: Optional[str] = Query(None, description="Default stream for rows without stream"),
    skip_errors: bool = Query(True, description="Skip rows with errors and import valid ones"),
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students")),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Import students from a bulk upload file.
    Creates student accounts with auto-generated usernames and passwords.
    """
    # Validate file type
    file_name = file.filename.lower()
    if not file_name.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload CSV or Excel (.xlsx, .xls) file."
        )
    
    # Check file size
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 5MB."
        )
    await file.seek(0)
    
    # Parse file
    try:
        df = await parse_upload_file(file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    if len(df) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many rows ({len(df)}). Maximum allowed is 500 students per upload."
        )
    
    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty. Please add student data."
        )
    
    # Get tenant info
    tenant_db = await get_tenant_db_or_403(db, current_user)
    admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
    
    # Get school settings
    try:
        settings_doc = await tenant_db["school_settings"].find_one({"admin_id": admin_id})
        valid_sections = settings_doc.get("sections", ["A", "B", "C", "D", "E", "F"]) if settings_doc else ["A", "B", "C", "D", "E", "F"]
    except Exception:
        valid_sections = ["A", "B", "C", "D", "E", "F"]
    
    # Get existing emails and usernames
    existing_students = await db.mongo_find("students", {"admin_id": admin_id}, projection={"email": 1, "username_lower": 1})
    existing_emails = {s.get("email", "").lower() for s in existing_students if s.get("email")}
    existing_usernames = {s.get("username_lower", "").lower() for s in existing_students if s.get("username_lower")}
    
    errors: List[BulkUploadError] = []
    credentials: List[StudentCredential] = []
    students_to_insert: List[Dict[str, Any]] = []
    
    # Track used usernames and emails in this import
    used_emails: set = set()
    used_usernames: set = set(existing_usernames)
    
    for idx, row in df.iterrows():
        row_num = idx + 2
        row_valid = True
        
        # Get values
        full_name = str(row.get('full_name', '')).strip()
        grade = str(row.get('grade', default_grade or '')).strip()
        section = str(row.get('section', default_section or '')).strip().upper() if row.get('section') or default_section else ''
        email = str(row.get('email', '')).strip().lower() if row.get('email') else ''
        phone = str(row.get('phone', '')).strip()
        stream = str(row.get('stream', default_stream or '')).strip().lower() if row.get('stream') or default_stream else ''
        gender = str(row.get('gender', '')).strip().lower() if row.get('gender') else ''
        date_of_birth = str(row.get('date_of_birth', '')).strip() if row.get('date_of_birth') else ''
        school = str(row.get('school', '')).strip() if row.get('school') else ''
        location = str(row.get('location', '')).strip() if row.get('location') else ''
        plan_types_str = str(row.get('plan_types', '')).strip() if row.get('plan_types') else ''
        subjects_str = str(row.get('subjects', '')).strip() if row.get('subjects') else ''
        
        # Validate required fields
        if not full_name or len(full_name) < 2 or len(full_name) > 100:
            errors.append(BulkUploadError(row=row_num, field="full_name", message="Invalid full name"))
            row_valid = False
        
        if not grade or not validate_grade(grade):
            errors.append(BulkUploadError(row=row_num, field="grade", message=f"Invalid grade: {grade}"))
            row_valid = False
        
        if email:
            if not validate_email(email):
                errors.append(BulkUploadError(row=row_num, field="email", message=f"Invalid email: {email}"))
                row_valid = False
            elif email in existing_emails or email in used_emails:
                errors.append(BulkUploadError(row=row_num, field="email", message=f"Duplicate email: {email}"))
                row_valid = False
        
        if not row_valid:
            if not skip_errors:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Validation error in row {row_num}. Use skip_errors=true to skip invalid rows."
                )
            continue
        
        # Generate unique username
        base_username = generate_username(full_name)
        username = base_username
        counter = 1
        while username.lower() in used_usernames:
            username = f"{base_username.rsplit('.', 1)[0]}.{counter}"
            counter += 1
        
        used_usernames.add(username.lower())
        if email:
            used_emails.add(email)
        
        # Generate password
        plain_password = generate_secure_password()
        password_hash = hash_password(plain_password)
        
        # Generate student_id
        import time
        student_id = f"STU_{username}_{int(time.time() * 1000) % 1000000}"
        
        # Parse list fields - Allow for ALL students (removed stream == 'science' check)
        plan_types = parse_list_field(plan_types_str)
        subjects = parse_list_field(subjects_str)
        
        # Auto-assign from settings if missing
        # If plan_types is empty, assign ALL valid plan_types from settings
        if not plan_types and settings_doc:
            settings_plan_types = settings_doc.get("plan_types", [])
            if settings_plan_types:
                plan_types = settings_plan_types
                
        # If subjects is empty, assign ALL valid subjects from settings
        if not subjects and settings_doc:
            settings_subjects = settings_doc.get("subjects", [])
            if settings_subjects:
                subjects = settings_subjects
        
        # Normalize fields
        if stream and stream not in ['science', 'commerce', 'arts', 'other']:
            stream = None # Invalid stream becomes None
        elif not stream:
            stream = None # Empty stream becomes None
            
        if gender and gender not in ['male', 'female', 'other']:
            gender = ''
        if section and section not in [s.upper() for s in valid_sections]:
            section = ''
        
        # Validate date format
        if date_of_birth:
            try:
                datetime.strptime(date_of_birth, '%Y-%m-%d')
            except ValueError:
                date_of_birth = ''
        
        # Create student document
        student_doc = {
            "admin_id": admin_id,
            "student_id": student_id,
            "username": username,
            "username_lower": username.lower(),
            "full_name": full_name,
            "name": full_name,  # Legacy field
            "email": email if email else None,
            "phone": phone if phone else None,
            "grade": grade,
            "section": section if section else None,
            "stream": None, # Force stream to None as per new policy
            "gender": gender if gender else None,
            "date_of_birth": date_of_birth if date_of_birth else None,
            "school": school if school else None,
            "location": location if location else None,
            "plan_types": plan_types if plan_types else None,
            "subjects": subjects if subjects else None,
            "password_hash": password_hash,
            "is_active": True,
            "requires_password_change": True,
            "created_at": datetime.utcnow(),
            "created_by": current_user["user_id"],
            "created_via": "bulk_upload"
        }
        
        students_to_insert.append(student_doc)
        credentials.append(StudentCredential(
            full_name=full_name,
            username=username,
            password=plain_password,
            grade=grade,
            section=section if section else None
        ))
    
    # Bulk insert
    total_imported = 0
    if students_to_insert:
        try:
            result = await db.mongo_db["students"].insert_many(students_to_insert, ordered=False)
            total_imported = len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Bulk insert error: {str(e)}")
            # Try to extract how many succeeded
            if hasattr(e, 'details') and 'nInserted' in e.details:
                total_imported = e.details['nInserted']
            errors.append(BulkUploadError(row=0, field="database", message=f"Database error: {str(e)[:100]}"))
    
    # Invalidate cache
    try:
        await cache.clear_pattern("students:*", "admin")
        await cache.delete("dashboard_stats", "admin")
    except Exception:
        pass
    
    total_skipped = len(df) - total_imported
    
    return BulkImportResponse(
        success=total_imported > 0,
        message=f"Successfully imported {total_imported} students. {total_skipped} rows skipped.",
        total_imported=total_imported,
        total_skipped=total_skipped,
        credentials=credentials[:total_imported],  # Only return credentials for actually imported students
        errors=errors
    )


@router.get("/students/bulk/credentials/download")
@limiter.limit("30/minute")
async def download_credentials(
    request: Request,
    credentials_json: str = Query(..., description="JSON array of credentials"),
    format: str = Query("xlsx", description="Download format: xlsx or csv"),
    current_user: Dict[str, Any] = Depends(require_admin_permission("manage_students"))
):
    """
    Download generated credentials as CSV or Excel file.
    Used after bulk import to save username/password combinations.
    """
    import json
    
    try:
        credentials = json.loads(credentials_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid credentials JSON")
    
    if not credentials:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No credentials to download")
    
    # Create DataFrame
    df = pd.DataFrame(credentials)
    
    # Rename columns for clarity
    df = df.rename(columns={
        'full_name': 'Full Name',
        'username': 'Username',
        'password': 'Password',
        'grade': 'Grade',
        'section': 'Section'
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format.lower() == "csv":
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=student_credentials_{timestamp}.csv"}
        )
    else:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Credentials', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Credentials']
            
            # Format header
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4A90D9',
                'font_color': 'white',
                'border': 1
            })
            
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 20)
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=student_credentials_{timestamp}.xlsx"}
        )
