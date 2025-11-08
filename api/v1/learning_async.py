"""
Learning Mode API - Async Routes
Provides endpoints for accessing Chapter Notes from document management
Integrated with student access control and activity tracking
Serves Admin-uploaded study materials (Document Type: "Chapter Notes")
Designed for 1000+ concurrent users with streaming and caching
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import aiofiles
import aiofiles.os

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


class LearningDocument(BaseModel):
    """Document information for learning mode"""
    document_id: str
    title: str
    subject: str
    standard: str
    course_plan: str
    document_type: str
    difficulty: Optional[str] = None
    file_path: str
    ocr_status: str
    created_at: Optional[datetime] = None


class CourseStructure(BaseModel):
    """Course structure response"""
    standards: List[str]
    subjects: Dict[str, List[str]]


@router.get("/health", tags=["Learning"])
async def health_check(db: DatabaseManager = Depends(get_database)):
    """Health check endpoint"""
    try:
        # Check if documents collection exists
        docs = await db.mongo_find("documents", {"document_type": "Chapter Notes"}, limit=1)
        doc_count = len(docs) if docs else 0
        return {
            "success": True,
            "message": "Learning Mode API is operational",
            "has_documents": doc_count > 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "success": False,
            "message": f"Health check failed: {str(e)}"
        }


@router.get("/structure", tags=["Learning"])
async def get_course_structure(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get the course structure for the logged-in user

    Behavior:
    - Admin users (viewing as student): Returns ALL Chapter Notes documents
    - Student users: Returns only documents matching their profile
    """
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        user_type = current_user.get("user_type", "student")

        # Build query based on user type - always filter by admin_id
        if user_type == "admin":
            # Admin viewing student panel - show Chapter Notes from their organization
            # Note: Chapter Notes don't require OCR processing
            query = {
                "document_type": "Chapter Notes",
                "admin_id": admin_id
            }
        else:
            # Actual student login - filter by profile
            student = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})

            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )

            # Get student's access parameters
            # Note: Student model uses 'grade' (e.g., "12") and 'plan_types' (e.g., ["JEE", "CBSE"])
            student_grade = student.get("grade")  # Maps to "standard" in documents
            student_plan_types = student.get("plan_types", [])  # Maps to "course_plan" in documents
            student_subjects = student.get("subjects", [])

            if not student_grade or not student_subjects:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Student profile incomplete. Please contact admin to set grade and subjects."
                )

            # Query documents with access control
            # Match documents where standard=grade AND course_plan is in student's plan_types
            # Note: Chapter Notes don't require OCR processing, so we don't filter by ocr_status
            query = {
                "document_type": "Chapter Notes",
                "admin_id": admin_id,  # Only show documents from student's admin
                "standard": student_grade,
                "subject": {"$in": student_subjects}
            }

            # If student has plan_types, filter by those as well
            if student_plan_types:
                query["course_plan"] = {"$in": student_plan_types}

        # Get all Chapter Notes documents based on query
        documents = await db.mongo_find("documents", query)

        # Organize by standard and subject
        standards_set = set()
        subjects_by_standard: Dict[str, set] = {}

        for doc in documents:
            std = doc.get("standard")
            subj = doc.get("subject")

            if std and subj:
                standards_set.add(std)
                if std not in subjects_by_standard:
                    subjects_by_standard[std] = set()
                subjects_by_standard[std].add(subj)

        # Convert sets to sorted lists
        standards = sorted(list(standards_set))
        subjects_dict = {
            std: sorted(list(subjects))
            for std, subjects in subjects_by_standard.items()
        }

        return {
            "success": True,
            "data": {
                "standards": standards,
                "subjects": subjects_dict
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get course structure: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve course structure: {str(e)}"
        )


@router.get("/chapters/{standard}/{subject}", tags=["Learning"])
async def get_chapters(
    standard: str,
    subject: str,
    course_plan: str = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all Chapter Notes documents for a specific standard and subject

    Behavior:
    - Admin users: Returns ALL documents for the requested standard/subject
    - Student users: Returns only documents matching their profile (with access verification)

    Args:
        standard: Standard/Grade (e.g., "11", "12")
        subject: Subject name (e.g., "Physics", "Chemistry", "Mathematics")
    """
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        user_type = current_user.get("user_type", "student")

        # Build query based on user type - always filter by admin_id
        if user_type == "admin":
            # Admin viewing student panel - show documents from their organization for this standard/subject
            query = {
                "document_type": "Chapter Notes",
                "admin_id": admin_id,
                "standard": standard,
                "subject": subject
            }
        else:
            # Actual student login - verify access and filter by profile
            student = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})

            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )

            # Verify student has access to this standard and subject
            # Use 'grade' field instead of 'standard'
            student_grade = student.get("grade")
            student_plan_types = student.get("plan_types", [])
            student_subjects = student.get("subjects", [])

            if student_grade != standard:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. You are enrolled in grade {student_grade}, not {standard}."
                )

            if subject not in student_subjects:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. You are not enrolled in {subject}."
                )

            # Query with access control
            query = {
                "document_type": "Chapter Notes",
                "admin_id": admin_id,  # Only show documents from student's admin
                "standard": standard,
                "subject": subject
            }

            # If course_plan is provided in the request, use it (must be in student's plan_types)
            if course_plan:
                if course_plan not in student_plan_types:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Access denied. You are not enrolled in {course_plan} plan."
                    )
                query["course_plan"] = course_plan
            # Otherwise, if student has plan_types, filter by all of them
            elif student_plan_types:
                query["course_plan"] = {"$in": student_plan_types}

        # Get Chapter Notes documents based on query
        documents = await db.mongo_find("documents", query, sort=[("title", 1)])

        # Convert MongoDB documents to response format
        documents_list = []
        for doc in documents:
            documents_list.append({
                "document_id": str(doc["_id"]),  # MongoDB uses _id as primary key
                "title": doc.get("title"),
                "subject": doc.get("subject"),
                "standard": doc.get("standard"),
                "course_plan": doc.get("course_plan"),
                "document_type": doc.get("document_type"),
                "difficulty": doc.get("difficulty"),
                "file_path": doc.get("file_path"),
                "ocr_status": doc.get("ocr_status"),
                "created_at": doc.get("created_at")
            })

        return {
            "success": True,
            "data": {
                "standard": standard,
                "subject": subject,
                "documents": documents_list,
                "total": len(documents_list)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chapters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chapters: {str(e)}"
        )


@router.get("/document/{document_id}", tags=["Learning"])
async def get_document_metadata(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get document metadata including PDF URL
    Returns document details with secure PDF URL
    """
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Get document from database (document_id is MongoDB's _id as string, filtered by admin_id)
        document = await db.mongo_find_one("documents", {"_id": ObjectId(document_id), "admin_id": admin_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        # Verify document type is Chapter Notes
        if document.get("document_type") != "Chapter Notes":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This document is not a Chapter Notes document"
            )

        # Access control based on user type
        user_type = current_user.get("user_type", "student")

        if user_type == "student":
            # For actual student login, verify access permissions
            student = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})

            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )

            # Verify student has access to this document
            student_grade = student.get("grade")
            student_plan_types = student.get("plan_types", [])
            student_subjects = student.get("subjects", [])

            doc_standard = document.get("standard")
            doc_course_plan = document.get("course_plan")
            doc_subject = document.get("subject")

            # Check if student's grade matches document standard
            if doc_standard != student_grade:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied. This document is not for your grade level."
                )

            # Check if student is enrolled in this subject
            if doc_subject not in student_subjects:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied. You are not enrolled in this subject."
                )

            # Check if document's course plan matches student's plan types
            if student_plan_types and doc_course_plan not in student_plan_types:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. This document is for {doc_course_plan} plan."
                )

        # Return document metadata with PDF URL
        # The PDF URL endpoint will also perform access control
        pdf_url = f"/api/learning/pdf/{document_id}"

        return {
            "success": True,
            "data": {
                "document_id": document_id,
                "title": document.get("title"),
                "subject": document.get("subject"),
                "standard": document.get("standard"),
                "course_plan": document.get("course_plan"),
                "pdf_url": pdf_url
            }
        }

        return {"success": True, "data": chapter}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}"
        )


@router.get("/pdf/{document_id}", tags=["Learning"])
async def get_chapter_pdf(
    document_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Serve the PDF file for a specific chapter with streaming support
    Optimized for 1000+ concurrent users with caching and range requests

    Args:
        document_id: Document ID from documents collection
        request: FastAPI request object for range support
    """
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)

        # Get document from database (document_id is MongoDB's _id as string, filtered by admin_id)
        document = await db.mongo_find_one("documents", {"_id": ObjectId(document_id), "admin_id": admin_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        # Verify document type is Chapter Notes
        if document.get("document_type") != "Chapter Notes":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This document is not a Chapter Notes document"
            )

        # Note: Chapter Notes don't require OCR processing, so we skip the OCR status check

        # Access control based on user type
        user_type = current_user.get("user_type", "student")

        if user_type == "student":
            # For actual student login, verify access permissions
            student = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})

            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )

            # Verify student has access to this document
            student_grade = student.get("grade")
            student_plan_types = student.get("plan_types", [])
            student_subjects = student.get("subjects", [])

            doc_standard = document.get("standard")
            doc_course_plan = document.get("course_plan")
            doc_subject = document.get("subject")

            # Check if student's grade matches document standard
            if doc_standard != student_grade:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied. This document is not for your grade level."
                )

            # Check if student is enrolled in this subject
            if doc_subject not in student_subjects:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied. You are not enrolled in this subject."
                )

            # Check if document's course plan matches student's plan types (if student has plan types)
            if student_plan_types and doc_course_plan not in student_plan_types:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. This document is for {doc_course_plan} plan."
                )
        # else: Admin users can access all documents

        # Get PDF path
        pdf_path = Path(document.get("file_path", ""))

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found for document: {document_id}"
            )

        # Get file stats for headers
        file_stat = await aiofiles.os.stat(pdf_path)
        file_size = file_stat.st_size

        # Production-ready headers for caching and performance
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=31536000, immutable",  # 1 year cache
            "Content-Type": "application/pdf",
            "Content-Disposition": f'inline; filename="{document.get("title", "chapter")}.pdf"',
            "X-Content-Type-Options": "nosniff",
        }

        # Check for range request (for seeking in PDF)
        range_header = request.headers.get("range")

        if range_header:
            # Parse range header (e.g., "bytes=0-1023")
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1

            # Validate range
            if start >= file_size or end >= file_size or start > end:
                raise HTTPException(
                    status_code=416,
                    detail="Requested range not satisfiable"
                )

            chunk_size = end - start + 1

            # Stream partial content
            async def stream_range():
                async with aiofiles.open(pdf_path, mode='rb') as f:
                    await f.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        read_size = min(65536, remaining)  # 64KB chunks
                        data = await f.read(read_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            headers["Content-Length"] = str(chunk_size)

            return StreamingResponse(
                stream_range(),
                status_code=206,
                headers=headers,
                media_type="application/pdf"
            )

        # Full file streaming (for initial load)
        async def stream_full():
            async with aiofiles.open(pdf_path, mode='rb') as f:
                while chunk := await f.read(65536):  # 64KB chunks
                    yield chunk

        headers["Content-Length"] = str(file_size)

        # Log viewing activity in background
        try:
            await db.mongo_insert_one("student_activity_log", {
                "student_id": ObjectId(current_user["user_id"]),
                "action": "chapter_viewed",
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "document_id": document_id,
                    "title": document.get("title"),
                    "subject": document.get("subject"),
                    "standard": document.get("standard"),
                    "course_plan": document.get("course_plan")
                }
            })
        except Exception as log_error:
            logger.error(f"Failed to log chapter view activity: {str(log_error)}")

        return StreamingResponse(
            stream_full(),
            headers=headers,
            media_type="application/pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve PDF: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve PDF file: {str(e)}"
        )
